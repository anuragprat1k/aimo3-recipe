#!/usr/bin/env python3
"""
Evaluate a checkpoint on all available benchmarks.

Usage:
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo --num-samples 16
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo --benchmarks math,aime

    # For LoRA adapter checkpoints (auto-detected and merged):
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math/checkpoint-142

    # Parallel evaluation across GPUs (8 GPUs, 3 benchmarks = 3 parallel processes):
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math/checkpoint-142 \\
        --benchmarks aime,math500,gsm8k --parallel --gpus-per-benchmark 2
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from aimo3_recipe.evaluation.evaluate import MathEvaluator, EvalConfig, verify_answers_parallel
from aimo3_recipe.evaluation.benchmarks import load_benchmark, get_benchmark, ALL_BENCHMARKS


def get_available_gpus() -> list[int]:
    """Get list of available GPU indices."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(x) for x in cuda_visible.split(",")]

    # Try to detect GPUs
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return [0]  # Default to single GPU


def is_lora_adapter(model_path: str) -> bool:
    """Check if the model path contains a LoRA adapter."""
    adapter_config = Path(model_path) / "adapter_config.json"
    return adapter_config.exists()


def merge_lora_adapter(adapter_path: str, output_path: str | None = None) -> str:
    """
    Merge a LoRA adapter with its base model.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        output_path: Where to save merged model (default: temp directory)

    Returns:
        Path to the merged model
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)

    # Load adapter config to find base model
    with open(adapter_path / "adapter_config.json") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("Could not find base_model_name_or_path in adapter_config.json")

    print(f"  Base model: {base_model_name}")
    print(f"  Loading and merging adapter...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged_model = model.merge_and_unload()

    # Determine output path
    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="merged_model_")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"  Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)

    # Save tokenizer (use from adapter if available, else from base)
    tokenizer_path = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Clean up GPU memory
    del model, merged_model
    torch.cuda.empty_cache()

    return str(output_path)


def run_single_benchmark_process(
    benchmark_name: str,
    model_path: str,
    output_dir: str,
    gpu_ids: list[int],
    num_samples: int = 1,
    max_samples: int | None = None,
    use_vllm: bool = True,
    num_workers: int = 0,
) -> dict:
    """
    Run evaluation on a single benchmark in a subprocess with specific GPUs.

    This function is designed to be called in a separate process.
    """
    # Set GPU visibility for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Log GPU assignment for this benchmark
    gpu_str = ",".join(str(g) for g in gpu_ids)
    print(f"[GPU {gpu_str}] Starting evaluation: {benchmark_name}")

    # Import here to avoid CUDA initialization in main process
    from aimo3_recipe.evaluation.evaluate import MathEvaluator, EvalConfig
    from aimo3_recipe.evaluation.benchmarks import load_benchmark, get_benchmark

    try:
        benchmark_config = get_benchmark(benchmark_name)
        dataset = load_benchmark(benchmark_config, max_samples)
        print(f"[GPU {gpu_str}] {benchmark_name}: Loaded {len(dataset)} samples")

        config = EvalConfig(
            model_name_or_path=model_path,
            use_vllm=use_vllm,
            num_samples=num_samples,
            max_samples=max_samples,
            output_dir=output_dir,
            num_workers=num_workers,
        )
        evaluator = MathEvaluator(config)
        results = evaluator.evaluate(dataset)

        print(f"[GPU {gpu_str}] {benchmark_name}: Completed - {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

        return {
            "benchmark": benchmark_name,
            "gpu_ids": gpu_ids,
            "success": True,
            "results": {
                "accuracy": results["accuracy"],
                "correct": results["correct"],
                "total": results["total"],
            },
            "full_results": results,
        }
    except Exception as e:
        print(f"[GPU {gpu_str}] {benchmark_name}: FAILED - {e}")
        return {
            "benchmark": benchmark_name,
            "gpu_ids": gpu_ids,
            "success": False,
            "error": str(e),
        }


def evaluate_parallel(
    benchmark_names: list[str],
    model_path: str,
    output_dir: str,
    gpus_per_benchmark: int = 1,
    num_samples: int = 1,
    max_samples: int | None = None,
    use_vllm: bool = True,
    num_workers: int = 0,
) -> dict:
    """
    Evaluate multiple benchmarks in parallel, each on its own GPU(s).

    Args:
        benchmark_names: List of benchmark names to evaluate
        model_path: Path to model checkpoint
        output_dir: Output directory for results
        gpus_per_benchmark: Number of GPUs to allocate per benchmark
        num_samples: Samples for self-consistency
        max_samples: Max samples per benchmark
        use_vllm: Whether to use vLLM
        num_workers: Workers for answer verification

    Returns:
        Dict mapping benchmark name -> results
    """
    available_gpus = get_available_gpus()
    num_gpus = len(available_gpus)
    max_parallel = num_gpus // gpus_per_benchmark

    if max_parallel == 0:
        raise ValueError(
            f"Not enough GPUs. Have {num_gpus}, need {gpus_per_benchmark} per benchmark."
        )

    print(f"Available GPUs: {available_gpus}")
    print(f"GPUs per benchmark: {gpus_per_benchmark}")
    print(f"Max parallel evaluations: {max_parallel}")
    print(f"Benchmarks to evaluate: {benchmark_names}")

    # Assign GPUs to benchmarks
    all_results = {}
    benchmark_queue = list(benchmark_names)

    # Use 'spawn' context for CUDA compatibility (fork doesn't work with CUDA)
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as executor:
        futures = {}
        gpu_assignments = {}

        # Submit initial batch
        for i, benchmark in enumerate(benchmark_queue[:max_parallel]):
            start_gpu = i * gpus_per_benchmark
            gpu_ids = available_gpus[start_gpu:start_gpu + gpus_per_benchmark]
            gpu_assignments[benchmark] = gpu_ids

            print(f"  Starting {benchmark} on GPUs {gpu_ids}")
            future = executor.submit(
                run_single_benchmark_process,
                benchmark,
                model_path,
                output_dir,
                gpu_ids,
                num_samples,
                max_samples,
                use_vllm,
                num_workers,
            )
            futures[future] = benchmark

        remaining = benchmark_queue[max_parallel:]

        # Process results and submit remaining benchmarks
        for future in as_completed(futures):
            benchmark = futures[future]
            try:
                result = future.result()
                all_results[benchmark] = result

                if result["success"]:
                    r = result["results"]
                    print(f"  Completed {benchmark}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

                    # Save per-benchmark results
                    benchmark_file = Path(output_dir) / f"{benchmark}_results.json"
                    with open(benchmark_file, "w") as f:
                        json.dump(result.get("full_results", result["results"]), f, indent=2)
                else:
                    print(f"  Failed {benchmark}: {result['error']}")

            except Exception as e:
                print(f"  Error {benchmark}: {e}")
                all_results[benchmark] = {"success": False, "error": str(e)}

            # Submit next benchmark if any remaining
            if remaining:
                next_benchmark = remaining.pop(0)
                gpu_ids = gpu_assignments[benchmark]  # Reuse freed GPUs

                print(f"  Starting {next_benchmark} on GPUs {gpu_ids}")
                new_future = executor.submit(
                    run_single_benchmark_process,
                    next_benchmark,
                    model_path,
                    output_dir,
                    gpu_ids,
                    num_samples,
                    max_samples,
                    use_vllm,
                    num_workers,
                )
                futures[new_future] = next_benchmark
                gpu_assignments[next_benchmark] = gpu_ids

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on all benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path (supports LoRA adapters)")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--merge-output", type=str, default=None, help="Where to save merged model (for LoRA adapters)")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples for self-consistency (1=greedy)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated list of benchmarks (default: all)")
    parser.add_argument("--use-vllm", action="store_true", default=True, help="Use vLLM for inference")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for parallel answer verification (0=auto, -1=disable)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks in parallel across GPUs (recommended for multi-GPU setups)",
    )
    parser.add_argument(
        "--gpus-per-benchmark",
        type=int,
        default=1,
        help="Number of GPUs to allocate per benchmark when using --parallel (default: 1)",
    )
    args = parser.parse_args()

    use_vllm = args.use_vllm and not args.no_vllm
    model_path = args.model

    # Check if this is a LoRA adapter and merge if needed
    if is_lora_adapter(model_path):
        print(f"Detected LoRA adapter at: {model_path}")
        model_path = merge_lora_adapter(model_path, args.merge_output)
        print(f"Using merged model: {model_path}")

    # Determine which benchmarks to run
    if args.benchmarks:
        benchmark_names = [b.strip() for b in args.benchmarks.split(",")]
        for name in benchmark_names:
            if name not in ALL_BENCHMARKS:
                print(f"Unknown benchmark: {name}")
                print(f"Available: {list(ALL_BENCHMARKS.keys())}")
                return
    else:
        benchmark_names = list(ALL_BENCHMARKS.keys())

    print(f"Model: {model_path}")
    print(f"Benchmarks: {benchmark_names}")
    print(f"Self-consistency samples: {args.num_samples}")
    print(f"Using vLLM: {use_vllm}")
    print(f"Parallel: {args.parallel}")
    if args.parallel:
        print(f"GPUs per benchmark: {args.gpus_per_benchmark}")
    print("=" * 60)

    # Print dataset sizes before evaluation
    print("\nDataset sizes:")
    for name in benchmark_names:
        try:
            benchmark_config = get_benchmark(name)
            dataset = load_benchmark(benchmark_config, args.max_samples)
            print(f"  {name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  {name}: Error loading - {e}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        # Parallel evaluation across GPUs
        print("\nRunning parallel evaluation across GPUs...")
        parallel_results = evaluate_parallel(
            benchmark_names=benchmark_names,
            model_path=model_path,
            output_dir=str(output_dir),
            gpus_per_benchmark=args.gpus_per_benchmark,
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            use_vllm=use_vllm,
            num_workers=args.num_workers,
        )

        # Convert parallel results format to standard format
        all_results = {}
        for name, result in parallel_results.items():
            if result.get("success"):
                all_results[name] = result.get("full_results", result["results"])
            else:
                all_results[name] = {"error": result.get("error", "Unknown error")}
    else:
        # Sequential evaluation (single GPU)
        print("\nRunning sequential evaluation...")
        config = EvalConfig(
            model_name_or_path=model_path,
            use_vllm=use_vllm,
            num_samples=args.num_samples,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
        evaluator = MathEvaluator(config)

        all_results = {}
        for name in benchmark_names:
            print(f"\nEvaluating on {name}...")
            try:
                benchmark_config = get_benchmark(name)
                dataset = load_benchmark(benchmark_config, args.max_samples)
                print(f"  Loaded {len(dataset)} problems")

                results = evaluator.evaluate(dataset)
                all_results[name] = results
                print(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

                # Save per-benchmark results
                benchmark_file = output_dir / f"{name}_results.json"
                with open(benchmark_file, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"  Error: {e}")
                all_results[name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)

    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<15} {'ERROR':>10}")
        else:
            acc = f"{result['accuracy']:.2%}"
            print(f"{name:<15} {acc:>10} {result['correct']:>10} {result['total']:>10}")

    # Save summary
    summary = {
        "model": args.model,
        "merged_model": model_path if model_path != args.model else None,
        "num_samples": args.num_samples,
        "parallel": args.parallel,
        "gpus_per_benchmark": args.gpus_per_benchmark if args.parallel else None,
        "timestamp": datetime.now().isoformat(),
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in all_results.items()
            if "error" not in v
        },
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
