#!/usr/bin/env python3
"""
Evaluate a checkpoint on all available benchmarks.

Usage:
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo --num-samples 16
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math_grpo --benchmarks math,aime

    # For LoRA adapter checkpoints (auto-detected and merged):
    python -m aimo3_recipe.evaluation.evaluate_all --model ./outputs/rl_math/checkpoint-142
"""

import argparse
import json
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from aimo3_recipe.evaluation.evaluate import MathEvaluator, EvalConfig, verify_answers_parallel
from aimo3_recipe.evaluation.benchmarks import load_benchmark, get_benchmark, ALL_BENCHMARKS


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


def evaluate_combined(
    evaluator: MathEvaluator,
    benchmark_datasets: dict,
    num_workers: int = 0,
) -> dict:
    """
    Evaluate all benchmarks in a single combined pass for maximum throughput.

    Combines all problems, runs inference once, then splits results back.

    Args:
        evaluator: Initialized MathEvaluator with model loaded
        benchmark_datasets: Dict mapping benchmark name -> dataset
        num_workers: Workers for parallel answer verification

    Returns:
        Dict mapping benchmark name -> results
    """
    # Combine all problems with tracking info
    all_problems = []
    all_solutions = []
    problem_to_benchmark = []  # Track which benchmark each problem belongs to

    for name, dataset in benchmark_datasets.items():
        for i in range(len(dataset)):
            all_problems.append(dataset[i]["problem"])
            all_solutions.append(dataset[i]["solution"])
            problem_to_benchmark.append(name)

    total_problems = len(all_problems)
    print(f"\nCombined {total_problems} problems from {len(benchmark_datasets)} benchmarks")

    # Ensure model is loaded
    if evaluator.model is None:
        evaluator.setup_model()

    # Format all prompts
    all_prompts = [evaluator.format_prompt(p) for p in all_problems]

    # Generate all responses in batches
    print("Running inference...")
    batch_size = 64 if evaluator.config.use_vllm else 1
    all_responses = []

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_responses = evaluator.generate(batch_prompts)

        # Apply majority voting if needed
        for responses in batch_responses:
            if evaluator.config.num_samples > 1:
                all_responses.append(evaluator.majority_vote(responses))
            else:
                all_responses.append(responses[0])

    # Verify all answers in parallel
    print("Verifying answers...")
    correctness = verify_answers_parallel(all_responses, all_solutions, num_workers=num_workers)

    # Split results back by benchmark
    results_by_benchmark = {name: {"correct": 0, "total": 0, "predictions": []} for name in benchmark_datasets}

    for idx, (problem, solution, response, is_correct, bench_name) in enumerate(
        zip(all_problems, all_solutions, all_responses, correctness, problem_to_benchmark)
    ):
        results_by_benchmark[bench_name]["total"] += 1
        results_by_benchmark[bench_name]["correct"] += int(is_correct)
        results_by_benchmark[bench_name]["predictions"].append({
            "problem": problem,
            "ground_truth": solution,
            "prediction": response,
            "correct": is_correct,
        })

    # Compute accuracies
    for name in results_by_benchmark:
        r = results_by_benchmark[name]
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0

    return results_by_benchmark


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
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for vLLM inference (default: 64)",
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
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all benchmark datasets first (can be parallelized)
    print("\nLoading datasets...")
    benchmark_datasets = {}

    def load_single_benchmark(name):
        benchmark_config = get_benchmark(name)
        dataset = load_benchmark(benchmark_config, args.max_samples)
        return name, dataset

    with ThreadPoolExecutor(max_workers=len(benchmark_names)) as executor:
        for name, dataset in executor.map(load_single_benchmark, benchmark_names):
            benchmark_datasets[name] = dataset
            print(f"  {name}: {len(dataset)} problems")

    # Initialize evaluator
    config = EvalConfig(
        model_name_or_path=model_path,
        use_vllm=use_vllm,
        num_samples=args.num_samples,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    evaluator = MathEvaluator(config)

    # Run combined evaluation (single pass through all problems)
    try:
        all_results = evaluate_combined(evaluator, benchmark_datasets, args.num_workers)
    except Exception as e:
        print(f"Combined evaluation failed: {e}")
        print("Falling back to sequential evaluation...")
        all_results = {}
        for name, dataset in benchmark_datasets.items():
            print(f"\nEvaluating on {name}...")
            results = evaluator.evaluate(dataset)
            all_results[name] = results

    # Print summary and save results
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

        # Save per-benchmark results
        benchmark_file = output_dir / f"{name}_results.json"
        with open(benchmark_file, "w") as f:
            json.dump(result, f, indent=2)

    # Save summary
    summary = {
        "model": args.model,
        "merged_model": model_path if model_path != args.model else None,
        "num_samples": args.num_samples,
        "timestamp": datetime.now().isoformat(),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in all_results.items()},
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
