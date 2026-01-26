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

from aimo3_recipe.evaluation.evaluate import MathEvaluator, EvalConfig
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
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator once (reuses model across benchmarks)
    config = EvalConfig(
        model_name_or_path=model_path,
        use_vllm=use_vllm,
        num_samples=args.num_samples,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    evaluator = MathEvaluator(config)

    # Run evaluation on each benchmark
    all_results = {}
    for name in benchmark_names:
        print(f"\nEvaluating on {name}...")
        try:
            benchmark_config = get_benchmark(name)
            dataset = load_benchmark(benchmark_config, args.max_samples)
            print(f"  Loaded {len(dataset)} problems")

            results = evaluator.evaluate(dataset)
            accuracy = results["accuracy"]
            correct = results["correct"]
            total = results["total"]

            all_results[name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
            print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")

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
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
