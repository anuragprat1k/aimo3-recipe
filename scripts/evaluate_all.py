#!/usr/bin/env python3
"""
Evaluate a checkpoint on all available benchmarks.

Usage:
    python scripts/evaluate_all.py --model ./outputs/rl_math_grpo
    python scripts/evaluate_all.py --model ./outputs/rl_math_grpo --num-samples 16
    python scripts/evaluate_all.py --model ./outputs/rl_math_grpo --benchmarks math,aime
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from aimo3_recipe.evaluation.evaluate import MathEvaluator, EvalConfig
from aimo3_recipe.evaluation.benchmarks import load_benchmark, get_benchmark, ALL_BENCHMARKS


def main():
    parser = argparse.ArgumentParser(description="Evaluate on all benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples for self-consistency (1=greedy)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated list of benchmarks (default: all)")
    parser.add_argument("--use-vllm", action="store_true", default=True, help="Use vLLM for inference")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM")
    args = parser.parse_args()

    use_vllm = args.use_vllm and not args.no_vllm

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

    print(f"Model: {args.model}")
    print(f"Benchmarks: {benchmark_names}")
    print(f"Self-consistency samples: {args.num_samples}")
    print(f"Using vLLM: {use_vllm}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator once (reuses model across benchmarks)
    config = EvalConfig(
        model_name_or_path=args.model,
        use_vllm=use_vllm,
        num_samples=args.num_samples,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
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
