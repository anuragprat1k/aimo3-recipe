#!/usr/bin/env python3
"""
Regenerate summary.json from individual benchmark result files.

Useful when evaluation was interrupted or summary is missing/incomplete.

Usage:
    python scripts/regenerate_summary.py --eval-dir eval_results/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Regenerate summary from result files")
    parser.add_argument("--eval-dir", type=str, required=True, help="Evaluation results directory")
    parser.add_argument("--model", type=str, default=None, help="Model path (for metadata)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory not found: {eval_dir}")
        return

    # Find all *_results.json files
    result_files = list(eval_dir.glob("*_results.json"))
    if not result_files:
        print(f"No result files found in {eval_dir}")
        return

    print(f"Found {len(result_files)} result files:")

    all_results = {}
    for result_file in sorted(result_files):
        # Extract benchmark name from filename (e.g., "math500_results.json" -> "math500")
        benchmark_name = result_file.stem.replace("_results", "")

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Check if it's a valid result (has accuracy, correct, total)
            if "accuracy" in data and "correct" in data and "total" in data:
                # Store without predictions to keep summary small
                all_results[benchmark_name] = {
                    k: v for k, v in data.items() if k != "predictions"
                }
                print(f"  {benchmark_name}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
            elif "error" in data:
                print(f"  {benchmark_name}: ERROR - {data['error']}")
            else:
                print(f"  {benchmark_name}: Unknown format, keys: {list(data.keys())}")
        except Exception as e:
            print(f"  {benchmark_name}: Failed to load - {e}")

    if not all_results:
        print("\nNo valid results found!")
        return

    # Create summary
    summary = {
        "model": args.model,
        "regenerated": True,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    # Try to preserve metadata from existing summary if present
    existing_summary = eval_dir / "summary.json"
    if existing_summary.exists():
        try:
            with open(existing_summary) as f:
                old_summary = json.load(f)
            # Preserve model info if not provided
            if args.model is None and "model" in old_summary:
                summary["model"] = old_summary["model"]
            if "merged_model" in old_summary:
                summary["merged_model"] = old_summary["merged_model"]
            if "num_samples" in old_summary:
                summary["num_samples"] = old_summary["num_samples"]
        except Exception:
            pass

    # Save new summary
    summary_file = eval_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary regenerated: {summary_file}")
    print(f"Total benchmarks: {len(all_results)}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 45)
    for name, result in sorted(all_results.items()):
        acc = f"{result['accuracy']:.2%}"
        print(f"{name:<15} {acc:>10} {result['correct']:>10} {result['total']:>10}")


if __name__ == "__main__":
    main()
