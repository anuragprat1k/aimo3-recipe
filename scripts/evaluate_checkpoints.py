#!/usr/bin/env python3
"""
Evaluate multiple checkpoints from a training run.

Usage:
    # Evaluate checkpoints 10-120 (step 10) on all benchmarks:
    python scripts/evaluate_checkpoints.py \
        --run-dir outputs/rl_math_qwen3/rl_math_qwen_judge \
        --start 10 --end 120 --step 10

    # Evaluate specific benchmarks:
    python scripts/evaluate_checkpoints.py \
        --run-dir outputs/rl_math_qwen3/rl_math_qwen_judge \
        --start 10 --end 120 --step 10 \
        --benchmarks math500,aime,gsm8k

    # Parallel evaluation across GPUs:
    python scripts/evaluate_checkpoints.py \
        --run-dir outputs/rl_math_qwen3/rl_math_qwen_judge \
        --start 10 --end 120 --step 10 \
        --parallel --gpus-per-benchmark 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_checkpoint_numbers(run_dir: Path, start: int, end: int, step: int) -> list[int]:
    """Get list of checkpoint numbers that exist in the run directory."""
    checkpoints = []
    for i in range(start, end + 1, step):
        checkpoint_path = run_dir / f"checkpoint-{i}"
        if checkpoint_path.exists():
            checkpoints.append(i)
        else:
            print(f"Warning: checkpoint-{i} not found, skipping")
    return checkpoints


def run_evaluation(
    checkpoint_path: Path,
    output_dir: Path,
    benchmarks: str | None,
    num_samples: int,
    parallel: bool,
    gpus_per_benchmark: int,
    max_samples: int | None,
) -> dict | None:
    """Run evaluate_all.py on a single checkpoint."""
    cmd = [
        sys.executable, "-m", "aimo3_recipe.evaluation.evaluate_all",
        "--model", str(checkpoint_path),
        "--output-dir", str(output_dir),
        "--num-samples", str(num_samples),
    ]

    if benchmarks:
        cmd.extend(["--benchmarks", benchmarks])

    if parallel:
        cmd.extend(["--parallel", "--gpus-per-benchmark", str(gpus_per_benchmark)])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)

        # Read the summary file
        summary_file = output_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                return json.load(f)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {checkpoint_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--start", type=int, required=True, help="Starting checkpoint number")
    parser.add_argument("--end", type=int, required=True, help="Ending checkpoint number (inclusive)")
    parser.add_argument("--step", type=int, default=10, help="Checkpoint step interval (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory (default: run_dir/eval_results)")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated benchmarks (default: all)")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples for self-consistency")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    parser.add_argument("--gpus-per-benchmark", type=int, default=1, help="GPUs per benchmark for parallel mode")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = run_dir / "eval_results"

    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Get available checkpoints
    checkpoints = get_checkpoint_numbers(run_dir, args.start, args.end, args.step)
    if not checkpoints:
        print("No checkpoints found in the specified range!")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints to evaluate: {checkpoints}")
    print(f"Benchmarks: {args.benchmarks or 'all'}")
    print(f"Output directory: {base_output_dir}")

    # Evaluate each checkpoint
    all_summaries = {}
    for ckpt_num in checkpoints:
        checkpoint_path = run_dir / f"checkpoint-{ckpt_num}"
        output_dir = base_output_dir / f"checkpoint-{ckpt_num}"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = run_evaluation(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            benchmarks=args.benchmarks,
            num_samples=args.num_samples,
            parallel=args.parallel,
            gpus_per_benchmark=args.gpus_per_benchmark,
            max_samples=args.max_samples,
        )

        if summary:
            all_summaries[f"checkpoint-{ckpt_num}"] = summary

    # Create combined summary
    print("\n" + "=" * 80)
    print("COMBINED RESULTS")
    print("=" * 80)

    # Get all benchmark names
    benchmark_names = set()
    for summary in all_summaries.values():
        if "results" in summary:
            benchmark_names.update(summary["results"].keys())
    benchmark_names = sorted(benchmark_names)

    # Print header
    header = f"{'Checkpoint':<15}"
    for name in benchmark_names:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))

    # Print results for each checkpoint
    for ckpt_name, summary in sorted(all_summaries.items()):
        row = f"{ckpt_name:<15}"
        for name in benchmark_names:
            if name in summary.get("results", {}):
                acc = summary["results"][name].get("accuracy", 0)
                row += f" {acc:>11.2%}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Save combined summary
    combined_summary = {
        "run_dir": str(run_dir),
        "checkpoints": checkpoints,
        "benchmarks": args.benchmarks,
        "num_samples": args.num_samples,
        "timestamp": datetime.now().isoformat(),
        "summaries": all_summaries,
    }

    combined_file = base_output_dir / "combined_summary.json"
    with open(combined_file, "w") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\nCombined summary saved to: {combined_file}")

    # Also create a CSV for easy analysis
    csv_file = base_output_dir / "results.csv"
    with open(csv_file, "w") as f:
        # Header
        f.write("checkpoint," + ",".join(benchmark_names) + "\n")
        # Data
        for ckpt_name, summary in sorted(all_summaries.items()):
            ckpt_num = ckpt_name.replace("checkpoint-", "")
            row = [ckpt_num]
            for name in benchmark_names:
                if name in summary.get("results", {}):
                    acc = summary["results"][name].get("accuracy", 0)
                    row.append(f"{acc:.4f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")
    print(f"CSV results saved to: {csv_file}")


if __name__ == "__main__":
    main()
