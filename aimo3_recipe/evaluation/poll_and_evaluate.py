#!/usr/bin/env python3
"""
Checkpoint Polling and Evaluation Script.

Continuously polls for new checkpoints from a training run and kicks off
evaluation on specified benchmarks (default: aime, olympiad) when new
checkpoints are available.

Usage:
    # Poll a local output directory
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math

    # Poll with a specific run name pattern
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs --run-name rl_math_grpo

    # Custom benchmarks and poll interval
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math \
        --benchmarks aime,olympiad,math_level5 --poll-interval 300

    # Parallel evaluation across GPUs
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math \
        --parallel --gpus-per-benchmark 2
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Handle graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    print("\nShutdown requested. Finishing current evaluation...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_checkpoints(output_dir: str) -> list[tuple[int, Path]]:
    """
    Find all checkpoints in the given output directory.

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step number.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    checkpoint_pattern = re.compile(r"checkpoint-(\d+)$")
    checkpoints = []

    for item in output_path.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_evaluated_checkpoints(state_file: Path) -> set[int]:
    """Load the set of already-evaluated checkpoint steps from state file."""
    if not state_file.exists():
        return set()

    try:
        with open(state_file) as f:
            data = json.load(f)
            return set(data.get("evaluated_steps", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_evaluated_checkpoints(state_file: Path, evaluated_steps: set[int]) -> None:
    """Save the set of evaluated checkpoint steps to state file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump({
            "evaluated_steps": sorted(evaluated_steps),
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


def run_evaluation(
    checkpoint_path: Path,
    benchmarks: list[str],
    eval_output_dir: Path,
    parallel: bool = False,
    gpus_per_benchmark: int = 1,
    num_samples: int = 1,
    max_samples: Optional[int] = None,
    use_vllm: bool = True,
) -> dict:
    """
    Run evaluation on a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint
        benchmarks: List of benchmark names to evaluate
        eval_output_dir: Directory to save evaluation results
        parallel: Whether to run benchmarks in parallel
        gpus_per_benchmark: GPUs to allocate per benchmark
        num_samples: Number of samples for self-consistency
        max_samples: Max samples per benchmark (for quick testing)
        use_vllm: Whether to use vLLM for inference

    Returns:
        Dict with evaluation results
    """
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, "-m", "aimo3_recipe.evaluation.evaluate_all",
        "--model", str(checkpoint_path),
        "--output-dir", str(eval_output_dir),
        "--benchmarks", ",".join(benchmarks),
        "--num-samples", str(num_samples),
    ]

    if parallel:
        cmd.extend(["--parallel", "--gpus-per-benchmark", str(gpus_per_benchmark)])

    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    if not use_vllm:
        cmd.append("--no-vllm")

    print(f"Running: {' '.join(cmd)}")

    # Run evaluation
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    # Load results
    summary_file = eval_output_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results = json.load(f)
            results["elapsed_seconds"] = elapsed
            return results
    else:
        return {
            "error": f"Evaluation failed with return code {result.returncode}",
            "elapsed_seconds": elapsed,
        }


def format_results_summary(results: dict) -> str:
    """Format evaluation results as a compact summary string."""
    if "error" in results:
        return f"ERROR: {results['error']}"

    benchmark_results = results.get("results", {})
    parts = []
    for name, data in benchmark_results.items():
        if isinstance(data, dict) and "accuracy" in data:
            acc = data["accuracy"] * 100
            parts.append(f"{name}: {acc:.1f}%")

    return " | ".join(parts) if parts else "No results"


def poll_and_evaluate(
    output_dir: str,
    run_name: Optional[str] = None,
    benchmarks: list[str] = None,
    eval_base_dir: str = "./eval_results",
    poll_interval: int = 60,
    parallel: bool = False,
    gpus_per_benchmark: int = 1,
    num_samples: int = 1,
    max_samples: Optional[int] = None,
    use_vllm: bool = True,
    max_polls: Optional[int] = None,
    eval_latest_only: bool = False,
) -> None:
    """
    Main polling loop to watch for new checkpoints and evaluate them.

    Args:
        output_dir: Directory to watch for checkpoints
        run_name: Optional run name to filter/locate checkpoints
        benchmarks: List of benchmark names (default: aime, olympiad)
        eval_base_dir: Base directory for evaluation results
        poll_interval: Seconds between polls
        parallel: Whether to run benchmark evaluation in parallel
        gpus_per_benchmark: GPUs per benchmark for parallel eval
        num_samples: Samples for self-consistency
        max_samples: Max samples per benchmark
        use_vllm: Whether to use vLLM
        max_polls: Maximum number of poll iterations (None = infinite)
        eval_latest_only: Only evaluate the latest checkpoint, skip earlier ones
    """
    global _shutdown_requested

    if benchmarks is None:
        benchmarks = ["aime", "olympiad"]

    # Resolve the checkpoint directory
    if run_name:
        checkpoint_dir = Path(output_dir) / run_name
    else:
        checkpoint_dir = Path(output_dir)

    if not checkpoint_dir.exists():
        print(f"Waiting for checkpoint directory to be created: {checkpoint_dir}")

    # State file to track evaluated checkpoints
    state_file = Path(eval_base_dir) / checkpoint_dir.name / ".poll_state.json"
    evaluated_steps = load_evaluated_checkpoints(state_file)

    if evaluated_steps:
        print(f"Previously evaluated checkpoints: {sorted(evaluated_steps)}")

    print(f"\nPolling for checkpoints in: {checkpoint_dir}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Poll interval: {poll_interval}s")
    print(f"Parallel evaluation: {parallel}")
    if parallel:
        print(f"GPUs per benchmark: {gpus_per_benchmark}")
    print("=" * 60)

    poll_count = 0
    while not _shutdown_requested:
        poll_count += 1
        if max_polls and poll_count > max_polls:
            print(f"\nReached max polls ({max_polls}). Exiting.")
            break

        # Check for checkpoints
        checkpoints = get_checkpoints(str(checkpoint_dir))

        if not checkpoints:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No checkpoints found. Waiting...")
        else:
            # Find new checkpoints
            new_checkpoints = [
                (step, path) for step, path in checkpoints
                if step not in evaluated_steps
            ]

            if new_checkpoints:
                # If eval_latest_only, only keep the latest
                if eval_latest_only and len(new_checkpoints) > 1:
                    skipped = new_checkpoints[:-1]
                    for step, _ in skipped:
                        print(f"Skipping checkpoint-{step} (--eval-latest-only)")
                        evaluated_steps.add(step)
                    new_checkpoints = [new_checkpoints[-1]]

                for step, ckpt_path in new_checkpoints:
                    if _shutdown_requested:
                        break

                    print(f"\n{'=' * 60}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"New checkpoint found: checkpoint-{step}")
                    print(f"{'=' * 60}")

                    # Set up evaluation output directory
                    eval_output_dir = Path(eval_base_dir) / checkpoint_dir.name / f"checkpoint-{step}"

                    # Run evaluation
                    results = run_evaluation(
                        checkpoint_path=ckpt_path,
                        benchmarks=benchmarks,
                        eval_output_dir=eval_output_dir,
                        parallel=parallel,
                        gpus_per_benchmark=gpus_per_benchmark,
                        num_samples=num_samples,
                        max_samples=max_samples,
                        use_vllm=use_vllm,
                    )

                    # Print summary
                    print(f"\nCheckpoint-{step} Results: {format_results_summary(results)}")
                    print(f"Elapsed time: {results.get('elapsed_seconds', 0):.1f}s")
                    print(f"Full results saved to: {eval_output_dir}")

                    # Mark as evaluated
                    evaluated_steps.add(step)
                    save_evaluated_checkpoints(state_file, evaluated_steps)
            else:
                latest_step = checkpoints[-1][0] if checkpoints else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"No new checkpoints. Latest: {latest_step}. Waiting...")

        # Wait for next poll
        if not _shutdown_requested:
            time.sleep(poll_interval)

    print("\nPolling stopped.")
    if evaluated_steps:
        print(f"Evaluated checkpoints: {sorted(evaluated_steps)}")


def main():
    parser = argparse.ArgumentParser(
        description="Poll for training checkpoints and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - poll a training output directory
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math

    # With a specific run name
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs --run-name my_run

    # Custom benchmarks and poll interval
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math \\
        --benchmarks aime,olympiad,math_level5 --poll-interval 300

    # Parallel evaluation on multi-GPU setup
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math \\
        --parallel --gpus-per-benchmark 2

    # Only evaluate latest checkpoint (skip older ones)
    python -m aimo3_recipe.evaluation.poll_and_evaluate --output-dir ./outputs/rl_math \\
        --eval-latest-only
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing training checkpoints (or parent dir if using --run-name)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Training run name (subdirectory under --output-dir)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="aime,olympiad",
        help="Comma-separated list of benchmarks to evaluate (default: aime,olympiad)",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="./eval_results",
        help="Base directory for evaluation results (default: ./eval_results)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between checkpoint polls (default: 60)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks in parallel across GPUs",
    )
    parser.add_argument(
        "--gpus-per-benchmark",
        type=int,
        default=1,
        help="Number of GPUs per benchmark when using --parallel (default: 1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples for self-consistency (1=greedy, default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per benchmark (for quick testing)",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM for inference",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=None,
        help="Maximum number of poll iterations before stopping (default: run forever)",
    )
    parser.add_argument(
        "--eval-latest-only",
        action="store_true",
        help="Only evaluate the latest checkpoint, skip older unevaluated ones",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (equivalent to --max-polls 1)",
    )

    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    max_polls = args.max_polls
    if args.once:
        max_polls = 1

    poll_and_evaluate(
        output_dir=args.output_dir,
        run_name=args.run_name,
        benchmarks=benchmarks,
        eval_base_dir=args.eval_output_dir,
        poll_interval=args.poll_interval,
        parallel=args.parallel,
        gpus_per_benchmark=args.gpus_per_benchmark,
        num_samples=args.num_samples,
        max_samples=args.max_samples,
        use_vllm=not args.no_vllm,
        max_polls=max_polls,
        eval_latest_only=args.eval_latest_only,
    )


if __name__ == "__main__":
    main()
