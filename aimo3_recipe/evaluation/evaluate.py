"""
Evaluation pipeline for math reasoning models.

Supports:
- Single-shot evaluation
- Self-consistency with majority voting
- Tool-Integrated Reasoning evaluation
- GenSelect evaluation
- Multiprocessing for faster answer verification
"""

import argparse
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional
from collections import Counter

from datasets import Dataset
from tqdm import tqdm

from aimo3_recipe.evaluation.answer_extraction import (
    extract_boxed_answer,
    verify_answer,
    normalize_answer,
)
from aimo3_recipe.evaluation.benchmarks import (
    load_benchmark,
    get_benchmark,
    ALL_BENCHMARKS,
)


logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    # Model
    model_name_or_path: str = "./outputs/rl_math"
    use_vllm: bool = True

    # Benchmark
    benchmark: str = "math"
    max_samples: Optional[int] = None

    # Generation
    max_new_tokens: int = 4096
    temperature: float = 0.0  # Greedy for single-shot
    top_p: float = 1.0

    # Self-consistency
    num_samples: int = 1  # >1 enables self-consistency
    sc_temperature: float = 0.7  # Temperature for self-consistency

    # TIR
    use_tir: bool = False
    max_code_executions: int = 5

    # Output
    output_dir: str = "./eval_results"
    save_predictions: bool = True

    # Multiprocessing
    num_workers: int = 0  # 0 = auto (use all CPUs), -1 = disable multiprocessing


def _verify_single_answer(args: tuple) -> tuple:
    """
    Verify a single answer. Used for multiprocessing.

    Args:
        args: Tuple of (index, response, ground_truth)

    Returns:
        Tuple of (index, is_correct)
    """
    idx, response, ground_truth = args
    is_correct = verify_answer(response, ground_truth)
    return idx, is_correct


# Global pool for reuse across calls (lazy initialization)
_WORKER_POOL = None
_POOL_SIZE = None


def _get_worker_pool(num_workers: int) -> mp.Pool:
    """Get or create a worker pool for parallel verification."""
    global _WORKER_POOL, _POOL_SIZE
    if _WORKER_POOL is None or _POOL_SIZE != num_workers:
        if _WORKER_POOL is not None:
            _WORKER_POOL.terminate()
        _POOL_SIZE = num_workers
        _WORKER_POOL = mp.Pool(processes=num_workers)
    return _WORKER_POOL


def verify_answers_parallel(
    responses: list[str],
    ground_truths: list[str],
    num_workers: int = 0,
) -> list[bool]:
    """
    Verify multiple answers in parallel using multiprocessing.

    Uses a persistent worker pool to avoid process spawn overhead on repeated calls.
    Only enables parallelization for batches >= 100 items where the benefit outweighs overhead.

    Args:
        responses: List of model responses
        ground_truths: List of ground truth answers
        num_workers: Number of worker processes. 0 = auto (all CPUs), -1 = sequential

    Returns:
        List of boolean correctness values
    """
    # Use sequential processing for small batches or when disabled
    # Multiprocessing overhead only pays off for larger batches
    min_batch_for_parallel = 100
    if num_workers == -1 or len(responses) < min_batch_for_parallel:
        return [verify_answer(r, gt) for r, gt in zip(responses, ground_truths)]

    # Determine number of workers
    if num_workers == 0:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers to reduce overhead
    num_workers = min(num_workers, len(responses))

    # Prepare arguments
    args = [(i, r, gt) for i, (r, gt) in enumerate(zip(responses, ground_truths))]

    # Process in parallel using persistent pool
    results = [False] * len(responses)
    try:
        pool = _get_worker_pool(num_workers)
        # Use larger chunksize for better efficiency
        chunksize = max(1, len(args) // (num_workers * 4))
        for idx, is_correct in pool.imap_unordered(_verify_single_answer, args, chunksize=chunksize):
            results[idx] = is_correct
    except Exception as e:
        # Fall back to sequential on any pool error
        logger.warning(f"Parallel verification failed, falling back to sequential: {e}")
        return [verify_answer(r, gt) for r, gt in zip(responses, ground_truths)]

    return results


class MathEvaluator:
    """
    Evaluator for math reasoning models.

    Supports greedy decoding, self-consistency, and TIR evaluation.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup_model(self) -> None:
        """Initialize model for inference."""
        if self.config.use_vllm:
            from vllm import LLM, SamplingParams
            self.model = LLM(
                model=self.config.model_name_or_path,
                trust_remote_code=True,
                dtype="bfloat16",
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    def generate(self, prompts: list[str]) -> list[list[str]]:
        """
        Generate responses for a batch of prompts.

        Returns list of list of responses (multiple samples per prompt for SC).
        """
        if self.config.use_vllm:
            from vllm import SamplingParams

            # Determine temperature and num samples
            if self.config.num_samples > 1:
                params = SamplingParams(
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.sc_temperature,
                    top_p=self.config.top_p,
                    n=self.config.num_samples,
                )
            else:
                params = SamplingParams(
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

            outputs = self.model.generate(prompts, params)
            return [[o.text for o in output.outputs] for output in outputs]
        else:
            # HuggingFace generation
            import torch

            all_responses = []
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                responses = []
                for _ in range(self.config.num_samples):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.sc_temperature if self.config.num_samples > 1 else self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.num_samples > 1 or self.config.temperature > 0,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    responses.append(response)

                all_responses.append(responses)

            return all_responses

    def majority_vote(self, responses: list[str]) -> str:
        """
        Apply majority voting across multiple responses.

        Extracts and normalizes answers, then returns the most common one.
        """
        answers = []
        for response in responses:
            answer = extract_boxed_answer(response)
            if answer:
                normalized = normalize_answer(answer)
                answers.append(normalized)

        if not answers:
            # Fallback to first response if no boxed answers found
            return responses[0] if responses else ""

        # Count occurrences
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]

        # Find a response with this answer
        for response in responses:
            answer = extract_boxed_answer(response)
            if answer and normalize_answer(answer) == most_common:
                return response

        return responses[0]

    def format_prompt(self, problem: str) -> str:
        """Format problem as chat prompt."""
        system = """You are a mathematical reasoning expert. Solve problems step-by-step, showing your work clearly. Always put your final answer in \\boxed{}."""

        return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

    def evaluate(self, dataset: Dataset) -> dict:
        """
        Evaluate model on a dataset.

        Returns metrics and predictions.
        Uses multiprocessing for faster answer verification.
        """
        if self.model is None:
            self.setup_model()

        results = {
            "correct": 0,
            "total": 0,
            "predictions": [],
        }

        # Process in batches
        batch_size = 32 if self.config.use_vllm else 1
        problems = dataset["problem"]
        solutions = dataset["solution"]

        for i in tqdm(range(0, len(problems), batch_size), desc="Evaluating"):
            batch_problems = problems[i:i+batch_size]
            batch_solutions = solutions[i:i+batch_size]

            # Format prompts
            prompts = [self.format_prompt(p) for p in batch_problems]

            # Generate responses
            all_responses = self.generate(prompts)

            # Apply majority voting if multiple samples
            final_responses = []
            for responses in all_responses:
                if self.config.num_samples > 1:
                    final_responses.append(self.majority_vote(responses))
                else:
                    final_responses.append(responses[0])

            # Verify answers in parallel for better performance
            correctness = verify_answers_parallel(
                final_responses,
                batch_solutions,
                num_workers=self.config.num_workers,
            )

            # Collect results
            for j, (problem, gt_solution, responses, final_response, is_correct) in enumerate(
                zip(batch_problems, batch_solutions, all_responses, final_responses, correctness)
            ):
                results["correct"] += int(is_correct)
                results["total"] += 1

                if self.config.save_predictions:
                    results["predictions"].append({
                        "problem": problem,
                        "ground_truth": gt_solution,
                        "prediction": final_response,
                        "all_responses": responses if self.config.num_samples > 1 else None,
                        "correct": is_correct,
                    })

        # Compute accuracy
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--benchmark", type=str, default="math", choices=list(ALL_BENCHMARKS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples for self-consistency")
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--use-vllm", action="store_true", default=True)
    parser.add_argument("--use-tir", action="store_true", help="Use Tool-Integrated Reasoning")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for parallel answer verification (0=auto, -1=disable)",
    )

    args = parser.parse_args()

    config = EvalConfig(
        model_name_or_path=args.model,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        use_tir=args.use_tir,
        num_workers=args.num_workers,
    )

    # Load benchmark
    benchmark_config = get_benchmark(config.benchmark)
    dataset = load_benchmark(benchmark_config, config.max_samples)
    logger.info(f"Loaded {len(dataset)} problems from {benchmark_config.name}")

    # Run evaluation
    evaluator = MathEvaluator(config)
    results = evaluator.evaluate(dataset)

    # Print results
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_config.name}")
    print(f"Model: {config.model_name_or_path}")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{benchmark_config.name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
