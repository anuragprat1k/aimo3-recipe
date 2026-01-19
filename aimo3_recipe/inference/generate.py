"""
Main generation interface for math problem solving.

Provides a unified API for different solving strategies:
- Greedy/single-shot generation
- Self-consistency (majority voting)
- Tool-Integrated Reasoning (TIR)
- GenSelect solution selection
"""

import argparse
from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class GenerationConfig:
    """Configuration for math problem generation."""

    # Model
    model_name_or_path: str = "./outputs/rl_math"
    use_vllm: bool = True

    # Strategy
    strategy: Literal["greedy", "majority_vote", "tir", "genselect", "weighted_vote"] = "greedy"

    # Generation parameters
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0

    # Self-consistency / voting parameters
    num_samples: int = 16
    sc_temperature: float = 0.7

    # TIR parameters
    max_code_executions: int = 5

    # GenSelect parameters
    selector_model_path: Optional[str] = None


class MathGenerator:
    """
    Unified math problem solver supporting multiple strategies.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.solver = None

    def setup(self) -> None:
        """Initialize model and solver based on strategy."""
        # Load model
        if self.config.use_vllm and self.config.strategy in ["greedy", "majority_vote"]:
            from vllm import LLM
            self.model = LLM(
                model=self.config.model_name_or_path,
                trust_remote_code=True,
                dtype="bfloat16",
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # Initialize solver based on strategy
        if self.config.strategy == "tir":
            from aimo3_recipe.inference.tir_executor import TIRExecutor
            self.solver = TIRExecutor(
                model=self.model,
                tokenizer=self.tokenizer,
                max_code_executions=self.config.max_code_executions,
            )

        elif self.config.strategy == "majority_vote":
            from aimo3_recipe.inference.solution_selection import MajorityVoteSolver
            self.solver = MajorityVoteSolver(
                model=self.model,
                tokenizer=self.tokenizer,
                num_samples=self.config.num_samples,
                temperature=self.config.sc_temperature,
            )

        elif self.config.strategy == "weighted_vote":
            from aimo3_recipe.inference.solution_selection import WeightedVoteSolver
            self.solver = WeightedVoteSolver(
                model=self.model,
                tokenizer=self.tokenizer,
                num_samples=self.config.num_samples,
                temperature=self.config.sc_temperature,
            )

        elif self.config.strategy == "genselect":
            from aimo3_recipe.inference.solution_selection import GenSelectSolver
            from transformers import AutoModelForCausalLM

            selector_path = self.config.selector_model_path or self.config.model_name_or_path
            selector_model = AutoModelForCausalLM.from_pretrained(
                selector_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self.solver = GenSelectSolver(
                generator_model=self.model,
                selector_model=selector_model,
                tokenizer=self.tokenizer,
                num_candidates=self.config.num_samples,
                temperature=self.config.sc_temperature,
            )

    def solve(self, problem: str) -> str:
        """
        Solve a math problem.

        Args:
            problem: Math problem text

        Returns:
            Solution with reasoning and boxed answer
        """
        if self.model is None:
            self.setup()

        if self.solver is not None:
            if self.config.strategy == "tir":
                return self.solver.solve_with_tir(problem)
            else:
                return self.solver.solve(problem)

        # Greedy generation
        return self._greedy_solve(problem)

    def _greedy_solve(self, problem: str) -> str:
        """Single-shot greedy generation."""
        prompt = self._format_prompt(problem)

        if self.config.use_vllm:
            from vllm import SamplingParams

            params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            outputs = self.model.generate([prompt], params)
            return outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            return self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

    def _format_prompt(self, problem: str) -> str:
        """Format problem as chat prompt."""
        return f"""<|im_start|>system
You are a mathematical reasoning expert. Solve problems step-by-step, showing your work clearly. Always put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

    def solve_batch(self, problems: list[str]) -> list[str]:
        """Solve multiple problems."""
        return [self.solve(p) for p in problems]


def main():
    parser = argparse.ArgumentParser(description="Solve math problems")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="greedy",
                        choices=["greedy", "majority_vote", "tir", "genselect", "weighted_vote"])
    parser.add_argument("--problem", type=str, help="Problem to solve")
    parser.add_argument("--input-file", type=str, help="File with problems (one per line)")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--use-vllm", action="store_true", default=True)

    args = parser.parse_args()

    config = GenerationConfig(
        model_name_or_path=args.model,
        strategy=args.strategy,
        num_samples=args.num_samples,
        use_vllm=args.use_vllm,
    )

    generator = MathGenerator(config)

    if args.problem:
        solution = generator.solve(args.problem)
        print(f"\nProblem: {args.problem}")
        print(f"\nSolution:\n{solution}")

    elif args.input_file:
        with open(args.input_file) as f:
            problems = [line.strip() for line in f if line.strip()]

        for i, problem in enumerate(problems):
            print(f"\n{'='*60}")
            print(f"Problem {i+1}: {problem}")
            print(f"{'='*60}")
            solution = generator.solve(problem)
            print(f"\nSolution:\n{solution}")


if __name__ == "__main__":
    main()
