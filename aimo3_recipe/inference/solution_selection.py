"""
Solution Selection Methods

Implements strategies for selecting the best solution from multiple candidates:
- Majority Voting: Select answer with most votes
- GenSelect: Train a model to select the best solution (NemoSkills AIMO2)
- Weighted Voting: Weight votes by model confidence
"""

import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter


from aimo3_recipe.evaluation.answer_extraction import (
    extract_boxed_answer,
    normalize_answer,
    is_equivalent_answer,
)


@dataclass
class SolutionCandidate:
    """A candidate solution with metadata."""
    response: str
    answer: Optional[str]
    normalized_answer: str
    log_prob: Optional[float] = None
    execution_verified: bool = False


class MajorityVoteSolver:
    """
    Majority voting solution selection.

    Generates multiple solutions and selects the answer that appears most frequently.
    This is the simplest but effective baseline approach.
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_samples: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str) -> str:
        """
        Solve a problem using majority voting.

        Args:
            problem: Math problem text

        Returns:
            Selected solution response
        """
        # Generate multiple solutions
        candidates = self._generate_candidates(problem)

        # Extract and normalize answers
        for candidate in candidates:
            candidate.answer = extract_boxed_answer(candidate.response)
            candidate.normalized_answer = normalize_answer(candidate.answer) if candidate.answer else ""

        # Count votes for each normalized answer
        answer_counts = Counter(
            c.normalized_answer for c in candidates
            if c.normalized_answer
        )

        if not answer_counts:
            # No valid answers found, return first response
            return candidates[0].response if candidates else ""

        # Get most common answer
        most_common_answer = answer_counts.most_common(1)[0][0]

        # Return a response with this answer (prefer longer/more detailed)
        matching_responses = [
            c for c in candidates
            if c.normalized_answer == most_common_answer
        ]
        matching_responses.sort(key=lambda c: len(c.response), reverse=True)

        return matching_responses[0].response

    def _generate_candidates(self, problem: str) -> list[SolutionCandidate]:
        """Generate multiple solution candidates."""
        import torch

        # Format prompt
        prompt = self._format_prompt(problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        candidates = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                response = self.tokenizer.decode(
                    outputs.sequences[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                # Compute average log probability
                log_prob = None
                if hasattr(outputs, 'scores') and outputs.scores:
                    import torch.nn.functional as F
                    log_probs = []
                    for i, score in enumerate(outputs.scores):
                        token_id = outputs.sequences[0, inputs["input_ids"].shape[1] + i]
                        log_prob_i = F.log_softmax(score[0], dim=-1)[token_id].item()
                        log_probs.append(log_prob_i)
                    log_prob = sum(log_probs) / len(log_probs) if log_probs else None

                candidates.append(SolutionCandidate(
                    response=response,
                    answer=None,
                    normalized_answer="",
                    log_prob=log_prob,
                ))

        return candidates

    def _format_prompt(self, problem: str) -> str:
        """Format problem as prompt."""
        return f"""<|im_start|>system
You are a mathematical reasoning expert. Solve problems step-by-step. Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""


class GenSelectSolver:
    """
    Generative Solution Selection (GenSelect) from NemoSkills (AIMO2 winner).

    Instead of majority voting, trains a model to select the best solution
    from multiple candidates. This can outperform majority voting by
    considering solution quality beyond just the final answer.
    """

    def __init__(
        self,
        generator_model,
        selector_model,
        tokenizer,
        num_candidates: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
    ):
        self.generator = generator_model
        self.selector = selector_model
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str) -> str:
        """
        Solve using GenSelect: generate candidates then select best.

        Args:
            problem: Math problem text

        Returns:
            Selected best solution
        """
        # Generate candidates
        candidates = self._generate_candidates(problem)

        if len(candidates) == 1:
            return candidates[0].response

        # Use selector model to choose best
        best_idx = self._select_best(problem, candidates)

        return candidates[best_idx].response

    def _generate_candidates(self, problem: str) -> list[SolutionCandidate]:
        """Generate candidate solutions."""
        import torch

        prompt = self._format_prompt(problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)

        candidates = []

        with torch.no_grad():
            for _ in range(self.num_candidates):
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                candidates.append(SolutionCandidate(
                    response=response,
                    answer=extract_boxed_answer(response),
                    normalized_answer=normalize_answer(extract_boxed_answer(response) or ""),
                ))

        return candidates

    def _select_best(self, problem: str, candidates: list[SolutionCandidate]) -> int:
        """
        Use selector model to choose the best candidate.

        The selector is prompted with the problem and all candidates,
        and asked to select the best solution.
        """
        import torch

        # Format selection prompt
        selection_prompt = self._format_selection_prompt(problem, candidates)
        inputs = self.tokenizer(selection_prompt, return_tensors="pt").to(self.selector.device)

        with torch.no_grad():
            outputs = self.selector.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        selection = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse selection (expect "Solution X" or just a number)
        match = re.search(r'(?:Solution\s*)?(\d+)', selection)
        if match:
            idx = int(match.group(1)) - 1  # Convert to 0-indexed
            if 0 <= idx < len(candidates):
                return idx

        # Fallback to majority voting if selection fails
        answer_counts = Counter(c.normalized_answer for c in candidates if c.normalized_answer)
        if answer_counts:
            most_common = answer_counts.most_common(1)[0][0]
            for i, c in enumerate(candidates):
                if c.normalized_answer == most_common:
                    return i

        return 0  # Default to first candidate

    def _format_prompt(self, problem: str) -> str:
        """Format problem for generation."""
        return f"""<|im_start|>system
You are a mathematical reasoning expert. Solve problems step-by-step. Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

    def _format_selection_prompt(self, problem: str, candidates: list[SolutionCandidate]) -> str:
        """Format prompt for solution selection."""
        solutions_text = ""
        for i, candidate in enumerate(candidates, 1):
            solutions_text += f"\n\n=== Solution {i} ===\n{candidate.response}"

        return f"""<|im_start|>system
You are evaluating mathematical solutions. Given a problem and multiple candidate solutions, select the solution that is most likely correct. Consider the reasoning quality, mathematical correctness, and final answer.<|im_end|>
<|im_start|>user
Problem: {problem}
{solutions_text}

Which solution is most likely correct? Reply with just the solution number (e.g., "Solution 1" or "1").<|im_end|>
<|im_start|>assistant
"""


class WeightedVoteSolver:
    """
    Weighted voting based on model confidence.

    Similar to majority voting but weights each vote by the
    sequence log probability, giving more weight to solutions
    the model is more confident about.
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_samples: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str) -> str:
        """Solve using weighted voting."""
        import math

        # Generate candidates with log probs
        candidates = self._generate_candidates_with_logprobs(problem)

        # Group by normalized answer
        answer_weights = {}
        answer_responses = {}

        for candidate in candidates:
            if not candidate.normalized_answer:
                continue

            answer = candidate.normalized_answer
            # Convert log prob to weight (use exp for probability)
            weight = math.exp(candidate.log_prob) if candidate.log_prob else 1.0

            if answer not in answer_weights:
                answer_weights[answer] = 0.0
                answer_responses[answer] = []

            answer_weights[answer] += weight
            answer_responses[answer].append(candidate.response)

        if not answer_weights:
            return candidates[0].response if candidates else ""

        # Select answer with highest total weight
        best_answer = max(answer_weights, key=answer_weights.get)

        # Return longest response with this answer
        responses = answer_responses[best_answer]
        return max(responses, key=len)

    def _generate_candidates_with_logprobs(self, problem: str) -> list[SolutionCandidate]:
        """Generate candidates with log probabilities."""
        import torch
        import torch.nn.functional as F

        prompt = f"""<|im_start|>system
You are a mathematical reasoning expert.<|im_end|>
<|im_start|>user
{problem}

Put your final answer in \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        candidates = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                response = self.tokenizer.decode(
                    outputs.sequences[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                # Compute mean log probability
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    token_id = outputs.sequences[0, inputs["input_ids"].shape[1] + i]
                    log_prob = F.log_softmax(score[0], dim=-1)[token_id].item()
                    log_probs.append(log_prob)

                mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0.0

                answer = extract_boxed_answer(response)
                candidates.append(SolutionCandidate(
                    response=response,
                    answer=answer,
                    normalized_answer=normalize_answer(answer) if answer else "",
                    log_prob=mean_log_prob,
                ))

        return candidates
