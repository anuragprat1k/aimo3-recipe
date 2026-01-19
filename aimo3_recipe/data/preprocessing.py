"""
Data preprocessing utilities for math reasoning datasets.
"""

import re
from dataclasses import dataclass
from typing import Optional
from datasets import Dataset


@dataclass
class MathPreprocessor:
    """
    Preprocessor for math problem datasets.

    Handles:
    - Problem text normalization
    - Solution formatting
    - Answer extraction and standardization
    """

    normalize_latex: bool = True
    remove_images: bool = True
    standardize_boxed: bool = True
    max_problem_length: int = 2048
    max_solution_length: int = 8192

    def __call__(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing to entire dataset."""
        return dataset.map(
            self._preprocess_example,
            desc="Preprocessing math dataset",
        )

    def _preprocess_example(self, example: dict) -> dict:
        """Preprocess a single example."""
        result = dict(example)

        if "problem" in example:
            result["problem"] = self.preprocess_problem(example["problem"])

        if "solution" in example:
            result["solution"] = self.preprocess_solution(example["solution"])

        return result

    def preprocess_problem(self, problem: str) -> str:
        """Clean and normalize problem text."""
        text = problem

        # Remove image references
        if self.remove_images:
            text = re.sub(r'\[asy\].*?\[/asy\]', '[diagram]', text, flags=re.DOTALL)
            text = re.sub(r'!\[.*?\]\(.*?\)', '[image]', text)

        # Normalize LaTeX
        if self.normalize_latex:
            text = self._normalize_latex(text)

        # Truncate if too long
        if len(text) > self.max_problem_length:
            text = text[:self.max_problem_length] + "..."

        return text.strip()

    def preprocess_solution(self, solution: str) -> str:
        """Clean and normalize solution text."""
        text = solution

        # Normalize LaTeX
        if self.normalize_latex:
            text = self._normalize_latex(text)

        # Standardize boxed format
        if self.standardize_boxed:
            text = self._standardize_boxed(text)

        # Truncate if too long
        if len(text) > self.max_solution_length:
            text = text[:self.max_solution_length]

        return text.strip()

    def _normalize_latex(self, text: str) -> str:
        """Normalize LaTeX expressions."""
        # Standardize math delimiters
        text = re.sub(r'\$\$(.+?)\$\$', r'\\[\1\\]', text, flags=re.DOTALL)

        # Fix common LaTeX issues
        text = text.replace(r'\left(', '(')
        text = text.replace(r'\right)', ')')
        text = text.replace(r'\cdot', r'\times')

        # Normalize fractions
        text = re.sub(r'\\frac\s*{([^}]+)}\s*{([^}]+)}', r'\\frac{\1}{\2}', text)

        return text

    def _standardize_boxed(self, text: str) -> str:
        """Ensure consistent \\boxed{} format for final answers."""
        # Convert various answer formats to boxed
        patterns = [
            (r'\\text{Answer:?\s*}(.+?)(?=\n|$)', r'\\boxed{\1}'),
            (r'(?:The )?[Aa]nswer is:?\s*\$?(.+?)\$?(?:\.|$)', r'\\boxed{\1}'),
            (r'\\fbox{(.+?)}', r'\\boxed{\1}'),
        ]

        for pattern, replacement in patterns:
            if '\\boxed' not in text:
                text = re.sub(pattern, replacement, text)

        return text


def filter_by_quality(
    dataset: Dataset,
    min_problem_length: int = 20,
    min_solution_length: int = 50,
    require_boxed: bool = True,
) -> Dataset:
    """
    Filter dataset for quality.

    Args:
        dataset: Input dataset
        min_problem_length: Minimum problem character length
        min_solution_length: Minimum solution character length
        require_boxed: Require \\boxed{} in solution

    Returns:
        Filtered dataset
    """

    def quality_filter(example):
        # Check lengths
        if len(example.get("problem", "")) < min_problem_length:
            return False
        if len(example.get("solution", "")) < min_solution_length:
            return False

        # Check for boxed answer
        if require_boxed and "\\boxed" not in example.get("solution", ""):
            return False

        return True

    return dataset.filter(quality_filter, desc="Filtering by quality")


def deduplicate_problems(dataset: Dataset, threshold: float = 0.9) -> Dataset:
    """
    Remove near-duplicate problems based on text similarity.

    Args:
        dataset: Input dataset
        threshold: Jaccard similarity threshold for duplicates

    Returns:
        Deduplicated dataset
    """
    seen_hashes = set()
    indices_to_keep = []

    def get_shingles(text: str, k: int = 5) -> set:
        """Get k-shingles (k-grams) of text."""
        text = text.lower().strip()
        return set(text[i:i+k] for i in range(len(text) - k + 1))

    def jaccard_similarity(set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    problem_shingles = []

    for idx, example in enumerate(dataset):
        problem = example.get("problem", "")
        shingles = get_shingles(problem)

        # Check against existing problems
        is_duplicate = False
        for prev_shingles in problem_shingles:
            if jaccard_similarity(shingles, prev_shingles) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            indices_to_keep.append(idx)
            problem_shingles.append(shingles)

    return dataset.select(indices_to_keep)
