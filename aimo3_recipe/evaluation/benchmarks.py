"""
Benchmark definitions for math evaluation.

Provides standardized benchmark configurations for:
- MATH (Hendrycks)
- MATH-500 (challenging subset)
- AIME (competition problems)
- GSM8K (grade school math)
"""

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset, concatenate_datasets


@dataclass
class BenchmarkConfig:
    """Configuration for a math benchmark."""
    name: str
    dataset_name: str
    split: str
    problem_column: str
    answer_column: str
    subset: Optional[str] = None  # Dataset subset/config name
    subsets: list[str] = field(default_factory=list)  # Multiple subsets to concatenate
    filter_fn: Optional[callable] = None
    description: str = ""


# Hendrycks MATH benchmark
MATH_BENCHMARK = BenchmarkConfig(
    name="MATH",
    dataset_name="hendrycks/competition_math",
    split="test",
    problem_column="problem",
    answer_column="solution",
    description="Hendrycks MATH benchmark - 5000 problems across 7 subjects",
)

# MATH by difficulty level
MATH_LEVEL_1 = BenchmarkConfig(
    name="MATH-Level1",
    dataset_name="hendrycks/competition_math",
    split="test",
    problem_column="problem",
    answer_column="solution",
    filter_fn=lambda x: x["level"] == "Level 1",
    description="MATH Level 1 (easiest)",
)

MATH_LEVEL_5 = BenchmarkConfig(
    name="MATH-Level5",
    dataset_name="hendrycks/competition_math",
    split="test",
    problem_column="problem",
    answer_column="solution",
    filter_fn=lambda x: x["level"] == "Level 5",
    description="MATH Level 5 (hardest)",
)

# AMC/AIME from NuminaMath
AMC_BENCHMARK = BenchmarkConfig(
    name="AMC",
    dataset_name="AI-MO/NuminaMath-CoT",
    split="train",
    problem_column="problem",
    answer_column="solution",
    filter_fn=lambda x: "amc" in x.get("source", "").lower(),
    description="AMC competition problems",
)

AIME_BENCHMARK = BenchmarkConfig(
    name="AIME",
    dataset_name="opencompass/AIME2025",
    split="test",
    problem_column="question",
    answer_column="answer",
    subsets=["AIME2025-I", "AIME2025-II"],
    description="AIME 2025 competition problems (I and II)",
)

# MATH-500 benchmark (challenging subset of MATH)
MATH500_BENCHMARK = BenchmarkConfig(
    name="MATH-500",
    dataset_name="HuggingFaceH4/MATH-500",
    split="test",
    problem_column="problem",
    answer_column="answer",
    description="MATH-500 benchmark - 500 challenging problems from MATH",
)

# GSM8K for basic math
GSM8K_BENCHMARK = BenchmarkConfig(
    name="GSM8K",
    dataset_name="openai/gsm8k",
    subset="main",
    split="test",
    problem_column="question",
    answer_column="answer",
    description="Grade school math word problems",
)


def load_benchmark(config: BenchmarkConfig, max_samples: Optional[int] = None) -> Dataset:
    """
    Load a benchmark dataset.

    Args:
        config: Benchmark configuration
        max_samples: Maximum number of samples to load

    Returns:
        Dataset ready for evaluation
    """
    if config.subsets:
        # Load multiple subsets and concatenate
        datasets = [
            load_dataset(config.dataset_name, subset, split=config.split)
            for subset in config.subsets
        ]
        dataset = concatenate_datasets(datasets)
    elif config.subset:
        dataset = load_dataset(config.dataset_name, config.subset, split=config.split)
    else:
        dataset = load_dataset(config.dataset_name, split=config.split)

    # Apply filter if specified
    if config.filter_fn:
        dataset = dataset.filter(config.filter_fn)

    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Standardize column names
    if config.problem_column != "problem":
        dataset = dataset.rename_column(config.problem_column, "problem")
    if config.answer_column != "solution":
        dataset = dataset.rename_column(config.answer_column, "solution")

    return dataset


# All available benchmarks
ALL_BENCHMARKS = {
    "math": MATH_BENCHMARK,
    "math_level1": MATH_LEVEL_1,
    "math_level5": MATH_LEVEL_5,
    "amc": AMC_BENCHMARK,
    "aime": AIME_BENCHMARK,
    "math500": MATH500_BENCHMARK,
    "gsm8k": GSM8K_BENCHMARK,
}


def get_benchmark(name: str) -> BenchmarkConfig:
    """Get benchmark config by name."""
    if name.lower() not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(ALL_BENCHMARKS.keys())}")
    return ALL_BENCHMARKS[name.lower()]
