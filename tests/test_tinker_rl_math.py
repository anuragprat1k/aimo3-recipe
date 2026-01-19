"""
Tests for tinker_rl_math module.

Tests the core utility functions for math RL training:
- extract_boxed_answer: Extract answers from LaTeX \boxed{} format
- normalize_math_answer: Normalize mathematical expressions for comparison
- compute_reward: Calculate rewards based on answer correctness
"""

import pytest

from aimo3_recipe.training.tinker_rl_math import (
    MathRLConfig,
    extract_boxed_answer,
    normalize_math_answer,
    compute_reward,
    get_renderer,
)


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer function."""

    def test_simple_boxed_number(self):
        """Extract a simple number from boxed."""
        text = r"The answer is \boxed{42}"
        assert extract_boxed_answer(text) == "42"

    def test_boxed_fraction(self):
        """Extract a fraction from boxed."""
        text = r"Therefore, \boxed{\frac{1}{2}}"
        assert extract_boxed_answer(text) == r"\frac{1}{2}"

    def test_boxed_with_nested_braces(self):
        """Extract content with nested braces."""
        text = r"The solution is \boxed{\frac{x+1}{2}}"
        assert extract_boxed_answer(text) == r"\frac{x+1}{2}"

    def test_multiple_boxed_returns_last(self):
        """When multiple boxed exist, return the last one."""
        text = r"First \boxed{wrong} and then \boxed{correct}"
        assert extract_boxed_answer(text) == "correct"

    def test_no_boxed_returns_none(self):
        """Return None when no boxed is found."""
        text = "The answer is 42"
        assert extract_boxed_answer(text) is None

    def test_empty_boxed(self):
        """Handle empty boxed content."""
        text = r"\boxed{}"
        assert extract_boxed_answer(text) == ""

    def test_boxed_with_negative_number(self):
        """Extract negative numbers."""
        text = r"\boxed{-5}"
        assert extract_boxed_answer(text) == "-5"

    def test_boxed_with_decimal(self):
        """Extract decimal numbers."""
        text = r"\boxed{3.14159}"
        assert extract_boxed_answer(text) == "3.14159"

    def test_boxed_with_expression(self):
        """Extract mathematical expressions."""
        text = r"\boxed{2x + 3}"
        assert extract_boxed_answer(text) == "2x + 3"

    def test_boxed_in_long_solution(self):
        """Extract boxed from a longer solution text."""
        text = r"""
        Let's solve this step by step.
        First, we compute 2 + 2 = 4.
        Then multiply by 3 to get 12.
        Therefore, the answer is \boxed{12}.
        """
        assert extract_boxed_answer(text) == "12"


class TestNormalizeMathAnswer:
    """Tests for normalize_math_answer function."""

    def test_simple_number(self):
        """Normalize a simple number."""
        assert normalize_math_answer("42") == "42"

    def test_strips_whitespace(self):
        """Strip leading and trailing whitespace."""
        assert normalize_math_answer("  42  ") == "42"

    def test_removes_internal_spaces(self):
        """Remove internal spaces."""
        assert normalize_math_answer("4 2") == "42"

    def test_lowercase_conversion(self):
        """Convert to lowercase."""
        assert normalize_math_answer("ABC") == "abc"

    def test_removes_text_command(self):
        """Remove LaTeX \\text{} command."""
        assert normalize_math_answer(r"\text{yes}") == "yes"

    def test_removes_mathrm_command(self):
        """Remove LaTeX \\mathrm{} command."""
        assert normalize_math_answer(r"\mathrm{cm}") == "cm"

    def test_removes_left_right(self):
        """Remove \\left and \\right delimiters."""
        assert normalize_math_answer(r"\left(\right)") == "()"

    def test_normalizes_frac(self):
        """Convert \\frac to simple fraction notation."""
        assert normalize_math_answer(r"\frac{1}{2}") == "1/2"

    def test_normalizes_dfrac(self):
        """Convert \\dfrac to simple fraction notation."""
        assert normalize_math_answer(r"\dfrac{3}{4}") == "3/4"

    def test_handles_none(self):
        """Return empty string for None input."""
        assert normalize_math_answer(None) == ""

    def test_complex_expression(self):
        """Normalize a complex expression."""
        result = normalize_math_answer(r"\text{The answer is } \frac{1}{2}")
        assert "1/2" in result


class TestComputeReward:
    """Tests for compute_reward function."""

    @pytest.fixture
    def config(self):
        """Create a default config for testing."""
        return MathRLConfig(
            correct_reward=1.0,
            incorrect_reward=-0.5,
            format_bonus=0.1,
        )

    def test_correct_answer_with_boxed(self, config):
        """Correct answer in boxed format gets full reward + bonus."""
        response = r"The answer is \boxed{42}"
        ground_truth = r"\boxed{42}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)  # 1.0 + 0.1 format bonus

    def test_incorrect_answer_with_boxed(self, config):
        """Incorrect answer in boxed format gets incorrect reward + bonus."""
        response = r"The answer is \boxed{43}"
        ground_truth = r"\boxed{42}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(-0.4)  # -0.5 + 0.1 format bonus

    def test_no_boxed_in_response(self, config):
        """No boxed in response gets incorrect reward, no bonus."""
        response = "The answer is 42"
        ground_truth = r"\boxed{42}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(-0.5)  # No format bonus

    def test_correct_fraction(self, config):
        """Correct fraction answer."""
        response = r"\boxed{\frac{1}{2}}"
        ground_truth = r"\boxed{1/2}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)

    def test_numeric_equivalence(self, config):
        """Numerically equivalent answers are correct."""
        response = r"\boxed{0.5}"
        ground_truth = r"\boxed{1/2}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)

    def test_ground_truth_without_boxed(self, config):
        """Handle ground truth that doesn't have boxed format."""
        response = r"\boxed{42}"
        ground_truth = "42"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)

    def test_negative_number(self, config):
        """Correct negative number answer."""
        response = r"\boxed{-5}"
        ground_truth = r"\boxed{-5}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)

    def test_case_insensitive_comparison(self, config):
        """Comparison should be case insensitive."""
        response = r"\boxed{YES}"
        ground_truth = r"\boxed{yes}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)

    def test_whitespace_normalization(self, config):
        """Whitespace differences should not affect correctness."""
        response = r"\boxed{ 42 }"
        ground_truth = r"\boxed{42}"
        reward = compute_reward(response, ground_truth, config)
        assert reward == pytest.approx(1.1)


class TestGetRenderer:
    """Tests for get_renderer function."""

    def test_qwen_model(self):
        """Get Qwen renderer for Qwen models."""
        renderer = get_renderer("Qwen/Qwen2.5-14B")
        assert renderer is not None
        assert "Qwen" in type(renderer).__name__

    def test_qwen_lowercase(self):
        """Handle lowercase qwen in model name."""
        renderer = get_renderer("some-qwen-model")
        assert renderer is not None
        assert "Qwen" in type(renderer).__name__

    def test_llama_model(self):
        """Get Llama renderer for Llama models."""
        renderer = get_renderer("meta-llama/Llama-3-8B")
        assert renderer is not None
        assert "Llama" in type(renderer).__name__

    def test_default_renderer(self):
        """Default to Qwen renderer for unknown models."""
        renderer = get_renderer("unknown-model")
        assert renderer is not None
        assert "Qwen" in type(renderer).__name__


class TestMathRLConfig:
    """Tests for MathRLConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MathRLConfig()
        assert config.model_name == "Qwen/Qwen2.5-14B"
        assert config.lora_rank == 64
        assert config.batch_size == 64
        assert config.group_size == 16
        assert config.learning_rate is None
        assert config.max_tokens == 4096
        assert config.correct_reward == 1.0
        assert config.incorrect_reward == -0.5
        assert config.format_bonus == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MathRLConfig(
            model_name="custom-model",
            lora_rank=32,
            batch_size=128,
            correct_reward=2.0,
        )
        assert config.model_name == "custom-model"
        assert config.lora_rank == 32
        assert config.batch_size == 128
        assert config.correct_reward == 2.0
