"""
Answer extraction and verification utilities for math problems.

Handles:
- Extracting final answers from \\boxed{} format
- Normalizing mathematical expressions
- Comparing answers for equivalence
"""

import re
from typing import Optional, Union
import sympy
from sympy.parsing.latex import parse_latex as sympy_parse_latex


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from \\boxed{} format.

    Handles nested braces and multiple boxed answers (returns last one).

    Args:
        text: Solution text containing \\boxed{answer}

    Returns:
        Extracted answer string or None if not found
    """
    if not text or "\\boxed" not in text:
        return None

    # Find all boxed expressions
    # This handles nested braces by matching balanced pairs
    answers = []
    i = 0
    while i < len(text):
        if text[i:i+7] == "\\boxed{":
            # Find matching closing brace
            start = i + 7
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                answers.append(text[start:j-1])
            i = j
        else:
            i += 1

    # Return last boxed answer (typically the final answer)
    return answers[-1].strip() if answers else None


def extract_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract answer using multiple strategies.

    Tries in order:
    1. \\boxed{} format
    2. "The answer is X" pattern
    3. "= X" at end of solution
    """
    # Try boxed first
    answer = extract_boxed_answer(solution)
    if answer:
        return answer

    # Try "The answer is" pattern
    patterns = [
        r"[Tt]he (?:final )?answer is[:\s]*\$?([^\$\n]+)\$?",
        r"[Aa]nswer[:\s]*\$?([^\$\n]+)\$?$",
        r"[Tt]herefore[,\s]+(?:the answer is[:\s]*)?\$?([^\$\n]+)\$?",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution)
        if match:
            return match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison.

    Removes formatting, standardizes notation, and simplifies expressions.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    text = answer.strip()

    # Remove LaTeX text commands
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)

    # Remove formatting commands
    text = re.sub(r'\\left|\\right', '', text)
    text = re.sub(r'\\,|\\;|\\:|\\!', '', text)
    text = re.sub(r'\\quad|\\qquad', ' ', text)

    # Normalize fractions
    text = re.sub(r'\\[dt]?frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)

    # Normalize square roots
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\sqrt\[(\d+)\]\{([^}]*)\}', r'(\2)^(1/\1)', text)

    # Normalize exponents
    text = re.sub(r'\^{([^}]*)}', r'^(\1)', text)

    # Remove dollar signs
    text = text.replace('$', '')

    # Normalize whitespace
    text = ' '.join(text.split())

    # Lowercase for case-insensitive comparison
    text = text.lower()

    return text


def parse_latex_to_sympy(latex: str) -> Optional[sympy.Basic]:
    """
    Parse LaTeX expression to SymPy for symbolic comparison.

    Args:
        latex: LaTeX math expression

    Returns:
        SymPy expression or None if parsing fails
    """
    try:
        # Clean up LaTeX for parsing
        latex = latex.strip().strip('$')
        latex = re.sub(r'\\text\{[^}]*\}', '', latex)
        latex = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', latex)

        expr = sympy_parse_latex(latex)
        return expr
    except Exception:
        return None


def is_equivalent_answer(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if two mathematical answers are equivalent.

    Uses multiple comparison strategies:
    1. Exact string match after normalization
    2. Numerical comparison for numeric answers
    3. Symbolic comparison using SymPy

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Tolerance for numerical comparison

    Returns:
        True if answers are equivalent
    """
    if not predicted or not ground_truth:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Try numerical comparison
    try:
        pred_val = eval_numeric(predicted)
        gt_val = eval_numeric(ground_truth)

        if pred_val is not None and gt_val is not None:
            if abs(pred_val - gt_val) < tolerance:
                return True
            # Check relative tolerance for large numbers
            if abs(gt_val) > 1:
                if abs(pred_val - gt_val) / abs(gt_val) < tolerance:
                    return True
    except Exception:
        pass

    # Try symbolic comparison
    try:
        pred_sym = parse_latex_to_sympy(predicted)
        gt_sym = parse_latex_to_sympy(ground_truth)

        if pred_sym is not None and gt_sym is not None:
            diff = sympy.simplify(pred_sym - gt_sym)
            if diff == 0:
                return True
    except Exception:
        pass

    return False


def eval_numeric(expr: str) -> Optional[float]:
    """
    Evaluate expression to numeric value.

    Handles fractions, simple arithmetic, and common constants.

    Args:
        expr: Mathematical expression string

    Returns:
        Float value or None if evaluation fails
    """
    try:
        # Clean up the expression
        expr = normalize_answer(expr)

        # Handle fractions written as a/b
        if '/' in expr and not any(c in expr for c in ['+', '-', '*', '^']):
            parts = expr.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip('()'))
                den = float(parts[1].strip('()'))
                return num / den

        # Replace common constants
        expr = expr.replace('pi', str(3.141592653589793))
        expr = expr.replace('e', str(2.718281828459045))

        # Handle sqrt
        expr = re.sub(r'sqrt\(([^)]+)\)', r'((\1)**0.5)', expr)

        # Evaluate
        result = eval(expr)
        return float(result)
    except Exception:
        return None


def verify_answer(response: str, ground_truth: str) -> bool:
    """
    Verify if a model response contains the correct answer.

    Extracts answer from response and compares with ground truth.

    Args:
        response: Full model response
        ground_truth: Ground truth answer or solution

    Returns:
        True if answer is correct
    """
    # Extract answers
    pred_answer = extract_boxed_answer(response) or extract_answer_from_solution(response)
    gt_answer = extract_boxed_answer(ground_truth) or extract_answer_from_solution(ground_truth) or ground_truth

    if not pred_answer:
        return False

    return is_equivalent_answer(pred_answer, gt_answer)
