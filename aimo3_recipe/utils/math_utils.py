"""
Mathematical utility functions for parsing and comparing answers.
"""

import re
from typing import Optional, Union

import sympy
from sympy.parsing.latex import parse_latex as sympy_parse_latex


def parse_latex(latex: str) -> Optional[sympy.Basic]:
    """
    Parse LaTeX expression to SymPy.

    Args:
        latex: LaTeX math string

    Returns:
        SymPy expression or None if parsing fails
    """
    try:
        # Clean the input
        latex = latex.strip().strip('$')
        latex = re.sub(r'\\text\{[^}]*\}', '', latex)
        latex = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', latex)
        latex = re.sub(r'\\left|\\right', '', latex)

        return sympy_parse_latex(latex)
    except Exception:
        return None


def latex_to_sympy(latex: str) -> Optional[sympy.Basic]:
    """Alias for parse_latex for backwards compatibility."""
    return parse_latex(latex)


def is_equivalent_answer(
    answer1: str,
    answer2: str,
    tolerance: float = 1e-9,
) -> bool:
    """
    Check if two mathematical answers are equivalent.

    Tries multiple comparison strategies:
    1. String comparison after normalization
    2. Numeric comparison
    3. Symbolic comparison

    Args:
        answer1: First answer
        answer2: Second answer
        tolerance: Numeric tolerance

    Returns:
        True if answers are equivalent
    """
    from aimo3_recipe.evaluation.answer_extraction import (
        normalize_answer,
        is_equivalent_answer as _is_equivalent,
    )
    return _is_equivalent(answer1, answer2, tolerance)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from solution text.

    Looks for \\boxed{} or common answer patterns.

    Args:
        text: Solution text

    Returns:
        Extracted answer or None
    """
    from aimo3_recipe.evaluation.answer_extraction import extract_boxed_answer

    # Try boxed first
    answer = extract_boxed_answer(text)
    if answer:
        return answer

    # Try other patterns
    patterns = [
        r'[Tt]he (?:final )?answer is[:\s]*\$?([^\$\n]+)\$?',
        r'[Aa]nswer[:\s]*\$?([^\$\n]+)\$?$',
        r'= \$?([^\$\n]+)\$?$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None


def simplify_expression(expr: str) -> str:
    """
    Simplify a mathematical expression using SymPy.

    Args:
        expr: Mathematical expression (LaTeX or plain)

    Returns:
        Simplified expression as string
    """
    try:
        sym_expr = parse_latex(expr)
        if sym_expr is not None:
            simplified = sympy.simplify(sym_expr)
            return str(simplified)
    except Exception:
        pass

    return expr


def evaluate_expression(expr: str) -> Optional[float]:
    """
    Evaluate a mathematical expression to a numeric value.

    Args:
        expr: Mathematical expression

    Returns:
        Float value or None
    """
    try:
        sym_expr = parse_latex(expr)
        if sym_expr is not None:
            result = float(sym_expr.evalf())
            return result
    except Exception:
        pass

    # Try direct Python evaluation
    try:
        # Clean and evaluate
        clean_expr = expr.replace('^', '**')
        clean_expr = re.sub(r'\\sqrt\{([^}]+)\}', r'((\1)**0.5)', clean_expr)
        clean_expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', clean_expr)
        return float(eval(clean_expr))
    except Exception:
        pass

    return None


def format_answer(answer: Union[str, int, float, sympy.Basic]) -> str:
    """
    Format an answer for display in \\boxed{}.

    Args:
        answer: Answer to format

    Returns:
        Formatted answer string
    """
    if isinstance(answer, sympy.Basic):
        return sympy.latex(answer)
    elif isinstance(answer, float):
        if answer == int(answer):
            return str(int(answer))
        return f"{answer:.6g}"
    else:
        return str(answer)


def parse_fraction(text: str) -> Optional[tuple[int, int]]:
    """
    Parse a fraction from text.

    Handles formats like:
    - "3/4"
    - "\\frac{3}{4}"
    - "3 over 4"

    Args:
        text: Text containing a fraction

    Returns:
        Tuple (numerator, denominator) or None
    """
    # Try LaTeX fraction
    match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', text)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try simple fraction
    match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try "over" format
    match = re.search(r'(\d+)\s+over\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def is_integer_answer(answer: str) -> bool:
    """Check if an answer is an integer."""
    try:
        val = evaluate_expression(answer)
        if val is not None:
            return val == int(val)
    except Exception:
        pass

    # Check string
    clean = answer.strip().lstrip('-')
    return clean.isdigit()


def greatest_common_divisor(a: int, b: int) -> int:
    """Compute GCD of two integers."""
    import math
    return math.gcd(a, b)


def simplify_fraction(num: int, den: int) -> tuple[int, int]:
    """Simplify a fraction to lowest terms."""
    gcd = greatest_common_divisor(abs(num), abs(den))
    return num // gcd, den // gcd
