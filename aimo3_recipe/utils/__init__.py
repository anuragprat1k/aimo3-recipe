"""Utility functions for AIMO3 recipe."""

from aimo3_recipe.utils.math_utils import (
    parse_latex,
    latex_to_sympy,
    is_equivalent_answer,
    extract_final_answer,
)
from aimo3_recipe.utils.code_execution import safe_execute_python, CodeExecutionResult
from aimo3_recipe.utils.safe_paths import (
    SafePathError,
    get_project_root,
    validate_path_within_project,
    safe_mkdir,
    safe_write_text,
    safe_open_for_write,
    safe_open_for_append,
)

__all__ = [
    "parse_latex",
    "latex_to_sympy",
    "is_equivalent_answer",
    "extract_final_answer",
    "safe_execute_python",
    "CodeExecutionResult",
    # Safe path utilities
    "SafePathError",
    "get_project_root",
    "validate_path_within_project",
    "safe_mkdir",
    "safe_write_text",
    "safe_open_for_write",
    "safe_open_for_append",
]
