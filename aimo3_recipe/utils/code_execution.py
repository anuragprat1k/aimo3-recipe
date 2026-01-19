"""
Safe code execution utilities for Tool-Integrated Reasoning.
"""

from dataclasses import dataclass
from typing import Optional

# Re-export from tir_executor for convenience
from aimo3_recipe.inference.tir_executor import (
    CodeExecutionResult,
    safe_execute_python,
    validate_code,
    create_safe_builtins,
)

__all__ = [
    "CodeExecutionResult",
    "safe_execute_python",
    "validate_code",
    "create_safe_builtins",
]
