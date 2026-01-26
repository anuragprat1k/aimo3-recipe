"""
Safe Path Utilities for RL Training Scripts

This module provides path validation and safe file operations to ensure:
1. All paths stay within the project folder
2. No deletion operations are allowed
3. Path traversal attacks are prevented

Usage:
    from aimo3_recipe.utils.safe_paths import (
        get_project_root,
        validate_path_within_project,
        safe_mkdir,
        safe_write_text,
        safe_open_for_write,
        SafePathError,
    )
"""

import os
from pathlib import Path
from typing import Union, TextIO
from contextlib import contextmanager


class SafePathError(Exception):
    """Exception raised when a path operation violates safety constraints."""
    pass


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    The project root is determined by finding the directory containing
    'aimo3_recipe' package, or falling back to the current working directory
    if running from within the project.

    Returns:
        Path: The absolute path to the project root directory.
    """
    # Try to find project root by looking for aimo3_recipe package
    current_file = Path(__file__).resolve()

    # Navigate up from utils/safe_paths.py -> utils -> aimo3_recipe -> project_root
    project_root = current_file.parent.parent.parent

    # Verify this looks like the project root
    if (project_root / "aimo3_recipe").is_dir():
        return project_root

    # Fall back to current working directory
    cwd = Path.cwd().resolve()

    # Check if cwd contains aimo3_recipe
    if (cwd / "aimo3_recipe").is_dir():
        return cwd

    # If we're inside the project, walk up to find root
    for parent in cwd.parents:
        if (parent / "aimo3_recipe").is_dir():
            return parent

    # Last resort: use cwd (may fail validation later)
    return cwd


def validate_path_within_project(
    path: Union[str, Path],
    project_root: Path = None,
) -> Path:
    """
    Validate that a path is within the project directory.

    This function resolves the path to an absolute path and checks that it
    is within the project root directory. It prevents path traversal attacks
    and ensures all file operations stay within the project.

    Args:
        path: The path to validate (can be relative or absolute).
        project_root: Optional project root override. If None, uses get_project_root().

    Returns:
        Path: The resolved absolute path.

    Raises:
        SafePathError: If the path is outside the project directory.
    """
    if project_root is None:
        project_root = get_project_root()

    project_root = project_root.resolve()

    # Convert to Path and resolve
    path = Path(path)

    # If relative, resolve relative to project root
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()

    # Check if path is within project root
    try:
        path.relative_to(project_root)
    except ValueError:
        raise SafePathError(
            f"Path '{path}' is outside the project directory '{project_root}'. "
            f"All file operations must stay within the project folder."
        )

    return path


def safe_mkdir(
    path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True,
    project_root: Path = None,
) -> Path:
    """
    Safely create a directory within the project folder.

    Args:
        path: The directory path to create.
        parents: If True, create parent directories as needed.
        exist_ok: If True, don't raise an error if directory exists.
        project_root: Optional project root override.

    Returns:
        Path: The created directory path.

    Raises:
        SafePathError: If the path is outside the project directory.
    """
    validated_path = validate_path_within_project(path, project_root)
    validated_path.mkdir(parents=parents, exist_ok=exist_ok)
    return validated_path


def safe_write_text(
    path: Union[str, Path],
    content: str,
    project_root: Path = None,
) -> Path:
    """
    Safely write text content to a file within the project folder.

    Args:
        path: The file path to write to.
        content: The text content to write.
        project_root: Optional project root override.

    Returns:
        Path: The written file path.

    Raises:
        SafePathError: If the path is outside the project directory.
    """
    validated_path = validate_path_within_project(path, project_root)

    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)

    validated_path.write_text(content)
    return validated_path


@contextmanager
def safe_open_for_write(
    path: Union[str, Path],
    mode: str = "w",
    project_root: Path = None,
) -> TextIO:
    """
    Safely open a file for writing within the project folder.

    This is a context manager that validates the path and opens the file.
    Only write modes are allowed ('w', 'a', 'x' and their binary variants).

    Args:
        path: The file path to open.
        mode: The file mode (must be a write mode).
        project_root: Optional project root override.

    Yields:
        TextIO: The opened file handle.

    Raises:
        SafePathError: If the path is outside the project directory.
        ValueError: If the mode is not a write mode.

    Example:
        with safe_open_for_write("outputs/samples.jsonl", "a") as f:
            f.write(json.dumps(sample) + "\\n")
    """
    # Validate mode is for writing
    valid_write_modes = {"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "x+"}
    if mode not in valid_write_modes:
        raise ValueError(
            f"Mode '{mode}' is not allowed. Only write modes are permitted: {valid_write_modes}"
        )

    validated_path = validate_path_within_project(path, project_root)

    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(validated_path, mode)
    try:
        yield f
    finally:
        f.close()


def safe_open_for_append(
    path: Union[str, Path],
    project_root: Path = None,
) -> TextIO:
    """
    Safely open a file for appending within the project folder.

    Note: This returns an open file handle that must be closed by the caller.
    For context manager usage, use safe_open_for_write() instead.

    Args:
        path: The file path to open.
        project_root: Optional project root override.

    Returns:
        TextIO: The opened file handle.

    Raises:
        SafePathError: If the path is outside the project directory.
    """
    validated_path = validate_path_within_project(path, project_root)

    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)

    return open(validated_path, "a")


# Explicitly block deletion operations
def _block_delete(*args, **kwargs):
    """Blocked function that raises an error."""
    raise SafePathError(
        "Deletion operations are not allowed in RL training scripts. "
        "This is a safety measure to prevent accidental data loss."
    )


# These are intentionally defined to block any accidental use
safe_remove = _block_delete
safe_rmtree = _block_delete
safe_unlink = _block_delete
safe_rmdir = _block_delete


__all__ = [
    "SafePathError",
    "get_project_root",
    "validate_path_within_project",
    "safe_mkdir",
    "safe_write_text",
    "safe_open_for_write",
    "safe_open_for_append",
    # Blocked operations (will raise errors if called)
    "safe_remove",
    "safe_rmtree",
    "safe_unlink",
    "safe_rmdir",
]
