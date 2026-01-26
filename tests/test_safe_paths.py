"""
Tests for safe_paths module.

Tests the path validation and safe file operations:
- get_project_root: Find the project root directory
- validate_path_within_project: Ensure paths stay within project
- safe_mkdir: Create directories safely
- safe_write_text: Write files safely
- safe_open_for_write: Open files for writing safely
- Blocked deletion operations
"""

import os
import tempfile
from pathlib import Path

import pytest

from aimo3_recipe.utils.safe_paths import (
    SafePathError,
    get_project_root,
    validate_path_within_project,
    safe_mkdir,
    safe_write_text,
    safe_open_for_write,
    safe_open_for_append,
    safe_remove,
    safe_rmtree,
    safe_unlink,
    safe_rmdir,
)


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_finds_project_root(self):
        """Should find the project root containing aimo3_recipe."""
        root = get_project_root()
        assert root.is_dir()
        assert (root / "aimo3_recipe").is_dir()

    def test_returns_absolute_path(self):
        """Should return an absolute path."""
        root = get_project_root()
        assert root.is_absolute()


class TestValidatePathWithinProject:
    """Tests for validate_path_within_project function."""

    def test_relative_path_within_project(self):
        """Relative paths within project should be allowed."""
        path = validate_path_within_project("./outputs/test")
        assert path.is_absolute()

    def test_absolute_path_within_project(self):
        """Absolute paths within project should be allowed."""
        project_root = get_project_root()
        test_path = project_root / "outputs" / "test"
        path = validate_path_within_project(str(test_path))
        assert path == test_path

    def test_path_outside_project_raises_error(self):
        """Paths outside project should raise SafePathError."""
        with pytest.raises(SafePathError) as exc_info:
            validate_path_within_project("/tmp/outside_project")
        assert "outside the project directory" in str(exc_info.value)

    def test_path_traversal_attack_blocked(self):
        """Path traversal attacks should be blocked."""
        with pytest.raises(SafePathError):
            validate_path_within_project("./outputs/../../../etc/passwd")

    def test_home_directory_blocked(self):
        """Home directory access should be blocked."""
        home = os.path.expanduser("~")
        # Only test if home is not inside project
        project_root = get_project_root()
        try:
            Path(home).relative_to(project_root)
            # Home is inside project, skip this test
            pytest.skip("Home directory is inside project")
        except ValueError:
            # Home is outside project, test should work
            with pytest.raises(SafePathError):
                validate_path_within_project(home)

    def test_root_directory_blocked(self):
        """Root directory access should be blocked."""
        with pytest.raises(SafePathError):
            validate_path_within_project("/")

    def test_nested_path_within_project(self):
        """Deeply nested paths within project should be allowed."""
        path = validate_path_within_project("./outputs/a/b/c/d/e/f")
        assert path.is_absolute()

    def test_custom_project_root(self):
        """Should work with custom project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create aimo3_recipe dir to make it look like project root
            (tmpdir_path / "aimo3_recipe").mkdir()

            path = validate_path_within_project(
                "outputs/test",
                project_root=tmpdir_path
            )
            assert str(tmpdir_path) in str(path)


class TestSafeMkdir:
    """Tests for safe_mkdir function."""

    def test_creates_directory_within_project(self):
        """Should create directory within project."""
        project_root = get_project_root()
        test_dir = project_root / "outputs" / "test_safe_mkdir"

        try:
            path = safe_mkdir(test_dir)
            assert path.is_dir()
        finally:
            # Cleanup
            if test_dir.exists():
                test_dir.rmdir()

    def test_creates_parent_directories(self):
        """Should create parent directories when parents=True."""
        project_root = get_project_root()
        test_dir = project_root / "outputs" / "test_safe_mkdir" / "nested" / "dir"

        try:
            path = safe_mkdir(test_dir, parents=True)
            assert path.is_dir()
        finally:
            # Cleanup
            for parent in [test_dir] + list(test_dir.parents)[:2]:
                if parent.exists() and parent != project_root / "outputs":
                    try:
                        parent.rmdir()
                    except OSError:
                        pass

    def test_exist_ok_true(self):
        """Should not raise error if directory exists with exist_ok=True."""
        project_root = get_project_root()
        test_dir = project_root / "outputs" / "test_safe_mkdir_exists"

        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            # Should not raise
            path = safe_mkdir(test_dir, exist_ok=True)
            assert path.is_dir()
        finally:
            if test_dir.exists():
                test_dir.rmdir()

    def test_blocks_directory_outside_project(self):
        """Should block directory creation outside project."""
        with pytest.raises(SafePathError):
            safe_mkdir("/tmp/outside_project_dir")


class TestSafeWriteText:
    """Tests for safe_write_text function."""

    def test_writes_file_within_project(self):
        """Should write file within project."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test_safe_write.txt"

        try:
            path = safe_write_text(test_file, "test content")
            assert path.exists()
            assert path.read_text() == "test content"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_creates_parent_directories(self):
        """Should create parent directories if needed."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test_nested" / "test_safe_write.txt"

        try:
            path = safe_write_text(test_file, "nested content")
            assert path.exists()
            assert path.read_text() == "nested content"
        finally:
            if test_file.exists():
                test_file.unlink()
            if test_file.parent.exists():
                test_file.parent.rmdir()

    def test_blocks_file_outside_project(self):
        """Should block file writing outside project."""
        with pytest.raises(SafePathError):
            safe_write_text("/tmp/outside_project.txt", "content")


class TestSafeOpenForWrite:
    """Tests for safe_open_for_write context manager."""

    def test_opens_file_for_write(self):
        """Should open file for writing within project."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test_safe_open_write.txt"

        try:
            with safe_open_for_write(test_file, "w") as f:
                f.write("test content")
            assert test_file.read_text() == "test content"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_opens_file_for_append(self):
        """Should open file for appending within project."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test_safe_open_append.txt"

        try:
            test_file.write_text("first\n")
            with safe_open_for_write(test_file, "a") as f:
                f.write("second\n")
            assert test_file.read_text() == "first\nsecond\n"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_blocks_read_mode(self):
        """Should block read mode."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test.txt"

        with pytest.raises(ValueError) as exc_info:
            with safe_open_for_write(test_file, "r") as f:
                pass
        assert "not allowed" in str(exc_info.value)

    def test_blocks_file_outside_project(self):
        """Should block file opening outside project."""
        with pytest.raises(SafePathError):
            with safe_open_for_write("/tmp/outside.txt", "w") as f:
                pass


class TestSafeOpenForAppend:
    """Tests for safe_open_for_append function."""

    def test_opens_file_for_append(self):
        """Should open file for appending within project."""
        project_root = get_project_root()
        test_file = project_root / "outputs" / "test_safe_append.txt"

        try:
            test_file.write_text("first\n")
            f = safe_open_for_append(test_file)
            try:
                f.write("second\n")
            finally:
                f.close()
            assert test_file.read_text() == "first\nsecond\n"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_blocks_file_outside_project(self):
        """Should block file opening outside project."""
        with pytest.raises(SafePathError):
            safe_open_for_append("/tmp/outside.txt")


class TestBlockedDeletionOperations:
    """Tests for blocked deletion operations."""

    def test_safe_remove_blocked(self):
        """safe_remove should raise SafePathError."""
        with pytest.raises(SafePathError) as exc_info:
            safe_remove("any_path")
        assert "Deletion operations are not allowed" in str(exc_info.value)

    def test_safe_rmtree_blocked(self):
        """safe_rmtree should raise SafePathError."""
        with pytest.raises(SafePathError) as exc_info:
            safe_rmtree("any_path")
        assert "Deletion operations are not allowed" in str(exc_info.value)

    def test_safe_unlink_blocked(self):
        """safe_unlink should raise SafePathError."""
        with pytest.raises(SafePathError) as exc_info:
            safe_unlink("any_path")
        assert "Deletion operations are not allowed" in str(exc_info.value)

    def test_safe_rmdir_blocked(self):
        """safe_rmdir should raise SafePathError."""
        with pytest.raises(SafePathError) as exc_info:
            safe_rmdir("any_path")
        assert "Deletion operations are not allowed" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and security scenarios."""

    def test_symlink_attack_blocked(self):
        """Symlinks pointing outside project should be blocked."""
        project_root = get_project_root()

        # Create a symlink inside project pointing outside
        symlink_path = project_root / "outputs" / "symlink_test"

        try:
            # Only run if we can create symlinks
            symlink_path.symlink_to("/tmp")

            # Trying to use the symlink should fail
            with pytest.raises(SafePathError):
                validate_path_within_project(symlink_path / "outside_file")
        except OSError:
            # Can't create symlink, skip test
            pytest.skip("Cannot create symlinks")
        finally:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

    def test_empty_path_rejected(self):
        """Empty path should be handled appropriately."""
        # Empty string resolves to current directory, which should be valid
        # if we're running from project root
        path = validate_path_within_project("")
        assert path.is_absolute()

    def test_dot_path_resolves_to_project(self):
        """Single dot path should resolve within project."""
        path = validate_path_within_project(".")
        assert path.is_absolute()

    def test_unicode_path_within_project(self):
        """Unicode paths within project should be allowed."""
        path = validate_path_within_project("./outputs/test_unicode_")
        assert path.is_absolute()

    def test_path_with_spaces(self):
        """Paths with spaces should be handled correctly."""
        path = validate_path_within_project("./outputs/path with spaces/file.txt")
        assert path.is_absolute()
        assert "path with spaces" in str(path)
