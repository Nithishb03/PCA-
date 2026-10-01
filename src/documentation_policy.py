"""Central policy for source files and code elements eligible for documentation."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import unquote

try:
    from .models import CodeElement
except ImportError:
    from models import CodeElement

PROTECTED_PATH_PREFIXES = (".github/", "config/")
PROTECTED_PATHS = frozenset(
    {
        ".gitignore",
        "requirements.txt",
        "src/comment_generator.py",
        "src/github_writer.py",
        "src/github_manager.py",
        "src/models.py",
        "src/repository_manager.py",
        "src/safety_validator.py",
    }
)
IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "__pypackages__",
        "build",
        "dist",
        "eggs",
        "env",
        "generated",
        "node_modules",
        "site-packages",
        "venv",
        ".venv",
        "virtualenv",
    }
)
TEST_DIRECTORY_NAMES = frozenset({"test", "tests", "testing"})
MIGRATION_DIRECTORY_NAMES = frozenset({"migration", "migrations"})
SUPPORTED_ELEMENT_TYPES = frozenset({"function", "async_function", "class", "method"})
MAX_SOURCE_CHARS = 100000


class DocumentationPolicy:
    """Decide which repository paths and analyzed elements may be documented."""

    def allows_directory(self, path: str) -> bool:
        """Return whether traversal may enter a repository-relative directory."""
        normalized = self._normalize_path(path)
        if normalized is None:
            return False
        parts = {part.lower() for part in PurePosixPath(normalized).parts}
        return not (
            parts & IGNORED_DIRECTORY_NAMES
            or parts & TEST_DIRECTORY_NAMES
            or parts & MIGRATION_DIRECTORY_NAMES
            or self.is_protected_path(normalized)
        )

    def allows_file(self, path: str, source_size: Optional[int] = None) -> bool:
        """Return whether a repository-relative file may be analyzed."""
        normalized = self._normalize_path(path)
        if normalized is None or not normalized.lower().endswith(".py"):
            return False
        if self.is_protected_path(normalized):
            return False
        parts = [part.lower() for part in PurePosixPath(normalized).parts]
        if (
            set(parts) & (IGNORED_DIRECTORY_NAMES | TEST_DIRECTORY_NAMES | MIGRATION_DIRECTORY_NAMES)
        ):
            return False
        filename = parts[-1]
        stem = filename[:-3]
        if (
            stem.startswith("test_")
            or stem.endswith("_test")
            or stem in {"test", "migration", "migrations"}
            or stem.startswith("generated_")
            or stem.endswith("_generated")
            or stem.endswith(".generated")
        ):
            return False
        if source_size is not None and source_size > MAX_SOURCE_CHARS:
            return False
        return True

    def allows_element(self, element: CodeElement) -> bool:
        """Return whether one analyzed element is eligible for documentation."""
        if element.element_type not in SUPPORTED_ELEMENT_TYPES:
            return False
        if element.name.startswith("_"):
            return False
        return not bool(element.docstring and element.docstring.strip())

    @staticmethod
    def is_protected_path(path: str) -> bool:
        normalized = path.replace("\\", "/").lstrip("/").lower()
        return normalized in PROTECTED_PATHS or normalized.startswith(PROTECTED_PATH_PREFIXES)

    @staticmethod
    def _normalize_path(path: str) -> Optional[str]:
        if not isinstance(path, str) or not path:
            return None
        normalized = path.replace("\\", "/")
        for _ in range(3):
            decoded = unquote(normalized)
            if decoded == normalized:
                break
            normalized = decoded
        if (
            normalized.startswith("/")
            or (len(normalized) > 1 and normalized[1] == ":")
            or "\x00" in normalized
        ):
            return None
        parts = PurePosixPath(normalized).parts
        if any(part in {"", ".", ".."} for part in parts):
            return None
        return normalized
