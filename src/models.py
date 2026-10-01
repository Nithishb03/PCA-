"""Shared data models for project-comment-automation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Repository:
    """A repository managed by the automation project."""

    name: str
    url: str
    schedule_day: str
    protected: bool


@dataclass(frozen=True)
class CodeElement:
    """A Python function, method, or class found during static analysis."""

    name: str
    element_type: str
    line_number: int
    end_line_number: Optional[int]
    has_docstring: bool
    docstring: Optional[str]
    is_private: bool
    parent_class: Optional[str]
    needs_documentation: bool


@dataclass(frozen=True)
class AnalysisResult:
    """The result of analyzing one source file."""

    file_path: str
    language: str
    elements: list[CodeElement]
    error: Optional[str] = None


@dataclass(frozen=True)
class CommentGenerationResult:
    """A proposed documentation result from an LLM provider."""

    success: bool
    documentation: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass(frozen=True)
class DocumentationChange:
    """A proposed, not-yet-applied documentation-only source change."""

    file_path: str
    element_name: str
    element_type: str
    original_source: str
    proposed_source: str
    documentation: Optional[str] = None
    line_number: Optional[int] = None
    parent_class: Optional[str] = None
    success: bool = True
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    """The conservative result of validating a proposed source change."""

    safe: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diff: str = ""


@dataclass(frozen=True)
class ValidatedChange:
    """A safety-validated change ready for optional controlled writing."""

    repository: Repository
    file_path: str
    original_source: str
    proposed_source: str
    element_name: str
    element_type: str
    validation_result: Optional[ValidationResult]
    generation_result: Optional[CommentGenerationResult] = None
    file_sha: Optional[str] = None
    branch: Optional[str] = None


@dataclass(frozen=True)
class WriteResult:
    """The result of a dry-run or controlled GitHub file write."""

    success: bool
    dry_run: bool
    repository: Optional[str] = None
    file_path: Optional[str] = None
    commit_sha: Optional[str] = None
    commit_url: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
