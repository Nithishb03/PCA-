"""Conservative validation for documentation-only Python source changes."""

from __future__ import annotations

import ast
import copy
import difflib
import io
import tokenize
from dataclasses import dataclass
from typing import Optional

try:
    from .models import DocumentationChange, ValidationResult
except ImportError:
    from models import DocumentationChange, ValidationResult


@dataclass(frozen=True)
class _DefinitionInfo:
    key: tuple[tuple[str, ...], str, str]
    name: str
    element_type: str
    node: ast.AST
    parent_class: Optional[str]


class _DefinitionCollector(ast.NodeVisitor):
    """Collect definitions using stable enclosing-definition paths."""

    def __init__(self) -> None:
        self.entries: dict[tuple[tuple[str, ...], str, str], _DefinitionInfo] = {}
        self.duplicates: set[tuple[tuple[str, ...], str, str]] = set()
        self.scope_stack: list[str] = []
        self.class_stack: list[str] = []
        self.body_context: list[bool] = []

    @property
    def in_class_body(self) -> bool:
        return bool(self.body_context and self.body_context[-1])

    def _add(self, node: ast.AST, element_type: str, parent_class: Optional[str]) -> None:
        name = getattr(node, "name")
        key = (tuple(self.scope_stack), element_type, name)
        info = _DefinitionInfo(key, name, element_type, node, parent_class)
        if key in self.entries:
            self.duplicates.add(key)
        self.entries[key] = info

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, "class", None)
        self.scope_stack.append(node.name)
        self.class_stack.append(node.name)
        for statement in node.body:
            self.body_context.append(True)
            self.visit(statement)
            self.body_context.pop()
        self.class_stack.pop()
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async_function")

    def _visit_function(self, node: ast.AST, function_type: str) -> None:
        is_method = bool(self.class_stack) and self.in_class_body
        element_type = "method" if is_method else function_type
        parent_class = self.class_stack[-1] if is_method else None
        self._add(node, element_type, parent_class)
        self.scope_stack.append(getattr(node, "name"))
        for statement in getattr(node, "body"):
            self.body_context.append(False)
            self.visit(statement)
            self.body_context.pop()
        self.scope_stack.pop()


class _DocumentationStripper(ast.NodeTransformer):
    """Remove definition docstrings while retaining all executable AST nodes."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        _remove_docstring(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        _remove_docstring(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        _remove_docstring(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        _remove_docstring(node)
        return node


def _remove_docstring(node: ast.AST) -> None:
    body = getattr(node, "body", None)
    if isinstance(body, list) and body and _is_docstring_node(body[0]):
        node.body = body[1:]


def _is_docstring_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _parse_source(source: str, label: str) -> ast.AST:
    return ast.parse(source, filename=label, type_comments=True)


def _normalized_dump(tree: ast.AST) -> str:
    normalized = _DocumentationStripper().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _module_docstring(tree: ast.Module) -> Optional[str]:
    return ast.get_docstring(tree, clean=False)


def _docstring_map(collector: _DefinitionCollector) -> dict[tuple[tuple[str, ...], str, str], Optional[str]]:
    return {key: ast.get_docstring(info.node, clean=False) for key, info in collector.entries.items()}


def _comment_tokens(source: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    )


class SafetyValidator:
    """Validate that a proposed Python change only updates one docstring."""

    def validate(self, change: DocumentationChange) -> ValidationResult:
        """Return a conservative validation result and never apply a change."""
        diff = self.generate_diff(change.original_source, change.proposed_source, change.file_path)
        errors = list(change.validation_errors)
        warnings = list(change.validation_warnings)

        if not change.success:
            errors.append("The documentation generation result was unsuccessful")
        if not change.file_path.lower().endswith(".py"):
            errors.append("Safety validation only supports Python source files")
        if errors:
            return ValidationResult(False, errors, warnings, diff)

        try:
            original_tree = _parse_source(change.original_source, change.file_path)
        except (SyntaxError, ValueError, TypeError) as error:
            errors.append(f"Invalid original Python source: {error}")
            return ValidationResult(False, errors, warnings, diff)

        try:
            proposed_tree = _parse_source(change.proposed_source, change.file_path)
        except (SyntaxError, ValueError, TypeError) as error:
            errors.append(f"Invalid proposed Python source: {error}")
            return ValidationResult(False, errors, warnings, diff)

        original_collector = _collect_definitions(original_tree)
        proposed_collector = _collect_definitions(proposed_tree)
        target = self._find_target(original_collector, change)
        if target is None:
            errors.append(f"Target element not found: {change.element_name}")
            return ValidationResult(False, errors, warnings, diff)
        if target.key not in proposed_collector.entries:
            errors.append(f"Target element not found in proposed source: {change.element_name}")
            return ValidationResult(False, errors, warnings, diff)
        if original_collector.duplicates or proposed_collector.duplicates:
            errors.append("Ambiguous duplicate definitions cannot be safely validated")
            return ValidationResult(False, errors, warnings, diff)

        if _normalized_dump(original_tree) != _normalized_dump(proposed_tree):
            errors.append("Executable or structural code changed")

        try:
            original_comments = _comment_tokens(change.original_source)
            proposed_comments = _comment_tokens(change.proposed_source)
        except (IndentationError, SyntaxError, tokenize.TokenError) as error:
            errors.append(f"Unable to compare source comments: {error}")
        else:
            if original_comments != proposed_comments:
                errors.append("Comments outside the target documentation changed")

        if _module_docstring(original_tree) != _module_docstring(proposed_tree):
            errors.append("Module documentation changed outside the target element")

        original_docs = _docstring_map(original_collector)
        proposed_docs = _docstring_map(proposed_collector)
        changed_docstrings = {
            key for key in original_docs if original_docs.get(key) != proposed_docs.get(key)
        }
        if target.key not in changed_docstrings and change.original_source != change.proposed_source:
            errors.append("No target documentation change was detected")
        if changed_docstrings - {target.key}:
            errors.append("Documentation changed outside the target element")
        if proposed_docs.get(target.key) is None:
            errors.append("Target documentation is missing from the proposed source")

        return ValidationResult(not errors, errors, warnings, diff)

    def validate_change(self, change: DocumentationChange) -> ValidationResult:
        """Alias for callers that prefer an explicit change-oriented method name."""
        return self.validate(change)

    @staticmethod
    def _find_target(
        collector: _DefinitionCollector,
        change: DocumentationChange,
    ) -> Optional[_DefinitionInfo]:
        candidates = [
            info
            for info in collector.entries.values()
            if info.name == change.element_name
            and _element_type_matches(change.element_type, info.element_type)
            and (change.parent_class is None or info.parent_class == change.parent_class)
        ]
        if change.line_number is not None:
            line_matches = [
                info for info in candidates if getattr(info.node, "lineno", None) == change.line_number
            ]
            if line_matches:
                candidates = line_matches
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def generate_diff(original_source: str, proposed_source: str, file_path: str = "source.py") -> str:
        """Return a unified diff without applying it anywhere."""
        return "".join(
            difflib.unified_diff(
                original_source.splitlines(keepends=True),
                proposed_source.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )


def _collect_definitions(tree: ast.AST) -> _DefinitionCollector:
    collector = _DefinitionCollector()
    collector.visit(tree)
    return collector


def _element_type_matches(requested: str, actual: str) -> bool:
    if requested == "function":
        return actual in {"function", "async_function"}
    if requested == "async_function":
        return actual == "async_function"
    return requested == actual
