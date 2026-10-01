"""AST-based static analysis for Python source files."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional

try:
    from .models import AnalysisResult, CodeElement
except ImportError:
    from models import AnalysisResult, CodeElement

IGNORED_DIRECTORIES = frozenset({".git", "venv", ".venv", "__pycache__", "generated"})


class _ElementVisitor(ast.NodeVisitor):
    """Collect documented and undocumented definitions from a Python AST."""

    def __init__(self) -> None:
        self.elements: list[CodeElement] = []
        self.class_stack: list[str] = []
        self.body_context: list[bool] = []

    @property
    def in_class_body(self) -> bool:
        return bool(self.body_context and self.body_context[-1])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.elements.append(_element_from_node(node, "class", None))
        self.class_stack.append(node.name)
        for statement in node.body:
            self.body_context.append(True)
            self.visit(statement)
            self.body_context.pop()
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self.class_stack) and self.in_class_body
        parent_class = self.class_stack[-1] if is_method else None
        element_type = "method" if is_method else "function"
        if isinstance(node, ast.AsyncFunctionDef) and not is_method:
            element_type = "async_function"
        self.elements.append(_element_from_node(node, element_type, parent_class))

        for statement in node.body:
            self.body_context.append(False)
            self.visit(statement)
            self.body_context.pop()


def _element_from_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    element_type: str,
    parent_class: Optional[str],
) -> CodeElement:
    docstring = ast.get_docstring(node)
    name = node.name
    return CodeElement(
        name=name,
        element_type=element_type,
        line_number=node.lineno,
        end_line_number=getattr(node, "end_lineno", None),
        has_docstring=docstring is not None,
        docstring=docstring,
        is_private=name.startswith("_"),
        parent_class=parent_class,
        needs_documentation=not name.startswith("_") and docstring is None,
    )


class CodeAnalyzer:
    """Analyze Python source without accessing or modifying repositories."""

    def analyze_python_file(self, source: str, file_path: str) -> AnalysisResult:
        """Analyze one Python source string and return a structured result."""
        if not file_path.lower().endswith(".py"):
            return AnalysisResult(
                file_path=file_path,
                language="unsupported",
                elements=[],
                error="Unsupported file extension; only Python files are supported",
            )

        try:
            tree = ast.parse(source, filename=file_path, type_comments=True)
        except (SyntaxError, ValueError, TypeError) as error:
            return AnalysisResult(
                file_path=file_path,
                language="python",
                elements=[],
                error=f"Invalid Python syntax: {error.msg if isinstance(error, SyntaxError) else error}",
            )

        visitor = _ElementVisitor()
        visitor.visit(tree)
        return AnalysisResult(file_path=file_path, language="python", elements=visitor.elements)

    def analyze_file(self, source: str, file_path: str) -> AnalysisResult:
        """Analyze a supported source file, reporting unsupported extensions."""
        return self.analyze_python_file(source, file_path)

    def analyze_files(
        self,
        files: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> list[AnalysisResult]:
        """Analyze supported files while ignoring non-Python and excluded paths."""
        file_items = files.items() if isinstance(files, Mapping) else files
        results = []
        for file_path, source in file_items:
            if self._should_ignore(file_path) or not file_path.lower().endswith(".py"):
                continue
            results.append(self.analyze_python_file(source, file_path))
        return results

    @staticmethod
    def _should_ignore(file_path: str) -> bool:
        return bool(set(Path(file_path).parts) & IGNORED_DIRECTORIES)
