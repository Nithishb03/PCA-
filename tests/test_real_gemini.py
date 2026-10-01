"""Opt-in live Gemini integration validation and quality/diff inspection.

Run explicitly with:
RUN_REAL_GEMINI_TEST=true and GEMINI_API_KEY set.
"""

from __future__ import annotations

import logging
import os

import pytest

from src.code_analyzer import CodeAnalyzer
from src.comment_generator import CommentGenerator
from src.main import _build_proposed_source
from src.models import DocumentationChange
from src.safety_validator import SafetyValidator

LOGGER = logging.getLogger(__name__)

SAMPLE_SOURCE = (
    "def calculate_total(items: list[float], tax_rate: float) -> float:\n"
    "    return sum(items) * (1.0 + tax_rate)\n"
    "\n\n"
    "class OrderSummary:\n"
    "    def format_receipt(self, currency: str) -> str:\n"
    '        return f"{currency} 0.00"\n'
)


@pytest.mark.skipif(
    os.getenv("RUN_REAL_GEMINI_TEST", "").lower() != "true",
    reason="Set RUN_REAL_GEMINI_TEST=true to run the opt-in live Gemini validation",
)
def test_real_gemini_documentation_quality_and_diff(monkeypatch):
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is unavailable")

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    LOGGER.info("Real Gemini documentation quality and diff validation started.")

    analyzer = CodeAnalyzer()
    generator = CommentGenerator()
    validator = SafetyValidator()

    analysis = analyzer.analyze_python_file(SAMPLE_SOURCE, "sample.py")
    assert not analysis.error
    assert len(analysis.elements) == 3

    element_types = [el.element_type for el in analysis.elements]
    assert "function" in element_types
    assert "class" in element_types
    assert "method" in element_types

    diffs = {}
    for element in analysis.elements:
        LOGGER.info("Generating documentation for %s (%s)...", element.name, element.element_type)
        generation = generator.generate_documentation(SAMPLE_SOURCE, element)

        assert generation.success is True
        assert generation.documentation
        assert "```" not in generation.documentation

        proposed_source = _build_proposed_source(
            SAMPLE_SOURCE,
            element,
            generation.documentation,
        )
        change = DocumentationChange(
            file_path="sample.py",
            element_name=element.name,
            element_type=element.element_type,
            original_source=SAMPLE_SOURCE,
            proposed_source=proposed_source,
            documentation=generation.documentation,
            line_number=element.line_number,
            parent_class=element.parent_class,
        )
        validation = validator.validate(change)

        assert validation.safe is True
        assert validation.diff
        diffs[element.name] = validation.diff

        # Safe output without exposing credentials
        print(f"\n=== Element: {element.name} ({element.element_type}) ===")
        print(f"Generated Docstring:\n{generation.documentation}\n")
        print(f"Proposed Diff:\n{validation.diff}")
        LOGGER.info("Generated docstring for %s:\n%s", element.name, generation.documentation)
        LOGGER.info("Unified diff for %s:\n%s", element.name, validation.diff)

    assert len(diffs) == 3
    LOGGER.info("All 3 elements successfully documented and verified by SafetyValidator.")
