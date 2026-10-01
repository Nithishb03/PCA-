import io
import json
from urllib.error import HTTPError

import pytest

import src.comment_generator as comment_generator_module
from src.comment_generator import (
    CommentGenerator,
    GroqProvider,
    LLMAuthenticationError,
    LLMProviderError,
    LLMResponseError,
)
from src.models import CodeElement, CommentGenerationResult


def make_element(element_type="function", name="calculate_total", is_private=False):
    return CodeElement(
        name=name,
        element_type=element_type,
        line_number=1,
        end_line_number=2,
        has_docstring=False,
        docstring=None,
        is_private=is_private,
        parent_class="Example" if element_type == "method" else None,
        needs_documentation=not is_private,
    )


class FakeProvider:
    def __init__(self, response="Useful documentation.", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_successful_generation_returns_structured_result():
    provider = FakeProvider("Calculate the total value of the provided items.")
    generator = CommentGenerator(provider=provider)

    result = generator.generate_documentation(
        "def calculate_total(items):\n    return sum(items)\n",
        make_element(),
        context="This helper receives numeric items.",
    )

    assert isinstance(result, CommentGenerationResult)
    assert result.success is True
    assert result.documentation == "Calculate the total value of the provided items."
    assert result.error is None
    assert "calculate_total" in provider.prompts[0]
    assert "Generate documentation only" in provider.prompts[0]
    assert "This helper receives numeric items." in provider.prompts[0]


def test_missing_api_key_returns_controlled_result(monkeypatch, caplog):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    caplog.set_level("INFO")

    result = CommentGenerator().generate_documentation("def function():\n    pass\n", make_element())

    assert result.success is False
    assert result.documentation is None
    assert "GROQ_API_KEY" in result.error
    assert "Groq API key unavailable to Python." in caplog.text
    assert "secret" not in caplog.text.lower()


def test_provider_api_failure_returns_safe_result():
    provider = FakeProvider(error=LLMProviderError("network failure with secret-value"))

    result = CommentGenerator(provider=provider).generate_documentation("source", make_element())

    assert result.success is False
    assert result.error == "LLM provider request failed"
    assert "secret-value" not in result.error


def test_empty_provider_response_returns_failure():
    result = CommentGenerator(provider=FakeProvider("   ")).generate_documentation(
        "source", make_element()
    )

    assert result.success is False
    assert result.error == "LLM provider returned empty documentation"


def test_markdown_fence_is_removed_from_response():
    result = CommentGenerator(provider=FakeProvider("```python\npass\n```")).generate_documentation(
        "source", make_element()
    )

    assert result.success is True
    assert result.documentation == "pass"


def test_surrounding_docstring_quotes_are_removed():
    result = CommentGenerator(provider=FakeProvider('"""Useful documentation."""')).generate_documentation(
        "source", make_element()
    )

    assert result.success is True
    assert result.documentation == "Useful documentation."


def test_unclosed_markdown_fence_returns_failure():
    result = CommentGenerator(provider=FakeProvider("```python\npass")).generate_documentation(
        "source", make_element()
    )

    assert result.success is False
    assert result.error == "LLM provider returned malformed documentation"


def test_non_text_response_returns_failure():
    result = CommentGenerator(provider=FakeProvider(response=None)).generate_documentation(
        "source", make_element()
    )

    assert result.success is False
    assert result.error == "LLM provider returned non-text documentation"


@pytest.mark.parametrize("element_type", ["function", "async_function", "class", "method"])
def test_supported_element_types_generate_documentation(element_type):
    result = CommentGenerator(provider=FakeProvider()).generate_documentation(
        "source", make_element(element_type=element_type)
    )

    assert result.success is True


@pytest.mark.parametrize("element_type", ["function", "async_function", "class", "method"])
def test_prompt_contains_element_specific_metadata_and_guidance(element_type):
    provider = FakeProvider()
    element = make_element(element_type=element_type, name="target_element")

    result = CommentGenerator(provider=provider).generate_documentation(
        "def target_element(value):\n    return value\n",
        element,
        context="Only the supplied source is authoritative.",
    )

    assert result.success is True
    prompt = provider.prompts[0]
    assert "target_element" in prompt
    assert f"element_type = {element_type}" in prompt
    assert "Only the supplied source is authoritative." in prompt
    assert "Do not invent parameters, return values, exceptions, side effects, or behavior." in prompt
    assert "Return only raw documentation text suitable for a Python docstring." in prompt


def test_private_element_is_skipped_without_calling_provider():
    provider = FakeProvider()

    result = CommentGenerator(provider=provider).generate_documentation(
        "def _private():\n    pass\n", make_element(name="_private", is_private=True)
    )

    assert result.success is False
    assert result.error == "Private code elements are not automatically documented"
    assert provider.prompts == []


def test_source_length_is_limited_in_prompt():
    provider = FakeProvider()
    source = "\n".join(f"line_{index} = {index}" for index in range(100))

    result = CommentGenerator(provider=provider, max_source_chars=100).generate_documentation(
        source, make_element()
    )

    assert result.success is True
    assert "source truncated" in provider.prompts[0]
    assert len(provider.prompts[0].split("Source:\n", 1)[1].split("\n\nElement:", 1)[0]) <= 100


def test_groq_provider_reads_model_from_environment(monkeypatch, caplog):
    captured = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        return FakeResponse({"choices": [{"message": {"content": "Documented."}}]})

    monkeypatch.setenv("GROQ_MODEL", "test-model")
    caplog.set_level("INFO")
    result = GroqProvider(api_key="secret-value", opener=opener).generate("prompt")

    assert result == "Documented."
    assert captured["body"]["model"] == "test-model"
    assert "secret-value" in captured["authorization"]
    assert "Groq request started." in caplog.text
    assert "Groq request succeeded and documentation was generated." in caplog.text
    assert "secret-value" not in caplog.text


def test_groq_provider_handles_api_failure_without_exposing_secret():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO())

    with pytest.raises(LLMProviderError) as raised:
        GroqProvider(api_key="secret-value", opener=opener).generate("prompt")

    assert "secret-value" not in str(raised.value)


def test_groq_provider_handles_malformed_response():
    def opener(request, timeout):
        return FakeResponse({"choices": []})

    with pytest.raises(LLMResponseError, match="no completion choices"):
        GroqProvider(api_key="secret-value", opener=opener).generate("prompt")


def test_groq_provider_handles_authentication_failure():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        GroqProvider(api_key="secret-value", opener=opener).generate("prompt")
