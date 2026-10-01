import io
import json
from urllib.error import HTTPError

import pytest

import src.comment_generator as comment_generator_module
from src.comment_generator import (
    CommentGenerator,
    GeminiProvider,
    GroqProvider,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceUnavailableError,
    MissingLLMAPIKeyError,
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


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.prompts = []
        self.calls = []

    def generate_content(self, model, contents, config=None, **kwargs):
        self.prompts.append((model, contents))
        self.calls.append({"model": model, "contents": contents, "config": config, **kwargs})
        if self.error:
            raise self.error
        return self.response


class FakeGeminiClient:
    def __init__(self, response=None, error=None):
        self.models = FakeGeminiModels(response=response, error=error)


class SequenceGeminiModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.prompts = []
        self.calls = []

    def generate_content(self, model, contents, config=None, **kwargs):
        self.prompts.append((model, contents))
        self.calls.append({"model": model, "contents": contents, "config": config, **kwargs})
        if not self.outcomes:
            raise RuntimeError("No more outcomes configured")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SequenceGeminiClient:
    def __init__(self, outcomes):
        self.models = SequenceGeminiModels(outcomes)


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
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    caplog.set_level("INFO")

    result = CommentGenerator().generate_documentation("def function():\n    pass\n", make_element())

    assert result.success is False
    assert result.documentation is None
    assert "GROQ_API_KEY" in result.error
    assert "Groq API key unavailable to Python." in caplog.text
    assert "secret" not in caplog.text.lower()


def test_gemini_provider_successful_generation(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = FakeGeminiClient(response=FakeGeminiResponse("Document the function."))

    result = GeminiProvider(api_key="test-key", client=client).generate("prompt")

    assert result == "Document the function."
    assert client.models.prompts[0][0] == "gemini-3.8-flash"
    assert client.models.prompts[0][1] == "prompt"
    assert client.models.calls[0]["config"] == {"automatic_function_calling": {"disable": True}}


def test_gemini_provider_disables_automatic_function_calling(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = FakeGeminiClient(response=FakeGeminiResponse("Generated doc."))
    provider = GeminiProvider(api_key="test-key", client=client)

    result = provider.generate("test prompt")

    assert result == "Generated doc."
    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["contents"] == "test prompt"
    assert call["config"] == {"automatic_function_calling": {"disable": True}}


def test_gemini_provider_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingLLMAPIKeyError, match="GEMINI_API_KEY"):
        GeminiProvider()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (type("GeminiAuthError", (RuntimeError,), {"status_code": 401})(), LLMAuthenticationError),
        (type("GeminiRateError", (RuntimeError,), {"status_code": 429})(), LLMRateLimitError),
        (type("GeminiUnavailableError", (RuntimeError,), {"status_code": 503, "status": "UNAVAILABLE"})(), LLMServiceUnavailableError),
        (ConnectionError("offline"), LLMProviderError),
    ],
)
def test_gemini_provider_maps_api_failures(error, expected):
    client = FakeGeminiClient(error=error)

    with pytest.raises(expected):
        GeminiProvider(api_key="test-key", client=client, sleeper=lambda _: None).generate("prompt")


def test_comment_generator_preserves_gemini_provider_error_identity():
    client = FakeGeminiClient(error=RuntimeError("API failure"))
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error.startswith("Gemini provider request failed: RuntimeError")
    assert "Groq" not in result.error


def test_gemini_api_error_preserves_sanitized_status_and_reason():
    class GeminiApiError(RuntimeError):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "Model not found for key=AIzaSECRET_VALUE"

    client = FakeGeminiClient(error=GeminiApiError())
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert "Gemini provider request failed" in result.error
    assert "status=400" in result.error
    assert "reason=INVALID_ARGUMENT" in result.error
    assert "Model not found" in result.error
    assert "AIzaSECRET_VALUE" not in result.error


def test_gemini_service_unavailable_503_classified_temporarily():
    class GeminiUnavailableError(RuntimeError):
        code = 503
        status = "UNAVAILABLE"
        message = "The model is overloaded. Please try again later. key=AIzaSECRET_KEY"

    client = FakeGeminiClient(error=GeminiUnavailableError())
    generator = CommentGenerator(
        provider=GeminiProvider(api_key="test-key", client=client, sleeper=lambda _: None)
    )

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert "temporary Gemini service availability failure" in result.error
    assert "status=503" in result.error
    assert "reason=UNAVAILABLE" in result.error
    assert "The model is overloaded" in result.error
    assert "AIzaSECRET_KEY" not in result.error


def test_gemini_provider_retries_without_switching_models_on_503():
    class GeminiUnavailableError(RuntimeError):
        code = 503
        status = "UNAVAILABLE"

    client = FakeGeminiClient(error=GeminiUnavailableError())
    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client=client,
        sleeper=lambda _: None,
    )

    with pytest.raises(LLMServiceUnavailableError):
        provider.generate("prompt")

    # Confirms exactly 3 calls (1 initial + 2 retries) were made with original model (no model switch)
    assert len(client.models.prompts) == 3
    assert all(p[0] == "gemini-3.8-flash" for p in client.models.prompts)
    assert provider.model == "gemini-3.8-flash"


def test_gemini_invalid_model_400_classified_as_request_failure():
    class GeminiInvalidModelError(RuntimeError):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "models/invalid-model is not found"

    client = FakeGeminiClient(error=GeminiInvalidModelError())
    generator = CommentGenerator(
        provider=GeminiProvider(api_key="test-key", model="invalid-model", client=client)
    )

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert "Gemini provider request failed" in result.error
    assert "status=400" in result.error
    assert "reason=INVALID_ARGUMENT" in result.error
    assert "invalid-model is not found" in result.error
    assert "temporary" not in result.error


def test_gemini_authentication_failure_handled_by_comment_generator():
    class GeminiAuthError(RuntimeError):
        status_code = 401

    client = FakeGeminiClient(error=GeminiAuthError())
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error == "LLM provider authentication failed"


def test_gemini_rate_limiting_handled_by_comment_generator():
    class GeminiRateError(RuntimeError):
        status_code = 429

    client = FakeGeminiClient(error=GeminiRateError())
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error == "LLM provider rate limit reached"


def test_gemini_successful_generation_via_comment_generator():
    client = FakeGeminiClient(response=FakeGeminiResponse("Calculate the total of items."))
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))

    result = generator.generate_documentation(
        "def calculate_total(items):\n    return sum(items)\n", make_element()
    )

    assert result.success is True
    assert result.documentation == "Calculate the total of items."
    assert result.error is None


def test_gemini_model_configuration_and_default(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-override")
    client = FakeGeminiClient(response=FakeGeminiResponse("Doc"))

    # Explicit constructor parameter has highest precedence
    provider_explicit = GeminiProvider(api_key="test-key", model="gemini-explicit", client=client)
    assert provider_explicit.model == "gemini-explicit"

    # Environment variable has second precedence
    provider_env = GeminiProvider(api_key="test-key", client=client)
    assert provider_env.model == "gemini-env-override"

    # Default fallback is gemini-3.8-flash
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider_default = GeminiProvider(api_key="test-key", client=client)
    assert provider_default.model == "gemini-3.8-flash"


def test_gemini_provider_rejects_malformed_response():
    client = FakeGeminiClient(response=object())

    with pytest.raises(LLMResponseError, match="non-text"):
        GeminiProvider(api_key="test-key", client=client).generate("prompt")


def test_gemini_provider_rejects_empty_response():
    client = FakeGeminiClient(response=FakeGeminiResponse("  "))

    with pytest.raises(LLMResponseError, match="empty"):
        GeminiProvider(api_key="test-key", client=client).generate("prompt")


def test_gemini_is_default_provider_without_real_sdk_call(monkeypatch):
    class StubGeminiProvider:
        def __init__(self, api_key=None, model=None):
            self.api_key = api_key

        def generate(self, prompt):
            return "Documentation."

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(comment_generator_module, "GeminiProvider", StubGeminiProvider)

    result = CommentGenerator().generate_documentation("source", make_element())

    assert result.success is True


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


def test_prompt_instructs_return_statement_inspection_and_semantics():
    provider = FakeProvider()
    element = make_element(element_type="function", name="scan_dir")

    result = CommentGenerator(provider=provider).generate_documentation(
        "def scan_dir(dir_path):\n    return rows, rejects, len(files)\n",
        element,
    )

    assert result.success is True
    prompt = provider.prompts[0]
    assert "Inspect the actual return statement(s)" in prompt
    assert "Do not invent return names or replace returned variables or values with raw implementation expressions" in prompt



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


def test_comment_generator_preserves_groq_provider_error_identity():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO())

    generator = CommentGenerator(provider=GroqProvider(api_key="secret-value", opener=opener))

    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error == "Groq provider request failed"


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


def test_gemini_provider_pacing_first_request_no_delay():
    clock_time = 100.0

    def clock():
        return clock_time

    sleep_calls = []

    def sleeper(seconds):
        sleep_calls.append(seconds)

    client = FakeGeminiClient(response=FakeGeminiResponse("Doc 1"))
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        request_interval=4.0,
        clock=clock,
        sleeper=sleeper,
    )

    result = provider.generate("prompt 1")
    assert result == "Doc 1"
    assert sleep_calls == []


def test_gemini_provider_pacing_second_request_before_interval_is_paced():
    current_time = [100.0]

    def clock():
        return current_time[0]

    sleep_calls = []

    def sleeper(seconds):
        sleep_calls.append(seconds)
        current_time[0] += seconds

    client = FakeGeminiClient(response=FakeGeminiResponse("Doc"))
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        request_interval=4.0,
        clock=clock,
        sleeper=sleeper,
    )

    # First request at t=100.0
    provider.generate("prompt 1")
    assert sleep_calls == []

    # Second request at t=101.5 (only 1.5s elapsed, interval 4.0s)
    current_time[0] = 101.5
    provider.generate("prompt 2")

    assert len(sleep_calls) == 1
    assert pytest.approx(sleep_calls[0]) == 2.5


def test_gemini_provider_pacing_second_request_after_interval_does_not_sleep():
    current_time = [100.0]

    def clock():
        return current_time[0]

    sleep_calls = []

    def sleeper(seconds):
        sleep_calls.append(seconds)

    client = FakeGeminiClient(response=FakeGeminiResponse("Doc"))
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        request_interval=4.0,
        clock=clock,
        sleeper=sleeper,
    )

    # First request at t=100.0
    provider.generate("prompt 1")

    # Second request at t=105.0 (5.0s elapsed >= 4.0s)
    current_time[0] = 105.0
    provider.generate("prompt 2")

    assert sleep_calls == []


def test_gemini_provider_request_interval_env_and_default(monkeypatch):
    client = FakeGeminiClient(response=FakeGeminiResponse("Doc"))

    # Default fallback is 4.0
    monkeypatch.delenv("GEMINI_REQUEST_INTERVAL_SECONDS", raising=False)
    p_default = GeminiProvider(api_key="test-key", client=client)
    assert p_default.request_interval == 4.0

    # Env var configuration
    monkeypatch.setenv("GEMINI_REQUEST_INTERVAL_SECONDS", "2.5")
    p_env = GeminiProvider(api_key="test-key", client=client)
    assert p_env.request_interval == 2.5

    # Invalid env var falls back to default 4.0
    monkeypatch.setenv("GEMINI_REQUEST_INTERVAL_SECONDS", "invalid")
    p_invalid = GeminiProvider(api_key="test-key", client=client)
    assert p_invalid.request_interval == 4.0

    # Explicit parameter overrides env var
    p_explicit = GeminiProvider(api_key="test-key", client=client, request_interval=1.0)
    assert p_explicit.request_interval == 1.0


def test_gemini_rate_limiting_result_has_error_type():
    class GeminiRateError(RuntimeError):
        status_code = 429

    client = FakeGeminiClient(error=GeminiRateError())
    generator = CommentGenerator(provider=GeminiProvider(api_key="test-key", client=client))
    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error_type == "rate_limited"


def test_gemini_service_unavailable_result_has_error_type():
    class GeminiUnavailableError(RuntimeError):
        code = 503
        status = "UNAVAILABLE"

    client = FakeGeminiClient(error=GeminiUnavailableError())
    generator = CommentGenerator(
        provider=GeminiProvider(api_key="test-key", client=client, sleeper=lambda _: None)
    )
    result = generator.generate_documentation("source", make_element())

    assert result.success is False
    assert result.error_type == "service_unavailable"


def test_gemini_first_request_succeeds_no_retry():
    sleep_calls = []
    client = SequenceGeminiClient([FakeGeminiResponse("Success doc")])
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        sleeper=lambda s: sleep_calls.append(s),
    )

    result = provider.generate("prompt")

    assert result == "Success doc"
    assert len(client.models.prompts) == 1
    assert sleep_calls == []


def test_gemini_first_request_503_second_succeeds():
    class Gemini503(RuntimeError):
        code = 503
        status = "UNAVAILABLE"

    current_time = [100.0]

    def clock():
        return current_time[0]

    sleep_calls = []

    def sleeper(s):
        sleep_calls.append(s)
        current_time[0] += s

    client = SequenceGeminiClient([
        Gemini503("Temporary failure"),
        FakeGeminiResponse("Recovered doc"),
    ])
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        retry_base_seconds=5.0,
        clock=clock,
        sleeper=sleeper,
    )

    result = provider.generate("prompt")

    assert result == "Recovered doc"
    assert len(client.models.prompts) == 2
    assert sleep_calls == [5.0]


def test_gemini_503_three_times_exhausts_retries_with_exponential_backoff():
    class Gemini503(RuntimeError):
        code = 503
        status = "UNAVAILABLE"

    current_time = [100.0]

    def clock():
        return current_time[0]

    sleep_calls = []

    def sleeper(s):
        sleep_calls.append(s)
        current_time[0] += s

    client = SequenceGeminiClient([
        Gemini503("Fail 1"),
        Gemini503("Fail 2"),
        Gemini503("Fail 3"),
    ])
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        max_retries=2,
        retry_base_seconds=5.0,
        clock=clock,
        sleeper=sleeper,
    )

    with pytest.raises(LLMServiceUnavailableError):
        provider.generate("prompt")

    assert len(client.models.prompts) == 3
    # Attempt 1 backoff: 5.0 * 2**0 = 5.0; Attempt 2 backoff: 5.0 * 2**1 = 10.0
    assert sleep_calls == [5.0, 10.0]


@pytest.mark.parametrize(
    "error_attrs,expected_exc",
    [
        ({"code": 400, "status": "INVALID_ARGUMENT", "message": "model not found"}, LLMProviderError),
        ({"status_code": 401, "message": "unauthorized"}, LLMAuthenticationError),
        ({"status_code": 403, "message": "forbidden"}, LLMAuthenticationError),
    ],
)
def test_gemini_permanent_error_no_retry(error_attrs, expected_exc):
    class PermanentError(RuntimeError):
        pass

    for key, val in error_attrs.items():
        setattr(PermanentError, key, val)

    sleep_calls = []
    client = FakeGeminiClient(error=PermanentError())
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        sleeper=lambda s: sleep_calls.append(s),
    )

    with pytest.raises(expected_exc):
        provider.generate("prompt")

    assert len(client.models.prompts) == 1
    assert sleep_calls == []


def test_gemini_429_rate_limit_not_retried():
    class Gemini429(RuntimeError):
        status_code = 429

    sleep_calls = []
    client = FakeGeminiClient(error=Gemini429())
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        sleeper=lambda s: sleep_calls.append(s),
    )

    with pytest.raises(LLMRateLimitError):
        provider.generate("prompt")

    assert len(client.models.prompts) == 1
    assert sleep_calls == []


def test_gemini_request_pacing_preserved_with_retries():
    current_time = [100.0]

    def clock():
        return current_time[0]

    sleep_calls = []

    def sleeper(seconds):
        sleep_calls.append(seconds)
        current_time[0] += seconds

    class Gemini503(RuntimeError):
        code = 503
        status = "UNAVAILABLE"

    # request_interval is 8.0s, retry base is 3.0s (3.0s < 8.0s)
    client = SequenceGeminiClient([
        Gemini503("Temporary failure"),
        FakeGeminiResponse("Recovered doc"),
    ])
    provider = GeminiProvider(
        api_key="test-key",
        client=client,
        request_interval=8.0,
        retry_base_seconds=3.0,
        clock=clock,
        sleeper=sleeper,
    )

    result = provider.generate("prompt")

    assert result == "Recovered doc"
    assert len(client.models.prompts) == 2
    # First attempt fails at t=100.0 -> retry sleep of 3.0s (clock advances to 103.0)
    # Next attempt starts -> elapsed is 3.0s, request_interval is 8.0s -> pacing sleep of 5.0s (clock advances to 108.0)
    assert sleep_calls == [3.0, 5.0]


def test_gemini_retry_env_configuration(monkeypatch):
    client = FakeGeminiClient(response=FakeGeminiResponse("Doc"))

    # Default fallback
    monkeypatch.delenv("GEMINI_MAX_RETRIES", raising=False)
    monkeypatch.delenv("GEMINI_RETRY_BASE_SECONDS", raising=False)
    p_def = GeminiProvider(api_key="test-key", client=client)
    assert p_def.max_retries == 2
    assert p_def.retry_base_seconds == 5.0

    # Env var configuration
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "4")
    monkeypatch.setenv("GEMINI_RETRY_BASE_SECONDS", "1.5")
    p_env = GeminiProvider(api_key="test-key", client=client)
    assert p_env.max_retries == 4
    assert p_env.retry_base_seconds == 1.5

    # Invalid env vars fallback to default
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "invalid")
    monkeypatch.setenv("GEMINI_RETRY_BASE_SECONDS", "invalid")
    p_inv = GeminiProvider(api_key="test-key", client=client)
    assert p_inv.max_retries == 2
    assert p_inv.retry_base_seconds == 5.0

    # Explicit constructor arguments override env vars
    p_exp = GeminiProvider(
        api_key="test-key",
        client=client,
        max_retries=1,
        retry_base_seconds=2.0,
    )
    assert p_exp.max_retries == 1
    assert p_exp.retry_base_seconds == 2.0


