"""Generate proposed documentation text through a replaceable LLM provider."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .models import CodeElement, CommentGenerationResult
except ImportError:
    from models import CodeElement, CommentGenerationResult

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_REQUEST_INTERVAL_SECONDS = 4.0
DEFAULT_GEMINI_MAX_RETRIES = 2
DEFAULT_GEMINI_RETRY_BASE_SECONDS = 5.0
SUPPORTED_ELEMENT_TYPES = frozenset({"function", "async_function", "class", "method"})
LOGGER = logging.getLogger(__name__)
ELEMENT_GUIDANCE = {
    "function": "Describe the function's purpose and only explicit inputs, outputs, and side effects.",
    "async_function": "Describe the asynchronous operation and only explicit inputs, outputs, and side effects.",
    "class": "Describe the class responsibility and only the behavior and state evident in the source.",
    "method": "Describe the method's role in its parent class and only explicit inputs, outputs, and side effects.",
}


class LLMProviderError(RuntimeError):
    """Base class for safe provider failures."""


class MissingLLMAPIKeyError(LLMProviderError):
    """Raised when the configured provider key is unavailable."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when the provider rejects authentication."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider rate limit is reached."""


class LLMResponseError(LLMProviderError):
    """Raised when the provider response is missing usable documentation."""


class LLMServiceUnavailableError(LLMProviderError):
    """Raised when the provider is temporarily unavailable or experiencing high demand."""


class LLMProvider(Protocol):
    """Interface required by CommentGenerator for any LLM provider."""

    def generate(self, prompt: str) -> str:
        """Return documentation text for a prompt."""
        ...


class GeminiProvider:
    """Google Gemini provider using the official google-genai SDK."""

    provider_name = "Gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        request_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_seconds: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            LOGGER.warning("Gemini API key unavailable to Python.")
            raise MissingLLMAPIKeyError(
                "LLM generation requires the GEMINI_API_KEY environment variable."
            )
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        if request_interval is not None:
            self.request_interval = float(request_interval)
        else:
            env_interval = os.getenv("GEMINI_REQUEST_INTERVAL_SECONDS")
            try:
                self.request_interval = (
                    float(env_interval)
                    if env_interval is not None
                    else DEFAULT_GEMINI_REQUEST_INTERVAL_SECONDS
                )
            except (ValueError, TypeError):
                self.request_interval = DEFAULT_GEMINI_REQUEST_INTERVAL_SECONDS

        if max_retries is not None:
            self.max_retries = int(max_retries)
        else:
            env_retries = os.getenv("GEMINI_MAX_RETRIES")
            try:
                self.max_retries = (
                    int(env_retries)
                    if env_retries is not None
                    else DEFAULT_GEMINI_MAX_RETRIES
                )
            except (ValueError, TypeError):
                self.max_retries = DEFAULT_GEMINI_MAX_RETRIES
        if self.max_retries < 0:
            self.max_retries = 0

        if retry_base_seconds is not None:
            self.retry_base_seconds = float(retry_base_seconds)
        else:
            env_base = os.getenv("GEMINI_RETRY_BASE_SECONDS")
            try:
                self.retry_base_seconds = (
                    float(env_base)
                    if env_base is not None
                    else DEFAULT_GEMINI_RETRY_BASE_SECONDS
                )
            except (ValueError, TypeError):
                self.retry_base_seconds = DEFAULT_GEMINI_RETRY_BASE_SECONDS
        if self.retry_base_seconds < 0:
            self.retry_base_seconds = 0.0

        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._last_request_start: Optional[float] = None

        if client is None:
            try:
                from google import genai
            except ImportError as error:
                raise LLMProviderError("Gemini SDK is unavailable.") from error
            client = genai.Client(api_key=self._api_key)
        self._client = client
        LOGGER.info(
            "Gemini provider initialized with model %s (max_retries=%d, retry_base=%.1fs).",
            self.model,
            self.max_retries,
            self.retry_base_seconds,
        )

    def generate(self, prompt: str) -> str:
        """Generate documentation with one Gemini request, retrying temporary 503 errors."""
        max_retries = max(0, self.max_retries)
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            now = self._clock()
            if self._last_request_start is not None and self.request_interval > 0:
                elapsed = now - self._last_request_start
                wait_time = self.request_interval - elapsed
                if wait_time > 0:
                    self._sleeper(wait_time)
                    now = self._clock()

            self._last_request_start = now
            LOGGER.info("Gemini request started.")
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"automatic_function_calling": {"disable": True}},
                )
            except Exception as error:
                status_code = getattr(error, "status_code", getattr(error, "code", None))
                error_name = type(error).__name__.lower()
                if status_code in {401, 403} or "auth" in error_name:
                    LOGGER.error("Gemini request returned an authentication error.")
                    raise LLMAuthenticationError("LLM provider authentication failed") from None
                if status_code == 429 or "rate" in error_name:
                    LOGGER.error("Gemini request returned a rate-limit error.")
                    raise LLMRateLimitError("LLM provider rate limit reached") from None
                if isinstance(error, (TimeoutError, URLError, OSError, ConnectionError)):
                    LOGGER.error("Gemini request returned a network error.")
                    raise LLMProviderError("Unable to reach the LLM provider") from None
                diagnostic = self._safe_error_detail(error)
                status = getattr(error, "status", None)
                status_str = str(status).upper() if status is not None else ""
                if (
                    status_code in {503, "503"}
                    or status_str == "UNAVAILABLE"
                    or "unavailable" in error_name
                    or "503" in error_name
                ):
                    if attempt < max_retries:
                        backoff = self.retry_base_seconds * (2 ** attempt)
                        LOGGER.warning(
                            "Gemini request returned temporary service availability error (503); "
                            "retrying in %.1fs (attempt %d of %d): %s",
                            backoff,
                            attempt + 1,
                            max_retries,
                            diagnostic,
                        )
                        if backoff > 0:
                            self._sleeper(backoff)
                        continue

                    LOGGER.error(
                        "Gemini request returned a temporary service availability error: %s",
                        diagnostic,
                    )
                    raise LLMServiceUnavailableError(diagnostic) from None

                LOGGER.error("Gemini request returned an API error: %s", diagnostic)
                raise LLMProviderError(diagnostic) from None

            documentation = getattr(response, "text", None)
            if not isinstance(documentation, str):
                LOGGER.error("Gemini response parsing failed: non-text response.")
                raise LLMResponseError("LLM provider returned non-text documentation")
            if not documentation.strip():
                LOGGER.error("Gemini response parsing failed: empty response.")
                raise LLMResponseError("LLM provider returned empty documentation")
            LOGGER.info("Gemini request succeeded and documentation was generated.")
            return documentation

    @staticmethod
    def _safe_error_detail(error: Exception) -> str:
        """Return bounded SDK error metadata with credential-like values removed."""
        status_code = getattr(error, "status_code", getattr(error, "code", None))
        status = getattr(error, "status", None)
        message = getattr(error, "message", None) or str(error)
        message = re.sub(r"AIza[0-9A-Za-z_-]+|gsk_[0-9A-Za-z_-]+", "[redacted]", message)
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", message)
        message = re.sub(r"(?i)(api[-_ ]?key|authorization)\s*[:=]\s*\S+", r"\1=[redacted]", message)
        message = re.sub(r"https?://\S+", "[url]", message)
        message = " ".join(message.split())[:300]
        details = [type(error).__name__]
        if status_code is not None:
            details.append(f"status={status_code}")
        if status:
            details.append(f"reason={status}")
        if message:
            details.append(message)
        return ": ".join(details)


class GroqProvider:
    """Minimal read-only Groq chat-completions client."""

    provider_name = "Groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self._api_key:
            LOGGER.warning("Groq API key unavailable to Python.")
            raise MissingLLMAPIKeyError(
                "LLM generation requires the GROQ_API_KEY environment variable."
            )
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self.timeout = timeout
        self._opener = opener or urlopen
        LOGGER.info("Groq API key available; provider initialized with model %s.", self.model)

    def generate(self, prompt: str) -> str:
        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        request = Request(
            GROQ_API_URL,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        LOGGER.info("Groq request started.")
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code in {401, 403}:
                LOGGER.error("Groq request returned an authentication error.")
                raise LLMAuthenticationError("LLM provider authentication failed") from error
            if error.code == 429:
                LOGGER.error("Groq request returned a rate-limit error.")
                raise LLMRateLimitError("LLM provider rate limit reached") from error
            LOGGER.error("Groq request returned an API error.")
            raise LLMProviderError(f"LLM provider returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError):
            LOGGER.error("Groq request returned a network error.")
            raise LLMProviderError("Unable to reach the LLM provider") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            LOGGER.error("Groq request succeeded but response parsing failed.")
            raise LLMResponseError("LLM provider returned malformed JSON") from None

        documentation = self._extract_documentation(payload)
        LOGGER.info("Groq request succeeded and documentation was generated.")
        return documentation

    @staticmethod
    def _extract_documentation(payload: Any) -> str:
        if not isinstance(payload, dict):
            LOGGER.error("Groq response parsing failed: malformed response.")
            raise LLMResponseError("LLM provider returned a malformed response")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            LOGGER.error("Groq response parsing failed: no completion choices.")
            raise LLMResponseError("LLM provider returned no completion choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            LOGGER.error("Groq response parsing failed: malformed completion.")
            raise LLMResponseError("LLM provider returned a malformed completion")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            LOGGER.error("Groq response parsing failed: malformed message.")
            raise LLMResponseError("LLM provider returned a malformed message")
        documentation = message.get("content")
        if not isinstance(documentation, str) or not documentation.strip():
            LOGGER.error("Groq response parsing failed: empty documentation.")
            raise LLMResponseError("LLM provider returned empty documentation")
        return documentation.strip()


class CommentGenerator:
    """Generate documentation text without modifying source code."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_source_chars: int = 12000,
        request_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_seconds: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        if max_source_chars < 1:
            raise ValueError("max_source_chars must be greater than zero")
        self.max_source_chars = max_source_chars
        self._provider = provider
        self._provider_error: Optional[str] = None
        if provider is None:
            try:
                provider_name = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
                if provider_name == "groq":
                    self._provider = GroqProvider(api_key=api_key, model=model)
                elif provider_name == "gemini":
                    gemini_kwargs: dict[str, Any] = {}
                    if request_interval is not None:
                        gemini_kwargs["request_interval"] = request_interval
                    if max_retries is not None:
                        gemini_kwargs["max_retries"] = max_retries
                    if retry_base_seconds is not None:
                        gemini_kwargs["retry_base_seconds"] = retry_base_seconds
                    if clock is not None:
                        gemini_kwargs["clock"] = clock
                    if sleeper is not None:
                        gemini_kwargs["sleeper"] = sleeper
                    self._provider = GeminiProvider(
                        api_key=api_key,
                        model=model,
                        **gemini_kwargs,
                    )
                else:
                    self._provider_error = f"Unsupported LLM_PROVIDER: {provider_name}"
            except MissingLLMAPIKeyError as error:
                self._provider_error = str(error)
                LOGGER.warning("CommentGenerator cannot call the selected LLM provider because its API key is unavailable.")

    def generate_documentation(
        self,
        source_code: str,
        code_element: CodeElement,
        context: Optional[str] = None,
    ) -> CommentGenerationResult:
        """Return proposed documentation text for one analyzed code element."""
        if code_element.element_type not in SUPPORTED_ELEMENT_TYPES:
            return CommentGenerationResult(
                success=False,
                error=f"Unsupported code element type: {code_element.element_type}",
            )
        if code_element.is_private:
            return CommentGenerationResult(
                success=False,
                error="Private code elements are not automatically documented",
            )
        if self._provider is None:
            return CommentGenerationResult(success=False, error=self._provider_error)

        prompt = self._build_prompt(source_code, code_element, context)
        try:
            documentation = self._normalize_documentation(self._provider.generate(prompt))
        except LLMAuthenticationError:
            LOGGER.error("Documentation generation failed: provider authentication error.")
            return CommentGenerationResult(
                success=False,
                error="LLM provider authentication failed",
                error_type="authentication",
            )
        except LLMRateLimitError:
            LOGGER.error("Documentation generation failed: provider rate-limit error.")
            return CommentGenerationResult(
                success=False,
                error="LLM provider rate limit reached",
                error_type="rate_limited",
            )
        except LLMServiceUnavailableError as error:
            provider_name = getattr(self._provider, "provider_name", "LLM")
            error_detail = str(error)
            LOGGER.error(
                "Documentation generation failed: %s temporary service availability error.",
                provider_name,
            )
            result_error = f"temporary {provider_name} service availability failure"
            if error_detail:
                result_error = f"{result_error}: {error_detail}"
            return CommentGenerationResult(
                success=False,
                error=result_error,
                error_type="service_unavailable",
            )
        except LLMResponseError as error:
            LOGGER.error("Documentation generation failed: provider response error.")
            return CommentGenerationResult(success=False, error=str(error))
        except LLMProviderError as error:
            provider_name = getattr(self._provider, "provider_name", "LLM")
            error_detail = str(error)
            LOGGER.error("Documentation generation failed: %s provider error.", provider_name)
            result_error = f"{provider_name} provider request failed"
            if provider_name == "Gemini" and error_detail:
                result_error = f"{result_error}: {error_detail}"
            return CommentGenerationResult(
                success=False,
                error=result_error,
            )
        except Exception:
            provider_name = getattr(self._provider, "provider_name", "LLM")
            LOGGER.error("Documentation generation failed: unexpected %s provider error.", provider_name)
            return CommentGenerationResult(
                success=False,
                error=f"{provider_name} provider request failed",
            )

        return CommentGenerationResult(success=True, documentation=documentation)

    @staticmethod
    def _normalize_documentation(documentation: Any) -> str:
        """Normalize one provider response into usable docstring content."""
        if not isinstance(documentation, str):
            raise LLMResponseError("LLM provider returned non-text documentation")
        normalized = documentation.strip()
        if not normalized:
            raise LLMResponseError("LLM provider returned empty documentation")

        lines = normalized.splitlines()
        if lines and lines[0].strip().startswith("```"):
            if len(lines) < 3 or lines[-1].strip() != "```":
                raise LLMResponseError("LLM provider returned malformed documentation")
            normalized = "\n".join(lines[1:-1]).strip()
        elif "```" in normalized:
            raise LLMResponseError("LLM provider returned malformed documentation")

        for quote in ('"""', "'''"):
            if normalized.startswith(quote) and normalized.endswith(quote):
                normalized = normalized[len(quote) : -len(quote)].strip()
                break
        if not normalized:
            raise LLMResponseError("LLM provider returned empty documentation")
        return normalized

    def _build_prompt(
        self,
        source_code: str,
        code_element: CodeElement,
        context: Optional[str],
    ) -> str:
        source_context = self._limit_source(source_code, code_element)
        extra_context = context.strip() if context else "No additional context provided."
        element_guidance = ELEMENT_GUIDANCE[code_element.element_type]
        return f"""Generate documentation only for the specified code element.
Understand the element before documenting it and describe only what the code actually does.
    Use only facts supported by the supplied source and context. Do not invent parameters, return values, exceptions, side effects, or behavior.
    Inspect the actual return statement(s) and describe the actual returned variable or value semantics from the supplied source code.
    Do not invent return names or replace returned variables or values with raw implementation expressions (such as len(files)); describe the semantic meaning of the returned values (e.g. total_files or count of files) rather than using code expressions as variable names.
    Do not suggest implementation changes or document unrelated code.
Keep the documentation concise and useful.
    {element_guidance}
    Return only raw documentation text suitable for a Python docstring. Do not return Markdown code fences, surrounding quote delimiters, or explanations outside the documentation.

Source:
{source_context}

Element:
name = {code_element.name}
element_type = {code_element.element_type}
line_number = {code_element.line_number}
end_line_number = {code_element.end_line_number}
parent_class = {code_element.parent_class or "None"}

Additional context:
{extra_context}
"""

    def _limit_source(self, source_code: str, code_element: CodeElement) -> str:
        if len(source_code) <= self.max_source_chars:
            return source_code

        marker = "\n... source truncated ...\n"
        available_chars = max(0, self.max_source_chars - len(marker))
        lines = source_code.splitlines(keepends=True)
        element_line_index = max(0, min(code_element.line_number - 1, len(lines) - 1))
        element_offset = sum(len(line) for line in lines[:element_line_index])
        start = max(0, min(element_offset - available_chars // 3, len(source_code) - available_chars))
        return marker + source_code[start : start + available_chars]
