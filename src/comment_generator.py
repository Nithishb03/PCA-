"""Generate proposed documentation text through a replaceable LLM provider."""

from __future__ import annotations

import json
import logging
import os
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
SUPPORTED_ELEMENT_TYPES = frozenset({"function", "async_function", "class", "method"})
LOGGER = logging.getLogger(__name__)


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


class LLMProvider(Protocol):
    """Interface required by CommentGenerator for any LLM provider."""

    def generate(self, prompt: str) -> str:
        """Return documentation text for a prompt."""
        ...


class GroqProvider:
    """Minimal read-only Groq chat-completions client."""

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
    ) -> None:
        if max_source_chars < 1:
            raise ValueError("max_source_chars must be greater than zero")
        self.max_source_chars = max_source_chars
        self._provider = provider
        self._provider_error: Optional[str] = None
        if provider is None:
            try:
                self._provider = GroqProvider(api_key=api_key, model=model)
            except MissingLLMAPIKeyError as error:
                self._provider_error = str(error)
                LOGGER.warning("CommentGenerator cannot call Groq because the API key is unavailable.")

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
            documentation = self._provider.generate(prompt).strip()
        except LLMAuthenticationError:
            LOGGER.error("Documentation generation failed: Groq authentication error.")
            return CommentGenerationResult(success=False, error="LLM provider authentication failed")
        except LLMRateLimitError:
            LOGGER.error("Documentation generation failed: Groq rate-limit error.")
            return CommentGenerationResult(success=False, error="LLM provider rate limit reached")
        except LLMResponseError as error:
            LOGGER.error("Documentation generation failed: Groq response error.")
            return CommentGenerationResult(success=False, error=str(error))
        except LLMProviderError:
            LOGGER.error("Documentation generation failed: Groq provider error.")
            return CommentGenerationResult(success=False, error="LLM provider request failed")
        except Exception:
            LOGGER.error("Documentation generation failed: unexpected provider error.")
            return CommentGenerationResult(success=False, error="LLM provider request failed")

        if not documentation:
            return CommentGenerationResult(success=False, error="LLM provider returned empty documentation")
        if "```" in documentation:
            return CommentGenerationResult(
                success=False,
                error="LLM provider returned malformed documentation",
            )
        return CommentGenerationResult(success=True, documentation=documentation)

    def _build_prompt(
        self,
        source_code: str,
        code_element: CodeElement,
        context: Optional[str],
    ) -> str:
        source_context = self._limit_source(source_code, code_element)
        extra_context = context.strip() if context else "No additional context provided."
        return f"""Generate documentation only for the specified code element.
Understand the element before documenting it and describe only what the code actually does.
Do not invent behavior, suggest implementation changes, or document unrelated code.
Keep the documentation concise and useful.
Return only the documentation text. Do not return Markdown code fences or explanations outside the documentation.

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
