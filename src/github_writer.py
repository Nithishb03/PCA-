"""Controlled, opt-in GitHub Contents API writes."""

from __future__ import annotations

import base64
import difflib
import json
import os
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen

try:
    from .documentation_policy import PROTECTED_PATH_PREFIXES, PROTECTED_PATHS
    from .github_manager import GitHubManager
    from .models import CommentGenerationResult, ValidatedChange, ValidationResult, WriteResult
except ImportError:
    from documentation_policy import PROTECTED_PATH_PREFIXES, PROTECTED_PATHS
    from github_manager import GitHubManager
    from models import CommentGenerationResult, ValidatedChange, ValidationResult, WriteResult

GITHUB_API_BASE_URL = "https://api.github.com"
COMMIT_MESSAGE = "docs: update Python documentation"
DEFAULT_MAX_DIFF_CHARS = 10000
PROTECTED_REPOSITORY_NAMES = frozenset({"project-comment-automation"})


class GitHubWriter:
    """Apply only explicitly enabled, safety-validated Python documentation changes."""

    def __init__(
        self,
        token: Optional[str] = None,
        opener: Optional[Callable[..., Any]] = None,
        write_enabled: Optional[bool] = None,
        max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    ) -> None:
        if max_diff_chars < 1:
            raise ValueError("max_diff_chars must be greater than zero")
        self._token = token or os.getenv("GITHUB_TOKEN")
        self._opener = opener or urlopen
        self.max_diff_chars = max_diff_chars
        self.write_enabled = (
            write_enabled
            if write_enabled is not None
            else os.getenv("GITHUB_WRITE_ENABLED", "").lower() == "true"
        )

    def write(self, change: ValidatedChange) -> WriteResult:
        """Dry-run or safely update one validated Python file."""
        if not isinstance(change, ValidatedChange):
            return WriteResult(
                success=False,
                dry_run=not self.write_enabled,
                error="A validated change is required.",
            )
        repository_name = change.repository.name
        if not self.write_enabled:
            return WriteResult(
                success=True,
                dry_run=True,
                repository=repository_name,
                file_path=change.file_path,
                message="Dry-run: GitHub write skipped.",
            )

        rejection = self._validate_write_request(change)
        if rejection:
            return WriteResult(
                success=False,
                dry_run=False,
                repository=repository_name,
                file_path=change.file_path,
                error=rejection,
            )

        try:
            full_name = GitHubManager.parse_repository_url(change.repository.url)
            metadata = self._request_json(
                "GET",
                self._contents_url(full_name, change.file_path, change.branch),
            )
            if not isinstance(metadata, dict) or metadata.get("type") not in {None, "file"}:
                return self._failure(change, "GitHub response did not describe a regular file.")
            current_sha = metadata.get("sha")
            if not isinstance(current_sha, str) or not current_sha:
                return self._failure(change, "GitHub response did not include a file SHA.")
            if current_sha != change.file_sha:
                return self._failure(change, "File changed since analysis; refusing to overwrite.")

            payload = {
                "message": COMMIT_MESSAGE,
                "content": base64.b64encode(change.proposed_source.encode("utf-8")).decode("ascii"),
                "sha": current_sha,
            }
            if change.branch:
                payload["branch"] = change.branch
            response = self._request_json(
                "PUT",
                self._contents_url(full_name, change.file_path, change.branch),
                payload,
            )
            commit = response.get("commit") if isinstance(response, dict) else None
            commit_sha = commit.get("sha") if isinstance(commit, dict) else None
            commit_url = commit.get("html_url") if isinstance(commit, dict) else None
            if not isinstance(commit_sha, str) or not commit_sha:
                return self._failure(change, "GitHub response did not include commit information.")
            return WriteResult(
                success=True,
                dry_run=False,
                repository=repository_name,
                file_path=change.file_path,
                commit_sha=commit_sha,
                commit_url=commit_url if isinstance(commit_url, str) else None,
                message=COMMIT_MESSAGE,
            )
        except HTTPError as error:
            return self._failure(change, self._http_error_message(error.code))
        except (URLError, TimeoutError, OSError):
            return self._failure(change, "Unable to reach GitHub.")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._failure(change, "GitHub returned a malformed response.")
        except ValueError:
            return self._failure(change, "Repository URL is invalid.")

    def _validate_write_request(self, change: ValidatedChange) -> Optional[str]:
        if self.is_protected_repository(change.repository):
            return "Repository is protected."
        path_error = self.validate_file_path(change.file_path)
        if path_error:
            return path_error
        if change.validation_result is None:
            return "Safety validation result is required."
        if not isinstance(change.validation_result, ValidationResult):
            return "Safety validation result is malformed or missing."
        if change.validation_result.safe is not True:
            return "Safety validation did not approve the change."
        if change.generation_result is not None and not isinstance(
            change.generation_result, CommentGenerationResult
        ):
            return "Documentation generation result is malformed."
        if change.generation_result is not None and change.generation_result.success is not True:
            return "Documentation generation was unsuccessful."
        if change.original_source == change.proposed_source:
            return "No source change to write."
        proposed_diff = "".join(
            difflib.unified_diff(
                change.original_source.splitlines(keepends=True),
                change.proposed_source.splitlines(keepends=True),
            )
        )
        if len(proposed_diff) > self.max_diff_chars:
            return "Proposed documentation diff exceeds the safety limit."
        if not self._token:
            return "GITHUB_TOKEN is required for enabled GitHub writes."
        if not change.file_sha:
            return "Original file SHA is required for stale-file protection."
        return None

    @staticmethod
    def validate_file_path(file_path: str) -> Optional[str]:
        """Return a rejection reason for paths the automation must not write."""
        if not isinstance(file_path, str) or not file_path:
            return "File path is invalid."
        normalized = file_path.replace("\\", "/")
        for _ in range(3):
            decoded = unquote(normalized)
            if decoded == normalized:
                break
            normalized = decoded
        if (
            normalized.startswith("/")
            or normalized.startswith("//")
            or (len(normalized) > 1 and normalized[1] == ":")
            or "\x00" in normalized
        ):
            return "File path is invalid."
        parts = PurePosixPath(normalized).parts
        if any(part in {"", ".", ".."} for part in parts):
            return "File path is invalid."
        lowered = normalized.lower()
        if lowered in PROTECTED_PATHS or lowered.startswith(PROTECTED_PATH_PREFIXES):
            return "File path is protected."
        if not lowered.endswith(".py"):
            return "Only Python files may be written."
        return None

    @staticmethod
    def is_protected_repository(repository: Any) -> bool:
        """Enforce self-repository protection independently of configuration."""
        if not hasattr(repository, "name") or not hasattr(repository, "url"):
            return True
        if getattr(repository, "protected", False) is True:
            return True
        if str(repository.name).lower() in PROTECTED_REPOSITORY_NAMES:
            return True
        try:
            full_name = GitHubManager.parse_repository_url(repository.url)
        except (TypeError, ValueError):
            return False
        return full_name.rsplit("/", 1)[-1].lower() in PROTECTED_REPOSITORY_NAMES

    @staticmethod
    def _contents_url(full_name: str, file_path: str, branch: Optional[str]) -> str:
        owner, repository = full_name.split("/", 1)
        encoded_name = f"{quote(owner, safe='')}/{quote(repository, safe='')}"
        encoded_path = quote(file_path.replace("\\", "/").lstrip("/"), safe="/")
        url = f"{GITHUB_API_BASE_URL}/repos/{encoded_name}/contents/{encoded_path}"
        if branch:
            url += f"?ref={quote(branch, safe='')}"
        return url

    def _request_json(self, method: str, url: str, payload: Optional[dict[str, Any]] = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            url,
            data=data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        with self._opener(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _http_error_message(status_code: int) -> str:
        if status_code in {401, 403}:
            return "GitHub authentication or write access was rejected."
        if status_code == 404:
            return "GitHub repository or file was not found."
        return f"GitHub returned HTTP {status_code}."

    @staticmethod
    def _failure(change: ValidatedChange, error: str) -> WriteResult:
        return WriteResult(
            success=False,
            dry_run=False,
            repository=change.repository.name,
            file_path=change.file_path,
            error=error,
        )
