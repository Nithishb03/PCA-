"""Read-only access to GitHub repository data."""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

GITHUB_API_BASE_URL = "https://api.github.com"


class GitHubManagerError(RuntimeError):
    """Base error for safe GitHub integration failures."""


class MissingGitHubTokenError(GitHubManagerError):
    """Raised when the GitHub token environment variable is missing."""


class GitHubAuthenticationError(GitHubManagerError):
    """Raised when GitHub rejects authentication or access permissions."""


class GitHubNotFoundError(GitHubManagerError):
    """Raised when a GitHub resource does not exist or is inaccessible."""


class GitHubAccessError(GitHubManagerError):
    """Raised when GitHub cannot be reached or returns an unexpected response."""


class GitHubManager:
    """Provide read-only GitHub API operations for repository URLs."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.getenv("GITHUB_TOKEN")
        if not self._token:
            raise MissingGitHubTokenError(
                "GitHub integration requires the GITHUB_TOKEN environment variable."
            )

    @staticmethod
    def parse_repository_url(repository_url: str) -> str:
        """Validate a GitHub repository URL and return its owner/name."""
        if not isinstance(repository_url, str):
            raise ValueError("Repository URL must be a string")

        parsed_url = urlparse(repository_url)
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if (
            parsed_url.scheme != "https"
            or parsed_url.netloc.lower() != "github.com"
            or len(path_parts) != 2
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("Repository URL must be an HTTPS GitHub repository URL")

        owner, repository = path_parts
        if repository.endswith(".git"):
            repository = repository[:-4]
        if not owner or not repository:
            raise ValueError("Repository URL must include an owner and repository name")
        return f"{owner}/{repository}"

    @staticmethod
    def _api_path(repository_url: str, suffix: str = "") -> str:
        full_name = GitHubManager.parse_repository_url(repository_url)
        encoded_name = "/".join(quote(part, safe="") for part in full_name.split("/"))
        return f"/repos/{encoded_name}{suffix}"

    def _request_json(self, api_path: str) -> Any:
        request = Request(
            f"{GITHUB_API_BASE_URL}{api_path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 404:
                raise GitHubNotFoundError("GitHub resource was not found or is inaccessible") from error
            if error.code in {401, 403}:
                raise GitHubAuthenticationError("GitHub authentication or access was rejected") from error
            raise GitHubAccessError(f"GitHub returned HTTP {error.code}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise GitHubAccessError("Unable to access GitHub") from error

    def get_repository(self, repository_url: str) -> dict[str, Any]:
        """Fetch basic metadata for a repository."""
        response = self._request_json(self._api_path(repository_url))
        if not isinstance(response, dict):
            raise GitHubAccessError("GitHub returned invalid repository metadata")
        return response

    def get_default_branch(self, repository_url: str) -> str:
        """Fetch the repository's default branch name."""
        repository = self.get_repository(repository_url)
        default_branch = repository.get("default_branch")
        if not isinstance(default_branch, str) or not default_branch:
            raise GitHubAccessError("GitHub response did not include a default branch")
        return default_branch

    def get_file(self, repository_url: str, path: str) -> str:
        """Fetch and decode a file from a repository's default branch."""
        if not path or path.endswith("/"):
            raise ValueError("File path must identify a file")
        response = self._request_json(self._api_path(repository_url, f"/contents/{quote(path, safe='/')}"))
        if not isinstance(response, dict) or response.get("type") != "file":
            raise GitHubAccessError("GitHub response did not contain a file")
        content = response.get("content")
        if not isinstance(content, str):
            raise GitHubAccessError("GitHub file response did not contain file content")
        try:
            return base64.b64decode(content.replace("\n", ""), validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise GitHubAccessError("GitHub returned invalid file content") from error

    def get_directory(self, repository_url: str, path: str = "") -> list[dict[str, Any]]:
        """Fetch directory entries from a repository's default branch."""
        suffix = "/contents"
        if path:
            suffix += f"/{quote(path.strip('/'), safe='/')}"
        response = self._request_json(self._api_path(repository_url, suffix))
        if not isinstance(response, list) or not all(isinstance(item, dict) for item in response):
            raise GitHubAccessError("GitHub response did not contain directory contents")
        return response
