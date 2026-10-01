import base64
import io
import json
from urllib.error import HTTPError

import pytest

import src.github_manager as github_manager_module
from src.github_manager import (
    GitHubAuthenticationError,
    GitHubManager,
    GitHubNotFoundError,
    MissingGitHubTokenError,
)


REPOSITORY_URL = "https://github.com/Nithishb03/repository-name"


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def mock_response(monkeypatch, payload):
    monkeypatch.setattr(
        github_manager_module,
        "urlopen",
        lambda request, timeout: FakeResponse(payload),
    )


def test_parses_valid_github_repository_url():
    assert GitHubManager.parse_repository_url(REPOSITORY_URL) == "Nithishb03/repository-name"


def test_rejects_invalid_repository_url():
    with pytest.raises(ValueError, match="HTTPS GitHub repository URL"):
        GitHubManager.parse_repository_url("http://gitlab.com/Nithishb03/repository-name")


def test_extracts_repository_full_name():
    assert GitHubManager.parse_repository_url(f"{REPOSITORY_URL}/") == "Nithishb03/repository-name"


def test_requires_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(MissingGitHubTokenError, match="GITHUB_TOKEN"):
        GitHubManager()


def test_fetches_repository_metadata_with_mocked_http(monkeypatch):
    mock_response(monkeypatch, {"full_name": "Nithishb03/repository-name", "private": False})
    manager = GitHubManager(token="test-token")

    metadata = manager.get_repository(REPOSITORY_URL)

    assert metadata["full_name"] == "Nithishb03/repository-name"


def test_handles_github_404(monkeypatch):
    def raise_not_found(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(github_manager_module, "urlopen", raise_not_found)

    with pytest.raises(GitHubNotFoundError, match="not found"):
        GitHubManager(token="test-token").get_repository(REPOSITORY_URL)


def test_handles_github_authentication_failure(monkeypatch):
    def raise_unauthorized(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(github_manager_module, "urlopen", raise_unauthorized)

    with pytest.raises(GitHubAuthenticationError, match="authentication"):
        GitHubManager(token="test-token").get_repository(REPOSITORY_URL)


def test_fetches_default_branch(monkeypatch):
    mock_response(monkeypatch, {"full_name": "Nithishb03/repository-name", "default_branch": "main"})

    assert GitHubManager(token="test-token").get_default_branch(REPOSITORY_URL) == "main"


def test_fetches_and_decodes_file(monkeypatch):
    content = base64.b64encode(b"print('hello')\n").decode("ascii")
    mock_response(monkeypatch, {"type": "file", "content": content})

    assert GitHubManager(token="test-token").get_file(REPOSITORY_URL, "src/main.py") == "print('hello')\n"


def test_fetches_directory(monkeypatch):
    entries = [{"name": "main.py", "type": "file"}, {"name": "tests", "type": "dir"}]
    mock_response(monkeypatch, entries)

    assert GitHubManager(token="test-token").get_directory(REPOSITORY_URL, "src") == entries
