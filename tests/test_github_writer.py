import base64
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from src.github_writer import GitHubWriter
from src.models import (
    CommentGenerationResult,
    DocumentationChange,
    Repository,
    ValidatedChange,
    ValidationResult,
)


REPOSITORY = Repository(
    name="example",
    url="https://github.com/Nithishb03/example",
    schedule_day="monday",
    protected=False,
)
PROTECTED_REPOSITORY = Repository(
    name="project-comment-automation",
    url="https://github.com/Nithishb03/project-comment-automation",
    schedule_day="friday",
    protected=True,
)
ORIGINAL = "def calculate_total(items):\n    return sum(items)\n"
PROPOSED = 'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n'


def make_change(
    repository=REPOSITORY,
    file_path="src/example.py",
    validation_result=ValidationResult(safe=True),
    generation_result=CommentGenerationResult(success=True, documentation="Calculate totals."),
    file_sha="old-sha",
):
    return ValidatedChange(
        repository=repository,
        file_path=file_path,
        original_source=ORIGINAL,
        proposed_source=PROPOSED,
        element_name="calculate_total",
        element_type="function",
        validation_result=validation_result,
        generation_result=generation_result,
        file_sha=file_sha,
    )


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_dry_run_is_default_and_makes_no_http_call(monkeypatch):
    monkeypatch.delenv("GITHUB_WRITE_ENABLED", raising=False)
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise AssertionError("dry-run must not call GitHub")

    result = GitHubWriter(token="secret", opener=opener).write(make_change())

    assert result.success is True
    assert result.dry_run is True
    assert result.message == "Dry-run: GitHub write skipped."
    assert result.commit_sha is None
    assert calls == []


def test_non_explicit_write_value_keeps_dry_run(monkeypatch):
    monkeypatch.setenv("GITHUB_WRITE_ENABLED", "1")

    result = GitHubWriter(token="secret").write(make_change())

    assert result.dry_run is True


def test_protected_repository_is_rejected_without_http_call():
    writer = GitHubWriter(token="secret", write_enabled=True, opener=lambda *args, **kwargs: pytest.fail())

    result = writer.write(make_change(repository=PROTECTED_REPOSITORY))

    assert result.success is False
    assert result.error == "Repository is protected."


@pytest.mark.parametrize(
    "file_path",
    [
        ".github/workflows/example.py",
        "config/settings.py",
        "src/github_writer.py",
        "src/github_manager.py",
        "src/repository_manager.py",
        "src/safety_validator.py",
        "src/comment_generator.py",
        "src/models.py",
        "requirements.txt",
        ".gitignore",
    ],
)
def test_protected_paths_are_rejected(file_path):
    result = GitHubWriter(token="secret", write_enabled=True).write(make_change(file_path=file_path))

    assert result.success is False
    assert result.error == "File path is protected."


def test_workflow_path_is_rejected():
    result = GitHubWriter(token="secret", write_enabled=True).write(
        make_change(file_path=".github/workflows/daily-comments.yml")
    )

    assert result.success is False
    assert result.error == "Only Python files may be written." or result.error == "File path is protected."


def test_non_python_file_is_rejected():
    result = GitHubWriter(token="secret", write_enabled=True).write(make_change(file_path="README.md"))

    assert result.success is False
    assert result.error == "Only Python files may be written."


def test_unsafe_validation_is_rejected():
    result = GitHubWriter(token="secret", write_enabled=True).write(
        make_change(validation_result=ValidationResult(safe=False, errors=["code changed"]))
    )

    assert result.success is False
    assert result.error == "Safety validation did not approve the change."


def test_missing_validation_is_rejected():
    result = GitHubWriter(token="secret", write_enabled=True).write(
        make_change(validation_result=None)
    )

    assert result.success is False
    assert result.error == "Safety validation result is required."


def test_unsuccessful_generation_is_rejected():
    result = GitHubWriter(token="secret", write_enabled=True).write(
        make_change(
            generation_result=CommentGenerationResult(success=False, error="generation failed")
        )
    )

    assert result.success is False
    assert result.error == "Documentation generation was unsuccessful."


def test_missing_token_is_rejected_before_http(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = GitHubWriter(write_enabled=True).write(make_change())

    assert result.success is False
    assert result.error == "GITHUB_TOKEN is required for enabled GitHub writes."


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "GitHub authentication or write access was rejected."),
        (403, "GitHub authentication or write access was rejected."),
        (404, "GitHub repository or file was not found."),
    ],
)
def test_github_http_errors_are_controlled(status, message):
    def opener(request, timeout):
        raise HTTPError(request.full_url, status, "failure", {}, io.BytesIO(b"secret-token"))

    result = GitHubWriter(token="secret-token", write_enabled=True, opener=opener).write(make_change())

    assert result.success is False
    assert result.error == message
    assert "secret-token" not in str(result)


def test_network_failure_is_controlled():
    def opener(request, timeout):
        raise URLError("network unavailable")

    result = GitHubWriter(token="secret-token", write_enabled=True, opener=opener).write(make_change())

    assert result.success is False
    assert result.error == "Unable to reach GitHub."
    assert "secret-token" not in str(result)


def test_malformed_response_is_controlled():
    def opener(request, timeout):
        return FakeResponse({"unexpected": True})

    result = GitHubWriter(token="secret-token", write_enabled=True, opener=opener).write(make_change())

    assert result.success is False
    assert result.error == "GitHub response did not include a file SHA."


def test_stale_sha_is_rejected_without_put():
    methods = []

    def opener(request, timeout):
        methods.append(request.get_method())
        return FakeResponse({"sha": "new-sha"})

    result = GitHubWriter(token="secret", write_enabled=True, opener=opener).write(make_change())

    assert result.success is False
    assert result.error == "File changed since analysis; refusing to overwrite."
    assert methods == ["GET"]


def test_successful_write_returns_commit_result_and_encodes_source():
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if request.get_method() == "GET":
            return FakeResponse({"sha": "old-sha"})
        return FakeResponse(
            {"commit": {"sha": "commit-sha", "html_url": "https://github.com/example/commit/commit-sha"}}
        )

    result = GitHubWriter(token="secret", write_enabled=True, opener=opener).write(make_change())

    assert result.success is True
    assert result.dry_run is False
    assert result.commit_sha == "commit-sha"
    assert result.commit_url.endswith("commit-sha")
    assert result.message == "docs: update Python documentation"
    assert [request.get_method() for request in requests] == ["GET", "PUT"]
    payload = json.loads(requests[1].data.decode("utf-8"))
    assert payload["message"] == "docs: update Python documentation"
    assert base64.b64decode(payload["content"]).decode("utf-8") == PROPOSED
    assert payload["sha"] == "old-sha"


def test_writer_does_not_expose_token_in_result():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 500, "failure", {}, io.BytesIO())

    result = GitHubWriter(token="never-print-this", write_enabled=True, opener=opener).write(make_change())

    assert "never-print-this" not in repr(result)


def test_self_repository_is_rejected_even_when_protection_flag_is_false():
    self_repository = Repository(
        name="other-name",
        url="https://github.com/Nithishb03/project-comment-automation",
        schedule_day="friday",
        protected=False,
    )

    result = GitHubWriter(token="secret", write_enabled=True).write(
        make_change(repository=self_repository)
    )

    assert result.success is False
    assert result.error == "Repository is protected."


@pytest.mark.parametrize(
    "file_path",
    ["../file.py", "../../file.py", "/absolute/file.py", "C:/absolute/file.py", "%2e%2e/file.py", "%252e%252e/file.py"],
)
def test_path_traversal_and_absolute_paths_are_rejected(file_path):
    result = GitHubWriter(token="secret", write_enabled=True).write(make_change(file_path=file_path))

    assert result.success is False
    assert result.error == "File path is invalid."


def test_malformed_validation_result_is_rejected():
    malformed_change = make_change(validation_result=object())

    result = GitHubWriter(token="secret", write_enabled=True).write(malformed_change)

    assert result.success is False
    assert result.error == "Safety validation result is malformed or missing."


def test_malformed_generation_result_is_rejected():
    malformed_change = make_change(generation_result=object())

    result = GitHubWriter(token="secret", write_enabled=True).write(malformed_change)

    assert result.success is False
    assert result.error == "Documentation generation result is malformed."


def test_oversized_diff_is_rejected_before_http():
    result = GitHubWriter(token="secret", write_enabled=True, max_diff_chars=10).write(make_change())

    assert result.success is False
    assert result.error == "Proposed documentation diff exceeds the safety limit."


def test_noop_change_is_rejected_before_http():
    change = make_change()
    noop = ValidatedChange(
        **{**change.__dict__, "proposed_source": change.original_source}
    )

    result = GitHubWriter(token="secret", write_enabled=True).write(noop)

    assert result.success is False
    assert result.error == "No source change to write."


def test_special_file_metadata_is_rejected():
    def opener(request, timeout):
        return FakeResponse({"sha": "old-sha", "type": "symlink"})

    result = GitHubWriter(token="secret", write_enabled=True, opener=opener).write(make_change())

    assert result.success is False
    assert result.error == "GitHub response did not describe a regular file."
