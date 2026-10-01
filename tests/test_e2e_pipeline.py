import base64
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.code_analyzer import CodeAnalyzer
from src.comment_generator import CommentGenerator, LLMProviderError
from src.documentation_policy import DocumentationPolicy
import src.github_manager as github_manager_module
from src.github_manager import GitHubManager
from src.github_writer import GitHubWriter
from src.main import _build_proposed_source, _process_repository
from src.models import DocumentationChange, Repository, ValidatedChange
from src.repository_manager import RepositoryManager
from src.safety_validator import SafetyValidator


@dataclass(frozen=True)
class E2EValidationReport:
    repositories_processed: int
    files_examined: int
    python_files_analyzed: int
    undocumented_elements: int
    documentation_generated: int
    generation_failures: int
    safe_changes: int
    unsafe_changes: int
    diffs_generated: int
    writes_attempted: int
    writes_completed: int

    @classmethod
    def from_stats(cls, stats):
        return cls(
            repositories_processed=1,
            files_examined=stats["files_examined"],
            python_files_analyzed=stats["files_analyzed"],
            undocumented_elements=stats["undocumented"],
            documentation_generated=stats["generated"],
            generation_failures=stats["generation_failures"],
            safe_changes=stats["safe"],
            unsafe_changes=stats["unsafe"],
            diffs_generated=stats["diffs_generated"],
            writes_attempted=stats["writes_attempted"],
            writes_completed=stats["writes"],
        )


class DeterministicDocumentationProvider:
    """Test-only LLMProvider implementation for deterministic E2E checks."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        element_type = next(
            line.split("=", 1)[1].strip()
            for line in prompt.splitlines()
            if line.startswith("element_type = ")
        )
        return {
            "function": "Return the calculated value.",
            "async_function": "Asynchronously fetch the requested value.",
            "class": "Provide calculator operations.",
            "method": "Add two values using the calculator.",
        }[element_type]


class FailingDocumentationProvider:
    def generate(self, prompt):
        raise LLMProviderError("deterministic provider failure")


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class FixtureGitHubOpener:
    def __init__(self, sources):
        self.sources = sources
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if "/contents" not in request.full_url:
            return FakeResponse({"full_name": "Nithishb03/e2e-fixture", "default_branch": "main"})
        path = request.full_url.split("/contents", 1)[-1]
        if path in {"", "/"}:
            return FakeResponse(
                [
                    {"name": file_path, "type": "file"}
                    for file_path in self.sources
                ]
            )
        file_path = path.lstrip("/").split("?", 1)[0]
        encoded = base64.b64encode(self.sources[file_path].encode("utf-8")).decode("ascii")
        return FakeResponse({"type": "file", "content": encoded})


class RecordingDryRunWriter(GitHubWriter):
    def __init__(self):
        super().__init__(write_enabled=False, opener=self._unexpected_http_call)
        self.changes = []

    @staticmethod
    def _unexpected_http_call(*args, **kwargs):
        raise AssertionError("Dry-run writer must not perform HTTP")

    def write(self, change):
        self.changes.append(change)
        return super().write(change)


def write_repository_config(tmp_path):
    config_path = Path(tmp_path) / "repositories.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "repositories": [
                    {
                        "name": "e2e-fixture",
                        "url": "https://github.com/Nithishb03/e2e-fixture",
                        "schedule_day": "monday",
                        "protected": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config_path


def fixture_sources():
    return {
        "first.py": (
            "def public_function(value):\n"
            "    return value\n\n"
            "async def fetch_value(client):\n"
            "    return await client.fetch()\n\n"
            "def documented():\n"
            "    \"\"\"Already documented.\"\"\"\n"
            "    return True\n\n"
            "def _private():\n"
            "    return False\n"
        ),
        "second.py": (
            "class Calculator:\n"
            "    def __init__(self):\n"
            "        self.value = 0\n\n"
            "    def add(self, left, right):\n"
            "        return left + right\n"
        ),
    }


def run_fixture_pipeline(tmp_path, provider, monkeypatch):
    repository_manager = RepositoryManager(write_repository_config(tmp_path))
    repository = repository_manager.repositories[0]
    sources = fixture_sources()
    opener = FixtureGitHubOpener(sources)
    monkeypatch.setattr(github_manager_module, "urlopen", opener)
    github_manager = GitHubManager(token="test-token")
    writer = RecordingDryRunWriter()
    stats = _process_repository(
        repository,
        github_manager,
        CodeAnalyzer(),
        CommentGenerator(provider=provider),
        SafetyValidator(),
        writer,
        DocumentationPolicy(),
    )
    return stats, writer, opener, provider


def test_complete_pipeline_uses_real_components_and_dry_run_writer(tmp_path, monkeypatch):
    stats, writer, opener, provider = run_fixture_pipeline(
        tmp_path,
        DeterministicDocumentationProvider(),
        monkeypatch,
    )
    report = E2EValidationReport.from_stats(stats)

    assert report.repositories_processed == 1
    assert report.files_examined == 2
    assert report.python_files_analyzed == 2
    assert report.undocumented_elements > 0
    assert report.documentation_generated > 0
    assert report.generation_failures == 0
    assert report.safe_changes > 0
    assert report.unsafe_changes == 0
    assert report.diffs_generated > 0
    assert report.writes_attempted == 0
    assert report.writes_completed == 0
    assert len(writer.changes) == report.safe_changes
    assert all(request.get_method() == "GET" for request in opener.requests)
    assert len(provider.prompts) == report.documentation_generated
    assert all("name = __init__" not in prompt for prompt in provider.prompts)


def test_failed_e2e_generation_is_reported_without_stopping_pipeline(tmp_path, monkeypatch):
    stats, writer, _, _ = run_fixture_pipeline(tmp_path, FailingDocumentationProvider(), monkeypatch)

    assert stats["undocumented"] > 0
    assert stats["generation_failures"] == stats["undocumented"]
    assert stats["generated"] == 0
    assert stats["safe"] == 0
    assert stats["diffs_generated"] == 0
    assert stats["writes_attempted"] == 0
    assert writer.changes == []


def test_unsafe_generated_change_is_rejected_by_real_safety_validator():
    source = "def public_function(value):\n    return value\n"
    repository = Repository("e2e-fixture", "https://github.com/Nithishb03/e2e-fixture", "monday", False)
    provider = DeterministicDocumentationProvider()
    element = CodeAnalyzer().analyze_python_file(source, "first.py").elements[0]
    result = CommentGenerator(provider=provider).generate_documentation(source, element)
    unsafe_source = "def public_function(value):\n    return 0\n"

    validation = SafetyValidator().validate(
        DocumentationChange(
            file_path="first.py",
            element_name=element.name,
            element_type=element.element_type,
            original_source=source,
            proposed_source=unsafe_source,
            documentation=result.documentation,
            line_number=element.line_number,
        )
    )

    assert result.success is True
    assert validation.safe is False
    assert validation.errors
    assert repository.protected is False


def test_controlled_single_element_workflow_scan_dir():
    """Verify single-element flow for ml_pipeline/build_dataset.py:scan_dir."""
    build_dataset_source = (
        "from pathlib import Path\n\n"
        "def scan_dir(dir_path, source_label, label_value):\n"
        "    dir_path = Path(dir_path)\n"
        "    files = sorted(dir_path.glob('*.parquet'))\n"
        "    rows, rejects = [], {}\n"
        "    return rows, rejects, len(files)\n"
    )
    file_path = "ml_pipeline/build_dataset.py"
    target_element_name = "scan_dir"
    expected_sha = "3550e040a36a07ce3f2e78a4815b4e67dcb6f94b"

    # 1. Analyze file with CodeAnalyzer
    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze_python_file(build_dataset_source, file_path)
    assert not analysis.error
    assert len(analysis.elements) == 1

    # 2. Locate exactly scan_dir
    matching = [e for e in analysis.elements if e.name == target_element_name]
    assert len(matching) == 1
    element = matching[0]
    assert element.name == "scan_dir"
    assert element.element_type == "function"
    assert element.needs_documentation is True

    # 3. Generate documentation with provider
    provider = DeterministicDocumentationProvider()
    generator = CommentGenerator(provider=provider)
    generation = generator.generate_documentation(build_dataset_source, element)
    assert generation.success is True
    assert generation.documentation

    # 4. Build proposed source
    proposed_source = _build_proposed_source(build_dataset_source, element, generation.documentation)

    # 5. Run SafetyValidator
    change = DocumentationChange(
        file_path=file_path,
        element_name=element.name,
        element_type=element.element_type,
        original_source=build_dataset_source,
        proposed_source=proposed_source,
        documentation=generation.documentation,
        line_number=element.line_number,
        parent_class=element.parent_class,
    )
    validator = SafetyValidator()
    validation = validator.validate(change)

    # 6. Verify diff
    assert validation.safe is True
    assert validation.diff
    assert f"def {target_element_name}" in validation.diff

    # 7. Verify safety & policy invariants
    policy = DocumentationPolicy()
    assert policy.allows_file(file_path, len(build_dataset_source)) is True
    assert policy.allows_element(element) is True
    assert not policy.is_protected_path(file_path)
    assert file_path.endswith(".py")
    assert len(build_dataset_source) <= 100000

    # 8. Verify dry-run writer enforces safety, SHA check, and never writes
    repository = Repository("IEH-AC", "https://github.com/Nithishb03/IEH-AC", "thursday", False)
    writer = GitHubWriter(write_enabled=False)
    validated_change = ValidatedChange(
        repository=repository,
        file_path=file_path,
        original_source=build_dataset_source,
        proposed_source=proposed_source,
        element_name=element.name,
        element_type=element.element_type,
        validation_result=validation,
        generation_result=generation,
        file_sha=expected_sha,
    )
    write_result = writer.write(validated_change)
    assert write_result.dry_run is True
    assert write_result.success is True

