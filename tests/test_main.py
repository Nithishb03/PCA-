from datetime import date as real_date
from pathlib import Path

import pytest

import src.repository_manager as repository_manager_module
from src import main as main_module
from src.comment_generator import CommentGenerator
from src.github_writer import GitHubWriter
from src.models import CommentGenerationResult, Repository, ValidationResult, WriteResult
from src.repository_manager import RepositoryManager
from src.safety_validator import SafetyValidator


CONFIG_PATH = Path(__file__).parents[1] / "config" / "repositories.yml"
EXPECTED_REPOSITORIES = {
	"badge": "https://github.com/Nithishb03/badge",
	"Beam_Crack_Detection_System": "https://github.com/Nithishb03/Beam_Crack_Detection_System",
	"carbon-credit-": "https://github.com/Nithishb03/carbon-credit-",
	"credit-card-fraud-detection": "https://github.com/Nithishb03/credit-card-fraud-detection",
	"emergency-blood-donar-finder": "https://github.com/Nithishb03/emergency-blood-donar-finder",
	"Health-Guard-Vision": "https://github.com/Nithishb03/Health-Guard-Vision",
	"IEH-AC": "https://github.com/Nithishb03/IEH-AC",
	"Nithishb03": "https://github.com/Nithishb03/Nithishb03",
	"picoCTF-writeups": "https://github.com/Nithishb03/picoCTF-writeups",
	"portfolio-": "https://github.com/Nithishb03/portfolio-",
	"project-comment-automation": "https://github.com/Nithishb03/project-comment-automation",
	"shiksha-sahayaka-RAG-": "https://github.com/Nithishb03/shiksha-sahayaka-RAG-",
	"smart-parking-system": "https://github.com/Nithishb03/smart-parking-system",
	"ZnO-piezoelectric-energy-harvesting": "https://github.com/Nithishb03/ZnO-piezoelectric-energy-harvesting",
}


def write_config(tmp_path, content):
	config_path = tmp_path / "repositories.yml"
	config_path.write_text(content, encoding="utf-8")
	return config_path


def test_loads_all_14_repositories_with_exact_names_and_urls():
	manager = RepositoryManager(CONFIG_PATH)

	assert len(manager.repositories) == 14
	assert {repository.name: repository.url for repository in manager.repositories} == EXPECTED_REPOSITORIES


def test_rejects_duplicate_repository_names(tmp_path):
	config_path = write_config(
		tmp_path,
		"repositories:\n"
		"  - name: duplicate\n"
		"    url: https://github.com/example/one\n"
		"    schedule_day: Monday\n"
		"    protected: false\n"
		"  - name: duplicate\n"
		"    url: https://github.com/example/two\n"
		"    schedule_day: Tuesday\n"
		"    protected: false\n",
	)

	with pytest.raises(ValueError, match="Duplicate repository name"):
		RepositoryManager(config_path)


def test_rejects_invalid_schedule_day(tmp_path):
	config_path = write_config(
		tmp_path,
		"repositories:\n"
		"  - name: example\n"
		"    url: https://github.com/example/repository\n"
		"    schedule_day: Funday\n"
		"    protected: false\n",
	)

	with pytest.raises(ValueError, match="invalid schedule_day"):
		RepositoryManager(config_path)


def test_normalizes_schedule_days_and_returns_multiple_repositories():
	manager = RepositoryManager(CONFIG_PATH)

	monday_repositories = manager.get_repositories_for_day("MONDAY")

	assert [repository.name for repository in monday_repositories] == [
		"badge",
		"Beam_Crack_Detection_System",
	]
	assert all(repository.schedule_day == repository.schedule_day.lower() for repository in manager.repositories)


def test_returns_empty_list_when_no_repositories_are_scheduled(tmp_path):
	config_path = write_config(
		tmp_path,
		"repositories:\n"
		"  - name: example\n"
		"    url: https://github.com/example/repository\n"
		"    schedule_day: Monday\n"
		"    protected: false\n",
	)

	manager = RepositoryManager(config_path)

	assert manager.get_repositories_for_day("Tuesday") == []


def test_identifies_protected_repository():
	manager = RepositoryManager(CONFIG_PATH)

	protected = manager.get_protected_repositories()

	assert [repository.name for repository in protected] == ["project-comment-automation"]


def test_selects_all_repositories_for_today(monkeypatch):
	class FixedDate:
		@classmethod
		def today(cls):
			return real_date(2026, 10, 2)

	monkeypatch.setattr(repository_manager_module, "date", FixedDate)
	manager = RepositoryManager(CONFIG_PATH)

	todays_repositories = manager.get_todays_repositories()

	assert [repository.name for repository in todays_repositories] == [
		"picoCTF-writeups",
		"portfolio-",
		"project-comment-automation",
	]


class FakeGitHubManager:
	def get_repository(self, repository_url):
		return {"full_name": "owner/example"}

	def get_default_branch(self, repository_url):
		return "main"

	def get_directory(self, repository_url, path=""):
		return [{"name": "module.py", "type": "file"}]

	def get_file(self, repository_url, path):
		return "def public_function():\n    return 1\n"


class FakeGenerator:
	def generate_documentation(self, source, element):
		return CommentGenerationResult(success=True, documentation="Return one.")


class SuccessfulProvider:
	def generate(self, prompt):
		return "Describe the function's result."


class CountingGenerator(FakeGenerator):
	def __init__(self):
		self.calls = 0

	def generate_documentation(self, source, element):
		self.calls += 1
		return super().generate_documentation(source, element)


class FailingGenerator:
	def generate_documentation(self, source, element):
		return CommentGenerationResult(
			success=False,
			error="LLM generation requires the GROQ_API_KEY environment variable.",
		)


class FakeValidator:
	def validate(self, change):
		return ValidationResult(safe=True, diff="safe diff")


class FakeWriter:
	write_enabled = False

	def __init__(self):
		self.changes = []

	def write(self, change):
		self.changes.append(change)
		return WriteResult(success=True, dry_run=True, message="Dry-run: GitHub write skipped.")


def test_process_repository_runs_the_full_pipeline_in_dry_run():
	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	writer = FakeWriter()

	stats = main_module._process_repository(
		repository,
		FakeGitHubManager(),
		main_module.CodeAnalyzer(),
		FakeGenerator(),
		FakeValidator(),
		writer,
	)

	assert stats["status"] == "DRY RUN"
	assert stats["files_analyzed"] == 1
	assert stats["undocumented"] == 1
	assert stats["generated"] == 1
	assert stats["safe"] == 1
	assert stats["unsafe"] == 0
	assert len(writer.changes) == 1


def test_process_repository_isolates_github_failures():
	class FailingGitHubManager:
		def get_repository(self, repository_url):
			raise main_module.GitHubManagerError("access failed")

	repository = Repository("example", "https://github.com/owner/example", "monday", False)

	stats = main_module._process_repository(
		repository,
		FailingGitHubManager(),
		main_module.CodeAnalyzer(),
		FakeGenerator(),
		FakeValidator(),
		FakeWriter(),
	)

	assert stats["status"] == "ERROR"
	assert stats["error"] == "access failed"


def test_process_repository_skips_oversized_source_safely():
	class LargeGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return "# large source\n" + ("value = 1\n" * 20000)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		LargeGitHubManager(),
		main_module.CodeAnalyzer(),
		CountingGenerator(),
		FakeValidator(),
		FakeWriter(),
	)

	assert stats["files_examined"] == 1
	assert stats["python_files"] == 0
	assert stats["files_skipped"] == 1
	assert stats["files_analyzed"] == 0


def test_process_repository_does_not_regenerate_existing_documentation():
	class DocumentedGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return 'def public_function():\n    """Already documented."""\n    return 1\n'

	generator = CountingGenerator()
	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		DocumentedGitHubManager(),
		main_module.CodeAnalyzer(),
		generator,
		FakeValidator(),
		FakeWriter(),
	)

	assert stats["undocumented"] == 0
	assert stats["generated"] == 0
	assert generator.calls == 0


def test_process_repository_reports_generation_failure_reason():
	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		FakeGitHubManager(),
		main_module.CodeAnalyzer(),
		FailingGenerator(),
		FakeValidator(),
		FakeWriter(),
	)

	assert stats["undocumented"] == 1
	assert stats["generated"] == 0
	assert stats["generation_failures"] == 1
	assert stats["generation_error_reasons"] == [
		"LLM generation requires the GROQ_API_KEY environment variable."
	]


def test_successful_generation_reaches_real_safety_validator_and_dry_run_writer():
	class MultiFileGitHubManager(FakeGitHubManager):
		def get_directory(self, repository_url, path=""):
			return [
				{"name": "first.py", "type": "file"},
				{"name": "second.py", "type": "file"},
			]

		def get_file(self, repository_url, path):
			return "def first():\n    return 1\n" if path == "first.py" else "def second():\n    return 2\n"

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	generator = CommentGenerator(provider=SuccessfulProvider())
	writer = GitHubWriter(write_enabled=False, opener=lambda *args, **kwargs: pytest.fail("dry-run called HTTP"))

	stats = main_module._process_repository(
		repository,
		MultiFileGitHubManager(),
		main_module.CodeAnalyzer(),
		generator,
		SafetyValidator(),
		writer,
	)

	assert stats["files_analyzed"] == 2
	assert stats["undocumented"] == 2
	assert stats["generated"] == 2
	assert stats["safe"] == 2
	assert stats["unsafe"] == 0
	assert stats["writes"] == 0


def test_generation_failure_does_not_stop_later_elements():
	class SelectiveGenerator:
		def generate_documentation(self, source, element):
			if element.name == "first":
				return CommentGenerationResult(success=False, error="provider failure")
			return CommentGenerationResult(success=True, documentation="Describe the result.")

	class TwoFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return "def first():\n    return 1\n\ndef second():\n    return 2\n"

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		TwoFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		SelectiveGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
	)

	assert stats["undocumented"] == 2
	assert stats["generation_failures"] == 1
	assert stats["generated"] == 1
	assert stats["safe"] == 1
