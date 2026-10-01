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


def test_process_repository_applies_documentation_file_policy_before_fetching():
	class PolicyGitHubManager(FakeGitHubManager):
		def __init__(self):
			self.fetched_paths = []

		def get_directory(self, repository_url, path=""):
			if not path:
				return [
					{"name": "tests", "type": "dir"},
					{"name": "src", "type": "dir"},
				]
			return [
				{"name": "module.py", "type": "file"},
				{"name": "test_module.py", "type": "file"},
				{"name": "generated_module.py", "type": "file"},
			]

		def get_file(self, repository_url, path):
			self.fetched_paths.append(path)
			return "def public_function():\n    return 1\n"

	github_manager = PolicyGitHubManager()
	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		github_manager,
		main_module.CodeAnalyzer(),
		FakeGenerator(),
		FakeValidator(),
		FakeWriter(),
	)

	assert github_manager.fetched_paths == ["src/module.py"]
	assert stats["files_examined"] == 1
	assert stats["undocumented"] == 1


def test_circuit_breaker_consecutive_failures_and_reset():
	cb = main_module.CircuitBreaker(max_consecutive_failures=3)
	assert not cb.is_open

	cb.record_failure(is_quota_or_availability=True)
	assert cb.consecutive_failures == 1
	assert not cb.is_open

	cb.record_failure(is_quota_or_availability=True)
	assert cb.consecutive_failures == 2
	assert not cb.is_open

	# Non-quota failure does not increment consecutive 429/503 count
	cb.record_failure(is_quota_or_availability=False)
	assert cb.consecutive_failures == 2
	assert not cb.is_open

	# Successful request resets consecutive count
	cb.record_success()
	assert cb.consecutive_failures == 0
	assert not cb.is_open

	# 3 consecutive 429/503 failures open the circuit
	cb.record_failure(is_quota_or_availability=True)
	cb.record_failure(is_quota_or_availability=True)
	cb.record_failure(is_quota_or_availability=True)
	assert cb.consecutive_failures == 3
	assert cb.is_open


def test_process_repository_records_429_rate_limited():
	class RateLimitGenerator:
		def generate_documentation(self, source, element):
			return CommentGenerationResult(
				success=False,
				error="LLM provider rate limit reached",
				error_type="rate_limited",
			)

	class SingleFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return "def only_fn():\n    return 1\n"

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		SingleFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		RateLimitGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
	)

	assert stats["undocumented"] == 1
	assert stats["rate_limited"] == 1
	assert stats["service_unavailable"] == 0
	assert stats["generation_failures"] == 0
	assert stats["quota_deferred"] == 0


def test_process_repository_records_503_service_unavailable():
	class ServiceUnavailableGenerator:
		def generate_documentation(self, source, element):
			return CommentGenerationResult(
				success=False,
				error="temporary Gemini service availability failure: status=503 reason=UNAVAILABLE",
				error_type="service_unavailable",
			)

	class SingleFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return "def only_fn():\n    return 1\n"

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	stats = main_module._process_repository(
		repository,
		SingleFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		ServiceUnavailableGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
	)

	assert stats["undocumented"] == 1
	assert stats["service_unavailable"] == 1
	assert stats["rate_limited"] == 0
	assert stats["generation_failures"] == 0
	assert stats["quota_deferred"] == 0


def test_process_repository_circuit_breaker_defers_after_three_failures():
	calls = []

	class SequenceGenerator:
		def generate_documentation(self, source, element):
			calls.append(element.name)
			if element.name in ("fn1", "fn2"):
				return CommentGenerationResult(
					success=False,
					error="LLM provider rate limit reached",
					error_type="rate_limited",
				)
			if element.name == "fn3":
				return CommentGenerationResult(
					success=False,
					error="temporary Gemini service availability failure: status=503",
					error_type="service_unavailable",
				)
			return CommentGenerationResult(
				success=True,
				documentation="Doc",
			)

	class FiveFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return (
				"def fn1():\n    return 1\n\n"
				"def fn2():\n    return 2\n\n"
				"def fn3():\n    return 3\n\n"
				"def fn4():\n    return 4\n\n"
				"def fn5():\n    return 5\n"
			)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	circuit_breaker = main_module.CircuitBreaker(max_consecutive_failures=3)

	stats = main_module._process_repository(
		repository,
		FiveFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		SequenceGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
		circuit_breaker=circuit_breaker,
	)

	assert stats["undocumented"] == 5
	assert stats["rate_limited"] == 2
	assert stats["service_unavailable"] == 1
	assert stats["quota_deferred"] == 2
	assert stats["generation_failures"] == 0
	assert stats["generated"] == 0
	assert circuit_breaker.is_open is True
	# After circuit opened on 3rd failure, fn4 and fn5 were NOT requested
	assert calls == ["fn1", "fn2", "fn3"]


def test_normal_generation_failure_does_not_open_circuit():
	class NormalFailureGenerator:
		def generate_documentation(self, source, element):
			return CommentGenerationResult(
				success=False,
				error="LLM provider returned empty documentation",
			)

	class ThreeFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return (
				"def fn1():\n    return 1\n\n"
				"def fn2():\n    return 2\n\n"
				"def fn3():\n    return 3\n"
			)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	circuit_breaker = main_module.CircuitBreaker(max_consecutive_failures=3)

	stats = main_module._process_repository(
		repository,
		ThreeFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		NormalFailureGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
		circuit_breaker=circuit_breaker,
	)

	assert stats["undocumented"] == 3
	assert stats["generation_failures"] == 3
	assert stats["rate_limited"] == 0
	assert stats["service_unavailable"] == 0
	assert stats["quota_deferred"] == 0
	assert circuit_breaker.is_open is False


def test_success_resets_consecutive_failures_preventing_circuit_open():
	class IntermittentGenerator:
		def generate_documentation(self, source, element):
			if element.name in ("fn1", "fn2"):
				return CommentGenerationResult(
					success=False,
					error="LLM provider rate limit reached",
					error_type="rate_limited",
				)
			if element.name == "fn3":
				# Success resets consecutive failure count
				return CommentGenerationResult(
					success=True,
					documentation="Document fn3",
				)
			if element.name == "fn4":
				return CommentGenerationResult(
					success=False,
					error="LLM provider rate limit reached",
					error_type="rate_limited",
				)
			return CommentGenerationResult(
				success=True,
				documentation="Document fn5",
			)

	class FiveFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return (
				"def fn1():\n    return 1\n\n"
				"def fn2():\n    return 2\n\n"
				"def fn3():\n    return 3\n\n"
				"def fn4():\n    return 4\n\n"
				"def fn5():\n    return 5\n"
			)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	circuit_breaker = main_module.CircuitBreaker(max_consecutive_failures=3)

	stats = main_module._process_repository(
		repository,
		FiveFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		IntermittentGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
		circuit_breaker=circuit_breaker,
	)

	assert stats["undocumented"] == 5
	assert stats["rate_limited"] == 3
	assert stats["generated"] == 2
	assert stats["quota_deferred"] == 0
	assert circuit_breaker.is_open is False


def test_circuit_breaker_works_after_retries_are_exhausted():
	class ExhaustedRetriesGenerator:
		def generate_documentation(self, source, element):
			if element.name in ("fn1", "fn2", "fn3"):
				return CommentGenerationResult(
					success=False,
					error="temporary Gemini service availability failure: status=503",
					error_type="service_unavailable",
				)
			return CommentGenerationResult(
				success=True,
				documentation="Doc",
			)

	class FourFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return (
				"def fn1():\n    return 1\n\n"
				"def fn2():\n    return 2\n\n"
				"def fn3():\n    return 3\n\n"
				"def fn4():\n    return 4\n"
			)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	circuit_breaker = main_module.CircuitBreaker(max_consecutive_failures=3)

	stats = main_module._process_repository(
		repository,
		FourFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		ExhaustedRetriesGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
		circuit_breaker=circuit_breaker,
	)

	assert stats["undocumented"] == 4
	assert stats["service_unavailable"] == 3
	assert stats["quota_deferred"] == 1
	assert stats["generated"] == 0
	assert circuit_breaker.is_open is True


def test_successful_retry_resets_consecutive_failure_state():
	class RetrySuccessGenerator:
		def generate_documentation(self, source, element):
			if element.name in ("fn1", "fn3", "fn4"):
				return CommentGenerationResult(
					success=False,
					error="temporary Gemini service availability failure: status=503",
					error_type="service_unavailable",
				)
			return CommentGenerationResult(
				success=True,
				documentation="Documented fn2 after retry",
			)

	class FourFunctionGitHubManager(FakeGitHubManager):
		def get_file(self, repository_url, path):
			return (
				"def fn1():\n    return 1\n\n"
				"def fn2():\n    return 2\n\n"
				"def fn3():\n    return 3\n\n"
				"def fn4():\n    return 4\n"
			)

	repository = Repository("example", "https://github.com/owner/example", "monday", False)
	circuit_breaker = main_module.CircuitBreaker(max_consecutive_failures=3)

	stats = main_module._process_repository(
		repository,
		FourFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		RetrySuccessGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
		circuit_breaker=circuit_breaker,
	)

	assert stats["undocumented"] == 4
	assert stats["service_unavailable"] == 3
	assert stats["generated"] == 1
	assert stats["quota_deferred"] == 0
	assert circuit_breaker.is_open is False
	assert circuit_breaker.consecutive_failures == 2


def test_process_repository_nested_function_diff_not_duplicated(capsys):
	class NestedFunctionGitHubManager(FakeGitHubManager):
		def get_directory(self, repository_url, path=""):
			if not path:
				return [{"name": "ml_pipeline", "type": "dir"}]
			if path == "ml_pipeline":
				return [{"name": "feature_extractor.py", "type": "file"}]
			return []

		def get_file(self, repository_url, path):
			return (
				'def extract_features(events):\n'
				'    """Outer documented function."""\n'
				'    def count_of(t):\n'
				'        return sum(1 for e in events if e["type"] == t)\n'
				'    return count_of("click")\n'
			)

	calls = []

	class TrackingGenerator:
		def generate_documentation(self, source, element):
			calls.append(element.name)
			return CommentGenerationResult(
				success=True,
				documentation="Count occurrences of event type in events list.",
			)

	repository = Repository("IEH-AC", "https://github.com/Nithishb03/IEH-AC", "thursday", False)
	stats = main_module._process_repository(
		repository,
		NestedFunctionGitHubManager(),
		main_module.CodeAnalyzer(),
		TrackingGenerator(),
		SafetyValidator(),
		GitHubWriter(write_enabled=False),
	)

	assert stats["undocumented"] == 1
	assert stats["generated"] == 1
	assert stats["safe"] == 1
	assert calls == ["count_of"]

	captured = capsys.readouterr().out
	header = "Proposed diff for ml_pipeline/feature_extractor.py:count_of"
	assert captured.count(header) == 1



