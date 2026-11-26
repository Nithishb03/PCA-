"""Run the scheduled read, analyze, generate, validate, and dry-run pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

try:
	from .code_analyzer import CodeAnalyzer
	from .comment_generator import CommentGenerator, _element_id
	from .documentation_policy import DocumentationPolicy
	from .github_manager import GitHubManager, GitHubManagerError
	from .github_writer import GitHubWriter
	from .models import CodeElement, DocumentationChange, Repository, ValidatedChange
	from .repository_manager import RepositoryManager
	from .safety_validator import SafetyValidator
except ImportError:
	from code_analyzer import CodeAnalyzer
	from comment_generator import CommentGenerator, _element_id
	from documentation_policy import DocumentationPolicy
	from github_manager import GitHubManager, GitHubManagerError
	from github_writer import GitHubWriter
	from models import CodeElement, DocumentationChange, Repository, ValidatedChange
	from repository_manager import RepositoryManager
	from safety_validator import SafetyValidator

CONFIG_PATH = Path(__file__).parents[1] / "config" / "repositories.yml"
LOGGER = logging.getLogger(__name__)


class CircuitBreaker:
	"""Track consecutive availability/quota failures and open when threshold is reached."""

	def __init__(self, max_consecutive_failures: int = 3) -> None:
		self.max_consecutive_failures = max_consecutive_failures
		self.consecutive_failures = 0
		self.is_open = False

	def record_success(self) -> None:
		"""Reset consecutive failures upon a successful generation."""
		self.consecutive_failures = 0

	def record_failure(self, is_quota_or_availability: bool = True) -> None:
		"""Record a failure; if quota/availability, increment count and open circuit if threshold met."""
		if is_quota_or_availability:
			self.consecutive_failures += 1
			if self.consecutive_failures >= self.max_consecutive_failures:
				self.is_open = True


def _fetch_python_files(
	github_manager: GitHubManager,
	repository_url: str,
	path: str = "",
	policy: DocumentationPolicy | None = None,
) -> dict[str, tuple[str, Optional[str]]]:
	"""Recursively fetch readable Python files and their blob SHAs via GitHubManager."""
	policy = policy or DocumentationPolicy()
	files: dict[str, tuple[str, Optional[str]]] = {}
	for entry in github_manager.get_directory(repository_url, path):
		entry_name = entry.get("name")
		entry_type = entry.get("type")
		if not isinstance(entry_name, str):
			continue
		entry_path = f"{path}/{entry_name}".strip("/")
		if entry_type == "dir":
			if policy.allows_directory(entry_path):
				files.update(_fetch_python_files(github_manager, repository_url, entry_path, policy))
		elif entry_type == "file" and policy.allows_file(entry_path):
			if hasattr(github_manager, "get_file_with_sha"):
				files[entry_path] = github_manager.get_file_with_sha(repository_url, entry_path)
			else:
				files[entry_path] = (github_manager.get_file(repository_url, entry_path), None)
	return files


def _fetch_python_sources(
	github_manager: GitHubManager,
	repository_url: str,
	path: str = "",
	policy: DocumentationPolicy | None = None,
) -> dict[str, str]:
	"""Recursively fetch readable Python files through the read-only GitHub manager."""
	return {
		file_path: content
		for file_path, (content, _) in _fetch_python_files(
			github_manager, repository_url, path=path, policy=policy
		).items()
	}


def _reconcile_element_in_source(
	analyzer: CodeAnalyzer,
	source: str,
	file_path: str,
	original_element: CodeElement,
	occurrence_index: int,
) -> CodeElement:
	"""Re-locate an AST element after prior edits shifted line numbers in the same file."""
	reanalysis = analyzer.analyze_python_file(source, file_path)
	if reanalysis.error:
		raise ValueError(reanalysis.error)
	matches = [
		el
		for el in reanalysis.elements
		if el.element_type == original_element.element_type
		and el.name == original_element.name
		and el.parent_class == original_element.parent_class
	]
	if occurrence_index >= len(matches):
		raise ValueError("Code element is no longer present in updated source")
	return matches[occurrence_index]


def _build_proposed_source(source: str, element: CodeElement, documentation: str) -> str:
	"""Build an in-memory docstring proposal; never write it to disk."""
	lines = source.splitlines(keepends=True)
	index = element.line_number - 1
	if index < 0 or index >= len(lines):
		raise ValueError("Code element line is outside the source")
	definition_line = lines[index]
	indentation = definition_line[: len(definition_line) - len(definition_line.lstrip())]
	docstring = f"{indentation}    {json.dumps(documentation)}\n"
	colon_pos = definition_line.find(":")
	rest = definition_line[colon_pos + 1 :].strip()
	if colon_pos != -1 and rest and not rest.startswith("#"):
		lines[index] = f"{definition_line[:colon_pos + 1]}\n"
		lines.insert(index + 1, docstring)
		lines.insert(index + 2, f"{indentation}    {rest}\n")
	else:
		lines.insert(index + 1, docstring)
	return "".join(lines)


def _new_repository_stats() -> dict[str, Any]:
	return {
		"status": "ERROR",
		"files_examined": 0,
		"python_files": 0,
		"files_analyzed": 0,
		"files_skipped": 0,
		"undocumented": 0,
		"generated": 0,
		"generation_failures": 0,
		"rate_limited": 0,
		"service_unavailable": 0,
		"quota_deferred": 0,
		"generation_error_reasons": [],
		"safe": 0,
		"unsafe": 0,
		"rejected": 0,
		"diffs_generated": 0,
		"writes_attempted": 0,
		"writes": 0,
		"write_failures": 0,
		"error": None,
	}


def _process_repository(
	repository: Repository,
	github_manager: GitHubManager,
	analyzer: CodeAnalyzer,
	generator: CommentGenerator,
	validator: SafetyValidator,
	writer: GitHubWriter,
	policy: DocumentationPolicy | None = None,
	circuit_breaker: CircuitBreaker | None = None,
) -> dict[str, Any]:
	"""Process one repository and isolate its failures from other repositories."""
	stats = _new_repository_stats()
	policy = policy or DocumentationPolicy()
	circuit_breaker = circuit_breaker or CircuitBreaker()
	if getattr(repository, "protected", False) or GitHubWriter.is_protected_repository(repository):
		stats["status"] = "SKIPPED"
		print(f"\nRepository: {repository.name}\nStatus: SKIPPED (protected)")
		return stats
	try:
		metadata = github_manager.get_repository(repository.url)
		default_branch = github_manager.get_default_branch(repository.url)
		full_name = metadata.get("full_name")
		if not isinstance(full_name, str):
			full_name = GitHubManager.parse_repository_url(repository.url)
		print(f"\nRepository: {repository.name}")
		print(f"Full name: {full_name}")
		print(f"Default branch: {default_branch}")

		fetched_files = _fetch_python_files(github_manager, repository.url, policy=policy)
		stats["files_examined"] = len(fetched_files)
		sources: dict[str, str] = {}
		file_shas: dict[str, Optional[str]] = {}
		for path, (source, file_sha) in fetched_files.items():
			if policy.allows_file(path, len(source)):
				sources[path] = source
				file_shas[path] = file_sha
		stats["files_skipped"] = len(fetched_files) - len(sources)
		stats["python_files"] = len(sources)
		analysis_results = analyzer.analyze_files(sources)
		stats["files_analyzed"] = len(analysis_results)
		stats["status"] = "DRY RUN" if not writer.write_enabled else "PROCESSED"

		batch_size = getattr(generator, "batch_size", 1)

		for analysis_result in analysis_results:
			if analysis_result.error:
				continue
			file_path = analysis_result.file_path
			original_source = sources[file_path]
			current_source = original_source
			current_sha = file_shas.get(file_path)

			element_occurrences: dict[int, int] = {}
			seen_keys: dict[tuple[str, str, Optional[str]], int] = {}
			for el in analysis_result.elements:
				key = (el.element_type, el.name, el.parent_class)
				occ = seen_keys.get(key, 0)
				seen_keys[key] = occ + 1
				element_occurrences[id(el)] = occ

			allowed_elements = [
				el for el in analysis_result.elements if policy.allows_element(el)
			]
			if not allowed_elements:
				continue

			for i in range(0, len(allowed_elements), batch_size):
				batch = allowed_elements[i : i + batch_size]
				for _ in batch:
					stats["undocumented"] += 1

				if circuit_breaker.is_open:
					for _ in batch:
						stats["quota_deferred"] += 1
					reason = "Gemini quota/availability circuit breaker open; request deferred."
					if reason not in stats["generation_error_reasons"]:
						stats["generation_error_reasons"].append(reason)
					continue

				if hasattr(generator, "generate_documentation_batch"):
					batch_results = generator.generate_documentation_batch(original_source, batch)
				else:
					batch_results = {}
					for el in batch:
						eid = getattr(generator, "_element_id", lambda x: f"{x.element_type}:{x.name}:{x.line_number}")(el)
						batch_results[eid] = generator.generate_documentation(original_source, el)

				batch_had_429 = False
				batch_had_503 = False
				batch_had_success = False

				for element in batch:
					eid = getattr(generator, "_element_id", lambda x: f"{x.element_type}:{x.name}:{x.line_number}")(element)
					generation = batch_results.get(eid)
					if generation is None:
						stats["generation_failures"] += 1
						continue

					if not generation.success:
						error_type = getattr(generation, "error_type", None)
						reason = generation.error or "Unknown documentation generation failure."
						is_429 = error_type == "rate_limited" or "rate limit" in reason.lower()
						is_503 = error_type == "service_unavailable" or "service availability" in reason.lower()

						if is_429:
							stats["rate_limited"] += 1
							batch_had_429 = True
						elif is_503:
							stats["service_unavailable"] += 1
							batch_had_503 = True
						else:
							stats["generation_failures"] += 1

						if reason not in stats["generation_error_reasons"]:
							stats["generation_error_reasons"].append(reason)
						continue

					batch_had_success = True
					stats["generated"] += 1
					try:
						target_element = (
							element
							if current_source == original_source
							else _reconcile_element_in_source(
								analyzer,
								current_source,
								file_path,
								element,
								element_occurrences[id(element)],
							)
						)
						proposed_source = _build_proposed_source(
							current_source,
							target_element,
							generation.documentation or "",
						)
					except ValueError:
						stats["unsafe"] += 1
						continue
					change = DocumentationChange(
						file_path=file_path,
						element_name=target_element.name,
						element_type=target_element.element_type,
						original_source=current_source,
						proposed_source=proposed_source,
						documentation=generation.documentation,
						line_number=target_element.line_number,
						parent_class=target_element.parent_class,
					)
					validation = validator.validate(change)
					if not validation.safe:
						stats["unsafe"] += 1
						stats["rejected"] += 1
						continue
					stats["safe"] += 1
					if validation.diff:
						stats["diffs_generated"] += 1
					print(f"\nProposed diff for {file_path}:{target_element.name}:")
					print(validation.diff)
					write_result = writer.write(
						ValidatedChange(
							repository=repository,
							file_path=file_path,
							original_source=current_source,
							proposed_source=proposed_source,
							element_name=target_element.name,
							element_type=target_element.element_type,
							validation_result=validation,
							generation_result=generation,
							file_sha=current_sha,
							branch=default_branch,
						)
					)
					if not write_result.dry_run:
						stats["writes_attempted"] += 1
					if write_result.success and write_result.dry_run:
						current_source = proposed_source
					elif write_result.success and not write_result.dry_run:
						stats["writes"] += 1
						if write_result.file_sha:
							current_source = proposed_source
							current_sha = write_result.file_sha
						elif hasattr(github_manager, "get_file_with_sha"):
							current_source, current_sha = github_manager.get_file_with_sha(
								repository.url, file_path
							)
						else:
							current_source = proposed_source
					elif not write_result.success:
						stats["rejected"] += 1
						if not write_result.dry_run:
							stats["write_failures"] += 1
						print(f"Write failed for {file_path}:{target_element.name}: {write_result.error}")

				if batch_had_success:
					circuit_breaker.record_success()
				elif batch_had_429 or batch_had_503:
					circuit_breaker.record_failure(is_quota_or_availability=True)
				else:
					circuit_breaker.record_failure(is_quota_or_availability=False)

		if writer.write_enabled:
			stats["status"] = "REJECTED" if stats["rejected"] else "SUCCESS"
		return stats
	except GitHubManagerError as error:
		LOGGER.error("GitHub error processing %s: %s", repository.name, error)
		stats["error"] = str(error)
		return stats
	except Exception:
		LOGGER.exception("Unexpected error processing repository %s.", repository.name)
		stats["error"] = "Unexpected repository processing error."
		return stats


def _print_repository_stats(repository: Repository, stats: dict[str, Any]) -> None:
	print(f"\nRepository: {repository.name}")
	if stats["error"]:
		print("Status: ERROR")
		print(f"Reason: {stats['error']}")
	else:
		print(f"Status: {stats['status']}")
	print(f"Files examined: {stats['files_examined']}")
	print(f"Python files: {stats['python_files']}")
	print(f"Files analyzed: {stats['files_analyzed']}")
	print(f"Files skipped: {stats['files_skipped']}")
	print(f"Undocumented elements: {stats['undocumented']}")
	print(f"Generated documentation: {stats['generated']}")
	print(f"Generation failures: {stats['generation_failures']}")
	print(f"Rate limited: {stats['rate_limited']}")
	print(f"Service unavailable: {stats['service_unavailable']}")
	print(f"Quota deferred: {stats['quota_deferred']}")
	for reason in stats["generation_error_reasons"]:
		print(f"Generation reason: {reason}")
	print(f"Safe changes: {stats['safe']}")
	print(f"Unsafe changes rejected: {stats['unsafe']}")
	print(f"Diffs generated: {stats['diffs_generated']}")
	print(f"Writes attempted: {stats['writes_attempted']}")
	print(f"Writes completed: {stats['writes']}")
	print("Write: SKIPPED" if stats.get("status") == "DRY RUN" or stats.get("writes") == 0 else "Write: CONTROLLED")


def main() -> None:
	"""Run every repository scheduled for today with isolated error handling."""
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	repositories = RepositoryManager(CONFIG_PATH).get_todays_repositories()
	if not repositories:
		print("No repositories scheduled today.")
		return

	print("Today's scheduled repositories:")
	for index, repository in enumerate(repositories, start=1):
		print(f"{index}. {repository.name} ({repository.url})")

	summary = {
		"processed": 0,
		"skipped": 0,
		"failed": 0,
		"files_examined": 0,
		"python_files": 0,
		"files_analyzed": 0,
		"files_skipped": 0,
		"undocumented": 0,
		"generated": 0,
		"generation_failures": 0,
		"rate_limited": 0,
		"service_unavailable": 0,
		"quota_deferred": 0,
		"generation_error_reasons": [],
		"safe": 0,
		"unsafe": 0,
		"rejected": 0,
		"diffs_generated": 0,
		"writes_attempted": 0,
		"writes": 0,
		"write_failures": 0,
	}
	if not os.getenv("GITHUB_TOKEN"):
		print("\nGitHub access unavailable; GITHUB_TOKEN is not configured.")
		for repository in repositories:
			print(f"\nRepository: {repository.name}\nStatus: SKIPPED\nReason: GitHub access unavailable.")
			summary["skipped"] += 1
		_print_summary(summary, len(repositories), False)
		return

	github_manager = GitHubManager()
	analyzer = CodeAnalyzer()
	generator = CommentGenerator()
	validator = SafetyValidator()
	writer = GitHubWriter()
	if not os.getenv("GEMINI_API_KEY"):
		print("\nLLM generation skipped because GEMINI_API_KEY is unavailable.")
	if not writer.write_enabled:
		print("GitHub write mode disabled; running in dry-run mode.")

	circuit_breaker = CircuitBreaker()
	for repository in repositories:
		stats = _process_repository(
			repository,
			github_manager,
			analyzer,
			generator,
			validator,
			writer,
			circuit_breaker=circuit_breaker,
		)
		_print_repository_stats(repository, stats)
		if stats["error"]:
			summary["failed"] += 1
		elif stats["status"] == "SKIPPED":
			summary["skipped"] += 1
		else:
			summary["processed"] += 1
		for key in (
			"files_examined",
			"python_files",
			"files_analyzed",
			"files_skipped",
			"undocumented",
			"generated",
			"generation_failures",
			"rate_limited",
			"service_unavailable",
			"quota_deferred",
			"safe",
			"unsafe",
			"rejected",
			"diffs_generated",
			"writes_attempted",
			"writes",
			"write_failures",
		):
			summary[key] += stats[key]
		for reason in stats["generation_error_reasons"]:
			if reason not in summary["generation_error_reasons"]:
				summary["generation_error_reasons"].append(reason)
	_print_summary(summary, len(repositories), writer.write_enabled)
	if summary["failed"] > 0 or summary["write_failures"] > 0:
		raise SystemExit(1)


def _print_summary(summary: dict[str, int], scheduled: int, writes_enabled: bool) -> None:
	print("\nSummary:")
	print(f"Repositories scheduled: {scheduled}")
	print(f"Repositories processed: {summary['processed']}")
	print(f"Repositories skipped: {summary['skipped']}")
	print(f"Repositories failed: {summary['failed']}")
	print(f"Files examined: {summary['files_examined']}")
	print(f"Python files: {summary['python_files']}")
	print(f"Python files analyzed: {summary['files_analyzed']}")
	print(f"Files skipped: {summary['files_skipped']}")
	print(f"Undocumented elements: {summary['undocumented']}")
	print(f"Documentation generated: {summary['generated']}")
	print(f"Generation failures: {summary['generation_failures']}")
	print(f"Rate limited: {summary['rate_limited']}")
	print(f"Service unavailable: {summary['service_unavailable']}")
	print(f"Quota deferred: {summary['quota_deferred']}")
	for reason in summary["generation_error_reasons"]:
		print(f"Generation reason: {reason}")
	print(f"Safe proposed changes: {summary['safe']}")
	print(f"Unsafe changes rejected: {summary['unsafe']}")
	print(f"Rejected changes: {summary['rejected']}")
	print(f"Diffs generated: {summary['diffs_generated']}")
	print(f"Writes attempted: {summary['writes_attempted']}")
	print(f"Writes completed: {summary['writes']}")
	print("GitHub writes: ENABLED" if writes_enabled else "GitHub writes: DISABLED / DRY RUN")


if __name__ == "__main__":
	main()