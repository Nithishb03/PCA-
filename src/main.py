"""Run the scheduled read, analyze, generate, validate, and dry-run pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
	from .code_analyzer import CodeAnalyzer
	from .comment_generator import CommentGenerator
	from .github_manager import GitHubManager, GitHubManagerError
	from .github_writer import GitHubWriter
	from .models import CodeElement, DocumentationChange, Repository, ValidatedChange
	from .repository_manager import RepositoryManager
	from .safety_validator import SafetyValidator
except ImportError:
	from code_analyzer import CodeAnalyzer
	from comment_generator import CommentGenerator
	from github_manager import GitHubManager, GitHubManagerError
	from github_writer import GitHubWriter
	from models import CodeElement, DocumentationChange, Repository, ValidatedChange
	from repository_manager import RepositoryManager
	from safety_validator import SafetyValidator

CONFIG_PATH = Path(__file__).parents[1] / "config" / "repositories.yml"
IGNORED_DIRECTORIES = {".git", "venv", ".venv", "__pycache__", "generated"}
MAX_ANALYZABLE_SOURCE_CHARS = 100000


def _fetch_python_sources(github_manager: GitHubManager, repository_url: str, path: str = "") -> dict[str, str]:
	"""Recursively fetch readable Python files through the read-only GitHub manager."""
	sources: dict[str, str] = {}
	for entry in github_manager.get_directory(repository_url, path):
		entry_name = entry.get("name")
		entry_type = entry.get("type")
		if not isinstance(entry_name, str):
			continue
		entry_path = f"{path}/{entry_name}".strip("/")
		if entry_type == "dir":
			if entry_name not in IGNORED_DIRECTORIES:
				sources.update(_fetch_python_sources(github_manager, repository_url, entry_path))
		elif entry_type == "file" and entry_name.lower().endswith(".py"):
			sources[entry_path] = github_manager.get_file(repository_url, entry_path)
	return sources


def _build_proposed_source(source: str, element: CodeElement, documentation: str) -> str:
	"""Build an in-memory docstring proposal; never write it to disk."""
	lines = source.splitlines(keepends=True)
	index = element.line_number - 1
	if index < 0 or index >= len(lines):
		raise ValueError("Code element line is outside the source")
	definition_line = lines[index]
	indentation = definition_line[: len(definition_line) - len(definition_line.lstrip())]
	docstring = f"{indentation}    {json.dumps(documentation)}\n"
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
		"safe": 0,
		"unsafe": 0,
		"rejected": 0,
		"writes": 0,
		"error": None,
	}


def _process_repository(
	repository: Repository,
	github_manager: GitHubManager,
	analyzer: CodeAnalyzer,
	generator: CommentGenerator,
	validator: SafetyValidator,
	writer: GitHubWriter,
) -> dict[str, Any]:
	"""Process one repository and isolate its failures from other repositories."""
	stats = _new_repository_stats()
	try:
		metadata = github_manager.get_repository(repository.url)
		default_branch = github_manager.get_default_branch(repository.url)
		full_name = metadata.get("full_name")
		if not isinstance(full_name, str):
			full_name = GitHubManager.parse_repository_url(repository.url)
		print(f"\nRepository: {repository.name}")
		print(f"Full name: {full_name}")
		print(f"Default branch: {default_branch}")

		fetched_sources = _fetch_python_sources(github_manager, repository.url)
		stats["files_examined"] = len(fetched_sources)
		sources = {
			path: source
			for path, source in fetched_sources.items()
			if len(source) <= MAX_ANALYZABLE_SOURCE_CHARS
		}
		stats["files_skipped"] = len(fetched_sources) - len(sources)
		stats["python_files"] = len(sources)
		analysis_results = analyzer.analyze_files(sources)
		stats["files_analyzed"] = len(analysis_results)
		stats["status"] = "DRY RUN" if not writer.write_enabled else "PROCESSED"

		for analysis_result in analysis_results:
			if analysis_result.error:
				continue
			source = sources[analysis_result.file_path]
			for element in analysis_result.elements:
				if not element.needs_documentation:
					continue
				stats["undocumented"] += 1
				generation = generator.generate_documentation(source, element)
				if not generation.success:
					continue
				stats["generated"] += 1
				try:
					proposed_source = _build_proposed_source(
						source,
						element,
						generation.documentation or "",
					)
				except ValueError:
					stats["unsafe"] += 1
					continue
				change = DocumentationChange(
					file_path=analysis_result.file_path,
					element_name=element.name,
					element_type=element.element_type,
					original_source=source,
					proposed_source=proposed_source,
					documentation=generation.documentation,
					line_number=element.line_number,
					parent_class=element.parent_class,
				)
				validation = validator.validate(change)
				if not validation.safe:
					stats["unsafe"] += 1
					stats["rejected"] += 1
					continue
				stats["safe"] += 1
				print(f"\nProposed diff for {analysis_result.file_path}:{element.name}:")
				print(validation.diff)
				write_result = writer.write(
					ValidatedChange(
						repository=repository,
						file_path=analysis_result.file_path,
						original_source=source,
						proposed_source=proposed_source,
						element_name=element.name,
						element_type=element.element_type,
						validation_result=validation,
						generation_result=generation,
					)
				)
				if write_result.success and not write_result.dry_run:
					stats["writes"] += 1
				elif not write_result.success:
					stats["rejected"] += 1
		if writer.write_enabled:
			stats["status"] = "REJECTED" if stats["rejected"] else "SUCCESS"
		return stats
	except GitHubManagerError as error:
		stats["error"] = str(error)
		return stats
	except Exception:
		stats["error"] = "Unexpected repository processing error."
		return stats


def _print_repository_stats(repository: Repository, stats: dict[str, Any]) -> None:
	print(f"\nRepository: {repository.name}")
	if stats["error"]:
		print("Status: ERROR")
		print(f"Reason: {stats['error']}")
		return
	print(f"Status: {stats['status']}")
	print(f"Files examined: {stats['files_examined']}")
	print(f"Python files: {stats['python_files']}")
	print(f"Files analyzed: {stats['files_analyzed']}")
	print(f"Files skipped: {stats['files_skipped']}")
	print(f"Undocumented elements: {stats['undocumented']}")
	print(f"Generated documentation: {stats['generated']}")
	print(f"Safe changes: {stats['safe']}")
	print(f"Unsafe changes rejected: {stats['unsafe']}")
	print(f"Writes: {stats['writes']}")
	print("Write: SKIPPED" if stats["status"] == "DRY RUN" else "Write: CONTROLLED")


def main() -> None:
	"""Run every repository scheduled for today with isolated error handling."""
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
		"safe": 0,
		"unsafe": 0,
		"rejected": 0,
		"writes": 0,
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
	if not os.getenv("GROQ_API_KEY"):
		print("\nLLM generation skipped because GROQ_API_KEY is unavailable.")
	if not writer.write_enabled:
		print("GitHub write mode disabled; running in dry-run mode.")

	for repository in repositories:
		stats = _process_repository(repository, github_manager, analyzer, generator, validator, writer)
		_print_repository_stats(repository, stats)
		if stats["error"]:
			summary["failed"] += 1
		else:
			summary["processed"] += 1
		for key in (
			"files_examined",
			"python_files",
			"files_analyzed",
			"files_skipped",
			"undocumented",
			"generated",
			"safe",
			"unsafe",
			"rejected",
			"writes",
		):
			summary[key] += stats[key]
	_print_summary(summary, len(repositories), writer.write_enabled)


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
	print(f"Safe proposed changes: {summary['safe']}")
	print(f"Unsafe changes rejected: {summary['unsafe']}")
	print(f"Rejected changes: {summary['rejected']}")
	print(f"Writes completed: {summary['writes']}")
	print("GitHub writes: ENABLED" if writes_enabled else "GitHub writes: DISABLED / DRY RUN")


if __name__ == "__main__":
	main()
