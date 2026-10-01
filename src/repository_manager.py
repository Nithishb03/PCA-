"""Configuration-driven repository scheduling for Project Comment Automation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

try:
    from .models import Repository
except ImportError:
    from models import Repository

SCHEDULE_DAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)
REQUIRED_FIELDS = frozenset({"name", "url", "schedule_day", "protected"})


class RepositoryManager:
    """Load, validate, and query repositories from a YAML configuration."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.repositories = self._load_repositories()

    def _load_repositories(self) -> list[Repository]:
        with self.config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping")

        entries = config.get("repositories")
        if not isinstance(entries, list):
            raise ValueError("Configuration must contain a 'repositories' list")

        repositories: list[Repository] = []
        names: set[str] = set()
        for index, entry in enumerate(entries):
            repository = self._parse_entry(entry, index)
            if repository.name in names:
                raise ValueError(f"Duplicate repository name: {repository.name}")
            names.add(repository.name)
            repositories.append(repository)
        return repositories

    @staticmethod
    def _parse_entry(entry: Any, index: int) -> Repository:
        if not isinstance(entry, dict):
            raise ValueError(f"Repository entry {index} must be a mapping")

        missing_fields = REQUIRED_FIELDS - entry.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Repository entry {index} is missing: {missing}")

        name = entry["name"]
        url = entry["url"]
        schedule_day = entry["schedule_day"]
        protected = entry["protected"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Repository entry {index} name must be a non-empty string")
        if not isinstance(url, str) or not RepositoryManager._is_http_url(url):
            raise ValueError(f"Repository entry {index} url must be an HTTP(S) URL")
        if not isinstance(schedule_day, str):
            raise ValueError(f"Repository entry {index} schedule_day must be a string")
        normalized_day = schedule_day.strip().lower()
        if normalized_day not in SCHEDULE_DAYS:
            raise ValueError(f"Repository entry {index} has invalid schedule_day: {schedule_day}")
        if type(protected) is not bool:
            raise ValueError(f"Repository entry {index} protected must be a boolean")

        return Repository(
            name=name,
            url=url,
            schedule_day=normalized_day,
            protected=protected,
        )

    @staticmethod
    def _is_http_url(url: str) -> bool:
        parsed_url = urlparse(url)
        return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)

    def get_repositories_for_day(self, day: str) -> list[Repository]:
        """Return all repositories scheduled for the requested day."""
        normalized_day = day.strip().lower()
        if normalized_day not in SCHEDULE_DAYS:
            raise ValueError(f"Invalid schedule day: {day}")
        return [repository for repository in self.repositories if repository.schedule_day == normalized_day]

    def get_todays_repositories(self) -> list[Repository]:
        """Return all repositories scheduled for today's day of the week."""
        return self.get_repositories_for_day(date.today().strftime("%A"))

    def get_protected_repositories(self) -> list[Repository]:
        """Return all repositories marked as protected."""
        return [repository for repository in self.repositories if repository.protected]
