from pathlib import Path

from src.repository_manager import RepositoryManager


CONFIG_PATH = Path(__file__).parents[1] / "config" / "repositories.yml"


def test_phase_2_configuration_still_loads_all_repositories():
    manager = RepositoryManager(CONFIG_PATH)

    assert len(manager.repositories) == 14
    assert [repository.name for repository in manager.get_repositories_for_day("Thursday")] == [
        "IEH-AC",
        "Nithishb03",
    ]


def test_phase_2_protected_repository_is_preserved():
    manager = RepositoryManager(CONFIG_PATH)

    assert [repository.name for repository in manager.get_protected_repositories()] == [
        "project-comment-automation"
    ]
