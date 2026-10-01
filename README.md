# Project Comment Automation

Project Comment Automation is a read-first, AI-assisted system for identifying undocumented Python code and proposing concise documentation changes.

## Architecture

The pipeline is configuration-driven:

`RepositoryManager` selects today's repositories, `GitHubManager` reads repository metadata and source files, `CodeAnalyzer` finds undocumented Python elements, `CommentGenerator` proposes documentation, `SafetyValidator` verifies that only documentation changed, and `GitHubWriter` performs an optional controlled write.

The system supports Python source files only. It never lets the LLM edit files directly. Proposed source and diffs remain in memory until they pass safety validation.

## Scheduling

Repository names, URLs, protection flags, and temporary schedule days live in `config/repositories.yml`. Multiple repositories may run on the same day. The GitHub Actions workflow starts `python src/main.py`; the Python application determines the daily selection.

## GitHub Actions

The workflow supports manual `workflow_dispatch` runs and a daily schedule. It uses Python 3.12 and installs the minimal dependency listed in `requirements.txt`.

The provider abstraction defaults to Gemini. Set `LLM_PROVIDER=groq` only when using the backward-compatible Groq provider.

Required GitHub Actions environment values are supplied through secrets:

- `GITHUB_TOKEN`: the built-in token used for read-only repository access.
- `LLM_PROVIDER`: `gemini` in the production workflow.
- `GEMINI_API_KEY`: repository secret used by the official `google-genai` SDK.
- `GEMINI_MODEL`: optional model override; the workflow uses the stable `gemini-3.8-flash` model.
- `GROQ_API_KEY`: optional repository secret when `LLM_PROVIDER=groq`.
- `GITHUB_WRITE_ENABLED`: explicitly set to `false` in the workflow.

The workflow runs safely without `GEMINI_API_KEY`; generation is skipped and read-only work continues when GitHub access is available. No API key belongs in source code, configuration, or this README.

## Safety

Dry-run mode is the default. Real writes require the explicit environment value `GITHUB_WRITE_ENABLED=true`, a `GITHUB_TOKEN`, a successful documentation result, and a successful `SafetyValidator` result.

The writer rejects the protected `project-comment-automation` repository, workflow and configuration paths, automation source files, non-Python files, traversal or absolute paths, stale file SHAs, no-op changes, oversized diffs, and malformed validation results. Protected logic is enforced at the writer boundary, not only by scheduling.

No API keys or tokens belong in source code, YAML, configuration, or this README. The project does not create commits, pushes, or pull requests unless an explicitly enabled future execution passes every write guard.

## Local checks

```powershell
python -m pytest -q
python src/main.py
```

Without credentials, the application reports skipped GitHub access and remains read-only.
