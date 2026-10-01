from src.documentation_policy import DocumentationPolicy
from src.models import CodeElement


POLICY = DocumentationPolicy()


def make_element(name="public", element_type="function", docstring=None, private=False):
    return CodeElement(
        name=name,
        element_type=element_type,
        line_number=1,
        end_line_number=2,
        has_docstring=docstring is not None,
        docstring=docstring,
        is_private=private,
        parent_class="Example" if element_type == "method" else None,
        needs_documentation=docstring is None and not private,
    )


def test_allows_python_files():
    assert POLICY.allows_file("src/module.py") is True
    assert POLICY.allows_file("src/module.PY") is True
    assert POLICY.allows_file("src/module.py", source_size=100000) is True


def test_skips_tests_and_test_directories():
    assert POLICY.allows_file("tests/module.py") is False
    assert POLICY.allows_file("src/test_module.py") is False
    assert POLICY.allows_file("src/module_test.py") is False
    assert POLICY.allows_directory("tests") is False


def test_skips_generated_cache_virtualenv_dependency_and_migration_paths():
    skipped_paths = [
        "venv/module.py",
        ".venv/module.py",
        "node_modules/package.py",
        "build/module.py",
        "dist/module.py",
        "generated/module.py",
        "__pycache__/module.py",
        "migrations/0001_initial.py",
        "src/generated_module.py",
        "src/module_generated.py",
    ]

    assert all(POLICY.allows_file(path) is False for path in skipped_paths)


def test_skips_protected_pca_automation_paths():
    protected_paths = [
        ".github/workflows/daily-comments.yml",
        "config/repositories.yml",
        "src/github_writer.py",
        "src/github_manager.py",
        "src/repository_manager.py",
        "src/safety_validator.py",
        "src/comment_generator.py",
        "src/models.py",
        "requirements.txt",
        ".gitignore",
    ]

    assert all(POLICY.allows_file(path) is False for path in protected_paths)
    assert POLICY.allows_directory("src") is True


def test_allows_public_functions_async_functions_classes_and_methods():
    elements = [
        make_element(element_type="function"),
        make_element(name="fetch", element_type="async_function"),
        make_element(name="Example", element_type="class"),
        make_element(name="method", element_type="method"),
    ]

    assert all(POLICY.allows_element(element) for element in elements)


def test_skips_private_and_dunder_elements():
    assert POLICY.allows_element(make_element(name="_private", private=True)) is False
    assert POLICY.allows_element(make_element(name="__init__", private=True)) is False
    assert POLICY.allows_element(make_element(name="__str__", private=True)) is False


def test_skips_meaningfully_documented_elements():
    assert POLICY.allows_element(make_element(docstring="Already documented.")) is False
    assert POLICY.allows_element(make_element(docstring="  Useful documentation.  ")) is False


def test_empty_docstrings_remain_eligible():
    assert POLICY.allows_element(make_element(docstring="")) is True


def test_preserves_size_and_path_security_protections():
    assert POLICY.allows_file("src/large.py", source_size=100001) is False
    assert POLICY.allows_file("../outside.py") is False
    assert POLICY.allows_file("../../outside.py") is False
    assert POLICY.allows_file("/absolute.py") is False
    assert POLICY.allows_file("C:/absolute.py") is False
    assert POLICY.allows_file("%2e%2e/outside.py") is False
