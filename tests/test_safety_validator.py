import pytest

from src.models import DocumentationChange, ValidationResult
from src.safety_validator import SafetyValidator


VALIDATOR = SafetyValidator()


def make_change(original, proposed, name="calculate_total", element_type="function", line=1):
    return DocumentationChange(
        file_path="example.py",
        element_name=name,
        element_type=element_type,
        original_source=original,
        proposed_source=proposed,
        documentation="Generated documentation.",
        line_number=line,
    )


def test_safe_function_docstring_insertion():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = 'def calculate_total(items):\n    """Calculate the total value of the items."""\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is True
    assert result.errors == []
    assert '"""Calculate the total value' in result.diff


def test_safe_function_docstring_update():
    original = 'def calculate_total(items):\n    """Calculate total."""\n    return sum(items)\n'
    proposed = 'def calculate_total(items):\n    """Calculate the total value of the items."""\n    return sum(items)\n'

    assert VALIDATOR.validate(make_change(original, proposed)).safe is True


def test_safe_class_docstring_insertion():
    original = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    proposed = 'class Calculator:\n    """Provides basic calculator operations."""\n\n    def add(self, a, b):\n        return a + b\n'

    result = VALIDATOR.validate(
        make_change(original, proposed, name="Calculator", element_type="class")
    )

    assert result.safe is True


def test_safe_method_docstring_insertion():
    original = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    proposed = 'class Calculator:\n    def add(self, a, b):\n        """Add two values."""\n        return a + b\n'

    result = VALIDATOR.validate(make_change(original, proposed, name="add", element_type="method", line=2))

    assert result.safe is True


def test_safe_async_function_documentation():
    original = "async def fetch_data(client):\n    return await client.fetch()\n"
    proposed = 'async def fetch_data(client):\n    """Fetch data from the client."""\n    return await client.fetch()\n'

    result = VALIDATOR.validate(
        make_change(original, proposed, name="fetch_data", element_type="async_function")
    )

    assert result.safe is True


def test_safe_nested_function_documentation():
    original = "def outer(value):\n    def inner():\n        return value\n    return inner()\n"
    proposed = 'def outer(value):\n    def inner():\n        """Return the enclosing value."""\n        return value\n    return inner()\n'

    result = VALIDATOR.validate(make_change(original, proposed, name="inner", line=2))

    assert result.safe is True


def test_safe_multiple_functions_only_target_documented():
    original = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    proposed = 'def first():\n    return 1\n\ndef second():\n    """Return the second value."""\n    return 2\n'

    result = VALIDATOR.validate(make_change(original, proposed, name="second", line=4))

    assert result.safe is True


@pytest.mark.parametrize(
    ("proposed", "message"),
    [
        ("def calculate_total(items):\n    return 0\n", "Executable or structural code changed"),
        ("def calculate_total(values):\n    return sum(values)\n", "Executable or structural code changed"),
        ("import os\n\ndef calculate_total(items):\n    return sum(items)\n", "Executable or structural code changed"),
        ("def calculate_total(items):\n    value = sum(items)\n    return value\n", "Executable or structural code changed"),
        ("def calculate_total(items):\n    if items:\n        return sum(items)\n    return 0\n", "Executable or structural code changed"),
        ("def calculate_total(items):\n    return list(items)\n", "Executable or structural code changed"),
    ],
)
def test_rejects_executable_changes(proposed, message):
    original = "def calculate_total(items):\n    return sum(items)\n"

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False
    assert message in result.errors


def test_rejects_decorator_change():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = '@cached\ndef calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False


def test_rejects_function_rename():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = 'def calculate_sum(items):\n    """Calculate totals."""\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False
    assert any("Target element not found" in error for error in result.errors)


def test_rejects_class_rename():
    original = "class Calculator:\n    pass\n"
    proposed = 'class AdvancedCalculator:\n    """Calculator class."""\n    pass\n'

    result = VALIDATOR.validate(make_change(original, proposed, name="Calculator", element_type="class"))

    assert result.safe is False


def test_rejects_function_addition():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = 'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n\ndef added():\n    return 1\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False


def test_rejects_function_removal():
    original = "def calculate_total(items):\n    return sum(items)\n\ndef other():\n    return 2\n"
    proposed = 'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False


def test_rejects_comment_change_outside_target():
    original = "# original comment\ndef calculate_total(items):\n    return sum(items)\n"
    proposed = '# changed comment\ndef calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed, line=2))

    assert result.safe is False
    assert "Comments outside the target documentation changed" in result.errors


def test_rejects_documentation_change_outside_target():
    original = 'def first():\n    """First."""\n    return 1\n\ndef second():\n    return 2\n'
    proposed = 'def first():\n    """Changed first."""\n    return 1\n\ndef second():\n    """Document second."""\n    return 2\n'

    result = VALIDATOR.validate(make_change(original, proposed, name="second", line=5))

    assert result.safe is False
    assert "Documentation changed outside the target element" in result.errors


def test_rejects_missing_target():
    original = "def other():\n    return 1\n"
    proposed = 'def other():\n    """Document other."""\n    return 1\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False
    assert "Target element not found" in result.errors[0]


def test_rejects_invalid_original_python():
    result = VALIDATOR.validate(make_change("def broken(:\n", "def broken(:\n"))

    assert result.safe is False
    assert any("Invalid original Python source" in error for error in result.errors)


def test_rejects_invalid_proposed_python():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = 'def calculate_total(items):\n    """Broken.\n    return sum(items)\n'

    result = VALIDATOR.validate(make_change(original, proposed))

    assert result.safe is False
    assert any("Invalid proposed Python source" in error for error in result.errors)


def test_rejects_empty_source_without_target():
    result = VALIDATOR.validate(make_change("", "", name="missing"))

    assert result.safe is False
    assert result.errors


def test_generates_unified_diff_without_applying_it():
    original = "def calculate_total(items):\n    return sum(items)\n"
    proposed = 'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n'

    diff = VALIDATOR.generate_diff(original, proposed, "module.py")

    assert "--- a/module.py" in diff
    assert "+++ b/module.py" in diff
    assert "+    \"\"\"Calculate totals.\"\"\"" in diff
    assert original == "def calculate_total(items):\n    return sum(items)\n"


def test_validation_result_structure():
    result = VALIDATOR.validate(
        make_change(
            "def calculate_total(items):\n    return sum(items)\n",
            'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n',
        )
    )

    assert isinstance(result, ValidationResult)
    assert result.safe is True
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.diff, str)


def test_conservatively_rejects_unsuccessful_change():
    change = make_change(
        "def calculate_total(items):\n    return sum(items)\n",
        'def calculate_total(items):\n    """Calculate totals."""\n    return sum(items)\n',
    )
    unsuccessful_change = DocumentationChange(
        **{**change.__dict__, "success": False}
    )

    result = VALIDATOR.validate(unsuccessful_change)

    assert result.safe is False
    assert "unsuccessful" in result.errors[0]
