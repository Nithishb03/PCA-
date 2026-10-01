from src.code_analyzer import CodeAnalyzer


ANALYZER = CodeAnalyzer()


def test_public_function_without_docstring_needs_documentation():
    result = ANALYZER.analyze_python_file("def public_function():\n    pass\n", "module.py")

    element = result.elements[0]
    assert (element.name, element.element_type) == ("public_function", "function")
    assert element.has_docstring is False
    assert element.needs_documentation is True
    assert element.is_private is False


def test_public_function_with_docstring_is_documented():
    result = ANALYZER.analyze_python_file(
        'def public_function():\n    """Documented function."""\n    pass\n',
        "module.py",
    )

    element = result.elements[0]
    assert element.has_docstring is True
    assert element.docstring == "Documented function."
    assert element.needs_documentation is False


def test_private_function_without_docstring_is_reported_but_not_required():
    result = ANALYZER.analyze_python_file("def _private_function():\n    pass\n", "module.py")

    element = result.elements[0]
    assert element.is_private is True
    assert element.needs_documentation is False


def test_public_class_without_docstring_needs_documentation():
    result = ANALYZER.analyze_python_file("class PublicClass:\n    pass\n", "module.py")

    element = result.elements[0]
    assert (element.name, element.element_type) == ("PublicClass", "class")
    assert element.needs_documentation is True


def test_class_docstring_is_extracted():
    result = ANALYZER.analyze_python_file(
        'class PublicClass:\n    """Class documentation."""\n    pass\n',
        "module.py",
    )

    element = result.elements[0]
    assert element.has_docstring is True
    assert element.docstring == "Class documentation."
    assert element.needs_documentation is False


def test_methods_include_parent_class():
    source = "class Example:\n    def method(self):\n        pass\n\n    def _private_method(self):\n        pass\n"
    result = ANALYZER.analyze_python_file(source, "module.py")

    methods = [element for element in result.elements if element.element_type == "method"]
    assert [(method.name, method.parent_class) for method in methods] == [
        ("method", "Example"),
        ("_private_method", "Example"),
    ]
    assert methods[0].needs_documentation is True
    assert methods[1].needs_documentation is False


def test_async_functions_are_detected():
    result = ANALYZER.analyze_python_file("async def fetch_data():\n    pass\n", "module.py")

    assert result.elements[0].name == "fetch_data"
    assert result.elements[0].element_type == "async_function"
    assert result.elements[0].needs_documentation is True


def test_multiple_functions_are_detected():
    source = "def first():\n    pass\n\ndef second():\n    pass\n"

    result = ANALYZER.analyze_python_file(source, "module.py")

    assert [element.name for element in result.elements] == ["first", "second"]


def test_nested_functions_are_detected():
    source = "def outer():\n    def inner():\n        pass\n    return inner\n"

    result = ANALYZER.analyze_python_file(source, "module.py")

    assert [element.name for element in result.elements] == ["outer", "inner"]
    assert all(element.element_type == "function" for element in result.elements)


def test_empty_python_file_returns_empty_analysis():
    result = ANALYZER.analyze_python_file("", "empty.py")

    assert result.language == "python"
    assert result.elements == []
    assert result.error is None


def test_invalid_python_returns_analysis_error():
    result = ANALYZER.analyze_python_file("def broken(:\n", "broken.py")

    assert result.elements == []
    assert result.error is not None
    assert result.error.startswith("Invalid Python syntax:")


def test_line_and_end_line_numbers_are_correct():
    source = "\n\ndef public_function():\n    value = 1\n    return value\n"

    result = ANALYZER.analyze_python_file(source, "module.py")

    assert result.elements[0].line_number == 3
    assert result.elements[0].end_line_number == 5


def test_existing_docstring_text_is_extracted():
    source = 'def documented():\n    """First line.\n\n    More detail.\n    """\n    pass\n'

    result = ANALYZER.analyze_python_file(source, "module.py")

    assert result.elements[0].docstring == "First line.\n\nMore detail."


def test_non_python_file_is_rejected_by_single_file_api():
    result = ANALYZER.analyze_file("plain text", "README.md")

    assert result.language == "unsupported"
    assert result.elements == []
    assert result.error == "Unsupported file extension; only Python files are supported"


def test_multiple_file_api_ignores_non_python_and_excluded_paths():
    results = ANALYZER.analyze_files(
        {
            "module.py": "def public_function():\n    pass\n",
            "README.md": "not Python",
            "venv/ignored.py": "def ignored():\n    pass\n",
        }
    )

    assert [result.file_path for result in results] == ["module.py"]
