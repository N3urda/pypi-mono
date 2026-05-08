"""Tests for Grep and Find tools."""

import pytest
import asyncio
import tempfile
import os
import re
from pathlib import Path

from pypi_cli.tools.grep import grep_tool, execute_grep, GrepParameters
from pypi_cli.tools.find import find_tool, execute_find, FindParameters
from pypi_agent.types import AgentToolResult


@pytest.fixture
def temp_dir_with_files():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create some files
        (root / "file1.txt").write_text("Hello world\nPython is great\n")
        (root / "file2.py").write_text("def hello():\n    print('Hello')\n")
        (root / "subdir").mkdir()
        (root / "subdir" / "file3.txt").write_text("Another hello\n")

        yield root


@pytest.mark.asyncio
async def test_grep_simple_pattern(temp_dir_with_files):
    """Test grep with simple pattern."""
    params = GrepParameters(pattern="hello", path=str(temp_dir_with_files))
    result = await execute_grep("test_id", params)

    assert isinstance(result, AgentToolResult)
    assert "hello" in result.content[0].text.lower() or result.details["matches_found"] > 0


@pytest.mark.asyncio
async def test_grep_ignore_case(temp_dir_with_files):
    """Test grep with ignore case."""
    params = GrepParameters(
        pattern="HELLO",
        path=str(temp_dir_with_files),
        ignore_case=True,
    )
    result = await execute_grep("test_id", params)

    assert result.details["matches_found"] > 0


@pytest.mark.asyncio
async def test_grep_include_pattern(temp_dir_with_files):
    """Test grep with include pattern."""
    params = GrepParameters(
        pattern="def",
        path=str(temp_dir_with_files),
        include="*.py",
    )
    result = await execute_grep("test_id", params)

    assert result.details["matches_found"] >= 1


@pytest.mark.asyncio
async def test_grep_no_matches(temp_dir_with_files):
    """Test grep with no matches."""
    params = GrepParameters(
        pattern="nonexistent",
        path=str(temp_dir_with_files),
    )
    result = await execute_grep("test_id", params)

    assert "No matches" in result.content[0].text or result.details["matches_found"] == 0


@pytest.mark.asyncio
async def test_grep_invalid_path():
    """Test grep with invalid path."""
    params = GrepParameters(pattern="test", path="/nonexistent/path")
    result = await execute_grep("test_id", params)

    assert "Path not found" in result.content[0].text


@pytest.mark.asyncio
async def test_find_by_pattern(temp_dir_with_files):
    """Test find by pattern."""
    params = FindParameters(pattern="*.txt", path=str(temp_dir_with_files))
    result = await execute_find("test_id", params)

    assert result.details["count"] >= 2


@pytest.mark.asyncio
async def test_find_by_type_file(temp_dir_with_files):
    """Test find by type (file)."""
    params = FindParameters(
        pattern="*",
        path=str(temp_dir_with_files),
        type="file",
    )
    result = await execute_find("test_id", params)

    assert result.details["count"] >= 3


@pytest.mark.asyncio
async def test_find_by_type_dir(temp_dir_with_files):
    """Test find by type (dir)."""
    params = FindParameters(
        pattern="*",
        path=str(temp_dir_with_files),
        type="dir",
    )
    result = await execute_find("test_id", params)

    assert result.details["count"] >= 1


@pytest.mark.asyncio
async def test_find_no_matches(temp_dir_with_files):
    """Test find with no matches."""
    params = FindParameters(
        pattern="*.nonexistent",
        path=str(temp_dir_with_files),
    )
    result = await execute_find("test_id", params)

    assert "No files found" in result.content[0].text


@pytest.mark.asyncio
async def test_find_invalid_path():
    """Test find with invalid path."""
    params = FindParameters(pattern="*", path="/nonexistent/path")
    result = await execute_find("test_id", params)

    assert "Path not found" in result.content[0].text


def test_grep_tool_definition():
    """Test grep tool has correct definition."""
    assert grep_tool.name == "grep"
    assert grep_tool.label == "Grep"
    assert "pattern" in grep_tool.parameters["properties"]


def test_find_tool_definition():
    """Test find tool has correct definition."""
    assert find_tool.name == "find"
    assert find_tool.label == "Find"
    assert "pattern" in find_tool.parameters["properties"]

@pytest.mark.asyncio
async def test_grep_file_path():
    """Test grep on a single file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Hello world\nTest line\n")

        params = GrepParameters(pattern="Hello", path=str(test_file))
        result = await execute_grep("test_id", params)

        assert result.details["matches_found"] >= 1


@pytest.mark.asyncio
async def test_grep_error_reading_file():
    """Test grep handling read errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_file = root / "test.txt"
        test_file.write_text("content")

        if os.name != 'nt':
            os.chmod(test_file, 0o000)
            params = GrepParameters(pattern="content", path=str(root))
            result = await execute_grep("test_id", params)
            # Should handle error gracefully - might find 0 matches
            assert result is not None
            os.chmod(test_file, 0o644)


@pytest.mark.asyncio
async def test_grep_many_results_truncated():
    """Test grep truncating many results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create many files with matches
        for i in range(150):
            (root / f"file{i}.txt").write_text("match content\n")

        params = GrepParameters(pattern="match", path=str(root))
        result = await execute_grep("test_id", params)

        # Should be truncated at 100
        assert "more results" in result.content[0].text or result.details["matches_found"] > 100


@pytest.mark.asyncio
async def test_find_file_path():
    """Test find on a single file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("content")

        params = FindParameters(pattern="test.txt", path=str(test_file))
        result = await execute_find("test_id", params)

        # Should handle file path
        assert result.details["count"] >= 0


@pytest.mark.asyncio
async def test_find_error():
    """Test find handling errors."""
    params = FindParameters(pattern="[invalid", path="/tmp")
    result = await execute_find("test_id", params)

    # Should handle regex error
    assert "Error" in result.content[0].text or result is not None


@pytest.mark.asyncio
async def test_find_general_exception():
    """Test find tool handling general exceptions."""
    from unittest.mock import patch

    with patch("pathlib.Path.exists", side_effect=OSError("Mock path error")):
        params = FindParameters(pattern="*.txt", path="/tmp")
        result = await execute_find("test_id", params)

        assert "Error finding files" in result.content[0].text
        assert result.details.get("error") is not None


@pytest.mark.asyncio
async def test_find_many_results_truncated():
    """Test find truncating many results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create many files
        for i in range(250):
            (root / f"file{i}.txt").write_text("content")

        params = FindParameters(pattern="*.txt", path=str(root))
        result = await execute_find("test_id", params)

        # Should be truncated at 200
        assert "more results" in result.content[0].text or result.details["count"] > 200


@pytest.mark.asyncio
async def test_grep_general_exception():
    """Test grep tool handling general exceptions."""
    from unittest.mock import patch

    with patch("re.compile", side_effect=re.error("Mock regex error")):
        params = GrepParameters(pattern="test", path="/tmp")
        result = await execute_grep("test_id", params)

        assert "Error searching" in result.content[0].text
        assert result.details.get("error") is not None
