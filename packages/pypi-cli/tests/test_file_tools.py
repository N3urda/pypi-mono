"""Tests for file tools."""

import pytest
import asyncio
import tempfile
import os

from pypi_cli.tools.read import read_tool, execute_read, ReadParameters
from pypi_cli.tools.write import write_tool, execute_write, WriteParameters
from pypi_cli.tools.edit import edit_tool, execute_edit, EditParameters
from pypi_agent.types import AgentToolResult


@pytest.fixture
def temp_file():
    """Create a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Line 1\nLine 2\nLine 3\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.mark.asyncio
async def test_read_file(temp_file):
    """Test reading file."""
    params = ReadParameters(file_path=temp_file)
    result = await execute_read("test_id", params)

    assert "Line 1" in result.content[0].text
    assert result.details["total_lines"] == 3


@pytest.mark.asyncio
async def test_read_file_with_offset(temp_file):
    """Test reading file with offset."""
    params = ReadParameters(file_path=temp_file, offset=2)
    result = await execute_read("test_id", params)

    # Should start from line 2
    assert "2\tLine 2" in result.content[0].text


@pytest.mark.asyncio
async def test_read_nonexistent_file():
    """Test reading nonexistent file."""
    params = ReadParameters(file_path="/nonexistent/file.txt")
    result = await execute_read("test_id", params)

    assert "File not found" in result.content[0].text


@pytest.mark.asyncio
async def test_write_file():
    """Test writing file."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as f:
        params = WriteParameters(file_path=f.name, content="Test content")
        result = await execute_write("test_id", params)

        assert "Successfully wrote" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_file(temp_file):
    """Test editing file."""
    params = EditParameters(
        file_path=temp_file,
        old_string="Line 2",
        new_string="Modified Line 2",
    )
    result = await execute_edit("test_id", params)

    assert "Successfully edited" in result.content[0].text

    # Verify change
    with open(temp_file) as f:
        content = f.read()
    assert "Modified Line 2" in content


@pytest.mark.asyncio
async def test_edit_not_unique(temp_file):
    """Test edit with non-unique string."""
    params = EditParameters(
        file_path=temp_file,
        old_string="Line",  # Appears multiple times
        new_string="Modified",
    )
    result = await execute_edit("test_id", params)

    assert "appears" in result.content[0].text or "not unique" in result.content[0].text


@pytest.mark.asyncio
async def test_edit_not_found(temp_file):
    """Test edit with string not found."""
    params = EditParameters(
        file_path=temp_file,
        old_string="NonexistentString",
        new_string="Replacement",
    )
    result = await execute_edit("test_id", params)

    assert "not found" in result.content[0].text


def test_tool_definitions():
    """Test all tools have correct definitions."""
    assert read_tool.name == "read"
    assert write_tool.name == "write"
    assert edit_tool.name == "edit"


@pytest.mark.asyncio
async def test_edit_file_not_found():
    """Test editing nonexistent file."""
    params = EditParameters(
        file_path="/nonexistent/file.txt",
        old_string="something",
        new_string="else",
    )
    result = await execute_edit("test_id", params)

    assert "File not found" in result.content[0].text


@pytest.mark.asyncio
async def test_write_file_to_new_path():
    """Test writing to a new file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_file = os.path.join(tmpdir, "new_file.txt")
        params = WriteParameters(file_path=new_file, content="New content")
        result = await execute_write("test_id", params)

        assert "Successfully wrote" in result.content[0].text

        # Verify file was created
        with open(new_file) as f:
            content = f.read()
        assert content == "New content"


@pytest.mark.asyncio
async def test_read_file_with_limit(temp_file):
    """Test reading file with limit."""
    params = ReadParameters(file_path=temp_file, limit=2)
    result = await execute_read("test_id", params)

    # Should have content even with limit
    assert result.content[0].text != ""


@pytest.mark.asyncio
async def test_read_directory():
    """Test reading a directory (should fail)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = ReadParameters(file_path=tmpdir)
        result = await execute_read("test_id", params)

        # Should handle the error
        assert "Error" in result.content[0].text or "not found" in result.content[0].text