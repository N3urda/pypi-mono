"""Tests for Bash tool."""

import pytest
import asyncio

from pypi_cli.tools.bash import bash_tool, execute_bash, BashParameters
from pypi_agent.types import AgentToolResult


@pytest.mark.asyncio
async def test_bash_simple_command():
    """Test simple bash command."""
    params = BashParameters(command="echo 'hello'")
    result = await execute_bash("test_id", params)

    assert isinstance(result, AgentToolResult)
    assert "hello" in result.content[0].text


@pytest.mark.asyncio
async def test_bash_exit_code():
    """Test exit code in result."""
    params = BashParameters(command="true")
    result = await execute_bash("test_id", params)

    assert result.details["exit_code"] == 0


@pytest.mark.asyncio
async def test_bash_error_command():
    """Test error command."""
    params = BashParameters(command="ls /nonexistent")
    result = await execute_bash("test_id", params)

    # Should still return output (stderr)
    assert result.content[0].text != ""


@pytest.mark.asyncio
async def test_bash_timeout():
    """Test timeout handling."""
    params = BashParameters(command="sleep 5", timeout=100)
    result = await execute_bash("test_id", params)

    assert result.details.get("timeout") is True


def test_bash_tool_definition():
    """Test bash tool has correct definition."""
    assert bash_tool.name == "bash"
    assert bash_tool.label == "Bash"
    assert "command" in bash_tool.parameters["properties"]