"""Tests for tool argument validation."""

import pytest

from pypi_agent.types import AgentTool, AgentToolResult
from pypi_ai.types import Tool, ToolCall, TextContent


def test_tool_call_arguments():
    """Test ToolCall with arguments."""
    call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test",
        arguments={"arg": "value"},
    )

    assert call.arguments["arg"] == "value"


def test_tool_parameters_schema():
    """Test tool with parameter schema."""
    tool = Tool(
        name="test",
        description="Test tool",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )

    schema = tool.to_json_schema()
    assert schema["name"] == "test"
    assert "query" in schema["parameters"]["properties"]


def test_agent_tool_prepare_arguments():
    """Test AgentTool prepare_arguments hook."""
    def prepare(args):
        # Transform arguments
        if "query" in args:
            args["query"] = args["query"].lower()
        return args

    tool = AgentTool(
        name="test",
        description="Test",
        parameters={},
        prepare_arguments=prepare,
    )

    result = tool.prepare_arguments({"query": "HELLO"})
    assert result["query"] == "hello"


def test_agent_tool_execute_returns_result():
    """Test tool execute returns AgentToolResult."""
    async def execute(tool_call_id, params, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Done")],
            details={"success": True},
        )

    tool = AgentTool(
        name="test",
        description="Test",
        parameters={},
        execute=execute,
    )

    assert tool.execute is not None


def test_tool_strict_mode():
    """Test tool strict mode."""
    tool = Tool(
        name="test",
        description="Test",
        parameters={},
        strict=True,
    )

    assert tool.strict is True


def test_agent_tool_result_content_types():
    """Test AgentToolResult with different content types."""
    # Text content
    result = AgentToolResult(
        content=[TextContent(type="text", text="Output")],
    )
    assert result.content[0].type == "text"

    # With details
    result_with_details = AgentToolResult(
        content=[TextContent(type="text", text="Output")],
        details={"key": "value"},
    )
    assert result_with_details.details["key"] == "value"

    # With terminate flag
    result_terminate = AgentToolResult(
        content=[TextContent(type="text", text="Done")],
        terminate=True,
    )
    assert result_terminate.terminate is True