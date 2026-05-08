"""Tests for hook behavior."""

import pytest
import asyncio

from pypi_agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    AfterToolCallResult,
    ToolExecutionMode,
)
from pypi_agent.loop import execute_tool
from pypi_ai.types import (
    Model,
    Api,
    ToolCall,
    TextContent,
)


@pytest.fixture
def sample_tool():
    """Create a sample tool."""
    async def execute(tool_call_id, params, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Executed with {params}")],
            details={"executed": True},
        )

    return AgentTool(
        name="test_tool",
        description="Test tool",
        parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        label="Test",
        execute=execute,
    )


@pytest.mark.asyncio
async def test_before_tool_call_hook_blocks(sample_tool):
    """Test beforeToolCall hook can block execution."""
    context = AgentContext(
        system_prompt="Test",
        tools=[sample_tool],
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    # Create config with blocking hook
    async def before_hook(ctx, signal):
        return BeforeToolCallResult(block=True, reason="Blocked by policy")

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        before_tool_call=before_hook,
    )

    result_msg = await execute_tool(tool_call, context, config, None, lambda e: None)

    assert result_msg.is_error is True
    assert "Blocked" in result_msg.content[0].text


@pytest.mark.asyncio
async def test_after_tool_call_hook_modifies(sample_tool):
    """Test afterToolCall hook can modify result."""
    context = AgentContext(
        system_prompt="Test",
        tools=[sample_tool],
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    # Create config with modifying hook
    async def after_hook(ctx, signal):
        return AfterToolCallResult(
            content=[TextContent(type="text", text="Modified result")],
            is_error=False,
        )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        after_tool_call=after_hook,
    )

    result_msg = await execute_tool(tool_call, context, config, None, lambda e: None)

    assert result_msg.content[0].text == "Modified result"


@pytest.mark.asyncio
async def test_tool_without_execute(sample_tool):
    """Test tool without execute function."""
    no_exec_tool = AgentTool(
        name="no_exec_tool",
        description="Tool without execute",
        parameters={},
        label="No Exec",
        execute=None,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[no_exec_tool],
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="no_exec_tool",
        arguments={},
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
    )

    result_msg = await execute_tool(tool_call, context, config, None, lambda e: None)

    assert result_msg.is_error is True
    assert "no execute function" in result_msg.content[0].text.lower()


@pytest.mark.asyncio
async def test_unknown_tool():
    """Test calling unknown tool."""
    context = AgentContext(system_prompt="Test", tools=[])

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="unknown_tool",
        arguments={},
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
    )

    result_msg = await execute_tool(tool_call, context, config, None, lambda e: None)

    assert result_msg.is_error is True
    assert "Unknown tool" in result_msg.content[0].text