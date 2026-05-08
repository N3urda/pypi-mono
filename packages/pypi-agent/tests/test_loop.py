"""Tests for agent loop."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from pypi_agent.loop import emit_event, execute_tool
from pypi_agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    ToolExecutionMode,
    BeforeToolCallResult,
    AfterToolCallResult,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
)
from pypi_agent.state import AgentState
from pypi_ai.types import (
    Model,
    Api,
    UserMessage,
    AssistantMessage,
    TextContent,
    StopReason,
    Usage,
    ToolCall,
    ToolResultMessage,
)


def test_tool_execution_mode():
    """Test ToolExecutionMode values."""
    assert ToolExecutionMode.SEQUENTIAL == "sequential"
    assert ToolExecutionMode.PARALLEL == "parallel"


@pytest.mark.asyncio
async def test_agent_tool_result():
    """Test AgentToolResult."""
    result = AgentToolResult(
        content=[TextContent(type="text", text="Success")],
        details={"key": "value"},
        terminate=False,
    )

    assert len(result.content) == 1
    assert result.terminate is False


@pytest.mark.asyncio
async def test_emit_event_sync():
    """Test emit_event with sync callback."""
    events = []

    def sync_sink(event):
        events.append(event)

    event = ToolExecutionStartEvent(
        tool_call_id="test",
        tool_name="test",
        args={},
    )

    await emit_event(sync_sink, event)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_emit_event_async():
    """Test emit_event with async callback."""
    events = []

    async def async_sink(event):
        events.append(event)

    event = ToolExecutionStartEvent(
        tool_call_id="test",
        tool_name="test",
        args={},
    )

    await emit_event(async_sink, event)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_tool_success():
    """Test execute_tool with successful execution."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Success")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    events = []
    async def emit(event):
        events.append(event)

    result = await execute_tool(tool_call, context, config, None, emit)

    assert result.tool_call_id == "call_1"
    assert not result.is_error
    assert len(events) == 2  # start and end


@pytest.mark.asyncio
async def test_execute_tool_with_prepare_arguments():
    """Test execute_tool with prepare_arguments."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Got: {args['modified']}")],
        )

    def prepare_args(args):
        return {"modified": args["arg"] + "_prepared"}

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        prepare_arguments=prepare_args,
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, lambda e: events.append(e))

    assert not result.is_error


@pytest.mark.asyncio
async def test_execute_tool_with_before_hook():
    """Test execute_tool with before_tool_call hook blocking."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Success")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")

    async def before_hook(ctx, signal):
        return BeforeToolCallResult(
            block=True,
            reason="Blocked by hook",
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        before_tool_call=before_hook,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, lambda e: events.append(e))

    assert result.is_error is True
    assert "Blocked" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_tool_with_after_hook():
    """Test execute_tool with after_tool_call hook."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Success")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")

    async def after_hook(ctx, signal):
        return AfterToolCallResult(
            content=[TextContent(type="text", text="Modified result")],
            details={"modified": True},
            is_error=False,
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        after_tool_call=after_hook,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, lambda e: events.append(e))

    assert result.content[0].text == "Modified result"
    assert result.details == {"modified": True}


@pytest.mark.asyncio
async def test_execute_tool_with_after_hook_error():
    """Test execute_tool with after_tool_call hook setting error."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Success")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")

    async def after_hook(ctx, signal):
        return AfterToolCallResult(
            content=[TextContent(type="text", text="Error result")],
            is_error=True,
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        after_tool_call=after_hook,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    result = await execute_tool(tool_call, context, config, None, lambda e: None)

    assert result.is_error is True

@pytest.mark.asyncio
async def test_execute_tool_with_before_hook_blocks():
    """Test execute_tool with before_tool_call hook that blocks."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Should not run")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[AssistantMessage(
            content=[],
            api=Api.ANTHROPIC_MESSAGES,
            provider="anthropic",
            model="test",
            usage=Usage(),
            stop_reason=StopReason.END,
        )],
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")

    async def before_hook(ctx, signal):
        return BeforeToolCallResult(
            block=True,
            reason="Blocked for testing",
        )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        before_tool_call=before_hook,
    )

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={"arg": "value"},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, lambda e: events.append(e))

    assert result.is_error is True
    assert "Blocked" in result.content[0].text
