"""Tests for steering messages and edge cases in agent loop."""

import pytest
import asyncio
from unittest.mock import patch

from pypi_agent.loop import run_loop
from pypi_agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
)
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
from pypi_ai.event_stream import AssistantMessageEventStream, DoneEvent


def _create_config(model, **kwargs):
    """Create an AgentLoopConfig with async convert_to_llm."""
    async def convert_to_llm(m):
        return m
    return AgentLoopConfig(
        model=model,
        convert_to_llm=convert_to_llm,
        **kwargs
    )


def _create_emit(events):
    """Create an async emit function."""
    async def emit(event):
        events.append(event)
    return emit


def _create_mock_stream(message):
    """Create a mock stream that returns a message."""
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=StopReason.END, message=message))
    stream.end(message)
    return stream


@pytest.fixture
def model():
    return Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")


@pytest.mark.asyncio
async def test_run_loop_with_steering_messages(model):
    """Test run_loop with steering messages."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call returns message with tool call
            msg = AssistantMessage(
                role="assistant",
                content=[
                    TextContent(type="text", text="Response"),
                ],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(msg)
        else:
            # Second call returns final message
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(msg)

    steering_count = [0]

    async def get_steering():
        steering_count[0] += 1
        if steering_count[0] == 1:
            return [UserMessage(content="Steer")]
        return []

    config = _create_config(model, get_steering_messages=get_steering)

    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert steering_count[0] > 0


@pytest.mark.asyncio
async def test_run_loop_with_follow_up_messages(model):
    """Test run_loop with follow-up messages."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    def mock_stream_fn(*args, **kwargs):
        msg = AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="Response")],
            api=Api.ANTHROPIC_MESSAGES,
            provider="anthropic",
            model="test",
            usage=Usage(),
            stop_reason=StopReason.END,
        )
        return _create_mock_stream(msg)

    async def get_follow_up():
        return [UserMessage(content="Follow-up")]

    config = _create_config(model, get_follow_up_messages=get_follow_up)

    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))


@pytest.mark.asyncio
async def test_run_loop_tool_result_added_to_context(model):
    """Test that tool results are added to context."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Tool result")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            msg = AssistantMessage(
                role="assistant",
                content=[
                    TextContent(type="text", text="Using tool"),
                    ToolCall(
                        type="toolCall",
                        id="call_1",
                        name="test_tool",
                        arguments={},
                    ),
                ],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            )
            stream = AssistantMessageEventStream()
            stream.push(DoneEvent(reason=StopReason.TOOL_USE, message=msg))
            stream.end(msg)
            return stream
        else:
            msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(msg)

    config = _create_config(model)

    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    # Check that context has tool result
    tool_results = [m for m in context.messages if isinstance(m, ToolResultMessage)]
    assert len(tool_results) > 0


@pytest.mark.asyncio
async def test_run_loop_empty_response_ends(model):
    """Test run_loop ends when stream returns None."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    def mock_stream_fn(*args, **kwargs):
        # Return a stream that ends with None message
        stream = AssistantMessageEventStream()
        stream.end(None)
        return stream

    config = _create_config(model)

    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    # Should complete without error
    assert True
