"""Tests for agent_loop and agent_loop_continue functions."""

import pytest
import asyncio
from unittest.mock import patch

from pypi_agent.loop import agent_loop, agent_loop_continue
from pypi_agent.types import (
    AgentContext,
    AgentLoopConfig,
    ToolExecutionMode,
)
from pypi_ai.types import (
    Model,
    Api,
    UserMessage,
    AssistantMessage,
    TextContent,
    StopReason,
    Usage,
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


def _create_mock_stream(message):
    """Create a mock stream that returns a message."""
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=StopReason.END, message=message))
    stream.end(message)
    return stream


@pytest.fixture
def model():
    return Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")


@pytest.fixture
def mock_assistant():
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Response")],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.END,
    )


@pytest.mark.asyncio
async def test_agent_loop_emits_events(model, mock_assistant):
    """Test agent_loop emits start and end events."""
    context = AgentContext(
        system_prompt="Test",
        messages=[],
    )

    config = _create_config(model)

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant)):
        events = []
        async for event in agent_loop([UserMessage(content="Hello")], context, config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_continue_emits_events(model, mock_assistant):
    """Test agent_loop_continue emits start and end events."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    config = _create_config(model)

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant)):
        events = []
        async for event in agent_loop_continue(context, config):
            events.append(event)

        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types


@pytest.mark.asyncio
async def test_agent_loop_continue_empty_messages_raises(model):
    """Test agent_loop_continue raises when no messages."""
    context = AgentContext(system_prompt="Test")
    config = _create_config(model)

    with pytest.raises(ValueError, match="no messages"):
        async for _ in agent_loop_continue(context, config):
            pass


@pytest.mark.asyncio
async def test_agent_loop_continue_assistant_last_raises(model):
    """Test agent_loop_continue raises when assistant is last message."""
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
    )
    config = _create_config(model)

    with pytest.raises(ValueError, match="Cannot continue"):
        async for _ in agent_loop_continue(context, config):
            pass


@pytest.mark.asyncio
async def test_agent_loop_multiple_prompts(model, mock_assistant):
    """Test agent_loop with multiple prompts."""
    context = AgentContext(
        system_prompt="Test",
        messages=[],
    )

    config = _create_config(model)

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant)):
        events = []
        async for event in agent_loop([
            UserMessage(content="First"),
            UserMessage(content="Second"),
        ], context, config):
            events.append(event)

        # Should have message events for both prompts
        message_starts = [e for e in events if e.type == "message_start"]
        assert len(message_starts) >= 2
