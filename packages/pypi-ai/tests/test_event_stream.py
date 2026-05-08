"""Tests for event stream protocol."""

import pytest
import asyncio

from pypi_ai.event_stream import (
    AssistantMessageEventStream,
    TextDeltaEvent,
    DoneEvent,
    StartEvent,
)
from pypi_ai.types import AssistantMessage, Api, StopReason, Usage


@pytest.mark.asyncio
async def test_event_stream_basic():
    """Test basic event stream functionality."""
    stream = AssistantMessageEventStream()

    # Push events
    stream.push(TextDeltaEvent(delta="Hello", content_index=0))

    # Create message
    msg = AssistantMessage(
        role="assistant",
        content=[],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="claude-test",
        usage=Usage(),
        stop_reason=StopReason.END,
    )

    stream.push(DoneEvent(reason=StopReason.END, message=msg))
    stream.end(msg)

    # Iterate
    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 2
    assert events[0].type == "text_delta"
    assert events[0].delta == "Hello"


@pytest.mark.asyncio
async def test_event_stream_result():
    """Test getting final result from stream."""
    stream = AssistantMessageEventStream()

    msg = AssistantMessage(
        role="assistant",
        content=[],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="claude-test",
        usage=Usage(input=10, output=5),
        stop_reason=StopReason.END,
    )

    stream.push(DoneEvent(reason=StopReason.END, message=msg))
    stream.end(msg)

    result = await stream.result()
    assert result.role == "assistant"
    assert result.usage.input == 10


@pytest.mark.asyncio
async def test_event_stream_empty():
    """Test empty stream raises error."""
    stream = AssistantMessageEventStream()
    stream.end()

    with pytest.raises(RuntimeError):
        await stream.result()


def test_event_stream_push():
    """Test push method."""
    stream = AssistantMessageEventStream()
    stream.push(TextDeltaEvent(delta="test", content_index=0))

    assert not stream.ended


def test_event_stream_end():
    """Test end method."""
    stream = AssistantMessageEventStream()
    stream.end()

    assert stream.ended