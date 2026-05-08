"""
Event stream protocol for streaming assistant responses.

This module defines:
- Event types for streaming (text_delta, thinking_delta, tool_call events, etc.)
- AssistantMessageEventStream class with asyncio.Queue backing
- Async iterator pattern for consuming events
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional, Union
import asyncio

from pypi_ai.types import (
    AssistantMessage,
    StopReason,
    ToolCall,
    TextContent,
    ThinkingContent,
    ImageContent,
)


# =============================================================================
# Event Types
# =============================================================================


@dataclass
class StartEvent:
    """Stream start event with partial assistant message."""

    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass
class TextStartEvent:
    """Text content start event."""

    content_index: int
    type: Literal["text_start"] = "text_start"


@dataclass
class TextDeltaEvent:
    """Text content delta event."""

    delta: str
    content_index: int
    type: Literal["text_delta"] = "text_delta"


@dataclass
class TextEndEvent:
    """Text content end event."""

    content: str
    content_index: int
    type: Literal["text_end"] = "text_end"


@dataclass
class ThinkingStartEvent:
    """Thinking content start event."""

    content_index: int
    type: Literal["thinking_start"] = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    """Thinking content delta event."""

    delta: str
    content_index: int
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ThinkingEndEvent:
    """Thinking content end event."""

    content: str
    content_index: int
    type: Literal["thinking_end"] = "thinking_end"


@dataclass
class ToolCallStartEvent:
    """Tool call start event."""

    content_index: int
    name: Optional[str] = None
    id: Optional[str] = None
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass
class ToolCallDeltaEvent:
    """Tool call arguments delta event."""

    delta: str
    content_index: int
    partial: Optional[AssistantMessage] = None
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass
class ToolCallEndEvent:
    """Tool call complete event."""

    tool_call: ToolCall
    content_index: int
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass
class DoneEvent:
    """Stream completion event."""

    reason: StopReason
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass
class ErrorEvent:
    """Stream error event."""

    reason: StopReason
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent = Union[
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]


# =============================================================================
# Event Stream
# =============================================================================


class AssistantMessageEventStream:
    """
    Async iterator for streaming assistant message events.

    Uses asyncio.Queue for buffering events to handle slow consumers.
    Provides push() for event emission and end() for completion.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[AssistantMessageEvent]] = asyncio.Queue()
        self._ended = False
        self._result: Optional[AssistantMessage] = None
        self._error: Optional[AssistantMessage] = None

    def push(self, event: AssistantMessageEvent) -> None:
        """
        Push an event to the stream.

        Args:
            event: The event to push to the stream.
        """
        if not self._ended:
            self._queue.put_nowait(event)

    def end(self, message: Optional[AssistantMessage] = None) -> None:
        """
        Signal stream completion.

        Args:
            message: The final assistant message (optional).
        """
        self._ended = True
        self._result = message
        # Signal end with None sentinel
        self._queue.put_nowait(None)

    def error(self, error_message: AssistantMessage) -> None:
        """
        Signal stream error.

        Args:
            error_message: The error assistant message.
        """
        self._error = error_message
        self._result = error_message
        self._ended = True
        self._queue.put_nowait(None)

    async def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        """
        Iterate over stream events.

        Yields:
            AssistantMessageEvent events until stream ends.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def result(self) -> AssistantMessage:
        """
        Consume all events and return the final message.

        Returns:
            The complete AssistantMessage.

        Raises:
            RuntimeError: If no message was produced.
        """
        # Consume any remaining events
        async for _ in self:
            pass

        if self._result is None:
            raise RuntimeError("Stream completed without producing a message")
        return self._result

    @property
    def ended(self) -> bool:
        """Check if stream has ended."""
        return self._ended


def create_error_message(
    api: str,
    provider: str,
    model: str,
    error_message: str,
) -> AssistantMessage:
    """
    Create an error assistant message.

    Args:
        api: The API type.
        provider: The provider name.
        model: The model ID.
        error_message: The error message.

    Returns:
        An AssistantMessage with error state.
    """
    from pypi_ai.types import Api, Usage, Cost

    return AssistantMessage(
        role="assistant",
        content=[],
        api=Api(api) if api in Api._value2member_map_ else api,
        provider=provider,
        model=model,
        usage=Usage(
            input=0,
            output=0,
            cost=Cost(),
        ),
        stop_reason=StopReason.ERROR,
        error_message=error_message,
    )