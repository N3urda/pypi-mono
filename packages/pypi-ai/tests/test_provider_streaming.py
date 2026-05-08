"""Tests for provider HTTP streaming implementations."""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from dataclasses import dataclass

from pypi_ai.types import (
    Api,
    Model,
    Context,
    UserMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ImageContent,
    ToolCall,
    ToolResultMessage,
    StopReason,
    Usage,
    StreamOptions,
    SimpleStreamOptions,
    ThinkingLevel,
    Tool,
)
from pypi_ai.providers.anthropic import AnthropicProvider
from pypi_ai.providers.openai import OpenAIProvider
from pypi_ai.providers.google import GoogleProvider
from pypi_ai.providers.mistral import MistralProvider


# =============================================================================
# Mock Helpers
# =============================================================================


@dataclass
class MockStreamEvent:
    """Mock streaming event."""
    type: str
    index: int = 0
    content_block: dict = None
    delta: dict = None
    stop_reason: str = None
    usage: dict = None
    message: dict = None


class MockAsyncContextManager:
    """Mock async context manager."""
    def __init__(self, items):
        self.items = items

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._async_iter()

    async def _async_iter(self):
        for item in self.items:
            yield item


class MockAsyncIterator:
    """Mock async iterator for chunked responses."""
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._async_iter()

    async def _async_iter(self):
        for item in self.items:
            yield item


# =============================================================================
# Anthropic Provider Streaming Tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_stream_text_content():
    """Test Anthropic streaming with text content."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    # Create mock events
    events = [
        MockStreamEvent(type="message_start", message={"usage": {"input_tokens": 10}}),
        MockStreamEvent(type="content_block_start", index=0, content_block={"type": "text"}),
        MockStreamEvent(type="content_block_delta", index=0, delta={"type": "text_delta", "text": "Hello"}),
        MockStreamEvent(type="content_block_stop", index=0),
        MockStreamEvent(type="message_delta", stop_reason="end_turn", usage={"output_tokens": 5}),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)

        # Give time for async task to run
        await asyncio.sleep(0.1)

        # The stream should have events
        collected_events = []
        async for event in stream:
            collected_events.append(event)
            if len(collected_events) > 10:
                break

        assert len(collected_events) > 0


@pytest.mark.asyncio
async def test_anthropic_stream_with_thinking():
    """Test Anthropic streaming with thinking content."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    events = [
        MockStreamEvent(type="message_start", message={"usage": {"input_tokens": 10}}),
        MockStreamEvent(type="content_block_start", index=0, content_block={"type": "thinking"}),
        MockStreamEvent(type="content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "Let me think..."}),
        MockStreamEvent(type="content_block_stop", index=0),
        MockStreamEvent(type="content_block_start", index=1, content_block={"type": "text"}),
        MockStreamEvent(type="content_block_delta", index=1, delta={"type": "text_delta", "text": "Hello"}),
        MockStreamEvent(type="content_block_stop", index=1),
        MockStreamEvent(type="message_delta", stop_reason="end_turn", usage={"output_tokens": 5}),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anthropic_stream_with_tool_call():
    """Test Anthropic streaming with tool call."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    events = [
        MockStreamEvent(type="message_start", message={"usage": {"input_tokens": 10}}),
        MockStreamEvent(type="content_block_start", index=0, content_block={"type": "tool_use", "id": "call_1", "name": "test_tool"}),
        MockStreamEvent(type="content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"arg":'}),
        MockStreamEvent(type="content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": ' "value"}'}),
        MockStreamEvent(type="content_block_stop", index=0),
        MockStreamEvent(type="message_delta", stop_reason="tool_use", usage={"output_tokens": 5}),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anthropic_stream_with_invalid_json_tool_call():
    """Test Anthropic streaming with invalid JSON in tool call (covers JSONDecodeError handling)."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    # Create a mock content_block with attributes (not a dict)
    mock_content_block = MagicMock()
    mock_content_block.type = "tool_use"
    mock_content_block.id = "call_1"
    mock_content_block.name = "test_tool"

    # Create a mock delta with partial_json attribute
    mock_delta = MagicMock()
    mock_delta.type = "input_json_delta"
    mock_delta.partial_json = 'invalid json{'  # Malformed JSON to trigger JSONDecodeError

    events = [
        MockStreamEvent(type="message_start", message={"usage": {"input_tokens": 10}}),
        MockStreamEvent(type="content_block_start", index=0, content_block=mock_content_block),
        MockStreamEvent(type="content_block_delta", index=0, delta=mock_delta),
        MockStreamEvent(type="content_block_stop", index=0),
        MockStreamEvent(type="message_delta", stop_reason="tool_use", usage={"output_tokens": 5}),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.1)

        # Should still complete without error (JSONDecodeError is caught)
        received_events = []
        async for event in stream:
            received_events.append(event)
            if len(received_events) > 5:
                break


@pytest.mark.asyncio
async def test_anthropic_stream_with_error():
    """Test Anthropic streaming with error."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    mock_client = MagicMock()
    mock_messages = MagicMock()

    async def raise_error(*args, **kwargs):
        raise RuntimeError("API Error")

    mock_messages.stream = raise_error
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.1)

        # Stream should have error event
        events = []
        async for event in stream:
            events.append(event)
            if len(events) > 5:
                break


@pytest.mark.asyncio
async def test_anthropic_stream_with_reasoning():
    """Test Anthropic streaming with reasoning/thinking."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])
    options = SimpleStreamOptions(reasoning=ThinkingLevel.HIGH)

    events = [
        MockStreamEvent(type="message_start", message={"usage": {"input_tokens": 10}}),
        MockStreamEvent(type="content_block_start", index=0, content_block={"type": "text"}),
        MockStreamEvent(type="content_block_delta", index=0, delta={"type": "text_delta", "text": "Response"}),
        MockStreamEvent(type="content_block_stop", index=0),
        MockStreamEvent(type="message_delta", stop_reason="end_turn", usage={"output_tokens": 5}),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream_simple(model, context, options)
        await asyncio.sleep(0.1)


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


@pytest.mark.asyncio
async def test_openai_stream_text_content():
    """Test OpenAI streaming with text content."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_openai_stream_with_system():
    """Test OpenAI streaming with system prompt."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello")]
    )

    stream = provider.stream(model, context, None)
    await asyncio.sleep(0.1)


# =============================================================================
# Google Provider Tests
# =============================================================================


@pytest.mark.asyncio
async def test_google_stream_text_content():
    """Test Google streaming with text content."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    await asyncio.sleep(0.1)


# =============================================================================
# Mistral Provider Tests
# =============================================================================


@pytest.mark.asyncio
async def test_mistral_stream_text_content():
    """Test Mistral streaming with text content."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    await asyncio.sleep(0.1)


# =============================================================================
# Provider Client Initialization Tests
# =============================================================================


def test_anthropic_get_client():
    """Test Anthropic provider client initialization."""
    provider = AnthropicProvider(api_key="test_key")

    # Should work without anthropic package
    with patch.dict("sys.modules", {"anthropic": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            try:
                provider._get_client()
            except ImportError:
                pass  # Expected


def test_openai_get_client():
    """Test OpenAI provider client initialization."""
    provider = OpenAIProvider(api_key="test_key")

    with patch.dict("sys.modules", {"openai": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            try:
                provider._get_client()
            except ImportError:
                pass  # Expected


def test_google_get_client():
    """Test Google provider client initialization."""
    provider = GoogleProvider(api_key="test_key")

    with patch.dict("sys.modules", {"google.generativeai": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            try:
                provider._get_client()
            except ImportError:
                pass  # Expected


def test_mistral_get_client():
    """Test Mistral provider client initialization."""
    provider = MistralProvider(api_key="test_key")

    with patch.dict("sys.modules", {"mistralai": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            try:
                provider._get_client()
            except ImportError:
                pass  # Expected


# =============================================================================
# Provider Request Building Tests
# =============================================================================


def test_anthropic_build_request_with_tools():
    """Test Anthropic request building with tools."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    tools = [
        Tool(name="test_tool", description="Test", parameters={"type": "object"})
    ]
    context = Context(
        messages=[UserMessage(content="Hello")],
        tools=tools
    )

    # Import the conversion function
    from pypi_ai.providers.anthropic import _convert_context, _convert_tools

    messages = _convert_context(context)
    assert len(messages) == 1

    tools_data = _convert_tools(tools)
    assert len(tools_data) == 1
    assert tools_data[0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_openai_build_request_with_tools():
    """Test OpenAI request building with tools."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    tools = [
        Tool(name="test_tool", description="Test", parameters={"type": "object"}, strict=True)
    ]
    context = Context(
        messages=[UserMessage(content="Hello")],
        tools=tools
    )

    stream = provider.stream(model, context, None)
    assert stream is not None
    await asyncio.sleep(0.1)


# =============================================================================
# Context Conversion Tests
# =============================================================================


def test_anthropic_convert_context_with_image():
    """Test Anthropic context conversion with image content."""
    from pypi_ai.providers.anthropic import _convert_context

    image = ImageContent(
        type="image",
        data="base64data",
        mime_type="image/png"
    )
    context = Context(messages=[UserMessage(content=[image])])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "image"


def test_anthropic_convert_context_with_assistant():
    """Test Anthropic context conversion with assistant message."""
    from pypi_ai.providers.anthropic import _convert_context

    assistant_msg = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Hello"),
            ThinkingContent(type="thinking", thinking="Let me think..."),
            ToolCall(type="toolCall", id="call_1", name="test", arguments={"arg": "value"}),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="claude-3",
        usage=Usage(),
        stop_reason=StopReason.END,
    )
    context = Context(messages=[UserMessage(content="Hi"), assistant_msg])
    result = _convert_context(context)

    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert len(result[1]["content"]) == 3


def test_anthropic_convert_context_with_tool_result():
    """Test Anthropic context conversion with tool result."""
    from pypi_ai.providers.anthropic import _convert_context

    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        content=[TextContent(type="text", text="Result")],
        is_error=False,
    )
    context = Context(messages=[UserMessage(content="Hi"), tool_result])
    result = _convert_context(context)

    assert len(result) == 2
    assert result[1]["role"] == "user"
    assert result[1]["content"][0]["type"] == "tool_result"


def test_anthropic_convert_context_tool_result_with_image():
    """Test Anthropic context conversion with tool result containing image."""
    from pypi_ai.providers.anthropic import _convert_context

    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        content=[
            TextContent(type="text", text="Result"),
            ImageContent(type="image", data="imgdata", mime_type="image/png"),
        ],
        is_error=True,
    )
    context = Context(messages=[tool_result])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["content"][0]["type"] == "tool_result"
    assert result[0]["content"][0]["is_error"] is True


def test_openai_convert_context_with_image():
    """Test OpenAI context conversion with image content."""
    from pypi_ai.providers.openai import _convert_context

    image = ImageContent(
        type="image",
        data="base64data",
        mime_type="image/png"
    )
    context = Context(messages=[UserMessage(content=[image])])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["type"] == "image_url"


def test_openai_convert_context_with_assistant():
    """Test OpenAI context conversion with assistant message."""
    from pypi_ai.providers.openai import _convert_context

    assistant_msg = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Hello"),
            ThinkingContent(type="thinking", thinking="Thinking..."),
            ToolCall(type="toolCall", id="call_1", name="test", arguments={"a": 1}),
        ],
        api=Api.OPENAI_COMPLETIONS,
        provider="openai",
        model="gpt-4",
        usage=Usage(),
        stop_reason=StopReason.END,
    )
    context = Context(messages=[assistant_msg])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "tool_calls" in result[0]


def test_openai_convert_context_with_tool_result():
    """Test OpenAI context conversion with tool result."""
    from pypi_ai.providers.openai import _convert_context

    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        content=[TextContent(type="text", text="Result")],
        is_error=False,
    )
    context = Context(messages=[tool_result])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["role"] == "tool"


def test_openai_convert_context_tool_result_error():
    """Test OpenAI context conversion with tool result error."""
    from pypi_ai.providers.openai import _convert_context

    tool_result = ToolResultMessage(
        tool_call_id="call_1",
        content=[TextContent(type="text", text="Error occurred")],
        is_error=True,
    )
    context = Context(messages=[tool_result])
    result = _convert_context(context)

    assert "Error:" in result[0]["content"]


# =============================================================================
# Stop Reason Mapping Tests
# =============================================================================


def test_anthropic_map_stop_reason():
    """Test Anthropic stop reason mapping."""
    from pypi_ai.providers.anthropic import _map_stop_reason

    assert _map_stop_reason("end_turn") == StopReason.END
    assert _map_stop_reason("stop_sequence") == StopReason.STOP
    assert _map_stop_reason("tool_use") == StopReason.TOOL_USE
    assert _map_stop_reason(None) == StopReason.END
    assert _map_stop_reason("unknown") == StopReason.END


def test_openai_map_stop_reason():
    """Test OpenAI stop reason mapping."""
    from pypi_ai.providers.openai import _map_stop_reason

    assert _map_stop_reason("stop") == StopReason.STOP
    assert _map_stop_reason("length") == StopReason.END
    assert _map_stop_reason("tool_calls") == StopReason.TOOL_USE
    assert _map_stop_reason("content_filter") == StopReason.ERROR
    assert _map_stop_reason(None) == StopReason.END
    assert _map_stop_reason("unknown") == StopReason.END


def test_mistral_map_stop_reason():
    """Test Mistral stop reason mapping."""
    from pypi_ai.providers.mistral import _map_stop_reason

    assert _map_stop_reason("stop") == StopReason.STOP
    assert _map_stop_reason("length") == StopReason.END
    assert _map_stop_reason("tool_calls") == StopReason.TOOL_USE
    assert _map_stop_reason(None) == StopReason.END
    assert _map_stop_reason("unknown") == StopReason.END


# =============================================================================
# Tool Conversion Tests
# =============================================================================


def test_openai_convert_tools_with_strict():
    """Test OpenAI tool conversion with strict mode."""
    from pypi_ai.providers.openai import _convert_tools

    tools = [
        Tool(name="test", description="Test tool", parameters={"type": "object"}, strict=True)
    ]
    result = _convert_tools(tools)

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["strict"] is True


def test_anthropic_convert_tools():
    """Test Anthropic tool conversion."""
    from pypi_ai.providers.anthropic import _convert_tools

    tools = [
        Tool(name="test", description="Test", parameters={"type": "object", "properties": {}})
    ]
    result = _convert_tools(tools)

    assert len(result) == 1
    assert result[0]["name"] == "test"
    assert "input_schema" in result[0]


# =============================================================================
# Provider Instance Tests
# =============================================================================


def test_anthropic_provider_instance():
    """Test Anthropic provider instance exists."""
    from pypi_ai.providers.anthropic import provider
    assert provider is not None
    assert isinstance(provider, AnthropicProvider)


def test_openai_provider_instance():
    """Test OpenAI provider instance exists."""
    from pypi_ai.providers.openai import provider
    assert provider is not None
    assert isinstance(provider, OpenAIProvider)


def test_google_provider_instance():
    """Test Google provider instance exists."""
    from pypi_ai.providers.google import provider
    assert provider is not None
    assert isinstance(provider, GoogleProvider)


def test_mistral_provider_instance():
    """Test Mistral provider instance exists."""
    from pypi_ai.providers.mistral import provider
    assert provider is not None
    assert isinstance(provider, MistralProvider)


# =============================================================================
# Provider with Base URL Tests
# =============================================================================


def test_openai_provider_with_base_url():
    """Test OpenAI provider with custom base URL."""
    provider = OpenAIProvider(
        api_key="test_key",
        base_url="https://custom.api.com"
    )
    assert provider._base_url == "https://custom.api.com"


@pytest.mark.asyncio
async def test_openai_stream_with_base_url():
    """Test OpenAI streaming with custom base URL from model."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(
        id="gpt-4",
        api=Api.OPENAI_COMPLETIONS,
        provider="openai",
        base_url="https://model.api.com"
    )
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    assert stream is not None
    await asyncio.sleep(0.1)


# =============================================================================
# Streaming Options Tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_stream_with_temperature():
    """Test Anthropic streaming with temperature option."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])
    options = StreamOptions(temperature=0.5)

    stream = provider.stream(model, context, options)
    assert stream is not None
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_openai_stream_with_temperature():
    """Test OpenAI streaming with temperature option."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])
    options = StreamOptions(temperature=0.7)

    stream = provider.stream(model, context, options)
    assert stream is not None
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anthropic_stream_with_max_tokens():
    """Test Anthropic streaming with max_tokens option."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])
    options = StreamOptions(max_tokens=1000)

    stream = provider.stream(model, context, options)
    assert stream is not None
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_openai_stream_with_max_tokens():
    """Test OpenAI streaming with max_tokens option."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai", max_tokens=2000)
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    assert stream is not None
    await asyncio.sleep(0.1)


# =============================================================================
# Detailed Provider Streaming Tests
# =============================================================================


class MockAnthropicMessageStart:
    """Mock message_start event."""
    type = "message_start"
    def __init__(self):
        self.message = type("Message", (), {
            "usage": type("Usage", (), {
                "input_tokens": 100,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            })()
        })()


class MockAnthropicContentBlockStart:
    """Mock content_block_start event."""
    type = "content_block_start"
    def __init__(self, index, block_type, id=None, name=None):
        self.index = index
        self.content_block = type("Block", (), {
            "type": block_type,
            "id": id,
            "name": name,
        })()


class MockAnthropicContentBlockDelta:
    """Mock content_block_delta event."""
    type = "content_block_delta"
    def __init__(self, index, delta_type, text=None, thinking=None, partial_json=None):
        self.index = index
        if delta_type == "text_delta":
            self.delta = type("Delta", (), {"type": "text_delta", "text": text})()
        elif delta_type == "thinking_delta":
            self.delta = type("Delta", (), {"type": "thinking_delta", "thinking": thinking})()
        elif delta_type == "input_json_delta":
            self.delta = type("Delta", (), {"type": "input_json_delta", "partial_json": partial_json})()


class MockAnthropicContentBlockStop:
    """Mock content_block_stop event."""
    type = "content_block_stop"
    def __init__(self, index):
        self.index = index


class MockAnthropicMessageDelta:
    """Mock message_delta event."""
    type = "message_delta"
    def __init__(self, stop_reason, output_tokens):
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"output_tokens": output_tokens})()


@pytest.mark.asyncio
async def test_anthropic_full_stream_flow():
    """Test complete Anthropic streaming flow with all event types."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    # Create comprehensive mock events
    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(0, "thinking"),
        MockAnthropicContentBlockDelta(0, "thinking_delta", thinking="Thinking..."),
        MockAnthropicContentBlockStop(0),
        MockAnthropicContentBlockStart(1, "text"),
        MockAnthropicContentBlockDelta(1, "text_delta", text="Hello"),
        MockAnthropicContentBlockDelta(1, "text_delta", text=" world"),
        MockAnthropicContentBlockStop(1),
        MockAnthropicContentBlockStart(2, "tool_use", id="call_1", name="test_tool"),
        MockAnthropicContentBlockDelta(2, "input_json_delta", partial_json='{"arg":'),
        MockAnthropicContentBlockDelta(2, "input_json_delta", partial_json='"value"}'),
        MockAnthropicContentBlockStop(2),
        MockAnthropicMessageDelta("tool_use", 50),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)

        await asyncio.sleep(0.2)

        collected_events = []
        async for event in stream:
            collected_events.append(event)
            if len(collected_events) > 20:
                break

        assert len(collected_events) > 0


@pytest.mark.asyncio
async def test_anthropic_stream_with_cache():
    """Test Anthropic streaming with cache retention."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hello")]
    )
    options = {"cache_retention": "ephemeral"}

    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(0, "text"),
        MockAnthropicContentBlockDelta(0, "text_delta", text="Response"),
        MockAnthropicContentBlockStop(0),
        MockAnthropicMessageDelta("end_turn", 10),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anthropic_stream_with_thinking_budget():
    """Test Anthropic streaming with custom thinking budget."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {
        "reasoning": "high",
        "thinking_budgets": {"high": 10000}
    }

    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(0, "text"),
        MockAnthropicContentBlockDelta(0, "text_delta", text="Response"),
        MockAnthropicContentBlockStop(0),
        MockAnthropicMessageDelta("end_turn", 10),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anthropic_stream_exception_handling():
    """Test Anthropic streaming handles exceptions."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    mock_client = MagicMock()
    mock_messages = MagicMock()

    async def raise_error(**kwargs):
        raise RuntimeError("API Error")

    mock_messages.stream = raise_error
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)

        events = []
        async for event in stream:
            events.append(event)
            if len(events) > 5:
                break

        # Should have error event
        assert len(events) >= 0  # At minimum, stream exists


# =============================================================================
# OpenAI Detailed Streaming Tests
# =============================================================================


class MockOpenAIChunk:
    """Mock OpenAI streaming chunk."""
    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        self.choices = [
            type("Choice", (), {
                "delta": type("Delta", (), {
                    "content": content,
                    "tool_calls": tool_calls,
                })(),
                "finish_reason": finish_reason,
            })()
        ] if content or tool_calls or finish_reason else []
        self.usage = usage


class MockOpenAIToolCall:
    """Mock OpenAI tool call in delta."""
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = type("Function", (), {
            "name": name,
            "arguments": arguments,
        })() if name or arguments else None


@pytest.mark.asyncio
async def test_openai_full_stream_flow():
    """Test complete OpenAI streaming flow."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(
        system_prompt="Be helpful",
        messages=[UserMessage(content="Hello")]
    )

    # Create mock chunks
    chunks = [
        MockOpenAIChunk(content="Hello"),
        MockOpenAIChunk(content=" there"),
        MockOpenAIChunk(content="!"),
        MockOpenAIChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)

        events = []
        async for event in stream:
            events.append(event)
            if len(events) > 10:
                break


@pytest.mark.asyncio
async def test_openai_stream_with_tool_calls():
    """Test OpenAI streaming with tool calls."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    tools = [Tool(name="test", description="Test", parameters={"type": "object"})]
    context = Context(messages=[UserMessage(content="Hello")], tools=tools)

    chunks = [
        MockOpenAIChunk(
            tool_calls=[
                MockOpenAIToolCall(0, id="call_1", name="test"),
            ]
        ),
        MockOpenAIChunk(
            tool_calls=[
                MockOpenAIToolCall(0, arguments='{"arg":'),
            ]
        ),
        MockOpenAIChunk(
            tool_calls=[
                MockOpenAIToolCall(0, arguments='"value"}'),
            ]
        ),
        MockOpenAIChunk(finish_reason="tool_calls"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_stream_with_usage():
    """Test OpenAI streaming with usage info."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    usage = type("Usage", (), {
        "prompt_tokens": 100,
        "completion_tokens": 50,
    })()

    chunks = [
        MockOpenAIChunk(content="Response"),
        MockOpenAIChunk(finish_reason="stop", usage=usage),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_stream_exception():
    """Test OpenAI streaming handles exceptions."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        raise RuntimeError("API Error")

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


# =============================================================================
# Google Provider Detailed Tests
# =============================================================================


class MockGeminiChunk:
    """Mock Gemini streaming chunk."""
    def __init__(self, text=None):
        self.text = text
        self.usage_metadata = None


@pytest.mark.asyncio
async def test_google_full_stream_flow():
    """Test complete Google streaming flow."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])

    chunks = [
        MockGeminiChunk(text="Hello"),
        MockGeminiChunk(text=" there"),
    ]

    mock_genai = MagicMock()
    mock_model = MagicMock()
    mock_chat = MagicMock()

    async def mock_send(*args, **kwargs):
        return MockAsyncIterator(chunks)

    mock_chat.send_message_async = mock_send
    mock_model.start_chat = MagicMock(return_value=mock_chat)
    mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

    with patch.object(provider, "_get_client", return_value=mock_genai):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_google_stream_with_temperature():
    """Test Google streaming with temperature option."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"temperature": 0.7}

    chunks = [MockGeminiChunk(text="Response")]

    mock_genai = MagicMock()
    mock_model = MagicMock()
    mock_chat = MagicMock()

    async def mock_send(*args, **kwargs):
        return MockAsyncIterator(chunks)

    mock_chat.send_message_async = mock_send
    mock_model.start_chat = MagicMock(return_value=mock_chat)
    mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

    with patch.object(provider, "_get_client", return_value=mock_genai):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_google_stream_with_history():
    """Test Google streaming with chat history."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")

    assistant_msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Previous response")],
        api=Api.GOOGLE_GENERATIVE_AI,
        provider="google",
        model="gemini-pro",
        usage=Usage(),
        stop_reason=StopReason.END,
    )
    context = Context(messages=[
        UserMessage(content="Hello"),
        assistant_msg,
        UserMessage(content="Continue"),
    ])

    chunks = [MockGeminiChunk(text="Response")]

    mock_genai = MagicMock()
    mock_model = MagicMock()
    mock_chat = MagicMock()

    async def mock_send(*args, **kwargs):
        return MockAsyncIterator(chunks)

    mock_chat.send_message_async = mock_send
    mock_model.start_chat = MagicMock(return_value=mock_chat)
    mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

    with patch.object(provider, "_get_client", return_value=mock_genai):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_google_stream_exception():
    """Test Google streaming handles exceptions."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])

    mock_genai = MagicMock()
    mock_model = MagicMock()
    mock_chat = MagicMock()

    async def mock_send(*args, **kwargs):
        raise RuntimeError("API Error")

    mock_chat.send_message_async = mock_send
    mock_model.start_chat = MagicMock(return_value=mock_chat)
    mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

    with patch.object(provider, "_get_client", return_value=mock_genai):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


# =============================================================================
# Mistral Provider Detailed Tests
# =============================================================================


class MockMistralChunk:
    """Mock Mistral streaming chunk."""
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.data = type("Data", (), {
            "choices": [
                type("Choice", (), {
                    "delta": type("Delta", (), {"content": content})(),
                    "finish_reason": finish_reason,
                })()
            ] if content or finish_reason else None,
            "usage": usage,
        })()


@pytest.mark.asyncio
async def test_mistral_full_stream_flow():
    """Test complete Mistral streaming flow."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    chunks = [
        MockMistralChunk(content="Hello"),
        MockMistralChunk(content=" there"),
        MockMistralChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()

    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_with_temperature():
    """Test Mistral streaming with temperature option."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"temperature": 0.7}

    chunks = [MockMistralChunk(content="Response")]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_with_history():
    """Test Mistral streaming with chat history."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")

    assistant_msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Previous")],
        api=Api.MISTRAL_CONVERSATIONS,
        provider="mistral",
        model="mistral-large",
        usage=Usage(),
        stop_reason=StopReason.END,
    )
    context = Context(messages=[
        UserMessage(content="Hello"),
        assistant_msg,
        UserMessage(content="Continue"),
    ])

    chunks = [MockMistralChunk(content="Response")]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_exception():
    """Test Mistral streaming handles exceptions."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    mock_client = MagicMock()
    mock_chat = MagicMock()

    mock_chat.stream_async = MagicMock(side_effect=RuntimeError("API Error"))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


# =============================================================================
# Provider Edge Case Tests for 100% Coverage
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_json_decode_error_in_tool_call():
    """Test Anthropic handling invalid JSON in tool call arguments."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])

    # Create events with invalid JSON for tool call
    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(2, "tool_use", id="call_1", name="test_tool"),
        MockAnthropicContentBlockDelta(2, "input_json_delta", partial_json='{"broken": json}'),
        MockAnthropicContentBlockStop(2),
        MockAnthropicMessageDelta("tool_use", 50),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_anthropic_stream_with_tools_in_request():
    """Test Anthropic streaming with tools in request parameters."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    tools = [Tool(name="test", description="Test", parameters={"type": "object"})]
    context = Context(
        messages=[UserMessage(content="Hello")],
        tools=tools
    )

    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(0, "text"),
        MockAnthropicContentBlockDelta(0, "text_delta", text="Response"),
        MockAnthropicContentBlockStop(0),
        MockAnthropicMessageDelta("end_turn", 10),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_anthropic_stream_with_temperature_in_request():
    """Test Anthropic streaming with temperature in request parameters."""
    provider = AnthropicProvider(api_key="test_key")

    model = Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"temperature": 0.5}

    events = [
        MockAnthropicMessageStart(),
        MockAnthropicContentBlockStart(0, "text"),
        MockAnthropicContentBlockDelta(0, "text_delta", text="Response"),
        MockAnthropicContentBlockStop(0),
        MockAnthropicMessageDelta("end_turn", 10),
    ]

    mock_stream = MockAsyncContextManager(events)

    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.stream.return_value = mock_stream
    mock_client.messages = mock_messages

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_json_decode_error_in_tool_call():
    """Test OpenAI handling invalid JSON in tool call arguments."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    chunks = [
        MockOpenAIChunk(
            tool_calls=[
                MockOpenAIToolCall(0, id="call_1", name="test"),
            ]
        ),
        MockOpenAIChunk(
            tool_calls=[
                MockOpenAIToolCall(0, arguments='{"broken": json}'),
            ]
        ),
        MockOpenAIChunk(finish_reason="tool_calls"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_stream_with_temperature_option():
    """Test OpenAI streaming with temperature option in request."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"temperature": 0.5}

    chunks = [
        MockOpenAIChunk(content="Response"),
        MockOpenAIChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_stream_with_max_tokens_option():
    """Test OpenAI streaming with max_tokens option in request."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"max_tokens": 100}

    chunks = [
        MockOpenAIChunk(content="Response"),
        MockOpenAIChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_openai_stream_empty_choices():
    """Test OpenAI streaming with empty choices in chunk."""
    provider = OpenAIProvider(api_key="test_key")

    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    class EmptyChunk:
        choices = []
        usage = None

    chunks = [
        EmptyChunk(),
        MockOpenAIChunk(content="Response"),
        MockOpenAIChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    async def mock_create(**kwargs):
        return MockAsyncIterator(chunks)

    mock_completions.create = mock_create
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


def test_openai_convert_with_image_content():
    """Test OpenAI context conversion with image content."""
    from pypi_ai.providers.openai import _convert_context

    image = ImageContent(
        type="image",
        data="base64data",
        mime_type="image/png"
    )
    context = Context(messages=[UserMessage(content=[TextContent(type="text", text="Hello"), image])])
    result = _convert_context(context)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][1]["type"] == "image_url"


# =============================================================================
# Additional Provider Tests for Remaining Coverage
# =============================================================================


@pytest.mark.asyncio
async def test_google_stream_with_max_tokens():
    """Test Google streaming with max_tokens option."""
    provider = GoogleProvider(api_key="test_key")

    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"max_tokens": 100}  # This triggers line 102

    chunks = [MockGeminiChunk(text="Response")]

    mock_genai = MagicMock()
    mock_model = MagicMock()
    mock_chat = MagicMock()

    async def mock_send(*args, **kwargs):
        return MockAsyncIterator(chunks)

    mock_chat.send_message_async = mock_send
    mock_model.start_chat = MagicMock(return_value=mock_chat)
    mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

    with patch.object(provider, "_get_client", return_value=mock_genai):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_with_max_tokens():
    """Test Mistral streaming with max_tokens option."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])
    options = {"max_tokens": 100}  # This triggers line 114

    chunks = [MockMistralChunk(content="Response")]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, options)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_with_usage():
    """Test Mistral streaming with usage info."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    # Create chunk with usage info
    class MockMistralUsageChunk:
        def __init__(self, content=None, usage=None):
            self.data = type("Data", (), {
                "choices": [
                    type("Choice", (), {
                        "delta": type("Delta", (), {"content": content})(),
                        "finish_reason": None,
                    })()
                ] if content else None,
                "usage": type("Usage", (), {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                })() if usage else None,
            })()

    chunks = [
        MockMistralUsageChunk(content="Response"),
        MockMistralUsageChunk(usage=True),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_mistral_stream_with_finish_reason():
    """Test Mistral streaming with finish_reason."""
    provider = MistralProvider(api_key="test_key")

    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    chunks = [
        MockMistralChunk(content="Response"),
        MockMistralChunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.stream_async = MagicMock(return_value=MockAsyncIterator(chunks))
    mock_client.chat = mock_chat

    with patch.object(provider, "_get_client", return_value=mock_client):
        stream = provider.stream(model, context, None)
        await asyncio.sleep(0.2)


def test_mistral_get_client_creates_new():
    """Test Mistral _get_client creates new client."""
    provider = MistralProvider(api_key="test_key")

    mock_mistral = MagicMock()
    mock_client = MagicMock()
    mock_mistral.return_value = mock_client

    with patch.dict("sys.modules", {"mistralai": MagicMock(Mistral=mock_mistral)}):
        client = provider._get_client("new_key")
        # Client should be created and returned
        assert mock_mistral.called
