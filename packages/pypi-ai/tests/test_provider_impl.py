"""Tests for provider implementations."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from pypi_ai.types import (
    Api,
    Model,
    Context,
    UserMessage,
    AssistantMessage,
    TextContent,
    StopReason,
    Usage,
    Tool,
)
from pypi_ai.providers.anthropic import (
    AnthropicProvider,
    _convert_context,
    _convert_tools,
    _create_assistant_message,
    _map_stop_reason,
    _get_thinking_budget,
    provider as anthropic_provider,
)
from pypi_ai.providers.openai import OpenAIProvider
from pypi_ai.providers.google import GoogleProvider
from pypi_ai.providers.mistral import MistralProvider


@pytest.fixture
def model():
    return Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")


@pytest.fixture
def context():
    return Context(
        system_prompt="You are helpful.",
        messages=[UserMessage(content="Hello")]
    )


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


def test_anthropic_provider_api():
    """Test Anthropic provider API property."""
    provider = AnthropicProvider()
    assert provider.api == Api.ANTHROPIC_MESSAGES


def test_anthropic_provider_instance():
    """Test Anthropic provider instance."""
    assert anthropic_provider is not None
    assert isinstance(anthropic_provider, AnthropicProvider)


def test_convert_context_user_message():
    """Test context conversion with user message."""
    context = Context(messages=[UserMessage(content="Hello")])
    result = _convert_context(context)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_convert_context_with_string_content():
    """Test context conversion with string content."""
    context = Context(messages=[UserMessage(content="Hello world")])
    result = _convert_context(context)
    assert result[0]["content"] == "Hello world"


def test_convert_context_with_content_list():
    """Test context conversion with content list."""
    content = [TextContent(type="text", text="Hello")]
    context = Context(messages=[UserMessage(content=content)])
    result = _convert_context(context)
    assert len(result) == 1
    assert "content" in result[0]
    assert isinstance(result[0]["content"], list)


def test_convert_tools():
    """Test tool conversion."""
    tools = [
        Tool(name="test", description="Test tool", parameters={"type": "object"})
    ]
    result = _convert_tools(tools)
    assert len(result) == 1
    assert result[0]["name"] == "test"
    assert "input_schema" in result[0]


def test_create_assistant_message(model):
    """Test assistant message creation."""
    content = [TextContent(type="text", text="Response")]
    usage = Usage(input=100, output=50)

    msg = _create_assistant_message(model, content, usage, StopReason.END)

    assert msg.role == "assistant"
    assert len(msg.content) == 1
    assert msg.api == Api.ANTHROPIC_MESSAGES


def test_create_assistant_message_with_error(model):
    """Test assistant message creation with error."""
    content = []
    usage = Usage()

    msg = _create_assistant_message(
        model, content, usage, StopReason.ERROR, error_message="Test error"
    )

    assert msg.stop_reason == StopReason.ERROR
    assert msg.error_message == "Test error"


def test_map_stop_reason():
    """Test stop reason mapping."""
    assert _map_stop_reason("end_turn") == StopReason.END
    assert _map_stop_reason("stop_sequence") == StopReason.STOP
    assert _map_stop_reason("tool_use") == StopReason.TOOL_USE
    assert _map_stop_reason(None) == StopReason.END
    assert _map_stop_reason("unknown") == StopReason.END


def test_get_thinking_budget_default():
    """Test default thinking budget values."""
    assert _get_thinking_budget("minimal", {}) == 1024
    assert _get_thinking_budget("low", {}) == 2048
    assert _get_thinking_budget("medium", {}) == 4096
    assert _get_thinking_budget("high", {}) == 8192
    assert _get_thinking_budget("xhigh", {}) == 16000


def test_get_thinking_budget_custom():
    """Test custom thinking budget values."""
    options = {"thinking_budgets": {"minimal": 500, "high": 10000}}
    assert _get_thinking_budget("minimal", options) == 500
    assert _get_thinking_budget("high", options) == 10000


@pytest.mark.asyncio
async def test_anthropic_stream_returns_stream(model, context):
    """Test Anthropic stream returns event stream."""
    provider = AnthropicProvider(api_key="test_key")
    # Mock the client to avoid actual API calls
    with patch.object(provider, "_get_client") as mock_client:
        mock_async_client = MagicMock()
        mock_client.return_value = mock_async_client

        stream = provider.stream(model, context, None)
        assert stream is not None
        assert hasattr(stream, '__aiter__')


@pytest.mark.asyncio
async def test_anthropic_stream_simple_returns_stream(model, context):
    """Test Anthropic stream_simple returns event stream."""
    provider = AnthropicProvider(api_key="test_key")
    with patch.object(provider, "_get_client") as mock_client:
        mock_async_client = MagicMock()
        mock_client.return_value = mock_async_client

        stream = provider.stream_simple(model, context, None)
        assert stream is not None
        assert hasattr(stream, '__aiter__')


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


def test_openai_provider_api():
    """Test OpenAI provider API property."""
    provider = OpenAIProvider()
    assert provider.api == Api.OPENAI_COMPLETIONS


def test_openai_provider_with_api_key():
    """Test OpenAI provider with API key."""
    provider = OpenAIProvider(api_key="test_key")
    assert provider._api_key == "test_key"


def test_openai_provider_with_base_url():
    """Test OpenAI provider with base URL."""
    provider = OpenAIProvider(base_url="https://custom.openai.com")
    assert provider._base_url == "https://custom.openai.com"


@pytest.mark.asyncio
async def test_openai_stream_returns_stream():
    """Test OpenAI stream returns event stream."""
    provider = OpenAIProvider(api_key="test_key")
    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


@pytest.mark.asyncio
async def test_openai_stream_simple_returns_stream():
    """Test OpenAI stream_simple returns event stream."""
    provider = OpenAIProvider(api_key="test_key")
    model = Model(id="gpt-4", api=Api.OPENAI_COMPLETIONS, provider="openai")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream_simple(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


# =============================================================================
# Google Provider Tests
# =============================================================================


def test_google_provider_api():
    """Test Google provider API property."""
    provider = GoogleProvider()
    assert provider.api == Api.GOOGLE_GENERATIVE_AI


def test_google_provider_with_api_key():
    """Test Google provider with API key."""
    provider = GoogleProvider(api_key="test_key")
    assert provider._api_key == "test_key"


@pytest.mark.asyncio
async def test_google_stream_returns_stream():
    """Test Google stream returns event stream."""
    provider = GoogleProvider(api_key="test_key")
    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


@pytest.mark.asyncio
async def test_google_stream_simple_returns_stream():
    """Test Google stream_simple returns event stream."""
    provider = GoogleProvider(api_key="test_key")
    model = Model(id="gemini-pro", api=Api.GOOGLE_GENERATIVE_AI, provider="google")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream_simple(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


# =============================================================================
# Mistral Provider Tests
# =============================================================================


def test_mistral_provider_api():
    """Test Mistral provider API property."""
    provider = MistralProvider()
    assert provider.api == Api.MISTRAL_CONVERSATIONS


def test_mistral_provider_with_api_key():
    """Test Mistral provider with API key."""
    provider = MistralProvider(api_key="test_key")
    assert provider._api_key == "test_key"


@pytest.mark.asyncio
async def test_mistral_stream_returns_stream():
    """Test Mistral stream returns event stream."""
    provider = MistralProvider(api_key="test_key")
    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


@pytest.mark.asyncio
async def test_mistral_stream_simple_returns_stream():
    """Test Mistral stream_simple returns event stream."""
    provider = MistralProvider(api_key="test_key")
    model = Model(id="mistral-large", api=Api.MISTRAL_CONVERSATIONS, provider="mistral")
    context = Context(messages=[UserMessage(content="Hello")])

    stream = provider.stream_simple(model, context, None)
    assert stream is not None
    assert hasattr(stream, '__aiter__')


# =============================================================================
# Provider Initialization Tests
# =============================================================================


def test_anthropic_provider_init_without_api_key():
    """Test Anthropic provider initialization without API key."""
    provider = AnthropicProvider()
    assert provider._api_key is None


def test_openai_provider_init_without_api_key():
    """Test OpenAI provider initialization without API key."""
    provider = OpenAIProvider()
    assert provider._api_key is None


def test_google_provider_init_without_api_key():
    """Test Google provider initialization without API key."""
    provider = GoogleProvider()
    assert provider._api_key is None


def test_mistral_provider_init_without_api_key():
    """Test Mistral provider initialization without API key."""
    provider = MistralProvider()
    assert provider._api_key is None
