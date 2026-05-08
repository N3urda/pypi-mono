"""Tests for stream module with full coverage."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from pypi_ai.stream import (
    stream,
    stream_simple,
    complete,
    complete_simple,
    get_model,
    _infer_api_type,
)
from pypi_ai.types import (
    Api,
    Model,
    Context,
    UserMessage,
    AssistantMessage,
    TextContent,
    StopReason,
    Usage,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevel,
)
from pypi_ai.event_stream import AssistantMessageEventStream, DoneEvent


@pytest.fixture
def model():
    return Model(id="claude-3-5-sonnet", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")


@pytest.fixture
def context():
    return Context(messages=[UserMessage(content="Hello")])


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


def test_infer_api_type_anthropic():
    """Test inferring API type for Anthropic."""
    assert _infer_api_type("anthropic") == Api.ANTHROPIC_MESSAGES


def test_infer_api_type_openai():
    """Test inferring API type for OpenAI."""
    assert _infer_api_type("openai") == Api.OPENAI_COMPLETIONS


def test_infer_api_type_google():
    """Test inferring API type for Google."""
    assert _infer_api_type("google") == Api.GOOGLE_GENERATIVE_AI


def test_infer_api_type_mistral():
    """Test inferring API type for Mistral."""
    assert _infer_api_type("mistral") == Api.MISTRAL_CONVERSATIONS


def test_infer_api_type_bedrock():
    """Test inferring API type for Bedrock."""
    assert _infer_api_type("amazon-bedrock") == Api.BEDROCK_CONVERSE_STREAM
    assert _infer_api_type("bedrock") == Api.BEDROCK_CONVERSE_STREAM


def test_infer_api_type_deepseek():
    """Test inferring API type for DeepSeek."""
    assert _infer_api_type("deepseek") == Api.OPENAI_COMPLETIONS


def test_infer_api_type_groq():
    """Test inferring API type for Groq."""
    assert _infer_api_type("groq") == Api.OPENAI_COMPLETIONS


def test_infer_api_type_cerebras():
    """Test inferring API type for Cerebras."""
    assert _infer_api_type("cerebras") == Api.OPENAI_COMPLETIONS


def test_infer_api_type_xai():
    """Test inferring API type for XAI."""
    assert _infer_api_type("xai") == Api.OPENAI_COMPLETIONS


def test_infer_api_type_unknown():
    """Test inferring API type for unknown provider."""
    assert _infer_api_type("unknown") == Api.OPENAI_COMPLETIONS


def test_get_model_anthropic():
    """Test getting Anthropic model."""
    model = get_model("anthropic", "claude-3-5-sonnet")
    assert model.api == Api.ANTHROPIC_MESSAGES
    assert model.id == "claude-3-5-sonnet"


def test_get_model_openai():
    """Test getting OpenAI model."""
    model = get_model("openai", "gpt-4")
    assert model.api == Api.OPENAI_COMPLETIONS
    assert model.id == "gpt-4"


def test_get_model_with_api():
    """Test getting model with explicit API type."""
    model = get_model("custom", "model-id", api=Api.AZURE_OPENAI_RESPONSES)
    assert model.api == Api.AZURE_OPENAI_RESPONSES


def test_get_model_with_base_url():
    """Test getting model with base URL."""
    model = get_model("custom", "model-id", base_url="https://api.custom.com")
    assert model.base_url == "https://api.custom.com"


def test_get_model_with_kwargs():
    """Test getting model with additional kwargs."""
    model = get_model("anthropic", "claude-3-5-sonnet", context_window=200000)
    assert model.context_window == 200000


def test_stream_returns_stream(model, context):
    """Test stream returns event stream."""
    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream.return_value = AssistantMessageEventStream()
        mock_resolve.return_value = mock_provider

        result = stream(model, context, None)
        assert isinstance(result, AssistantMessageEventStream)


def test_stream_simple_returns_stream(model, context):
    """Test stream_simple returns event stream."""
    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream_simple.return_value = AssistantMessageEventStream()
        mock_resolve.return_value = mock_provider

        result = stream_simple(model, context, None)
        assert isinstance(result, AssistantMessageEventStream)


@pytest.mark.asyncio
async def test_complete_returns_message(model, context, mock_assistant):
    """Test complete returns message."""
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=StopReason.END, message=mock_assistant))
    stream.end(mock_assistant)

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream.return_value = stream
        mock_resolve.return_value = mock_provider

        result = await complete(model, context, None)
        assert isinstance(result, AssistantMessage)
        assert result.role == "assistant"


@pytest.mark.asyncio
async def test_complete_simple_returns_message(model, context, mock_assistant):
    """Test complete_simple returns message."""
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=StopReason.END, message=mock_assistant))
    stream.end(mock_assistant)

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream_simple.return_value = stream
        mock_resolve.return_value = mock_provider

        result = await complete_simple(model, context, None)
        assert isinstance(result, AssistantMessage)


@pytest.mark.asyncio
async def test_complete_no_message_raises():
    """Test complete raises when no message produced."""
    test_model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    test_context = Context(messages=[UserMessage(content="Hello")])
    stream = AssistantMessageEventStream()
    stream.end(None)

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream.return_value = stream
        mock_resolve.return_value = mock_provider

        with pytest.raises(RuntimeError, match="without producing"):
            await complete(test_model, test_context, None)


@pytest.mark.asyncio
async def test_complete_simple_no_message_raises():
    """Test complete_simple raises when no message produced."""
    test_model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="test")
    test_context = Context(messages=[UserMessage(content="Hello")])
    stream = AssistantMessageEventStream()
    stream.end(None)

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream_simple.return_value = stream
        mock_resolve.return_value = mock_provider

        with pytest.raises(RuntimeError, match="without producing"):
            await complete_simple(test_model, test_context, None)


def test_stream_with_options(model, context):
    """Test stream with options."""
    options = StreamOptions(
        temperature=0.7,
        max_tokens=1024,
    )

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream.return_value = AssistantMessageEventStream()
        mock_resolve.return_value = mock_provider

        result = stream(model, context, options)
        assert isinstance(result, AssistantMessageEventStream)


def test_stream_simple_with_options(model, context):
    """Test stream_simple with options."""
    options = SimpleStreamOptions(
        reasoning=ThinkingLevel.HIGH,
    )

    with patch("pypi_ai.stream.resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.stream_simple.return_value = AssistantMessageEventStream()
        mock_resolve.return_value = mock_provider

        result = stream_simple(model, context, options)
        assert isinstance(result, AssistantMessageEventStream)
