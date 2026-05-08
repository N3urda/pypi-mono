"""Tests for stream functions."""

import pytest

from pypi_ai.stream import stream, stream_simple, complete, complete_simple, get_model, _infer_api_type
from pypi_ai.types import Api, Context, Model, UserMessage
from pypi_ai.registry import register_provider, clear_providers


class MockProvider:
    """Mock provider for testing."""

    api = Api.ANTHROPIC_MESSAGES

    def stream(self, model, context, options=None):
        from pypi_ai.event_stream import AssistantMessageEventStream
        return AssistantMessageEventStream()

    def stream_simple(self, model, context, options=None):
        from pypi_ai.event_stream import AssistantMessageEventStream
        return AssistantMessageEventStream()


@pytest.fixture(autouse=True)
def setup_provider():
    """Register mock provider for tests."""
    clear_providers()
    register_provider(MockProvider())
    yield
    clear_providers()


def test_get_model():
    """Test get_model function."""
    model = get_model("anthropic", "claude-test")
    assert model.provider == "anthropic"
    assert model.id == "claude-test"


def test_get_model_with_options():
    """Test get_model with options."""
    model = get_model("anthropic", "claude-test", base_url="https://custom.url")
    assert model.base_url == "https://custom.url"


def test_infer_api_type():
    """Test API type inference."""
    assert _infer_api_type("anthropic") == Api.ANTHROPIC_MESSAGES
    assert _infer_api_type("openai") == Api.OPENAI_COMPLETIONS
    assert _infer_api_type("google") == Api.GOOGLE_GENERATIVE_AI
    assert _infer_api_type("mistral") == Api.MISTRAL_CONVERSATIONS
    assert _infer_api_type("unknown") == Api.OPENAI_COMPLETIONS


def test_stream_returns_event_stream():
    """Test stream returns event stream."""
    model = get_model("anthropic", "test")
    context = Context(messages=[UserMessage(content="Hello")])

    result = stream(model, context)
    assert result is not None


def test_stream_simple_returns_event_stream():
    """Test stream_simple returns event stream."""
    model = get_model("anthropic", "test")
    context = Context(messages=[UserMessage(content="Hello")])

    result = stream_simple(model, context)
    assert result is not None


@pytest.mark.asyncio
async def test_complete_returns_message():
    """Test complete returns assistant message."""
    from pypi_ai.event_stream import AssistantMessageEventStream, DoneEvent
    from pypi_ai.types import AssistantMessage, StopReason, Usage

    # Create a mock provider that returns immediately
    class CompletingProvider:
        api = Api.ANTHROPIC_MESSAGES

        def stream(self, model, context, options=None):
            s = AssistantMessageEventStream()
            msg = AssistantMessage(
                role="assistant",
                content=[],
                api=Api.ANTHROPIC_MESSAGES,
                provider="test",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            s.push(DoneEvent(reason=StopReason.END, message=msg))
            s.end(msg)
            return s

        def stream_simple(self, model, context, options=None):
            return self.stream(model, context, options)

    clear_providers()
    register_provider(CompletingProvider())

    model = get_model("anthropic", "test")
    context = Context(messages=[UserMessage(content="Hello")])

    result = await complete(model, context)
    assert result.role == "assistant"


def test_stream_raises_for_missing_provider():
    """Test stream raises when no provider is registered."""
    clear_providers()

    # Use a valid API but ensure no provider is registered for it
    model = Model(
        id="test",
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
    )
    context = Context(messages=[])

    with pytest.raises(ValueError, match="No provider registered"):
        stream(model, context)