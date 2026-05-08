"""
Public API for streaming LLM responses.

This module provides:
- stream() for full-featured streaming
- stream_simple() for simplified interface
- complete() for non-streaming responses
- complete_simple() for simplified non-streaming
"""

from typing import Optional, Any

from pypi_ai.types import (
    Api,
    Context,
    Model,
    StreamOptions,
    SimpleStreamOptions,
    AssistantMessage,
)
from pypi_ai.event_stream import AssistantMessageEventStream
from pypi_ai.registry import resolve_provider


def stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    """
    Stream a response from an LLM provider.

    Args:
        model: The model configuration.
        context: The context with messages and tools.
        options: Optional streaming options.

    Returns:
        An event stream of AssistantMessageEvent events.
    """
    provider = resolve_provider(model.api)
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessage:
    """
    Complete a request and return the full response.

    Args:
        model: The model configuration.
        context: The context with messages and tools.
        options: Optional streaming options.

    Returns:
        The complete assistant message.
    """
    event_stream = stream(model, context, options)
    return await event_stream.result()


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    """
    Stream a response with simplified options.

    This function adds reasoning/thinking support through the
    SimpleStreamOptions interface.

    Args:
        model: The model configuration.
        context: The context with messages and tools.
        options: Optional simplified streaming options.

    Returns:
        An event stream of AssistantMessageEvent events.
    """
    provider = resolve_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessage:
    """
    Complete a request with simplified options.

    Args:
        model: The model configuration.
        context: The context with messages and tools.
        options: Optional simplified streaming options.

    Returns:
        The complete assistant message.
    """
    event_stream = stream_simple(model, context, options)
    return await event_stream.result()


def get_model(
    provider: str,
    model_id: str,
    api: Optional[Api] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Model:
    """
    Get a model configuration.

    Args:
        provider: The provider name.
        model_id: The model ID.
        api: Optional API type (inferred if not provided).
        base_url: Optional base URL for OpenAI-compatible providers.
        **kwargs: Additional model configuration.

    Returns:
        A Model configuration object.
    """
    # Infer API type from provider if not provided
    if api is None:
        api = _infer_api_type(provider)

    return Model(
        id=model_id,
        provider=provider,
        api=api,
        base_url=base_url,
        **kwargs,
    )


def _infer_api_type(provider: str) -> Api:
    """Infer the API type from a provider name."""
    provider_lower = provider.lower()

    if provider_lower in ("anthropic",):
        return Api.ANTHROPIC_MESSAGES
    elif provider_lower in ("openai", "deepseek", "groq", "cerebras", "xai"):
        return Api.OPENAI_COMPLETIONS
    elif provider_lower in ("google",):
        return Api.GOOGLE_GENERATIVE_AI
    elif provider_lower in ("mistral",):
        return Api.MISTRAL_CONVERSATIONS
    elif provider_lower in ("amazon-bedrock", "bedrock"):
        return Api.BEDROCK_CONVERSE_STREAM
    else:
        # Default to OpenAI completions for compatible providers
        return Api.OPENAI_COMPLETIONS