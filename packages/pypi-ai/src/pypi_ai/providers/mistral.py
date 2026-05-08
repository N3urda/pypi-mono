"""
Mistral provider implementation.

This module provides streaming support for Mistral's Conversations API.
"""

from typing import Any, Optional
import json

from pypi_ai.types import (
    Api,
    Context,
    Model,
    StreamOptions,
    SimpleStreamOptions,
    AssistantMessage,
    TextContent,
    ToolCall,
    Usage,
    Cost,
    StopReason,
)
from pypi_ai.event_stream import (
    AssistantMessageEventStream,
    create_error_message,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
)


class MistralProvider:
    """Mistral provider implementation."""

    api = Api.MISTRAL_CONVERSATIONS

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Mistral provider."""
        self._api_key = api_key
        self._client = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create the Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral
                key = api_key or self._api_key
                self._client = Mistral(api_key=key)
            except ImportError:
                raise ImportError(
                    "mistralai package is required. Install with: pip install mistralai"
                )
        return self._client

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Stream a response from Mistral."""
        return self._stream_impl(model, context, options or {})

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Stream a response with simplified options."""
        return self._stream_impl(model, context, options or {})

    def _stream_impl(
        self,
        model: Model,
        context: Context,
        options: dict[str, Any],
    ) -> AssistantMessageEventStream:
        """Internal streaming implementation."""
        stream = AssistantMessageEventStream()

        async def run_stream():
            try:
                client = self._get_client(options.get("api_key"))

                # Build messages
                messages = []
                for msg in context.messages:
                    if hasattr(msg, "role"):
                        if msg.role == "user":
                            content = msg.content if isinstance(msg.content, str) else ""
                            messages.append({"role": "user", "content": content})
                        elif msg.role == "assistant":
                            content = ""
                            for c in msg.content:
                                if c.type == "text":
                                    content += c.text
                            messages.append({"role": "assistant", "content": content})

                # Build request
                request_params = {
                    "model": model.id,
                    "messages": messages,
                }

                if options.get("temperature") is not None:
                    request_params["temperature"] = options["temperature"]
                if options.get("max_tokens") is not None:
                    request_params["max_tokens"] = options["max_tokens"]

                # Stream response
                text_content = ""
                usage_data = Usage()
                stop_reason = StopReason.END

                async_response = client.chat.stream_async(**request_params)

                async for chunk in async_response:
                    if chunk.data:
                        data = chunk.data
                        if hasattr(data, "choices") and data.choices:
                            choice = data.choices[0]
                            if hasattr(choice, "delta") and choice.delta:
                                delta = choice.delta
                                if hasattr(delta, "content") and delta.content:
                                    if not text_content:
                                        stream.push(TextStartEvent(content_index=0))
                                    stream.push(TextDeltaEvent(
                                        delta=delta.content,
                                        content_index=0
                                    ))
                                    text_content += delta.content

                            if hasattr(choice, "finish_reason") and choice.finish_reason:
                                stop_reason = _map_stop_reason(choice.finish_reason)

                        if hasattr(data, "usage") and data.usage:
                            usage_data.input = getattr(data.usage, "prompt_tokens", 0) or 0
                            usage_data.output = getattr(data.usage, "completion_tokens", 0) or 0

                # Finalize
                if text_content:
                    stream.push(TextEndEvent(content=text_content, content_index=0))

                usage_data.total_tokens = usage_data.input + usage_data.output
                usage_data.cost = Cost(
                    input=usage_data.input * 0.0002 / 1000,
                    output=usage_data.output * 0.0006 / 1000,
                )
                usage_data.cost.total = usage_data.cost.input + usage_data.cost.output

                final_content = [TextContent(type="text", text=text_content)]

                message = AssistantMessage(
                    role="assistant",
                    content=final_content,
                    api=Api.MISTRAL_CONVERSATIONS,
                    provider=model.provider,
                    model=model.id,
                    usage=usage_data,
                    stop_reason=stop_reason,
                )
                stream.push(DoneEvent(reason=stop_reason, message=message))
                stream.end(message)

            except Exception as e:
                error_msg = create_error_message(
                    Api.MISTRAL_CONVERSATIONS.value,
                    model.provider,
                    model.id,
                    str(e),
                )
                stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_msg))
                stream.error(error_msg)

        import asyncio
        asyncio.create_task(run_stream())

        return stream


def _map_stop_reason(reason: Optional[str]) -> StopReason:
    """Map Mistral stop reason to our enum."""
    if reason is None:
        return StopReason.END
    mapping = {
        "stop": StopReason.STOP,
        "length": StopReason.END,
        "tool_calls": StopReason.TOOL_USE,
    }
    return mapping.get(reason, StopReason.END)


# Provider instance
provider = MistralProvider()