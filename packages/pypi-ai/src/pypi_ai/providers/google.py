"""
Google Gemini provider implementation.

This module provides streaming support for Google's Gemini API.
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
    ThinkingContent,
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
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
)


class GoogleProvider:
    """Google Gemini provider implementation."""

    api = Api.GOOGLE_GENERATIVE_AI

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Google provider."""
        self._api_key = api_key
        self._client = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create the Google client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                key = api_key or self._api_key
                genai.configure(api_key=key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Stream a response from Google Gemini."""
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

                # Build generation config
                generation_config = {}
                if options.get("temperature") is not None:
                    generation_config["temperature"] = options["temperature"]
                if options.get("max_tokens") is not None:
                    generation_config["max_output_tokens"] = options["max_tokens"]

                # Create model instance
                gemini_model = client.GenerativeModel(
                    model_name=model.id,
                    generation_config=generation_config,
                )

                # Build chat history
                history = []
                for msg in context.messages:
                    if hasattr(msg, "role"):
                        if msg.role == "user":
                            content = msg.content if isinstance(msg.content, str) else msg.content
                            history.append({"role": "user", "parts": [content]})
                        elif msg.role == "assistant":
                            content = ""
                            for c in msg.content:
                                if c.type == "text":
                                    content += c.text
                            history.append({"role": "model", "parts": [content]})

                # Start chat
                chat = gemini_model.start_chat(history=history)

                # Stream response
                response = await chat.send_message_async(
                    context.messages[-1].content if context.messages else "",
                    stream=True,
                )

                text_content = ""
                usage_data = Usage()
                stop_reason = StopReason.END

                async for chunk in response:
                    # Handle text
                    if chunk.text:
                        if not text_content:
                            stream.push(TextStartEvent(content_index=0))
                        stream.push(TextDeltaEvent(delta=chunk.text, content_index=0))
                        text_content += chunk.text

                    # Handle usage
                    if hasattr(chunk, "usage_metadata"):
                        meta = chunk.usage_metadata
                        usage_data.input = getattr(meta, "prompt_token_count", 0) or 0
                        usage_data.output = getattr(meta, "candidates_token_count", 0) or 0

                # Finalize
                if text_content:
                    stream.push(TextEndEvent(content=text_content, content_index=0))

                usage_data.total_tokens = usage_data.input + usage_data.output
                usage_data.cost = Cost(
                    input=usage_data.input * 0.00025 / 1000,
                    output=usage_data.output * 0.00125 / 1000,
                )
                usage_data.cost.total = usage_data.cost.input + usage_data.cost.output

                final_content = [TextContent(type="text", text=text_content)]

                message = AssistantMessage(
                    role="assistant",
                    content=final_content,
                    api=Api.GOOGLE_GENERATIVE_AI,
                    provider=model.provider,
                    model=model.id,
                    usage=usage_data,
                    stop_reason=stop_reason,
                )
                stream.push(DoneEvent(reason=stop_reason, message=message))
                stream.end(message)

            except Exception as e:
                error_msg = create_error_message(
                    Api.GOOGLE_GENERATIVE_AI.value,
                    model.provider,
                    model.id,
                    str(e),
                )
                stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_msg))
                stream.error(error_msg)

        import asyncio
        asyncio.create_task(run_stream())

        return stream


# Provider instance
provider = GoogleProvider()