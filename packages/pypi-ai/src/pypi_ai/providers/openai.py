"""
OpenAI provider implementation.

This module provides streaming support for OpenAI's Chat Completions API
with support for:
- Text content
- Tool calling (function calling)
- Custom base URL for OpenAI-compatible providers (DeepSeek, Groq, etc.)
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
    UserMessage,
    ToolResultMessage,
    TextContent,
    ToolCall,
    Usage,
    Cost,
    StopReason,
    Tool,
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


def _convert_context(context: Context) -> list[dict[str, Any]]:
    """Convert pypi-ai Context to OpenAI message format."""
    messages = []

    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                # Convert content list
                parts = []
                for c in content:
                    if c.type == "text":
                        parts.append({"type": "text", "text": c.text})
                    elif c.type == "image":
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{c.mime_type};base64,{c.data}",
                            },
                        })
                messages.append({"role": "user", "content": parts})
        elif hasattr(msg, "role") and msg.role == "assistant":
            content_parts = []
            tool_calls = []
            for c in msg.content:
                if c.type == "text":
                    content_parts.append(c.text)
                elif c.type == "thinking":
                    # Wrap thinking in XML tags for OpenAI
                    content_parts.append(f"<thinking>{c.thinking}</thinking>")
                elif c.type == "toolCall":
                    tool_calls.append({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        },
                    })
            msg_dict = {"role": "assistant"}
            if content_parts:
                msg_dict["content"] = "\n".join(content_parts)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            messages.append(msg_dict)
        elif isinstance(msg, ToolResultMessage):
            content_str = ""
            for c in msg.content:
                if c.type == "text":
                    content_str += c.text
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content_str if not msg.is_error else f"Error: {content_str}",
            })

    return messages


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pypi-ai Tools to OpenAI function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict,
            },
        }
        for tool in tools
    ]


class OpenAIProvider:
    """OpenAI provider implementation."""

    api = Api.OPENAI_COMPLETIONS

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the OpenAI provider."""
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                key = api_key or self._api_key
                url = base_url or self._base_url
                self._client = AsyncOpenAI(api_key=key, base_url=url)
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Stream a response from OpenAI."""
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
                # Use model's base_url if provided, or options override
                base_url = options.get("base_url") or model.base_url
                client = self._get_client(options.get("api_key"), base_url)

                # Build request
                request_params = {
                    "model": model.id,
                    "messages": _convert_context(context),
                    "stream": True,
                }

                if context.tools:
                    request_params["tools"] = _convert_tools(context.tools)

                if options.get("temperature") is not None:
                    request_params["temperature"] = options["temperature"]

                if options.get("max_tokens") is not None:
                    request_params["max_tokens"] = options["max_tokens"]
                elif model.max_tokens:
                    request_params["max_tokens"] = model.max_tokens

                # Stream the response
                content_blocks: list[dict] = []
                tool_calls: dict[int, dict] = {}
                usage_data = Usage()
                stop_reason = StopReason.END

                response = await client.chat.completions.create(**request_params)

                text_started = False
                current_text = ""

                async for chunk in response:
                    # Handle usage
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data.input = chunk.usage.prompt_tokens or 0
                        usage_data.output = chunk.usage.completion_tokens or 0

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]

                    # Handle finish reason
                    if choice.finish_reason:
                        stop_reason = _map_stop_reason(choice.finish_reason)

                    delta = choice.delta

                    # Handle text content
                    if delta.content:
                        if not text_started:
                            text_started = True
                            stream.push(TextStartEvent(content_index=0))
                            content_blocks.append({"type": "text", "text": ""})

                        stream.push(TextDeltaEvent(delta=delta.content, content_index=0))
                        current_text += delta.content
                        content_blocks[0]["text"] = current_text

                    # Handle tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index

                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                                stream.push(ToolCallStartEvent(
                                    content_index=idx + 1,
                                    id=tc.id,
                                    name=None,
                                ))

                            if tc.function:
                                if tc.function.name:
                                    tool_calls[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[idx]["arguments"] += tc.function.arguments
                                    stream.push(ToolCallDeltaEvent(
                                        delta=tc.function.arguments,
                                        content_index=idx + 1,
                                    ))

                # Finalize text
                if text_started:
                    stream.push(TextEndEvent(content=current_text, content_index=0))

                # Finalize tool calls
                final_content = []
                if current_text:
                    final_content.append(TextContent(type="text", text=current_text))

                for idx, tc in sorted(tool_calls.items()):
                    try:
                        arguments = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    tool_call = ToolCall(
                        type="toolCall",
                        id=tc["id"],
                        name=tc["name"],
                        arguments=arguments,
                    )
                    stream.push(ToolCallEndEvent(tool_call=tool_call, content_index=idx + 1))
                    final_content.append(tool_call)

                # Calculate cost (approximate)
                usage_data.cost = Cost(
                    input=usage_data.input * 0.005 / 1000,
                    output=usage_data.output * 0.015 / 1000,
                )
                usage_data.total_tokens = usage_data.input + usage_data.output
                usage_data.cost.total = usage_data.cost.input + usage_data.cost.output

                message = AssistantMessage(
                    role="assistant",
                    content=final_content,
                    api=Api.OPENAI_COMPLETIONS,
                    provider=model.provider,
                    model=model.id,
                    usage=usage_data,
                    stop_reason=stop_reason,
                )
                stream.push(DoneEvent(reason=stop_reason, message=message))
                stream.end(message)

            except Exception as e:
                error_msg = create_error_message(
                    Api.OPENAI_COMPLETIONS.value,
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
    """Map OpenAI stop reason to our enum."""
    if reason is None:
        return StopReason.END
    mapping = {
        "stop": StopReason.STOP,
        "length": StopReason.END,
        "tool_calls": StopReason.TOOL_USE,
        "content_filter": StopReason.ERROR,
    }
    return mapping.get(reason, StopReason.END)


# Provider instance
provider = OpenAIProvider()