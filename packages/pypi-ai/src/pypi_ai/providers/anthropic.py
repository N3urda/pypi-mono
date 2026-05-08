"""
Anthropic Claude provider implementation.

This module provides streaming support for Anthropic's Messages API
with support for:
- Text and thinking content
- Tool calling with function execution
- Prompt caching
- Extended thinking (reasoning)
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
    ThinkingContent,
    ToolCall,
    Usage,
    Cost,
    StopReason,
    Tool,
)
from pypi_ai.event_stream import (
    AssistantMessageEventStream,
    create_error_message,
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
)


def _convert_context(context: Context) -> dict[str, Any]:
    """Convert pypi-ai Context to Anthropic message format."""
    messages = []

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                # Convert content list
                blocks = []
                for c in content:
                    if hasattr(c, "type"):
                        if c.type == "text":
                            blocks.append({"type": "text", "text": c.text})
                        elif c.type == "image":
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": c.mime_type,
                                    "data": c.data,
                                },
                            })
                messages.append({"role": "user", "content": blocks})
        elif hasattr(msg, "role") and msg.role == "assistant":
            # Convert assistant message
            content = []
            for c in msg.content:
                if c.type == "text":
                    content.append({"type": "text", "text": c.text})
                elif c.type == "thinking":
                    content.append({"type": "thinking", "thinking": c.thinking})
                elif c.type == "toolCall":
                    content.append({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    })
            messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, ToolResultMessage):
            content = []
            for c in msg.content:
                if c.type == "text":
                    content.append({"type": "text", "text": c.text})
                elif c.type == "image":
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": c.mime_type,
                            "data": c.data,
                        },
                    })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": content,
                    "is_error": msg.is_error,
                }],
            })

    return messages


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pypi-ai Tools to Anthropic tool format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]


def _create_assistant_message(
    model: Model,
    content: list,
    usage: Usage,
    stop_reason: StopReason,
    error_message: Optional[str] = None,
) -> AssistantMessage:
    """Create an AssistantMessage from Anthropic response."""
    return AssistantMessage(
        role="assistant",
        content=content,
        api=Api.ANTHROPIC_MESSAGES,
        provider=model.provider,
        model=model.id,
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
    )


class AnthropicProvider:
    """Anthropic Claude provider implementation."""

    api = Api.ANTHROPIC_MESSAGES

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic provider."""
        self._api_key = api_key
        self._client = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                key = api_key or self._api_key
                self._client = anthropic.AsyncAnthropic(api_key=key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
        return self._client

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Stream a response from Anthropic."""
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

                # Build request
                request_params = {
                    "model": model.id,
                    "messages": _convert_context(context),
                    "max_tokens": options.get("max_tokens", model.max_tokens),
                }

                if context.system_prompt:
                    request_params["system"] = context.system_prompt

                if context.tools:
                    request_params["tools"] = _convert_tools(context.tools)

                if options.get("temperature") is not None:
                    request_params["temperature"] = options["temperature"]

                # Handle caching
                cache_control = []
                if options.get("cache_retention") != "none":
                    # Add cache control to system prompt if present
                    if context.system_prompt:
                        request_params["system"] = [
                            {"type": "text", "text": context.system_prompt, "cache_control": {"type": "ephemeral"}}
                        ]

                # Handle extended thinking
                reasoning = options.get("reasoning")
                if reasoning:
                    request_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": _get_thinking_budget(reasoning, options),
                    }

                # Stream the response
                content_blocks: list[dict] = []
                current_block_index = -1
                usage_data = Usage()
                stop_reason = StopReason.END

                async with client.messages.stream(**request_params) as response_stream:
                    async for event in response_stream:
                        event_type = event.type

                        if event_type == "content_block_start":
                            block = event.content_block
                            current_block_index = event.index
                            content_blocks.append({})

                            if block.type == "text":
                                stream.push(TextStartEvent(content_index=current_block_index))
                            elif block.type == "thinking":
                                stream.push(ThinkingStartEvent(content_index=current_block_index))
                            elif block.type == "tool_use":
                                content_blocks[current_block_index] = {
                                    "id": block.id,
                                    "name": block.name,
                                    "input": "",
                                }
                                stream.push(ToolCallStartEvent(
                                    content_index=current_block_index,
                                    id=block.id,
                                    name=block.name,
                                ))

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            idx = event.index

                            if delta.type == "text_delta":
                                stream.push(TextDeltaEvent(delta=delta.text, content_index=idx))
                                content_blocks[idx]["text"] = content_blocks[idx].get("text", "") + delta.text
                            elif delta.type == "thinking_delta":
                                stream.push(ThinkingDeltaEvent(delta=delta.thinking, content_index=idx))
                                content_blocks[idx]["thinking"] = content_blocks[idx].get("thinking", "") + delta.thinking
                            elif delta.type == "input_json_delta":
                                partial_json = delta.partial_json
                                stream.push(ToolCallDeltaEvent(delta=partial_json, content_index=idx))
                                content_blocks[idx]["input"] = content_blocks[idx].get("input", "") + partial_json

                        elif event_type == "content_block_stop":
                            idx = event.index
                            block = content_blocks[idx]

                            if "text" in block:
                                stream.push(TextEndEvent(content=block["text"], content_index=idx))
                            elif "thinking" in block:
                                stream.push(ThinkingEndEvent(content=block["thinking"], content_index=idx))
                            elif "name" in block:
                                # Complete tool call
                                try:
                                    arguments = json.loads(block.get("input", "{}"))
                                except json.JSONDecodeError:
                                    arguments = {}

                                tool_call = ToolCall(
                                    type="toolCall",
                                    id=block["id"],
                                    name=block["name"],
                                    arguments=arguments,
                                )
                                stream.push(ToolCallEndEvent(tool_call=tool_call, content_index=idx))
                                content_blocks[idx]["tool_call"] = tool_call

                        elif event_type == "message_delta":
                            if hasattr(event, "usage"):
                                usage_data.output = event.usage.output_tokens or 0
                            if hasattr(event, "stop_reason"):
                                stop_reason = _map_stop_reason(event.stop_reason)

                        elif event_type == "message_start":
                            if hasattr(event, "message") and hasattr(event.message, "usage"):
                                u = event.message.usage
                                usage_data.input = u.input_tokens or 0
                                usage_data.cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                                usage_data.cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0

                # Build final content
                final_content = []
                for block in content_blocks:
                    if "text" in block:
                        final_content.append(TextContent(type="text", text=block["text"]))
                    elif "thinking" in block:
                        final_content.append(ThinkingContent(type="thinking", thinking=block["thinking"]))
                    elif "tool_call" in block:
                        final_content.append(block["tool_call"])

                # Calculate cost
                usage_data.cost = Cost(
                    input=usage_data.input * 0.003 / 1000,  # Approximate
                    output=usage_data.output * 0.015 / 1000,
                    cache_read=usage_data.cache_read * 0.0003 / 1000,
                    cache_write=usage_data.cache_write * 0.00375 / 1000,
                )
                usage_data.total_tokens = usage_data.input + usage_data.output
                usage_data.cost.total = (
                    usage_data.cost.input +
                    usage_data.cost.output +
                    usage_data.cost.cache_read +
                    usage_data.cost.cache_write
                )

                message = _create_assistant_message(model, final_content, usage_data, stop_reason)
                stream.push(DoneEvent(reason=stop_reason, message=message))
                stream.end(message)

            except Exception as e:
                error_msg = create_error_message(
                    Api.ANTHROPIC_MESSAGES.value,
                    model.provider,
                    model.id,
                    str(e),
                )
                stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_msg))
                stream.error(error_msg)

        import asyncio
        asyncio.create_task(run_stream())

        return stream


def _get_thinking_budget(level: str, options: dict) -> int:
    """Get thinking budget tokens for a reasoning level."""
    budgets = options.get("thinking_budgets", {})
    default_budgets = {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16000,
    }
    return budgets.get(level, default_budgets.get(level, 4096))


def _map_stop_reason(reason: Optional[str]) -> StopReason:
    """Map Anthropic stop reason to our enum."""
    if reason is None:
        return StopReason.END
    mapping = {
        "end_turn": StopReason.END,
        "stop_sequence": StopReason.STOP,
        "tool_use": StopReason.TOOL_USE,
    }
    return mapping.get(reason, StopReason.END)


# Provider instance
provider = AnthropicProvider()