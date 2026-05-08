"""
pypi-ai: Unified LLM API abstraction with multi-provider support.

This package provides:
- Unified streaming interface for multiple LLM providers
- Model configuration and discovery
- Token and cost tracking
- Event stream protocol for streaming responses
"""

__version__ = "0.0.1"

# Import and register built-in providers first
from pypi_ai.providers.register_builtins import register_builtin_providers
register_builtin_providers()

from pypi_ai.types import (
    Api,
    Provider,
    ThinkingLevel,
    StopReason,
    TextContent,
    ThinkingContent,
    ImageContent,
    ToolCall,
    Content,
    Usage,
    Cost,
    AssistantMessage,
    UserMessage,
    ToolResultMessage,
    Message,
    Context,
    StreamOptions,
    Model,
)
from pypi_ai.stream import stream, stream_simple, complete, complete_simple, get_model
from pypi_ai.registry import register_provider, get_provider, get_all_providers

__all__ = [
    # Types
    "Api",
    "Provider",
    "ThinkingLevel",
    "StopReason",
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    "Content",
    "Usage",
    "Cost",
    "AssistantMessage",
    "UserMessage",
    "ToolResultMessage",
    "Message",
    "Context",
    "StreamOptions",
    "Model",
    # Functions
    "stream",
    "stream_simple",
    "complete",
    "complete_simple",
    "register_provider",
    "get_provider",
    "get_all_providers",
]