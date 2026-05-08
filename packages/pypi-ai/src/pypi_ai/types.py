"""
Core types for pypi-ai.

This module defines all the fundamental types used across the LLM API abstraction:
- Enums for API types, providers, thinking levels, and stop reasons
- Content types for messages (text, thinking, image, tool calls)
- Message types (assistant, user, tool result)
- Context and options for LLM requests
- Model configuration
"""

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class Api(str, Enum):
    """Known API types for LLM providers."""

    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    AZURE_OPENAI_RESPONSES = "azure-openai-responses"
    OPENAI_CODEX_RESPONSES = "openai-codex-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"
    BEDROCK_CONVERSE_STREAM = "bedrock-converse-stream"
    GOOGLE_GENERATIVE_AI = "google-generative-ai"
    GOOGLE_VERTEX = "google-vertex"
    MISTRAL_CONVERSATIONS = "mistral-conversations"


class Provider(str, Enum):
    """Known LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GOOGLE_VERTEX = "google-vertex"
    MISTRAL = "mistral"
    AMAZON_BEDROCK = "amazon-bedrock"
    AZURE_OPENAI = "azure-openai"
    OPENAI_CODEX = "openai-codex"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"
    VERCEL_AI_GATEWAY = "vercel-ai-gateway"
    XAI = "xai"
    GITHUB_COPILOT = "github-copilot"


class ThinkingLevel(str, Enum):
    """Thinking/reasoning intensity levels."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ModelThinkingLevel(str, Enum):
    """Thinking level including off state."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class StopReason(str, Enum):
    """Reason for response completion."""

    END = "end"
    STOP = "stop"
    TOOL_USE = "tool_use"
    ERROR = "error"
    ABORTED = "aborted"


class CacheRetention(str, Enum):
    """Prompt cache retention preference."""

    NONE = "none"
    SHORT = "short"
    LONG = "long"


class Transport(str, Enum):
    """Transport type for providers that support multiple."""

    SSE = "sse"
    WEBSOCKET = "websocket"
    WEBSOCKET_CACHED = "websocket-cached"
    AUTO = "auto"


# =============================================================================
# Content Types
# =============================================================================


class TextContent(BaseModel):
    """Text content in a message."""

    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = None


class ThinkingContent(BaseModel):
    """Thinking/reasoning content in a message."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = None
    redacted: Optional[bool] = None


class ImageContent(BaseModel):
    """Image content in a message."""

    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str  # e.g., "image/jpeg", "image/png"


class ToolCall(BaseModel):
    """Tool call in an assistant message."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: Optional[str] = None


Content = Union[TextContent, ThinkingContent, ImageContent, ToolCall]


# =============================================================================
# Usage and Cost
# =============================================================================


class Cost(BaseModel):
    """Cost breakdown for a request."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    """Token usage for a request."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = Field(default_factory=Cost)


# =============================================================================
# Messages
# =============================================================================


class AssistantMessage(BaseModel):
    """Response from the assistant."""

    role: Literal["assistant"] = "assistant"
    content: list[Content] = Field(default_factory=list)
    api: Api
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(__import__("time").time() * 1000))


class UserMessage(BaseModel):
    """Message from the user."""

    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: Optional[int] = None


class ToolResultMessage(BaseModel):
    """Result from a tool execution."""

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    is_error: bool = False
    details: Optional[Any] = None


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# =============================================================================
# Context and Options
# =============================================================================


class Context(BaseModel):
    """Context for an LLM request."""

    system_prompt: Optional[str] = None
    messages: list[Message] = Field(default_factory=list)
    tools: list["Tool"] = Field(default_factory=list)


class StreamOptions(BaseModel):
    """Options for streaming LLM requests."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    transport: Optional[Transport] = None
    cache_retention: CacheRetention = CacheRetention.SHORT
    session_id: Optional[str] = None
    headers: Optional[dict[str, str]] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_retry_delay_ms: Optional[int] = 60000
    metadata: Optional[dict[str, Any]] = None
    signal: Optional[Any] = None  # AbortSignal equivalent


class ThinkingBudgets(BaseModel):
    """Token budgets for each thinking level."""

    minimal: Optional[int] = None
    low: Optional[int] = None
    medium: Optional[int] = None
    high: Optional[int] = None


class SimpleStreamOptions(StreamOptions):
    """Extended options with reasoning support."""

    reasoning: Optional[ThinkingLevel] = None
    thinking_budgets: Optional[ThinkingBudgets] = None


# =============================================================================
# Model Configuration
# =============================================================================


class Model(BaseModel):
    """Model configuration for LLM requests."""

    id: str
    name: Optional[str] = None
    api: Api
    provider: str
    base_url: Optional[str] = None
    reasoning: bool = False
    context_window: int = 128000
    max_tokens: int = 4096
    cost: Cost = Field(default_factory=Cost)
    thinking_levels: Optional[list[ModelThinkingLevel]] = None


# =============================================================================
# Tool Definition
# =============================================================================


class Tool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    strict: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for LLM providers."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# Update forward references
Context.model_rebuild()