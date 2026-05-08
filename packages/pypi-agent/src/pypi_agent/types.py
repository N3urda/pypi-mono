"""
Agent types for pypi-agent.

This module defines:
- AgentMessage union type
- AgentContext for agent execution
- AgentTool and AgentToolResult for tool execution
- AgentEvent types for agent lifecycle
- AgentLoopConfig for agent configuration
"""

from enum import StrEnum
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)
from pydantic import BaseModel, ConfigDict, Field

from pypi_ai.types import (
    Message,
    TextContent,
    ImageContent,
    Model,
    Tool,
    StopReason,
)
from pypi_ai.event_stream import AssistantMessageEvent


# =============================================================================
# Tool Execution Mode
# =============================================================================


class ToolExecutionMode(StrEnum):
    """Tool execution mode."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# =============================================================================
# Agent Message
# =============================================================================


AgentMessage = Message  # For now, can be extended with custom messages


# =============================================================================
# Agent Context
# =============================================================================


class AgentContext(BaseModel):
    """Context for agent execution."""

    system_prompt: str = ""
    messages: list[AgentMessage] = Field(default_factory=list)
    tools: list["AgentTool"] = Field(default_factory=list)


# =============================================================================
# Tool Types
# =============================================================================


class AgentToolResult(BaseModel):
    """Result from a tool execution."""

    content: list[TextContent | ImageContent] = Field(default_factory=list)
    details: Any = None
    terminate: bool = False


class AgentToolUpdateCallback(BaseModel):
    """Callback for streaming tool updates."""

    __slots__ = ("callback",)

    def __init__(self, callback: Callable[[AgentToolResult], None]):
        self.callback = callback

    def __call__(self, partial_result: AgentToolResult) -> None:
        self.callback(partial_result)


class AgentTool(Tool):
    """Extended tool definition for agents."""

    label: str = ""
    prepare_arguments: Optional[Callable[[dict], dict]] = None
    execute: Optional[Callable[[str, dict, Any, Any], AgentToolResult]] = None
    execution_mode: Optional[ToolExecutionMode] = None


# =============================================================================
# Agent Events
# =============================================================================


class AgentStartEvent(BaseModel):
    """Agent start event."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Agent end event."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class TurnStartEvent(BaseModel):
    """Turn start event."""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    """Turn end event."""

    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[Any]


class MessageStartEvent(BaseModel):
    """Message start event."""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(BaseModel):
    """Message update event."""

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class MessageEndEvent(BaseModel):
    """Message end event."""

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    """Tool execution start event."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(BaseModel):
    """Tool execution update event."""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class ToolExecutionEndEvent(BaseModel):
    """Tool execution end event."""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]


# =============================================================================
# Agent Loop Config
# =============================================================================


class BeforeToolCallResult(BaseModel):
    """Result from beforeToolCall hook."""

    block: bool = False
    reason: Optional[str] = None


class AfterToolCallResult(BaseModel):
    """Result from afterToolCall hook."""

    content: Optional[list[TextContent | ImageContent]] = None
    details: Any = None
    is_error: bool = False
    terminate: bool = False


class BeforeToolCallContext(BaseModel):
    """Context passed to before_tool_call hook."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: Optional[AgentMessage] = None
    tool_call: Any = None
    args: dict = Field(default_factory=dict)
    context: Optional["AgentContext"] = None


class AfterToolCallContext(BaseModel):
    """Context passed to after_tool_call hook."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: Optional[AgentMessage] = None
    tool_call: Any = None
    args: dict = Field(default_factory=dict)
    result: Optional[AgentToolResult] = None
    is_error: bool = False
    context: Optional["AgentContext"] = None


class AgentLoopConfig(BaseModel):
    """Configuration for the agent loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]]
    transform_context: Optional[Callable[[list[AgentMessage]], list[AgentMessage]]] = None
    get_api_key: Optional[Callable[[str], Optional[str]]] = None
    should_stop_after_turn: Optional[Callable[[Any], bool]] = None
    get_steering_messages: Optional[Callable[[], list[AgentMessage]]] = None
    get_follow_up_messages: Optional[Callable[[], list[AgentMessage]]] = None
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL
    before_tool_call: Optional[Callable] = None
    after_tool_call: Optional[Callable] = None