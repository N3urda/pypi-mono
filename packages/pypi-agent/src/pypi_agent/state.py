"""
Agent state management.

This module provides:
- AgentState class for managing agent state
- Properties for system prompt, model, messages, tools
- Tracking of streaming state and pending tool calls
"""

from typing import Any, Callable, Optional, Set
import copy

from pypi_ai.types import Model, ThinkingLevel
from pypi_agent.types import AgentMessage, AgentTool, ToolExecutionMode


class AgentState:
    """
    Agent state management class.

    Manages:
    - System prompt
    - Current model and thinking level
    - Messages history
    - Available tools
    - Streaming state
    - Pending tool calls
    - Error state
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: Optional[Model] = None,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
        tools: Optional[list[AgentTool]] = None,
        messages: Optional[list[AgentMessage]] = None,
    ) -> None:
        """Initialize agent state."""
        self._system_prompt = system_prompt
        self._model = model
        self._thinking_level = thinking_level
        self._tools: list[AgentTool] = list(tools or [])
        self._messages: list[AgentMessage] = list(messages or [])
        self._is_streaming = False
        self._streaming_message: Optional[AgentMessage] = None
        self._pending_tool_calls: Set[str] = set()
        self._error_message: Optional[str] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._system_prompt = value

    @property
    def model(self) -> Optional[Model]:
        """Get the current model."""
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        """Set the current model."""
        self._model = value

    @property
    def thinking_level(self) -> ThinkingLevel:
        """Get the thinking level."""
        return self._thinking_level

    @thinking_level.setter
    def thinking_level(self, value: ThinkingLevel) -> None:
        """Set the thinking level."""
        self._thinking_level = value

    @property
    def tools(self) -> list[AgentTool]:
        """Get the tools list (returns a copy)."""
        return list(self._tools)

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        """Set the tools list (copies the input)."""
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        """Get the messages list (returns a copy)."""
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        """Set the messages list (copies the input)."""
        self._messages = list(value)

    @property
    def is_streaming(self) -> bool:
        """Check if the agent is currently streaming."""
        return self._is_streaming

    @property
    def streaming_message(self) -> Optional[AgentMessage]:
        """Get the current streaming message if any."""
        return self._streaming_message

    @property
    def pending_tool_calls(self) -> Set[str]:
        """Get the set of pending tool call IDs."""
        return self._pending_tool_calls

    @property
    def error_message(self) -> Optional[str]:
        """Get the last error message if any."""
        return self._error_message

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the history."""
        self._messages.append(message)

    def add_messages(self, messages: list[AgentMessage]) -> None:
        """Add multiple messages to the history."""
        self._messages.extend(messages)

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool."""
        self._tools.append(tool)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        for i, tool in enumerate(self._tools):
            if tool.name == name:
                self._tools.pop(i)
                return True
        return False

    def set_streaming(self, is_streaming: bool, message: Optional[AgentMessage] = None) -> None:
        """Set the streaming state."""
        self._is_streaming = is_streaming
        self._streaming_message = message if is_streaming else None

    def add_pending_tool_call(self, tool_call_id: str) -> None:
        """Add a pending tool call."""
        self._pending_tool_calls.add(tool_call_id)

    def remove_pending_tool_call(self, tool_call_id: str) -> None:
        """Remove a pending tool call."""
        self._pending_tool_calls.discard(tool_call_id)

    def clear_pending_tool_calls(self) -> None:
        """Clear all pending tool calls."""
        self._pending_tool_calls.clear()

    def set_error(self, error_message: Optional[str]) -> None:
        """Set the error message."""
        self._error_message = error_message

    def clear_error(self) -> None:
        """Clear the error message."""
        self._error_message = None

    def copy(self) -> "AgentState":
        """Create a deep copy of the state."""
        return AgentState(
            system_prompt=self._system_prompt,
            model=self._model,
            thinking_level=self._thinking_level,
            tools=list(self._tools),
            messages=list(self._messages),
        )

    def to_context(self) -> dict[str, Any]:
        """Convert to context dict for agent loop."""
        return {
            "system_prompt": self._system_prompt,
            "messages": list(self._messages),
            "tools": list(self._tools),
        }