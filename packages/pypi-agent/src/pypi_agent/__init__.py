"""
pypi-agent: Agent runtime with tool execution loop.

This package provides:
- Agent state management
- Agent loop with turn management
- Tool execution system with parallel/sequential modes
- Hook system for before/after tool execution
"""

__version__ = "0.0.1"

from pypi_agent.types import (
    AgentMessage,
    AgentContext,
    AgentTool,
    AgentToolResult,
    AgentEvent,
    AgentLoopConfig,
    ToolExecutionMode,
)
from pypi_agent.state import AgentState
from pypi_agent.loop import agent_loop, agent_loop_continue

__all__ = [
    # Types
    "AgentMessage",
    "AgentContext",
    "AgentTool",
    "AgentToolResult",
    "AgentEvent",
    "AgentLoopConfig",
    "ToolExecutionMode",
    # State
    "AgentState",
    # Functions
    "agent_loop",
    "agent_loop_continue",
]