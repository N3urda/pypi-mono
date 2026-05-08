"""Tests for agent types."""

import pytest

from pypi_agent.types import (
    AgentContext,
    AgentTool,
    AgentToolResult,
    AgentEvent,
    AgentLoopConfig,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    BeforeToolCallResult,
    AfterToolCallResult,
    ToolExecutionMode,
    AgentToolUpdateCallback,
)
from pypi_ai.types import (
    Model,
    Api,
    TextContent,
    UserMessage,
)


def test_agent_context():
    """Test AgentContext model."""
    ctx = AgentContext(
        system_prompt="You are helpful",
        messages=[],
    )

    assert ctx.system_prompt == "You are helpful"
    assert len(ctx.messages) == 0


def test_agent_tool():
    """Test AgentTool definition."""
    tool = AgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object"},
        label="Test",
    )

    assert tool.name == "test_tool"
    assert tool.label == "Test"


def test_agent_tool_result():
    """Test AgentToolResult model."""
    result = AgentToolResult(
        content=[TextContent(type="text", text="Done")],
        details={"status": "success"},
        terminate=True,
    )

    assert result.terminate is True
    assert result.details["status"] == "success"


def test_agent_start_event():
    """Test AgentStartEvent."""
    event = AgentStartEvent()
    assert event.type == "agent_start"


def test_agent_end_event():
    """Test AgentEndEvent."""
    event = AgentEndEvent(messages=[])
    assert event.type == "agent_end"
    assert len(event.messages) == 0


def test_turn_start_event():
    """Test TurnStartEvent."""
    event = TurnStartEvent()
    assert event.type == "turn_start"


def test_turn_end_event():
    """Test TurnEndEvent."""
    msg = UserMessage(content="test")
    event = TurnEndEvent(message=msg, tool_results=[])
    assert event.type == "turn_end"


def test_message_start_event():
    """Test MessageStartEvent."""
    msg = UserMessage(content="test")
    event = MessageStartEvent(message=msg)
    assert event.type == "message_start"


def test_message_end_event():
    """Test MessageEndEvent."""
    msg = UserMessage(content="test")
    event = MessageEndEvent(message=msg)
    assert event.type == "message_end"


def test_tool_execution_start_event():
    """Test ToolExecutionStartEvent."""
    event = ToolExecutionStartEvent(
        tool_call_id="call_1",
        tool_name="test",
        args={"arg": "value"},
    )
    assert event.type == "tool_execution_start"
    assert event.tool_name == "test"


def test_tool_execution_end_event():
    """Test ToolExecutionEndEvent."""
    event = ToolExecutionEndEvent(
        tool_call_id="call_1",
        tool_name="test",
        result={"content": "done"},
        is_error=False,
    )
    assert event.type == "tool_execution_end"
    assert event.is_error is False


def test_before_tool_call_result():
    """Test BeforeToolCallResult."""
    result = BeforeToolCallResult(block=True, reason="Blocked")
    assert result.block is True


def test_after_tool_call_result():
    """Test AfterToolCallResult."""
    result = AfterToolCallResult(
        content=[TextContent(type="text", text="Modified")],
        is_error=True,
    )
    assert result.is_error is True


def test_agent_loop_config():
    """Test AgentLoopConfig."""
    model = Model(
        id="test",
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
    )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda m: m,
        tool_execution=ToolExecutionMode.PARALLEL,
    )

    assert config.tool_execution == ToolExecutionMode.PARALLEL