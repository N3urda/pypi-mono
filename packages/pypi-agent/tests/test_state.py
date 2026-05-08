"""Tests for agent state."""

import pytest

from pypi_agent.state import AgentState
from pypi_agent.types import AgentTool, AgentToolResult
from pypi_ai.types import ThinkingLevel, Model, Api


@pytest.fixture
def model():
    """Create a test model."""
    return Model(
        id="test-model",
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
    )


def test_agent_state_init():
    """Test AgentState initialization."""
    state = AgentState()

    assert state.system_prompt == ""
    assert state.model is None
    assert state.thinking_level == ThinkingLevel.OFF
    assert len(state.tools) == 0
    assert len(state.messages) == 0


def test_agent_state_with_model(model):
    """Test AgentState with model."""
    state = AgentState(model=model)

    assert state.model is not None
    assert state.model.id == "test-model"


def test_agent_state_system_prompt():
    """Test system prompt property."""
    state = AgentState()

    state.system_prompt = "Test prompt"
    assert state.system_prompt == "Test prompt"


def test_agent_state_thinking_level():
    """Test thinking level property."""
    state = AgentState()

    state.thinking_level = ThinkingLevel.MEDIUM
    assert state.thinking_level == ThinkingLevel.MEDIUM


def test_agent_state_tools():
    """Test tools property with copy semantics."""
    state = AgentState()
    tool = AgentTool(
        name="test_tool",
        description="Test tool",
        parameters={},
    )

    state.tools = [tool]
    assert len(state.tools) == 1

    # Get returns copy
    tools = state.tools
    tools.append(tool)
    assert len(state.tools) == 1  # Original unchanged


def test_agent_state_messages():
    """Test messages property with copy semantics."""
    state = AgentState()

    state.messages = [{"role": "user", "content": "test"}]
    assert len(state.messages) == 1

    # Get returns copy
    msgs = state.messages
    msgs.append({"role": "assistant", "content": "response"})
    assert len(state.messages) == 1


def test_agent_state_add_tool():
    """Test add_tool method."""
    state = AgentState()
    tool = AgentTool(
        name="test_tool",
        description="Test tool",
        parameters={},
    )

    state.add_tool(tool)
    assert len(state.tools) == 1
    assert state.tools[0].name == "test_tool"


def test_agent_state_remove_tool():
    """Test remove_tool method."""
    state = AgentState()
    tool = AgentTool(
        name="test_tool",
        description="Test tool",
        parameters={},
    )

    state.add_tool(tool)
    result = state.remove_tool("test_tool")
    assert result is True
    assert len(state.tools) == 0


def test_agent_state_remove_tool_not_found():
    """Test remove_tool returns False when not found."""
    state = AgentState()

    result = state.remove_tool("nonexistent")
    assert result is False


def test_agent_state_add_message():
    """Test add_message method."""
    state = AgentState()

    state.add_message({"role": "user", "content": "Hello"})
    assert len(state.messages) == 1


def test_agent_state_add_messages():
    """Test add_messages method."""
    state = AgentState()

    state.add_messages([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])
    assert len(state.messages) == 2


def test_agent_state_clear_messages():
    """Test clear_messages method."""
    state = AgentState()
    state.add_message({"role": "user", "content": "Hello"})
    assert len(state.messages) == 1

    state.clear_messages()
    assert len(state.messages) == 0


def test_agent_state_streaming():
    """Test streaming state methods."""
    state = AgentState()

    assert not state.is_streaming

    state.set_streaming(True, {"role": "assistant", "content": "Streaming"})
    assert state.is_streaming
    assert state.streaming_message is not None

    state.set_streaming(False)
    assert not state.is_streaming
    assert state.streaming_message is None


def test_agent_state_pending_tool_calls():
    """Test pending tool call methods."""
    state = AgentState()

    state.add_pending_tool_call("call_1")
    assert "call_1" in state.pending_tool_calls

    state.remove_pending_tool_call("call_1")
    assert "call_1" not in state.pending_tool_calls

    state.add_pending_tool_call("call_2")
    state.clear_pending_tool_calls()
    assert len(state.pending_tool_calls) == 0


def test_agent_state_error():
    """Test error state methods."""
    state = AgentState()

    state.set_error("Something went wrong")
    assert state.error_message == "Something went wrong"

    state.clear_error()
    assert state.error_message is None


def test_agent_state_copy():
    """Test copy method."""
    state = AgentState(
        system_prompt="Test",
        thinking_level=ThinkingLevel.HIGH,
    )
    tool = AgentTool(name="test", description="Test", parameters={})
    state.add_tool(tool)

    copy = state.copy()

    assert copy.system_prompt == state.system_prompt
    assert copy.thinking_level == state.thinking_level
    assert len(copy.tools) == 1

    # Modify copy doesn't affect original
    copy.system_prompt = "Changed"
    assert state.system_prompt == "Test"


def test_agent_state_to_context():
    """Test to_context method."""
    state = AgentState(system_prompt="Test prompt")

    context = state.to_context()

    assert context["system_prompt"] == "Test prompt"
    assert len(context["messages"]) == 0
    assert len(context["tools"]) == 0