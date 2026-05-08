"""Tests for agent loop with full coverage."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from pypi_agent.loop import (
    run_loop,
    execute_tool,
    emit_event,
)
from pypi_agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    ToolExecutionMode,
    BeforeToolCallResult,
    AfterToolCallResult,
)
from pypi_ai.types import (
    Model,
    Api,
    UserMessage,
    AssistantMessage,
    TextContent,
    StopReason,
    Usage,
    ToolCall,
)
from pypi_ai.event_stream import AssistantMessageEventStream, DoneEvent


@pytest.fixture
def model():
    """Create a test model."""
    return Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")


@pytest.fixture
def mock_assistant_message():
    """Create a mock assistant message."""
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Test response")],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.END,
    )


def _create_mock_stream(message):
    """Create a mock stream that returns a message."""
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=StopReason.END, message=message))
    stream.end(message)
    return stream


def _create_config(model, **kwargs):
    """Create an AgentLoopConfig with async convert_to_llm."""
    async def convert_to_llm(m):
        return m

    return AgentLoopConfig(
        model=model,
        convert_to_llm=convert_to_llm,
        **kwargs
    )


def _create_emit(events):
    """Create an async emit function."""
    async def emit(event):
        events.append(event)
    return emit


# =============================================================================
# Execute Tool Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execute_tool_with_error():
    """Test execute_tool when tool raises error."""
    async def execute_fn(tool_id, args, signal):
        raise RuntimeError("Tool error")

    tool = AgentTool(
        name="error_tool",
        description="Raises error",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="error_tool",
        arguments={},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, _create_emit(events))

    assert result.is_error
    assert "Error" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_tool_cancelled():
    """Test execute_tool when cancelled."""
    async def execute_fn(tool_id, args, signal):
        raise asyncio.CancelledError()

    tool = AgentTool(
        name="cancel_tool",
        description="Gets cancelled",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="cancel_tool",
        arguments={},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, _create_emit(events))

    assert result.is_error
    assert "cancelled" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_execute_tool_parallel_mode():
    """Test execute_tool with parallel execution mode."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Success")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        tools=[tool],
    )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model, tool_execution=ToolExecutionMode.PARALLEL)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={},
    )

    events = []
    result = await execute_tool(tool_call, context, config, None, _create_emit(events))

    assert not result.is_error


# =============================================================================
# Hook Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execute_tool_before_hook_with_message():
    """Test execute_tool with before_tool_call hook that has message."""
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=lambda tid, args, sig: AgentToolResult(content=[]),
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[AssistantMessage(
            content=[],
            api=Api.ANTHROPIC_MESSAGES,
            provider="anthropic",
            model="test",
            usage=Usage(),
            stop_reason=StopReason.END,
        )],
        tools=[tool],
    )

    hook_called = [False]

    async def before_hook(ctx, signal):
        hook_called[0] = True
        assert ctx.assistant_message is not None
        return None  # Don't block

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model, before_tool_call=before_hook)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={},
    )

    await execute_tool(tool_call, context, config, None, _create_emit([]))

    assert hook_called[0]


@pytest.mark.asyncio
async def test_execute_tool_after_hook_modifies_content():
    """Test execute_tool with after_tool_call hook modifying content."""
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=lambda tid, args, sig: AgentToolResult(
            content=[TextContent(type="text", text="Original")]
        ),
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[],
        tools=[tool],
    )

    async def after_hook(ctx, signal):
        return AfterToolCallResult(
            content=[TextContent(type="text", text="Modified")],
        )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model, after_tool_call=after_hook)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={},
    )

    result = await execute_tool(tool_call, context, config, None, _create_emit([]))

    assert result.content[0].text == "Modified"


@pytest.mark.asyncio
async def test_execute_tool_after_hook_sets_error():
    """Test execute_tool with after_tool_call hook setting error."""
    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=lambda tid, args, sig: AgentToolResult(
            content=[TextContent(type="text", text="Success")]
        ),
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[],
        tools=[tool],
    )

    async def after_hook(ctx, signal):
        return AfterToolCallResult(
            is_error=True,
        )

    model = Model(id="test", api=Api.ANTHROPIC_MESSAGES, provider="anthropic")
    config = _create_config(model, after_tool_call=after_hook)

    tool_call = ToolCall(
        type="toolCall",
        id="call_1",
        name="test_tool",
        arguments={},
    )

    result = await execute_tool(tool_call, context, config, None, _create_emit([]))

    assert result.is_error


# =============================================================================
# Run Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_loop_basic(model, mock_assistant_message):
    """Test basic run_loop execution."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    config = _create_config(model)
    events = []

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant_message)):
        await run_loop(context, [], config, None, _create_emit(events))

    assert len(events) >= 2  # message_start and message_end


@pytest.mark.asyncio
async def test_run_loop_with_should_stop(model, mock_assistant_message):
    """Test run_loop with should_stop_after_turn callback."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    stop_called = [False]

    async def should_stop(ctx):
        stop_called[0] = True
        return True

    config = _create_config(model, should_stop_after_turn=should_stop)
    events = []

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant_message)):
        await run_loop(context, [], config, None, _create_emit(events))

    assert stop_called[0]


@pytest.mark.asyncio
async def test_run_loop_with_transform_context(model, mock_assistant_message):
    """Test run_loop with context transformation."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    transformed = [False]

    async def transform_context(messages):
        transformed[0] = True
        return messages

    config = _create_config(model, transform_context=transform_context)
    events = []

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant_message)):
        await run_loop(context, [], config, None, _create_emit(events))

    assert transformed[0]


@pytest.mark.asyncio
async def test_run_loop_with_tool_calls(model):
    """Test run_loop with tool calls."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Tool result")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    assistant_with_tool = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Using tool"),
            ToolCall(
                type="toolCall",
                id="call_1",
                name="test_tool",
                arguments={"arg": "value"},
            ),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE,
    )

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _create_mock_stream(assistant_with_tool)
        else:
            final_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(final_msg)

    config = _create_config(model)
    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert call_count[0] == 2  # Tool call + final response


@pytest.mark.asyncio
async def test_run_loop_parallel_tool_execution(model):
    """Test run_loop with parallel tool execution."""
    async def execute_fn(tool_id, args, signal):
        await asyncio.sleep(0.01)  # Simulate work
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Result {tool_id}")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    assistant_with_tools = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Using tools"),
            ToolCall(
                type="toolCall",
                id="call_1",
                name="test_tool",
                arguments={},
            ),
            ToolCall(
                type="toolCall",
                id="call_2",
                name="test_tool",
                arguments={},
            ),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE,
    )

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            stream = AssistantMessageEventStream()
            stream.push(DoneEvent(reason=StopReason.TOOL_USE, message=assistant_with_tools))
            stream.end(assistant_with_tools)
            return stream
        else:
            final_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(final_msg)

    config = _create_config(model, tool_execution=ToolExecutionMode.PARALLEL)
    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    # Check that we have tool results
    tool_results = [e for e in events if hasattr(e, "type") and "tool" in e.type]
    assert len(tool_results) > 0


@pytest.mark.asyncio
async def test_run_loop_sequential_tool_execution(model):
    """Test run_loop with sequential tool execution."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Result {tool_id}")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    assistant_with_tools = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Using tools"),
            ToolCall(
                type="toolCall",
                id="call_1",
                name="test_tool",
                arguments={},
            ),
            ToolCall(
                type="toolCall",
                id="call_2",
                name="test_tool",
                arguments={},
            ),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE,
    )

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            stream = AssistantMessageEventStream()
            stream.push(DoneEvent(reason=StopReason.TOOL_USE, message=assistant_with_tools))
            stream.end(assistant_with_tools)
            return stream
        else:
            final_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(final_msg)

    config = _create_config(model, tool_execution=ToolExecutionMode.SEQUENTIAL)
    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_run_loop_with_steering_messages(model, mock_assistant_message):
    """Test run_loop with get_steering_messages callback."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    steering_called = [False]
    steering_count = [0]

    async def get_steering():
        steering_count[0] += 1
        if steering_count[0] == 1:
            steering_called[0] = True
            return [UserMessage(content="Steer this")]
        return None

    config = _create_config(model, get_steering_messages=get_steering)
    events = []

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        return _create_mock_stream(mock_assistant_message)

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert steering_called[0]
    # Should be called at least twice due to steering message
    assert call_count[0] >= 2


@pytest.mark.asyncio
async def test_run_loop_should_stop_context(model, mock_assistant_message):
    """Test run_loop with should_stop_after_turn receiving context."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    received_ctx = [None]

    async def should_stop(ctx):
        received_ctx[0] = ctx
        return True

    config = _create_config(model, should_stop_after_turn=should_stop)
    events = []

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant_message)):
        await run_loop(context, [], config, None, _create_emit(events))

    # Verify context was passed correctly
    assert received_ctx[0] is not None
    assert hasattr(received_ctx[0], 'message')
    assert hasattr(received_ctx[0], 'tool_results')
    assert hasattr(received_ctx[0], 'context')
    assert hasattr(received_ctx[0], 'new_messages')


@pytest.mark.asyncio
async def test_run_loop_should_stop_false(model, mock_assistant_message):
    """Test run_loop with should_stop_after_turn returning False."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    call_count = [0]

    async def should_stop(ctx):
        call_count[0] += 1
        return call_count[0] >= 2  # Stop after 2 calls

    config = _create_config(model, should_stop_after_turn=should_stop)
    events = []

    call_count_stream = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count_stream[0] += 1
        return _create_mock_stream(mock_assistant_message)

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    # Should have been called at least once
    assert call_count[0] >= 1


@pytest.mark.asyncio
async def test_run_loop_steering_adds_messages(model, mock_assistant_message):
    """Test run_loop with steering messages being added."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    steering_count = [0]

    async def get_steering():
        steering_count[0] += 1
        if steering_count[0] == 1:
            return [UserMessage(content="Steering message")]
        return None

    config = _create_config(model, get_steering_messages=get_steering)
    events = []

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        return _create_mock_stream(mock_assistant_message)

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    # Steering should have been checked
    assert steering_count[0] >= 1


@pytest.mark.asyncio
async def test_run_loop_steering_none(model, mock_assistant_message):
    """Test run_loop with steering returning None."""
    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
    )

    async def get_steering():
        return None

    config = _create_config(model, get_steering_messages=get_steering)
    events = []

    with patch("pypi_agent.loop.stream_simple", return_value=_create_mock_stream(mock_assistant_message)):
        await run_loop(context, [], config, None, _create_emit(events))

    # Should complete without issues
    assert True


@pytest.mark.asyncio
async def test_run_loop_with_tools_and_should_stop(model):
    """Test run_loop with tool calls and should_stop_after_turn."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Tool result")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    assistant_with_tool = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Using tool"),
            ToolCall(
                type="toolCall",
                id="call_1",
                name="test_tool",
                arguments={"arg": "value"},
            ),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE,
    )

    stop_called = [False]

    async def should_stop(ctx):
        stop_called[0] = True
        return True

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _create_mock_stream(assistant_with_tool)
        else:
            final_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(final_msg)

    config = _create_config(model, should_stop_after_turn=should_stop)
    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert stop_called[0]


@pytest.mark.asyncio
async def test_run_loop_with_tools_and_steering(model):
    """Test run_loop with tool calls and get_steering_messages."""
    async def execute_fn(tool_id, args, signal):
        return AgentToolResult(
            content=[TextContent(type="text", text="Tool result")],
        )

    tool = AgentTool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        execute=execute_fn,
    )

    context = AgentContext(
        system_prompt="Test",
        messages=[UserMessage(content="Hello")],
        tools=[tool],
    )

    assistant_with_tool = AssistantMessage(
        role="assistant",
        content=[
            TextContent(type="text", text="Using tool"),
            ToolCall(
                type="toolCall",
                id="call_1",
                name="test_tool",
                arguments={"arg": "value"},
            ),
        ],
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
        model="test",
        usage=Usage(),
        stop_reason=StopReason.TOOL_USE,
    )

    steering_called = [False]
    steering_count = [0]

    async def get_steering():
        steering_count[0] += 1
        if steering_count[0] == 1:
            steering_called[0] = True
            return [UserMessage(content="Steering")]
        return None

    call_count = [0]

    def mock_stream_fn(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _create_mock_stream(assistant_with_tool)
        else:
            final_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Final")],
                api=Api.ANTHROPIC_MESSAGES,
                provider="anthropic",
                model="test",
                usage=Usage(),
                stop_reason=StopReason.END,
            )
            return _create_mock_stream(final_msg)

    config = _create_config(model, get_steering_messages=get_steering)
    events = []

    with patch("pypi_agent.loop.stream_simple", side_effect=mock_stream_fn):
        await run_loop(context, [], config, None, _create_emit(events))

    assert steering_called[0]
