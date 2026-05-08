"""
Agent loop implementation.

This module provides:
- agent_loop() for starting new agent runs
- agent_loop_continue() for retrying/resuming
- Event emission for agent lifecycle
- Tool execution with parallel/sequential modes
"""

from typing import AsyncIterator, Callable, Optional, Any
import asyncio

from pypi_ai import stream_simple, Context, Model
from pypi_ai.types import AssistantMessage, ToolResultMessage, StopReason
from pypi_ai.event_stream import AssistantMessageEventStream

from pypi_agent.types import (
    AgentMessage,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    ToolExecutionMode,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    BeforeToolCallContext,
    AfterToolCallContext,
    BeforeToolCallResult,
    AfterToolCallResult,
)


async def emit_event(
    event_sink: Callable[[AgentEvent], Any],
    event: AgentEvent
) -> None:
    """Emit an event to the sink."""
    result = event_sink(event)
    if asyncio.iscoroutine(result):
        await result


async def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.CancelledError] = None,
) -> AsyncIterator[AgentEvent]:
    """
    Start an agent loop with new prompts.

    Args:
        prompts: Initial prompt messages
        context: Agent context with state
        config: Loop configuration
        signal: Optional cancellation signal

    Yields:
        AgentEvent events for UI updates
    """
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await queue.put(event)

    async def run() -> list[AgentMessage]:
        new_messages = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=[*context.messages, *prompts],
            tools=context.tools,
        )

        await emit(AgentStartEvent())
        await emit(TurnStartEvent())

        for prompt in prompts:
            await emit(MessageStartEvent(message=prompt))
            await emit(MessageEndEvent(message=prompt))

        await run_loop(
            current_context,
            new_messages,
            config,
            signal,
            emit,
        )

        await emit(AgentEndEvent(messages=new_messages))
        return new_messages

    task = asyncio.create_task(run())

    while True:
        event = await queue.get()
        yield event
        if event.type == "agent_end":
            break

    await task


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.CancelledError] = None,
) -> AsyncIterator[AgentEvent]:
    """
    Continue an agent loop from current context.

    Used for retries - context already has user message or tool results.

    Args:
        context: Agent context with existing messages
        config: Loop configuration
        signal: Optional cancellation signal

    Yields:
        AgentEvent events for UI updates
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last_msg = context.messages[-1]
    if hasattr(last_msg, "role") and last_msg.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await queue.put(event)

    async def run() -> list[AgentMessage]:
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        await emit(AgentStartEvent())
        await emit(TurnStartEvent())

        await run_loop(
            current_context,
            new_messages,
            config,
            signal,
            emit,
        )

        await emit(AgentEndEvent(messages=new_messages))
        return new_messages

    task = asyncio.create_task(run())

    while True:
        event = await queue.get()
        yield event
        if event.type == "agent_end":
            break

    await task


async def run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: Optional[asyncio.CancelledError],
    emit: Callable[[AgentEvent], Any],
) -> None:
    """
    Core agent loop logic.

    Handles:
    - LLM calls with streamSimple
    - Tool call detection and execution
    - Turn management
    - Hook invocation
    """
    while True:
        # Convert messages for LLM
        llm_messages = await config.convert_to_llm(context.messages)

        # Apply context transformation if configured
        if config.transform_context:
            transformed = await config.transform_context(llm_messages)
            llm_messages = transformed

        # Build LLM context
        llm_context = Context(
            system_prompt=context.system_prompt,
            messages=llm_messages,
            tools=[],
        )

        # Stream response
        response_stream = stream_simple(config.model, llm_context)

        # Collect response
        assistant_msg: Optional[AssistantMessage] = None
        async for event in response_stream:
            if event.type == "done":
                assistant_msg = event.message

        if assistant_msg is None:
            break

        # Add assistant message
        context.messages.append(assistant_msg)
        new_messages.append(assistant_msg)

        await emit(MessageStartEvent(message=assistant_msg))
        await emit(MessageEndEvent(message=assistant_msg))

        # Check for tool calls
        tool_calls = [
            c for c in assistant_msg.content
            if hasattr(c, "type") and c.type == "toolCall"
        ]

        if not tool_calls:
            # No tool calls, check for steering or finish
            if config.should_stop_after_turn:
                ctx_obj = type("Ctx", (), {
                    "message": assistant_msg,
                    "tool_results": [],
                    "context": context,
                    "new_messages": new_messages,
                })()
                if await config.should_stop_after_turn(ctx_obj):
                    break

            # Check steering messages
            if config.get_steering_messages:
                steering = await config.get_steering_messages()
                if steering:
                    context.messages.extend(steering)
                    continue

            break

        # Execute tools
        tool_results: list[ToolResultMessage] = []

        if config.tool_execution == ToolExecutionMode.SEQUENTIAL:
            for tc in tool_calls:
                result_msg = await execute_tool(
                    tc, context, config, signal, emit
                )
                tool_results.append(result_msg)
                context.messages.append(result_msg)
                new_messages.append(result_msg)
        else:
            # Parallel execution
            tasks = [
                execute_tool(tc, context, config, signal, emit)
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks)
            for result_msg in results:
                tool_results.append(result_msg)
                context.messages.append(result_msg)
                new_messages.append(result_msg)

        # Emit turn end
        await emit(TurnEndEvent(message=assistant_msg, tool_results=tool_results))

        # Check if we should stop after this turn
        if config.should_stop_after_turn:
            ctx_obj = type("Ctx", (), {
                "message": assistant_msg,
                "tool_results": tool_results,
                "context": context,
                "new_messages": new_messages,
            })()
            if await config.should_stop_after_turn(ctx_obj):
                break

        # Check steering messages
        if config.get_steering_messages:
            steering = await config.get_steering_messages()
            if steering:
                context.messages.extend(steering)
                await emit(TurnStartEvent())


async def execute_tool(
    tool_call: Any,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.CancelledError],
    emit: Callable[[AgentEvent], Any],
) -> ToolResultMessage:
    """Execute a single tool call."""
    tool_id = tool_call.id
    tool_name = tool_call.name
    args = tool_call.arguments

    # Find the tool
    tool: Optional[AgentTool] = None
    for t in context.tools:
        if t.name == tool_name:
            tool = t
            break

    if tool is None:
        return ToolResultMessage(
            tool_call_id=tool_id,
            content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            is_error=True,
        )

    # Prepare arguments if hook exists
    if tool.prepare_arguments:
        args = tool.prepare_arguments(args)

    # Emit start event
    await emit_event(emit, ToolExecutionStartEvent(
        tool_call_id=tool_id,
        tool_name=tool_name,
        args=args,
    ))

    # Before tool call hook
    if config.before_tool_call:
        ctx = BeforeToolCallContext(
            assistant_message=context.messages[-1] if context.messages else None,
            tool_call=tool_call,
            args=args,
            context=context,
        )
        result = await config.before_tool_call(ctx, signal)
        if result and result.block:
            error_msg = ToolResultMessage(
                tool_call_id=tool_id,
                content=[{"type": "text", "text": result.reason or "Tool blocked"}],
                is_error=True,
            )
            await emit_event(emit, ToolExecutionEndEvent(
                tool_call_id=tool_id,
                tool_name=tool_name,
                result=error_msg,
                is_error=True,
            ))
            return error_msg

    # Execute
    is_error = False
    result_content = []
    result_details = None

    try:
        if tool.execute:
            result: AgentToolResult = await tool.execute(tool_id, args, signal)
            result_content = result.content
            result_details = result.details
        else:
            result_content = [{"type": "text", "text": "Tool has no execute function"}]
            is_error = True
    except asyncio.CancelledError:
        result_content = [{"type": "text", "text": "Tool execution cancelled"}]
        is_error = True
    except Exception as e:
        result_content = [{"type": "text", "text": f"Error: {str(e)}"}]
        is_error = True

    # After tool call hook
    if config.after_tool_call:
        ctx = AfterToolCallContext(
            assistant_message=context.messages[-1] if context.messages else None,
            tool_call=tool_call,
            args=args,
            result=AgentToolResult(content=result_content, details=result_details),
            is_error=is_error,
            context=context,
        )
        hook_result = await config.after_tool_call(ctx, signal)
        if hook_result:
            if hook_result.content is not None:
                result_content = hook_result.content
            if hook_result.details is not None:
                result_details = hook_result.details
            is_error = hook_result.is_error

    # Build result message
    result_msg = ToolResultMessage(
        tool_call_id=tool_id,
        content=result_content,
        is_error=is_error,
        details=result_details,
    )

    await emit_event(emit, ToolExecutionEndEvent(
        tool_call_id=tool_id,
        tool_name=tool_name,
        result=result_msg,
        is_error=is_error,
    ))

    return result_msg