## ADDED Requirements

### Requirement: Agent state management
The system SHALL maintain agent state including system prompt, messages, tools, and model configuration.

#### Scenario: Initialize agent state
- **WHEN** agent is created
- **THEN** system initializes empty message history
- **AND** sets system prompt, model, and thinking level

#### Scenario: Update agent messages
- **WHEN** messages are assigned to agent state
- **THEN** system copies message array to prevent mutation
- **AND** validates message types before storage

### Requirement: Agent loop execution
The system SHALL execute an agent loop that processes prompts and generates responses.

#### Scenario: Run agent with single prompt
- **WHEN** user provides a prompt message
- **THEN** agent adds prompt to context
- **AND** generates assistant response via LLM call

#### Scenario: Handle tool calls in response
- **WHEN** assistant response contains tool calls
- **THEN** agent executes tools in configured mode (parallel/sequential)
- **AND** adds tool results to context for next turn

### Requirement: Turn management
The system SHALL manage conversation turns with clear start/end boundaries.

#### Scenario: Start new turn
- **WHEN** agent begins processing
- **THEN** system emits turn_start event
- **AND** tracks turn state throughout processing

#### Scenario: Complete turn with tool results
- **WHEN** tool execution finishes
- **THEN** system emits turn_end event with assistant message and tool results
- **AND** prepares for next turn or termination

### Requirement: Tool execution modes
The system SHALL support sequential and parallel tool execution modes.

#### Scenario: Sequential tool execution
- **WHEN** tool execution mode is "sequential"
- **THEN** system executes each tool call one at a time
- **AND** waits for previous tool to complete before starting next

#### Scenario: Parallel tool execution
- **WHEN** tool execution mode is "parallel"
- **THEN** system prepares all tool calls
- **AND** executes allowed tools concurrently

### Requirement: Agent events
The system SHALL emit events for agent lifecycle, messages, and tool execution.

#### Scenario: Emit agent lifecycle events
- **WHEN** agent loop starts and ends
- **THEN** system emits agent_start and agent_end events
- **AND** includes final messages in agent_end

#### Scenario: Emit message events
- **WHEN** message is added or streamed
- **THEN** system emits message_start, message_update, message_end events
- **AND** provides message content in each event

#### Scenario: Emit tool execution events
- **WHEN** tool is executed
- **THEN** system emits tool_execution_start, tool_execution_update, tool_execution_end
- **AND** includes tool name, arguments, and result