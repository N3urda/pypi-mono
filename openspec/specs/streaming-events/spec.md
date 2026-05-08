## ADDED Requirements

### Requirement: Event types for streaming
The system SHALL define standard event types for streaming assistant responses.

#### Scenario: Text delta event
- **WHEN** LLM emits text increment
- **THEN** system emits text_delta event with partial text
- **AND** includes content index for multi-part responses

#### Scenario: Thinking delta event
- **WHEN** LLM emits thinking/reasoning increment
- **THEN** system emits thinking_delta event with partial thinking
- **AND** marks redacted thinking when safety filters apply

#### Scenario: Tool call delta event
- **WHEN** LLM emits partial tool call arguments
- **THEN** system emits toolcall_delta event
- **AND** provides partial arguments as JSON string

#### Scenario: Tool call complete event
- **WHEN** tool call arguments are complete
- **THEN** system emits toolcall_end event
- **AND** includes validated tool call object

### Requirement: Stream protocol completion
The system SHALL emit final event when response completes.

#### Scenario: Normal completion
- **WHEN** LLM finishes response normally
- **THEN** system emits done event with stop reason "end" or "stop"
- **AND** includes complete AssistantMessage

#### Scenario: Tool use stop
- **WHEN** LLM stops to request tool execution
- **THEN** system emits done event with stop reason "tool_use"
- **AND** assistant message contains tool calls

#### Scenario: Error completion
- **WHEN** response fails
- **THEN** system emits done event with stop reason "error"
- **AND** includes error message in assistant message

### Requirement: Async iterator pattern
The system SHALL provide event streams as async iterators.

#### Scenario: Iterate stream events
- **WHEN** consumer uses async for on stream
- **THEN** system yields events in order
- **AND** completes when done event is emitted

#### Scenario: Collect final result
- **WHEN** consumer calls stream.result()
- **THEN** system consumes all events
- **AND** returns complete AssistantMessage

### Requirement: Stream buffering
The system SHALL buffer events to handle slow consumers.

#### Scenario: Fast producer slow consumer
- **WHEN** LLM produces events faster than consumer processes
- **THEN** system buffers events in asyncio.Queue
- **AND** prevents event loss

#### Scenario: Consumer cancellation
- **WHEN** consumer cancels iteration mid-stream
- **THEN** stream gracefully closes
- **AND** releases buffer resources