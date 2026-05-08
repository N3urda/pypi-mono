## ADDED Requirements

### Requirement: Tool definition with schema
The system SHALL support tool definition with Pydantic parameter schema and execution function.

#### Scenario: Define tool with parameters
- **WHEN** developer creates a tool
- **THEN** tool includes name, description, and Pydantic parameter model
- **AND** parameter model generates JSON Schema for LLM

#### Scenario: Define tool with optional parameters
- **WHEN** tool has optional parameters
- **THEN** JSON Schema marks fields as optional
- **AND** LLM may omit optional arguments in tool calls

### Requirement: Tool argument validation
The system SHALL validate tool arguments against schema before execution.

#### Scenario: Valid arguments passed
- **WHEN** tool call has valid arguments matching schema
- **THEN** system executes tool with validated parameters
- **AND** tool receives typed Pydantic model instance

#### Scenario: Invalid arguments passed
- **WHEN** tool call has invalid arguments
- **THEN** system returns error tool result
- **AND** includes validation error message

### Requirement: Tool execution hooks
The system SHALL support before and after tool execution hooks for customization.

#### Scenario: Block tool execution
- **WHEN** beforeToolCall hook returns block=true
- **THEN** system prevents tool execution
- **AND** returns error tool result with block reason

#### Scenario: Modify tool result
- **WHEN** afterToolCall hook returns modified content
- **THEN** system replaces original result content
- **AND** preserves other fields unless explicitly overridden

### Requirement: Tool result format
The system SHALL return tool results with content and optional details.

#### Scenario: Successful tool execution
- **WHEN** tool executes successfully
- **THEN** tool returns content array (text/image)
- **AND** optionally includes structured details

#### Scenario: Tool execution error
- **WHEN** tool execution raises exception
- **THEN** system captures error message
- **AND** returns tool result with isError=true

### Requirement: Abort signal support
The system SHALL support cancellation of tool execution via abort signal.

#### Scenario: Cancel running tool
- **WHEN** abort signal is triggered during tool execution
- **THEN** tool receives asyncio.CancelledError
- **AND** gracefully terminates execution

#### Scenario: Tool respects abort signal
- **WHEN** tool is properly implemented
- **THEN** tool checks signal periodically during long operations
- **AND** raises CancelledError when signal triggers