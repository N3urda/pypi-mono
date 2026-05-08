## ADDED Requirements

### Requirement: CLI entry point
The system SHALL provide a command-line interface entry point for the coding agent.

#### Scenario: Start interactive mode
- **WHEN** user runs CLI without arguments
- **THEN** system starts interactive REPL mode
- **AND** displays welcome message and prompt

#### Scenario: Process single prompt
- **WHEN** user runs CLI with prompt argument
- **THEN** system processes prompt and outputs response
- **AND** exits after response completes

### Requirement: Configuration management
The system SHALL support configuration via config files and CLI arguments.

#### Scenario: Load configuration file
- **WHEN** CLI starts
- **THEN** system loads configuration from ~/.pypi/settings.json
- **AND** applies default values for missing settings

#### Scenario: Override via CLI arguments
- **WHEN** user provides CLI arguments
- **THEN** system overrides config file settings
- **AND** validates argument values

### Requirement: Built-in tools
The system SHALL provide built-in tools for file operations and command execution.

#### Scenario: Bash tool execution
- **WHEN** agent calls Bash tool with command
- **THEN** system executes command in sandbox
- **AND** returns stdout/stderr as tool result

#### Scenario: Read tool execution
- **WHEN** agent calls Read tool with file path
- **THEN** system reads file content
- **AND** returns content with line numbers

#### Scenario: Write tool execution
- **WHEN** agent calls Write tool with file path and content
- **THEN** system writes content to file
- **AND** creates file if it doesn't exist

#### Scenario: Edit tool execution
- **WHEN** agent calls Edit tool with old_string and new_string
- **THEN** system performs exact string replacement
- **AND** validates old_string uniqueness

### Requirement: Session management
The system SHALL support session persistence and recovery.

#### Scenario: Save session
- **WHEN** session ends or user requests save
- **THEN** system saves message history to file
- **AND** includes model, thinking level, and configuration

#### Scenario: Resume session
- **WHEN** user starts CLI with session argument
- **THEN** system loads previous session state
- **AND** restores message history and configuration

### Requirement: Interactive mode
The system SHALL provide interactive prompt mode with real-time streaming.

#### Scenario: Stream response in TUI
- **WHEN** user submits prompt in interactive mode
- **THEN** system streams response events to TUI
- **AND** displays text/thinking/tool calls incrementally

#### Scenario: Handle user interruption
- **WHEN** user cancels response mid-stream
- **THEN** system aborts current request
- **AND** returns to prompt ready state