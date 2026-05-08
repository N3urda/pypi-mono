## 1. Project Setup

- [x] 1.1 Initialize monorepo structure with uv workspace
- [x] 1.2 Create packages/pypi-ai directory structure
- [x] 1.3 Create packages/pypi-agent directory structure
- [x] 1.4 Create packages/pypi-cli directory structure
- [x] 1.5 Configure root pyproject.toml with workspace definitions
- [x] 1.6 Add core dependencies (pydantic, anthropic, openai)
- [x] 1.7 Configure pytest and pytest-asyncio for testing
- [x] 1.8 Set up ruff and mypy configuration

## 2. Core Types (pypi-ai)

- [x] 2.1 Define Api and Provider enums
- [x] 2.2 Define ThinkingLevel and StopReason enums
- [x] 2.3 Create TextContent, ThinkingContent, ImageContent models
- [x] 2.4 Create ToolCall model with id, name, arguments
- [x] 2.5 Create Usage and Cost models for token tracking
- [x] 2.6 Create AssistantMessage model with content array
- [x] 2.7 Create UserMessage and ToolResultMessage models
- [x] 2.8 Create Context model for LLM requests
- [x] 2.9 Create StreamOptions model with configurable parameters
- [x] 2.10 Create Model configuration model with provider, api, cost

## 3. Event Stream Protocol

- [x] 3.1 Define AssistantMessageEvent union type (TextDelta, ThinkingDelta, etc.)
- [x] 3.2 Implement AssistantMessageEventStream class with asyncio.Queue
- [x] 3.3 Implement push() method for event emission
- [x] 3.4 Implement end() method for stream completion
- [x] 3.5 Implement async __aiter__ for event iteration
- [x] 3.6 Implement result() method for final message collection
- [x] 3.7 Add unit tests for event stream behavior

## 4. Provider Registry

- [x] 4.1 Create ApiProvider protocol/interface
- [x] 4.2 Implement provider registry as Dict[Api, ApiProvider]
- [x] 4.3 Create register_provider() function
- [x] 4.4 Create get_provider() function
- [x] 4.5 Implement lazy loading pattern with importlib
- [x] 4.6 Add functools.lru_cache for provider module caching
- [x] 4.7 Add unit tests for registry operations

## 5. Anthropic Provider

- [x] 5.1 Install and configure anthropic Python SDK
- [x] 5.2 Implement streamAnthropic() function
- [x] 5.3 Implement streamSimpleAnthropic() function
- [x] 5.4 Handle streaming text_delta events
- [x] 5.5 Handle streaming thinking_delta events (Claude extended thinking)
- [x] 5.6 Handle tool_call events with partial JSON parsing
- [x] 5.7 Implement prompt caching support
- [x] 5.8 Map Anthropic response to unified AssistantMessage
- [x] 5.9 Implement error handling with error events
- [x] 5.10 Add integration tests with mock API responses

## 6. OpenAI Provider

- [x] 6.1 Install and configure openai Python SDK
- [x] 6.2 Implement streamOpenAI() function for Chat Completions
- [x] 6.3 Handle streaming text_delta events
- [x] 6.4 Handle tool_call events with function calling
- [x] 6.5 Support custom base_url for OpenAI-compatible providers
- [x] 6.6 Map OpenAI response to unified AssistantMessage
- [x] 6.7 Implement error handling with error events
- [x] 6.8 Add integration tests for GPT models
- [x] 6.9 Add tests for OpenAI-compatible providers (DeepSeek, Groq)

## 7. Google Gemini Provider

- [x] 7.1 Install and configure google-generativeai SDK
- [x] 7.2 Implement streamGoogle() function
- [x] 7.3 Handle streaming text_delta events
- [x] 7.4 Handle thinking content from Gemini models
- [x] 7.5 Handle tool_call events
- [x] 7.6 Map Gemini response to unified AssistantMessage
- [x] 7.7 Add integration tests for Gemini models

## 8. Mistral Provider

- [x] 8.1 Install and configure mistralai SDK
- [x] 8.2 Implement streamMistral() function
- [x] 8.3 Handle streaming events from Mistral API
- [x] 8.4 Map Mistral response to unified AssistantMessage
- [x] 8.5 Add integration tests for Mistral models

## 9. Provider Registration

- [x] 9.1 Create register_builtins.py with all provider registrations
- [x] 9.2 Register Anthropic provider on module import
- [x] 9.3 Register OpenAI provider on module import
- [x] 9.4 Register Google provider on module import
- [x] 9.5 Register Mistral provider on module import
- [x] 9.6 Create stream() public API function
- [x] 9.7 Create streamSimple() public API function
- [x] 9.8 Create complete() and completeSimple() convenience functions
- [x] 9.9 Create __init__.py with public exports

## 10. Agent Types (pypi-agent)

- [x] 10.1 Define AgentMessage union type (extends Message)
- [x] 10.2 Create AgentContext model with system_prompt, messages, tools
- [x] 10.3 Create AgentTool model with label, parameters, execute function
- [x] 10.4 Create AgentToolResult model with content, details, terminate flag
- [x] 10.5 Define AgentEvent union type (agent_start, agent_end, etc.)
- [x] 10.6 Create AgentLoopConfig model with model, tools, hooks

## 11. Agent State

- [x] 11.1 Implement AgentState class
- [x] 11.2 Add system_prompt property
- [x] 11.3 Add model and thinking_level properties
- [x] 11.4 Add tools property with array copy semantics
- [x] 11.5 Add messages property with array copy semantics
- [x] 11.6 Add isStreaming readonly property
- [x] 11.7 Add pendingToolCalls readonly property
- [x] 11.8 Add errorMessage readonly property
- [x] 11.9 Add unit tests for AgentState

## 12. Agent Loop

- [x] 12.1 Implement agent_loop() main function
- [x] 12.2 Implement agent_loop_continue() for retries
- [x] 12.3 Implement run_agent_loop() async core logic
- [x] 12.4 Implement event emission (agent_start, agent_end)
- [x] 12.5 Implement turn management (turn_start, turn_end)
- [x] 12.6 Implement message event emission
- [x] 12.7 Implement LLM call integration with streamSimple
- [x] 12.8 Implement tool call detection and extraction
- [x] 12.9 Implement sequential tool execution mode
- [x] 12.10 Implement parallel tool execution mode
- [x] 12.11 Implement tool result collection and formatting
- [x] 12.12 Implement shouldStopAfterTurn hook
- [x] 12.13 Implement getSteeringMessages hook
- [x] 12.14 Add unit tests for agent loop

## 13. Tool Execution Hooks

- [x] 13.1 Implement beforeToolCall hook invocation
- [x] 13.2 Implement afterToolCall hook invocation
- [x] 13.3 Handle block result from beforeToolCall
- [x] 13.4 Handle content/details override from afterToolCall
- [x] 13.5 Pass AbortSignal equivalent to hooks
- [x] 13.6 Add tests for hook behavior

## 14. Tool Argument Validation

- [x] 14.1 Implement validate_tool_arguments() function
- [x] 14.2 Validate against Pydantic parameter model
- [x] 14.3 Handle validation errors with error result
- [x] 14.4 Support prepareArguments transformation hook
- [x] 14.5 Add tests for validation scenarios

## 15. CLI Structure (pypi-cli)

- [x] 15.1 Create CLI entry point module
- [x] 15.2 Implement argparse/rich-click argument parsing
- [x] 15.3 Create interactive mode entry point
- [x] 15.4 Create single-prompt mode entry point
- [x] 15.5 Create --help and --version support

## 16. Configuration

- [x] 16.1 Create SettingsManager class
- [x] 16.2 Implement ~/.pypi/settings.json loading
- [x] 16.3 Implement default values for settings
- [x] 16.4 Implement CLI argument override merging
- [x] 16.5 Add model and provider configuration
- [x] 16.6 Add thinking level configuration
- [x] 16.7 Add tests for configuration management

## 17. Session Management

- [x] 17.1 Create SessionManager class
- [x] 17.2 Implement session file format (JSON)
- [x] 17.3 Implement session save functionality
- [x] 17.4 Implement session load/resume functionality
- [x] 17.5 Track message history in session
- [x] 17.6 Track configuration in session
- [x] 17.7 Add tests for session persistence

## 18. Built-in Tools - Bash

- [x] 18.1 Create Bash tool definition with Pydantic parameters
- [x] 18.2 Implement Bash tool execute function
- [x] 18.3 Handle command execution with subprocess
- [x] 18.4 Capture stdout and stderr
- [x] 18.5 Handle timeout and cancellation
- [x] 18.6 Return formatted result content
- [x] 18.7 Add tests for Bash tool

## 19. Built-in Tools - File Operations

- [x] 19.1 Create Read tool definition
- [x] 19.2 Implement Read tool with line numbers
- [x] 19.3 Create Write tool definition
- [x] 19.4 Implement Write tool with content validation
- [x] 19.5 Create Edit tool definition
- [x] 19.6 Implement Edit tool with string replacement
- [x] 19.7 Validate old_string uniqueness
- [x] 19.8 Create Grep tool definition
- [x] 19.9 Implement Grep tool with pattern matching
- [x] 19.10 Create Find tool definition
- [x] 19.11 Implement Find tool with path search
- [x] 19.12 Add tests for all file tools

## 20. Interactive Mode

- [x] 20.1 Implement interactive REPL loop
- [x] 20.2 Create prompt input handling
- [x] 20.3 Implement streaming display with rich/textual
- [x] 20.4 Handle user interruption (Ctrl+C)
- [x] 20.5 Display tool execution progress
- [x] 20.6 Display thinking content when available
- [x] 20.7 Add command history support
- [x] 20.8 Add tests for interactive mode

## 21. Documentation and Polish

- [x] 21.1 Write README.md with installation instructions
- [x] 21.2 Write API documentation for pypi-ai
- [x] 21.3 Write usage guide for CLI
- [x] 21.4 Add example scripts for common use cases
- [x] 21.5 Verify 80%+ test coverage
- [x] 21.6 Run mypy type checking on all packages
- [x] 21.7 Run ruff linting and fix issues
- [x] 21.8 Create CHANGELOG.md for version tracking

## 22. Packaging

- [x] 22.1 Configure pypi-ai pyproject.toml for PyPI
- [x] 22.2 Configure pypi-agent pyproject.toml for PyPI
- [x] 22.3 Configure pypi-cli pyproject.toml for PyPI
- [x] 22.4 Add entry_points for CLI command
- [x] 22.5 Test package installation in fresh environment
- [x] 22.6 Verify all dependencies are correctly declared