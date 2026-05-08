## ADDED Requirements

### Requirement: Unified LLM API interface
The system SHALL provide a unified interface for calling multiple LLM providers through a single abstraction layer.

#### Scenario: Stream response from Anthropic Claude
- **WHEN** user requests streaming response using Anthropic Claude model
- **THEN** system returns an async iterator of text/thinking/tool-call events
- **AND** events follow the standard AssistantMessageEvent protocol

#### Scenario: Stream response from OpenAI GPT
- **WHEN** user requests streaming response using OpenAI GPT model
- **THEN** system returns an async iterator of text/tool-call events
- **AND** response format matches Anthropic response structure

### Requirement: Provider registration
The system SHALL support dynamic registration of LLM providers with lazy loading.

#### Scenario: Register custom provider
- **WHEN** developer registers a new provider with stream function
- **THEN** system adds provider to registry
- **AND** provider is available for subsequent requests

#### Scenario: Lazy load provider module
- **WHEN** first request to a provider is made
- **THEN** system imports provider module on-demand
- **AND** module is cached for subsequent requests

### Requirement: Model configuration
The system SHALL support model configuration with provider, API type, and parameters.

#### Scenario: Configure model with custom settings
- **WHEN** user creates a model configuration
- **THEN** system validates model against provider capabilities
- **AND** model includes context window, max tokens, cost information

#### Scenario: Use custom base URL for OpenAI-compatible providers
- **WHEN** user specifies custom base URL
- **THEN** system routes requests to custom endpoint
- **AND** maintains OpenAI API compatibility

### Requirement: Token and cost tracking
The system SHALL track token usage and costs for each request.

#### Scenario: Track input/output tokens
- **WHEN** LLM request completes
- **THEN** system records input tokens, output tokens, cache read/write
- **AND** calculates cost based on model pricing

#### Scenario: Aggregate usage across conversation
- **WHEN** multiple requests are made in a session
- **THEN** system provides cumulative token and cost totals
- **AND** maintains per-message breakdown

### Requirement: Error handling
The system SHALL encode errors in the stream protocol instead of throwing exceptions.

#### Scenario: Provider returns error response
- **WHEN** LLM provider returns an error
- **THEN** stream emits error event with stop reason "error"
- **AND** includes error message in final AssistantMessage

#### Scenario: Request is aborted
- **WHEN** user cancels request mid-stream
- **THEN** stream emits aborted event with stop reason "aborted"
- **AND** gracefully closes stream without exception