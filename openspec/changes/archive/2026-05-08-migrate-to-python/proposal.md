## Why

The pi-mono TypeScript project provides a sophisticated AI Agent framework with unified LLM API abstraction, agent runtime, and coding agent CLI. Migrating to Python enables:
- Better integration with Python's AI/ML ecosystem (PyTorch, LangChain, etc.)
- Accessibility to a broader developer audience
- Leveraging Python's async/await patterns for streaming responses

This migration creates a Python monorepo (pypi-mono) that preserves the core architecture while adapting to Python idioms and best practices.

## What Changes

### Core Framework Migration
- **Type System**: TypeScript TypeBox → Python Pydantic v2 for runtime validation and JSON Schema generation
- **Streaming Protocol**: AsyncIterable/EventStream → AsyncIterator with asyncio.Queue
- **Provider Registry**: Module-level lazy loading with functools caching
- **Agent Loop**: Tool execution loop with parallel/sequential modes

### Packages Structure
- `pypi-ai`: LLM API abstraction layer (equivalent to `packages/ai`)
- `pypi-agent`: Agent runtime and tool system (equivalent to `packages/agent`)
- `pypi-cli`: Coding agent CLI application (equivalent to `packages/coding-agent`)
- `pypi-tui`: Terminal UI components (equivalent to `packages/tui`)
- `pypi-web`: Web UI and API (equivalent to `packages/web-ui`)

### Provider Support
- Anthropic Messages API (anthropic SDK)
- OpenAI Chat Completions (openai SDK)
- Google Gemini (google-generativeai)
- Mistral (mistralai SDK)
- AWS Bedrock (boto3)
- OpenAI-compatible providers (DeepSeek, Groq, etc.)

## Capabilities

### New Capabilities
- `llm-api-abstraction`: Multi-provider LLM API with unified streaming interface, model discovery, token tracking
- `agent-runtime`: Agent loop with turn management, tool execution (parallel/sequential), state management
- `tool-system`: Pydantic-based tool definition, argument validation, execution hooks
- `streaming-events`: Event stream protocol for text/thinking/tool-call deltas, async iterator pattern
- `cli-application`: Interactive coding agent CLI with session management, configuration, built-in tools

### Modified Capabilities
<!-- No existing capabilities to modify - this is a new project migration -->

## Impact

### Dependencies
- **Pydantic v2**: Core type system (replaces TypeBox)
- **anthropic**: Anthropic Claude API
- **openai**: OpenAI GPT API
- **google-generativeai**: Google Gemini API
- **mistralai**: Mistral API
- **boto3**: AWS Bedrock integration
- **asyncio/anyio**: Async runtime
- **rich/textual**: TUI components
- **FastAPI**: Web API framework
- **pytest/pytest-asyncio**: Testing

### Architecture Changes
- Python monorepo structure using `uv` workspace or separate packages
- Pyproject.toml for package configuration
- Type hints throughout with mypy verification
- PEP 8 compliance with ruff linting

### Breaking Considerations
- **BREAKING**: Complete rewrite - not backward compatible with TypeScript version
- Python naming conventions (snake_case) vs TypeScript (camelCase)
- Module-based organization vs TypeScript's explicit exports