# pypi-mono

A Python AI Agent framework with unified LLM API abstraction, migrated from [pi-mono](https://github.com/nicknisi/pi) TypeScript project.

## Overview

pypi-mono is a Python monorepo that provides:
- **pypi-ai**: Unified LLM API abstraction with multi-provider support
- **pypi-agent**: Agent runtime with tool execution loop
- **pypi-cli**: Interactive coding agent CLI
- **pypi-tui**: Terminal UI components
- **pypi-web**: Web UI and API

## Features

### Unified LLM API (pypi-ai)
- Stream responses from multiple providers (Anthropic, OpenAI, Google, Mistral)
- Event stream protocol for streaming text/thinking/tool calls
- Token and cost tracking
- Lazy-loaded providers for efficiency

### Agent Runtime (pypi-agent)
- Agent loop with turn management
- Parallel and sequential tool execution modes
- Before/after tool execution hooks
- State management with message history

### CLI Application (pypi-cli)
- Interactive REPL mode
- Single-prompt mode
- Built-in tools: Bash, Read, Write, Edit
- Session persistence

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     pypi-cli                         │
│              (Interactive coding agent)              │
├─────────────────────────────────────────────────────┤
│                    pypi-agent                        │
│          (Agent runtime, tool execution)             │
├─────────────────────────────────────────────────────┤
│                     pypi-ai                          │
│      (Unified LLM API, streaming, providers)         │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
pypi-mono/
├── packages/
│   ├── pypi-ai/           # LLM API abstraction layer
│   │   ├── src/pypi_ai/
│   │   │   ├── types.py       # Core types (Pydantic)
│   │   │   ├── event_stream.py # Event stream protocol
│   │   │   ├── registry.py    # Provider registry
│   │   │   ├── stream.py      # Public API
│   │   │   └── providers/     # Provider implementations
│   │   └── tests/
│   ├── pypi-agent/        # Agent runtime
│   │   ├── src/pypi_agent/
│   │   │   ├── types.py       # Agent types
│   │   │   ├── state.py       # Agent state
│   │   │   └── loop.py        # Agent loop
│   │   └── tests/
│   ├── pypi-cli/          # CLI application
│   │   ├── src/pypi_cli/
│   │   │   ├── cli.py         # CLI entry point
│   │   │   ├── config.py      # Configuration
│   │   │   ├── session.py     # Session management
│   │   │   └── tools/         # Built-in tools
│   │   └── tests/
│   ├── pypi-tui/          # Terminal UI
│   └── pypi-web/          # Web UI/API
├── pyproject.toml         # Workspace config
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/pypi-mono
cd pypi-mono

# Install with uv
uv sync

# Or install packages individually
pip install pypi-ai pypi-agent pypi-cli
```

## Quick Start

### Using the CLI

```bash
# Interactive mode
pypi

# Single prompt
pypi "Write a Python function to calculate fibonacci"

# Specify model
pypi --model claude-sonnet-4-20250514 --provider anthropic "Explain async/await"
```

### Using pypi-ai

```python
from pypi_ai import stream_simple, get_model, Context

# Create model
model = get_model("anthropic", "claude-sonnet-4-20250514")

# Create context
context = Context(
    system_prompt="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Stream response
async for event in stream_simple(model, context):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "done":
        print("\n---")
```

### Using pypi-agent

```python
from pypi_agent import AgentState, agent_loop, AgentTool
from pypi_ai import get_model

# Create agent state
state = AgentState(
    model=get_model("anthropic", "claude-sonnet-4"),
    system_prompt="You are a coding assistant.",
)

# Run agent loop
async for event in agent_loop([user_message], context, config):
    print(event)
```

## Supported Providers

| Provider | SDK | API Type |
|----------|-----|----------|
| Anthropic | anthropic | Messages API |
| OpenAI | openai | Chat Completions |
| OpenAI-compatible | openai | Chat Completions |
| Google | google-generativeai | Gemini API |
| Mistral | mistralai | Conversations API |

## Configuration

Settings are stored in `~/.pypi/settings.json`:

```json
{
  "default_model": "claude-sonnet-4-20250514",
  "default_provider": "anthropic",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

API keys can be set via environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `MISTRAL_API_KEY`

## Development

```bash
# Run tests
pytest

# Type checking
mypy packages/

# Linting
ruff check .
```

## Migration from pi-mono

This project migrates the TypeScript [pi-mono](https://github.com/nicknisi/pi) framework to Python:

| TypeScript | Python |
|------------|--------|
| TypeBox | Pydantic v2 |
| AsyncIterable | AsyncIterator |
| Map<string, Provider> | Dict[Api, Provider] |
| AbortSignal | asyncio.CancelledError |

## License

MIT License