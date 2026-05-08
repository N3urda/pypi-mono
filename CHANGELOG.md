# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 2026-05-07

### Added
- **pypi-ai**: Initial release of unified LLM API abstraction
  - Core types (Api, Provider, ThinkingLevel, StopReason enums)
  - Content types (TextContent, ThinkingContent, ImageContent, ToolCall)
  - Message types (AssistantMessage, UserMessage, ToolResultMessage)
  - Event stream protocol with asyncio.Queue backing
  - Provider registry with lazy loading
  - Anthropic provider with extended thinking support
  - OpenAI provider with Chat Completions API

- **pypi-agent**: Initial release of agent runtime
  - Agent types (AgentMessage, AgentContext, AgentTool, AgentToolResult)
  - AgentState class with state management
  - Agent loop with turn management
  - Tool execution (parallel and sequential modes)
  - Before/after tool execution hooks

- **pypi-cli**: Initial release of CLI application
  - Interactive REPL mode
  - Single-prompt mode
  - Built-in tools (Bash, Read, Write, Edit)
  - Configuration management (~/.pypi/settings.json)
  - Session persistence

- **Project setup**
  - Monorepo structure with uv workspace
  - pyproject.toml configurations for all packages
  - ruff and mypy configuration
  - pytest configuration
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- CI/CD workflow with Python 3.10-3.12 testing
- Issue templates (bug report, feature request)
- Pull request template
- Security policy
- Contributing guidelines
- Dependabot configuration
- CODEOWNERS file
- Release workflow for PyPI publishing

## [0.1.0] - 2026-05-08

### Added
- Initial release
- Basic project structure
- MIT License
- pyproject.toml configuration
