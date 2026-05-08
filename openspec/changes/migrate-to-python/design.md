## Context

This document describes the technical design for migrating the pi-mono TypeScript AI Agent framework to Python (pypi-mono).

**Source Project**: pi-mono (TypeScript monorepo)
- Core LLM API abstraction with multi-provider support
- Agent runtime with tool execution loop
- Coding agent CLI application
- Built on TypeBox for runtime validation, AsyncIterable for streaming

**Target Project**: pypi-mono (Python monorepo)
- Preserve architecture and capabilities
- Adapt to Python idioms and ecosystem
- Leverage Pydantic v2, asyncio, and Python's strong AI/ML tooling

**Key Stakeholders**: Python developers wanting a unified LLM API and agent framework

## Goals / Non-Goals

**Goals:**
- Migrate core architecture: LLM API abstraction, agent runtime, tool system
- Implement streaming event protocol with asyncio
- Support 5+ LLM providers (Anthropic, OpenAI, Google, Mistral, Bedrock)
- Create Python CLI application equivalent to coding-agent
- Achieve 80%+ test coverage
- Follow Python best practices (PEP 8, type hints, Pydantic)

**Non-Goals:**
- 1:1 TypeScript feature parity (omit OAuth, model auto-generation in Phase 1)
- Web UI migration (Phase 4+)
- Backward compatibility with TypeScript version
- Real-time collaboration features
- Multi-language support

## Decisions

### D1: Type System - Pydantic v2

**Decision**: Use Pydantic v2 for all type definitions, validation, and JSON Schema generation.

**Rationale**:
- TypeBox provides runtime validation + JSON Schema in TypeScript
- Pydantic v2 is the Python equivalent with similar capabilities
- Native integration with FastAPI, excellent performance
- Supports discriminated unions for content types

**Alternatives Considered**:
- dataclasses: No runtime validation, no JSON Schema generation
- attrs + cattrs: More verbose, less ecosystem support
- marshmallow: Older, slower than Pydantic v2

### D2: Async Runtime - asyncio with anyio

**Decision**: Use asyncio as the primary async runtime with anyio for cross-backend compatibility.

**Rationale**:
- Native Python async/await support
- asyncio.Queue for event stream buffering
- CancelledError as AbortSignal equivalent
- anyio enables trio compatibility if needed

**Alternatives Considered**:
- trio: Smaller ecosystem, less mainstream
- curio: Minimal adoption

### D3: Monorepo Structure - uv workspace

**Decision**: Use uv workspace for monorepo management with separate packages under `packages/`.

**Rationale**:
- Modern, fast Python package manager
- Native workspace support (like npm workspaces)
- Lock file for reproducible builds
- Easy publishing to PyPI

**Alternatives Considered**:
- Poetry with plugins: More complex setup
- pip-tools: No native workspace support
- Hatch: Less mature workspace features

### D4: Provider Registry - Dict with Lazy Loading

**Decision**: Simple dict registry with optional lazy loading via importlib.

**Rationale**:
- TypeScript version uses Map with lazy import
- Python's importlib provides equivalent functionality
- functools.lru_cache for provider module caching
- Explicit registration function pattern

**Alternatives Considered**:
- Plugin system with entry_points: Overkill for built-in providers
- Class-based registry: More boilerplate

### D5: Event Stream - AsyncIterator Pattern

**Decision**: Implement event stream as AsyncIterator[Event] with asyncio.Queue backing.

**Rationale**:
- Matches TypeScript's AsyncIterable pattern
- Native Python async for support
- Queue provides buffering for slow consumers
- Can be consumed with `async for`

**Alternatives Considered**:
- Callback pattern: Less ergonomic
- RxPY (ReactiveX): Additional dependency, learning curve

### D6: Tool System - Pydantic BaseModel for Parameters

**Decision**: Tools define parameters as Pydantic BaseModel subclasses.

**Rationale**:
- TypeBox schemas → Pydantic models
- Automatic JSON Schema generation for tool definitions
- Built-in validation on execution
- Clear separation of parameters and result types

**Alternatives Considered**:
- TypedDict: No runtime validation
- dataclasses: No JSON Schema without additional library

## Risks / Trade-offs

### R1: Streaming Complexity
**Risk**: Python's async streaming is less mature than Node.js streams.
**Mitigation**: Use asyncio.Queue with careful backpressure handling. Test with high-throughput scenarios.

### R2: Provider SDK Differences
**Risk**: Python SDKs may have different APIs than TypeScript equivalents.
**Mitigation**: Create adapter layer for each provider. Document any behavior differences.

### R3: Performance
**Risk**: Python may be slower than TypeScript for I/O-bound operations.
**Mitigation**: Use anyio for concurrent operations. Profile critical paths.

### R4: Type Safety
**Risk**: Python's runtime type checking is less strict than TypeScript.
**Mitigation**: Use mypy strict mode. Pydantic for runtime validation. Pre-commit hooks for CI.

### R5: Ecosystem Fragmentation
**Risk**: Python AI/ML ecosystem has many overlapping tools.
**Mitigation**: Focus on core LLM APIs first. Avoid unnecessary dependencies.

## Migration Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. Set up monorepo structure with uv workspace
2. Implement core types (Pydantic models)
3. Create event stream protocol
4. Build provider registry

### Phase 2: LLM Providers (Week 2-4)
1. Implement Anthropic provider (highest priority)
2. Implement OpenAI provider
3. Implement Google/Mistral providers
4. Add Bedrock provider

### Phase 3: Agent Runtime (Week 4-6)
1. Implement AgentState and AgentContext
2. Build agent loop with turn management
3. Create tool execution system
4. Add before/after hooks

### Phase 4: CLI Application (Week 6-8)
1. Build CLI entry point
2. Implement interactive mode
3. Create built-in tools (Bash, Read, Write, Edit)
4. Add session management

### Rollback Strategy
- Each phase is independently versioned
- Can fall back to previous package version if issues arise
- Phase 1-2 are foundational; issues require immediate fix

## Open Questions

1. **OAuth Support**: Should we implement OAuth flows (Anthropic, GitHub Copilot) in Phase 1 or defer?
2. **Model Auto-Discovery**: Should we auto-generate model list from models.dev like TypeScript version?
3. **TUI Framework**: textual vs rich for terminal UI?
4. **Minimum Python Version**: 3.10 (union types) or 3.11 (performance improvements)?
