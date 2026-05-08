"""
Provider registry for LLM API providers.

This module provides:
- ApiProvider protocol for provider implementations
- Registration functions for adding providers
- Lazy loading support via importlib
"""

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pypi_ai.types import Api, Context, Model, StreamOptions, SimpleStreamOptions
from pypi_ai.event_stream import AssistantMessageEventStream


# =============================================================================
# Types
# =============================================================================

StreamFunction = Callable[
    [Model, Context, Optional[StreamOptions]],
    AssistantMessageEventStream,
]

SimpleStreamFunction = Callable[
    [Model, Context, Optional[SimpleStreamOptions]],
    AssistantMessageEventStream,
]


@runtime_checkable
class ApiProvider(Protocol):
    """Protocol for LLM API providers."""

    api: Api
    stream: StreamFunction
    stream_simple: SimpleStreamFunction


# =============================================================================
# Registry
# =============================================================================

_registry: dict[Api, ApiProvider] = {}
_lazy_loaders: dict[Api, Callable[[], ApiProvider]] = {}


def register_provider(provider: ApiProvider, source_id: Optional[str] = None) -> None:
    """
    Register a provider in the registry.

    Args:
        provider: The provider to register.
        source_id: Optional source identifier for tracking.
    """
    _registry[provider.api] = provider


def register_lazy_provider(
    api: Api,
    loader: Callable[[], ApiProvider],
) -> None:
    """
    Register a lazy-loaded provider.

    The provider will be loaded on first access via get_provider().

    Args:
        api: The API type for the provider.
        loader: A function that loads and returns the provider.
    """
    _lazy_loaders[api] = loader


def get_provider(api: Api) -> Optional[ApiProvider]:
    """
    Get a provider by API type.

    If the provider is lazily loaded, it will be loaded on first access
    and then cached.

    Args:
        api: The API type.

    Returns:
        The provider if found, None otherwise.
    """
    # Check if already loaded
    if api in _registry:
        return _registry[api]

    # Check for lazy loader
    if api in _lazy_loaders:
        provider = _lazy_loaders[api]()
        _registry[api] = provider
        del _lazy_loaders[api]
        return provider

    return None


def get_all_providers() -> list[ApiProvider]:
    """
    Get all registered providers.

    This will trigger loading of any lazy-loaded providers.

    Returns:
        List of all registered providers.
    """
    # Load all lazy providers
    for api, loader in list(_lazy_loaders.items()):
        if api not in _registry:
            provider = loader()
            _registry[api] = provider
    _lazy_loaders.clear()

    return list(_registry.values())


def unregister_provider(api: Api) -> bool:
    """
    Unregister a provider.

    Args:
        api: The API type to unregister.

    Returns:
        True if the provider was removed, False if not found.
    """
    if api in _registry:
        del _registry[api]
        return True
    if api in _lazy_loaders:
        del _lazy_loaders[api]
        return True
    return False


def clear_providers() -> None:
    """Clear all registered providers."""
    _registry.clear()
    _lazy_loaders.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def resolve_provider(api: Api) -> ApiProvider:
    """
    Resolve a provider, raising an error if not found.

    Args:
        api: The API type.

    Returns:
        The provider.

    Raises:
        ValueError: If no provider is registered for the API type.
    """
    provider = get_provider(api)
    if provider is None:
        raise ValueError(f"No provider registered for API type: {api}")
    return provider