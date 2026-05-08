"""
Register built-in LLM providers.

This module registers all built-in providers on import.
"""

from pypi_ai.types import Api
from pypi_ai.registry import register_provider, register_lazy_provider


def _lazy_load_anthropic():
    """Lazy load the Anthropic provider."""
    from pypi_ai.providers.anthropic import provider
    return provider


def _lazy_load_openai():
    """Lazy load the OpenAI provider."""
    from pypi_ai.providers.openai import provider
    return provider


def _lazy_load_google():
    """Lazy load the Google provider."""
    from pypi_ai.providers.google import provider
    return provider


def _lazy_load_mistral():
    """Lazy load the Mistral provider."""
    from pypi_ai.providers.mistral import provider
    return provider


def register_builtin_providers() -> None:
    """Register all built-in providers."""
    # Register Anthropic provider (lazy)
    register_lazy_provider(Api.ANTHROPIC_MESSAGES, _lazy_load_anthropic)

    # Register OpenAI provider (lazy)
    register_lazy_provider(Api.OPENAI_COMPLETIONS, _lazy_load_openai)

    # Register Google provider (lazy)
    register_lazy_provider(Api.GOOGLE_GENERATIVE_AI, _lazy_load_google)

    # Register Mistral provider (lazy)
    register_lazy_provider(Api.MISTRAL_CONVERSATIONS, _lazy_load_mistral)


def clear_builtin_providers() -> None:
    """Clear and re-register built-in providers."""
    from pypi_ai.registry import clear_providers
    clear_providers()
    register_builtin_providers()


# Register on import
register_builtin_providers()