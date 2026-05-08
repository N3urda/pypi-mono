"""Tests for provider registration."""

import pytest

from pypi_ai.registry import (
    register_provider,
    get_provider,
    unregister_provider,
    clear_providers,
    get_all_providers,
    resolve_provider,
)
from pypi_ai.providers.register_builtins import register_builtin_providers
from pypi_ai.types import Api


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    clear_providers()
    yield
    clear_providers()


class MockProvider:
    """Mock provider for testing."""
    api = Api.ANTHROPIC_MESSAGES


def test_register_builtin_providers():
    """Test registering builtin providers."""
    register_builtin_providers()

    # Check that providers are registered
    all_providers = get_all_providers()
    assert len(all_providers) > 0


def test_register_builtin_providers_idempotent():
    """Test that register_builtin_providers can be called multiple times."""
    register_builtin_providers()
    first_count = len(get_all_providers())

    register_builtin_providers()
    second_count = len(get_all_providers())

    # Should not duplicate registrations
    assert first_count == second_count


def test_register_provider():
    """Test registering a provider."""
    provider = MockProvider()
    register_provider(provider)

    result = get_provider(Api.ANTHROPIC_MESSAGES)
    assert result is not None
    assert result.api == Api.ANTHROPIC_MESSAGES


def test_unregister_provider():
    """Test unregistering a provider."""
    provider = MockProvider()
    register_provider(provider)

    result = unregister_provider(Api.ANTHROPIC_MESSAGES)
    assert result is True

    result = get_provider(Api.ANTHROPIC_MESSAGES)
    assert result is None


def test_unregister_provider_not_found():
    """Test unregistering a non-existent provider."""
    result = unregister_provider(Api.ANTHROPIC_MESSAGES)
    assert result is False


def test_get_all_providers():
    """Test getting all providers."""
    register_builtin_providers()

    providers = get_all_providers()
    assert len(providers) > 0

    # Each provider should have an api attribute
    for provider in providers:
        assert hasattr(provider, "api")


def test_resolve_provider():
    """Test resolving a provider."""
    provider = MockProvider()
    register_provider(provider)

    result = resolve_provider(Api.ANTHROPIC_MESSAGES)
    assert result is not None
    assert result.api == Api.ANTHROPIC_MESSAGES


def test_resolve_provider_not_found():
    """Test resolving a non-existent provider raises error."""
    with pytest.raises(ValueError, match="No provider registered"):
        resolve_provider(Api.ANTHROPIC_MESSAGES)


def test_clear_providers():
    """Test clearing all providers."""
    provider = MockProvider()
    register_provider(provider)

    clear_providers()

    result = get_provider(Api.ANTHROPIC_MESSAGES)
    assert result is None


def test_register_multiple_providers():
    """Test registering multiple providers."""
    class Provider1:
        api = Api.ANTHROPIC_MESSAGES

    class Provider2:
        api = Api.OPENAI_COMPLETIONS

    register_provider(Provider1())
    register_provider(Provider2())

    assert get_provider(Api.ANTHROPIC_MESSAGES) is not None
    assert get_provider(Api.OPENAI_COMPLETIONS) is not None


def test_replace_provider():
    """Test replacing a provider."""
    class Provider1:
        api = Api.ANTHROPIC_MESSAGES
        name = "first"

    class Provider2:
        api = Api.ANTHROPIC_MESSAGES
        name = "second"

    register_provider(Provider1())
    register_provider(Provider2())

    result = get_provider(Api.ANTHROPIC_MESSAGES)
    assert result.name == "second"


def test_provider_lazy_loading():
    """Test lazy loading registration."""
    from pypi_ai.registry import register_lazy_provider

    loaded = [False]

    def loader():
        loaded[0] = True
        return MockProvider()

    register_lazy_provider(Api.GOOGLE_GENERATIVE_AI, loader)

    # Provider should not be loaded yet
    assert not loaded[0]

    # Access the provider
    provider = get_provider(Api.GOOGLE_GENERATIVE_AI)
    assert provider is not None
    assert loaded[0]


def test_lazy_loader_caches_result():
    """Test that lazy loader caches the result."""
    from pypi_ai.registry import register_lazy_provider, clear_providers

    clear_providers()

    load_count = [0]

    def loader():
        load_count[0] += 1
        return MockProvider()

    register_lazy_provider(Api.ANTHROPIC_MESSAGES, loader)

    # Access multiple times
    get_provider(Api.ANTHROPIC_MESSAGES)
    get_provider(Api.ANTHROPIC_MESSAGES)
    get_provider(Api.ANTHROPIC_MESSAGES)

    # Should only load once
    assert load_count[0] == 1


def test_clear_builtin_providers():
    """Test clear_builtin_providers function."""
    from pypi_ai.providers.register_builtins import clear_builtin_providers, register_builtin_providers
    from pypi_ai.registry import get_all_providers

    # Ensure providers are registered
    register_builtin_providers()
    first_count = len(get_all_providers())

    # Clear and re-register
    clear_builtin_providers()
    second_count = len(get_all_providers())

    # Should have same count after clear and re-register
    assert second_count == first_count


def test_unregister_lazy_provider():
    """Test unregistering a lazy provider."""
    from pypi_ai.registry import register_lazy_provider, unregister_provider, get_provider, clear_providers

    clear_providers()

    loaded = [False]

    def loader():
        loaded[0] = True
        return type("MockProvider", (), {"api": Api.ANTHROPIC_MESSAGES})()

    register_lazy_provider(Api.ANTHROPIC_MESSAGES, loader)

    # Unregister before loading
    result = unregister_provider(Api.ANTHROPIC_MESSAGES)
    assert result is True

    # Provider should be gone
    assert get_provider(Api.ANTHROPIC_MESSAGES) is None
    assert not loaded[0]  # Should never have been loaded
