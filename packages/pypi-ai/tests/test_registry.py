"""Tests for provider registry."""

import pytest

from pypi_ai.registry import (
    register_provider,
    get_provider,
    get_all_providers,
    unregister_provider,
    clear_providers,
)
from pypi_ai.types import Api
from pypi_ai.event_stream import AssistantMessageEventStream


class MockProvider:
    """Mock provider for testing."""

    api = Api.ANTHROPIC_MESSAGES

    def stream(self, model, context, options=None):
        return AssistantMessageEventStream()

    def stream_simple(self, model, context, options=None):
        return AssistantMessageEventStream()


def test_register_provider():
    """Test registering a provider."""
    clear_providers()
    provider = MockProvider()

    register_provider(provider)

    result = get_provider(Api.ANTHROPIC_MESSAGES)
    assert result is not None
    assert result.api == Api.ANTHROPIC_MESSAGES


def test_get_provider_not_found():
    """Test getting non-existent provider."""
    clear_providers()

    result = get_provider(Api.OPENAI_RESPONSES)
    assert result is None


def test_unregister_provider():
    """Test unregistering a provider."""
    clear_providers()
    provider = MockProvider()

    register_provider(provider)
    assert get_provider(Api.ANTHROPIC_MESSAGES) is not None

    result = unregister_provider(Api.ANTHROPIC_MESSAGES)
    assert result is True

    assert get_provider(Api.ANTHROPIC_MESSAGES) is None


def test_clear_providers():
    """Test clearing all providers."""
    clear_providers()
    provider = MockProvider()

    register_provider(provider)
    clear_providers()

    assert get_provider(Api.ANTHROPIC_MESSAGES) is None


def test_get_all_providers():
    """Test getting all providers."""
    clear_providers()

    register_provider(MockProvider())

    providers = get_all_providers()
    assert len(providers) >= 1