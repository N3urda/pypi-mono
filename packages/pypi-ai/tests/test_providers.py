"""Tests for provider implementations."""

import pytest

from pypi_ai.providers.anthropic import provider, AnthropicProvider, _get_thinking_budget, _map_stop_reason
from pypi_ai.providers.openai import provider as openai_provider, OpenAIProvider, _map_stop_reason as openai_map_stop_reason
from pypi_ai.providers.google import provider as google_provider, GoogleProvider
from pypi_ai.providers.mistral import provider as mistral_provider, MistralProvider, _map_stop_reason as mistral_map_stop_reason
from pypi_ai.types import Api, StopReason


def test_anthropic_provider_api():
    """Test Anthropic provider API type."""
    assert AnthropicProvider.api == Api.ANTHROPIC_MESSAGES


def test_anthropic_provider_instance():
    """Test Anthropic provider instance."""
    assert provider.api == Api.ANTHROPIC_MESSAGES


def test_get_thinking_budget():
    """Test thinking budget calculation."""
    budget = _get_thinking_budget("medium", {})
    assert budget == 4096

    budget = _get_thinking_budget("high", {"thinking_budgets": {"high": 10000}})
    assert budget == 10000


def test_map_stop_reason_anthropic():
    """Test Anthropic stop reason mapping."""
    assert _map_stop_reason("end_turn") == StopReason.END
    assert _map_stop_reason("stop_sequence") == StopReason.STOP
    assert _map_stop_reason("tool_use") == StopReason.TOOL_USE
    assert _map_stop_reason(None) == StopReason.END


def test_openai_provider_api():
    """Test OpenAI provider API type."""
    assert OpenAIProvider.api == Api.OPENAI_COMPLETIONS


def test_openai_provider_instance():
    """Test OpenAI provider instance."""
    assert openai_provider.api == Api.OPENAI_COMPLETIONS


def test_map_stop_reason_openai():
    """Test OpenAI stop reason mapping."""
    assert openai_map_stop_reason("stop") == StopReason.STOP
    assert openai_map_stop_reason("length") == StopReason.END
    assert openai_map_stop_reason("tool_calls") == StopReason.TOOL_USE
    assert openai_map_stop_reason(None) == StopReason.END


def test_google_provider_api():
    """Test Google provider API type."""
    assert GoogleProvider.api == Api.GOOGLE_GENERATIVE_AI


def test_google_provider_instance():
    """Test Google provider instance."""
    assert google_provider.api == Api.GOOGLE_GENERATIVE_AI


def test_mistral_provider_api():
    """Test Mistral provider API type."""
    assert MistralProvider.api == Api.MISTRAL_CONVERSATIONS


def test_mistral_provider_instance():
    """Test Mistral provider instance."""
    assert mistral_provider.api == Api.MISTRAL_CONVERSATIONS


def test_map_stop_reason_mistral():
    """Test Mistral stop reason mapping."""
    assert mistral_map_stop_reason("stop") == StopReason.STOP
    assert mistral_map_stop_reason("length") == StopReason.END
    assert mistral_map_stop_reason("tool_calls") == StopReason.TOOL_USE
    assert mistral_map_stop_reason(None) == StopReason.END


def test_anthropic_provider_with_api_key():
    """Test Anthropic provider with custom API key."""
    p = AnthropicProvider(api_key="test-key")
    assert p._api_key == "test-key"


def test_openai_provider_with_base_url():
    """Test OpenAI provider with custom base URL."""
    p = OpenAIProvider(api_key="test-key", base_url="https://custom.url")
    assert p._base_url == "https://custom.url"


def test_google_provider_with_api_key():
    """Test Google provider with custom API key."""
    p = GoogleProvider(api_key="test-key")
    assert p._api_key == "test-key"


def test_mistral_provider_with_api_key():
    """Test Mistral provider with custom API key."""
    p = MistralProvider(api_key="test-key")
    assert p._api_key == "test-key"