"""Pytest configuration for pypi-ai tests."""

import pytest


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set mock API key for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")