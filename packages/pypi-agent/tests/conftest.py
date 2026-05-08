"""Pytest configuration for pypi-agent tests."""

import pytest


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    from pypi_ai.types import Model, Api
    return Model(
        id="test-model",
        api=Api.ANTHROPIC_MESSAGES,
        provider="anthropic",
    )