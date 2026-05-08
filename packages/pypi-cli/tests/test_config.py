"""Tests for configuration management."""

import pytest
import tempfile
import json
from pathlib import Path

from pypi_cli.config import Settings, SettingsManager


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.default_model == "claude-sonnet-4-20250514"
    assert settings.default_provider == "anthropic"
    assert settings.temperature == 0.7


def test_settings_custom_values():
    """Test settings with custom values."""
    settings = Settings(
        default_model="gpt-4",
        default_provider="openai",
        temperature=0.5,
    )

    assert settings.default_model == "gpt-4"
    assert settings.default_provider == "openai"
    assert settings.temperature == 0.5


def test_settings_manager_load_empty():
    """Test loading settings from nonexistent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))
        settings = manager.load()

        assert settings.default_model == "claude-sonnet-4-20250514"


def test_settings_manager_save_and_load():
    """Test saving and loading settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Save custom settings
        settings = Settings(default_model="test-model")
        manager.save(settings)

        # Load and verify
        loaded = manager.load()
        assert loaded.default_model == "test-model"


def test_settings_manager_update():
    """Test updating settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        manager.update(default_model="updated-model")

        assert manager.settings.default_model == "updated-model"


def test_settings_manager_api_key():
    """Test API key management."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        manager.set_api_key("anthropic", "test-key")
        key = manager.get_api_key("anthropic")

        assert key == "test-key"


def test_settings_manager_reset():
    """Test resetting settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        manager.update(default_model="custom-model")
        manager.reset()

        assert manager.settings.default_model == "claude-sonnet-4-20250514"


def test_settings_model_dump():
    """Test settings serialization."""
    settings = Settings(default_model="test-model")
    data = settings.model_dump()

    assert isinstance(data, dict)
    assert data["default_model"] == "test-model"


def test_settings_manager_load_invalid_json():
    """Test loading settings with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Write invalid JSON
        settings_file = manager.config_dir / "settings.json"
        settings_file.write_text("not valid json")

        # Should return defaults
        settings = manager.load()
        assert settings.default_model == "claude-sonnet-4-20250514"


def test_settings_manager_load_invalid_data():
    """Test loading settings with invalid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Write invalid data
        settings_file = manager.config_dir / "settings.json"
        settings_file.write_text('{"default_model": 12345}')  # Invalid type

        # Should return defaults
        settings = manager.load()
        assert settings.default_model == "claude-sonnet-4-20250514"


def test_settings_manager_get_api_key_from_env():
    """Test getting API key from environment variable."""
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Set environment variable
        os.environ["ANTHROPIC_API_KEY"] = "env-test-key"

        try:
            key = manager.get_api_key("anthropic")
            assert key == "env-test-key"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


def test_settings_manager_get_api_key_case_insensitive():
    """Test getting API key with case-insensitive provider name."""
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        os.environ["OPENAI_API_KEY"] = "openai-env-key"

        try:
            key = manager.get_api_key("OpenAI")
            assert key == "openai-env-key"
        finally:
            del os.environ["OPENAI_API_KEY"]


def test_settings_manager_get_api_key_unknown_provider():
    """Test getting API key for unknown provider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        key = manager.get_api_key("unknown_provider")
        assert key is None


def test_settings_manager_get_api_key_priority():
    """Test that settings API key takes priority over env var."""
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Set both settings key and env var
        manager.set_api_key("google", "settings-key")
        os.environ["GOOGLE_API_KEY"] = "env-key"

        try:
            key = manager.get_api_key("google")
            assert key == "settings-key"  # Settings takes priority
        finally:
            del os.environ["GOOGLE_API_KEY"]


def test_settings_manager_properties():
    """Test SettingsManager properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SettingsManager(config_dir=Path(tmpdir))

        # Access settings property
        settings = manager.settings
        assert settings is not None

        # Access again (should be cached)
        settings2 = manager.settings
        assert settings2 is settings