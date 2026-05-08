"""
Configuration management for pypi-cli.

Handles loading/saving settings from ~/.pypi/settings.json
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from pypi_ai.types import ThinkingLevel


class Settings(BaseModel):
    """Application settings."""

    model_config = ConfigDict(use_enum_values=True)

    default_model: str = "claude-sonnet-4-20250514"
    default_provider: str = "anthropic"
    thinking_level: ThinkingLevel = ThinkingLevel.OFF
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_ms: int = 60000
    api_keys: dict[str, str] = Field(default_factory=dict)


class SettingsManager:
    """Manager for application settings."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize settings manager."""
        self.config_dir = config_dir or Path.home() / ".pypi"
        self.settings_file = self.config_dir / "settings.json"
        self._settings: Optional[Settings] = None

    @property
    def settings(self) -> Settings:
        """Get current settings, loading if necessary."""
        if self._settings is None:
            self._settings = self.load()
        return self._settings

    def load(self) -> Settings:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                return Settings(**data)
            except (json.JSONDecodeError, ValueError):
                pass
        return Settings()

    def save(self, settings: Optional[Settings] = None) -> None:
        """Save settings to file."""
        settings = settings or self._settings or Settings()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.settings_file, "w") as f:
            json.dump(settings.model_dump(), f, indent=2)

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        # Check settings first
        key = self.settings.api_keys.get(provider)
        if key:
            return key

        # Check environment variable
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        env_var = env_map.get(provider.lower())
        if env_var:
            return os.environ.get(env_var)

        return None

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        self.settings.api_keys[provider] = api_key
        self.save()

    def update(self, **kwargs: Any) -> Settings:
        """Update settings with new values."""
        current = self.settings.model_dump()
        current.update(kwargs)
        self._settings = Settings(**current)
        self.save()
        return self._settings

    def reset(self) -> Settings:
        """Reset settings to defaults."""
        self._settings = Settings()
        self.save()
        return self._settings