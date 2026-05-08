"""
Session management for pypi-cli.

Handles saving and loading conversation sessions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field

from pypi_ai.types import Message, Model
from pypi_cli.config import Settings


class Session(BaseModel):
    """Session data model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_id: str = ""
    provider: str = ""
    messages: list[dict] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)


class SessionManager:
    """Manager for conversation sessions."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        """Initialize session manager."""
        self.sessions_dir = sessions_dir or Path.home() / ".pypi" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        model_id: str,
        provider: str,
        settings: Optional[Settings] = None,
    ) -> Session:
        """Create a new session."""
        session = Session(
            model_id=model_id,
            provider=provider,
            settings=settings.model_dump() if settings else {},
        )
        return session

    def save_session(self, session: Session) -> Path:
        """Save session to file."""
        session.updated_at = datetime.now().isoformat()
        filename = f"session_{session.id}.json"
        path = self.sessions_dir / filename
        with open(path, "w") as f:
            json.dump(session.model_dump(), f, indent=2)
        return path

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load session by ID."""
        filename = f"session_{session_id}.json"
        path = self.sessions_dir / filename
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return Session(**data)
        return None

    def load_session_by_path(self, path: Path) -> Optional[Session]:
        """Load session from a specific path."""
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return Session(**data)
        return None

    def list_sessions(self) -> list[Session]:
        """List all saved sessions."""
        sessions = []
        for path in self.sessions_dir.glob("session_*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                sessions.append(Session(**data))
            except (json.JSONDecodeError, ValueError):
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        filename = f"session_{session_id}.json"
        path = self.sessions_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def add_message(self, session: Session, message: Message) -> Session:
        """Add a message to the session."""
        msg_dict = message.model_dump()
        session.messages.append(msg_dict)
        return session

    def get_messages(self, session: Session) -> list[dict]:
        """Get all messages from session."""
        return session.messages