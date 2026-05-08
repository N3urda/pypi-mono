"""Tests for session management."""

import pytest
import tempfile
from pathlib import Path

from pypi_cli.session import Session, SessionManager


def test_session_creation():
    """Test creating a session."""
    session = Session(
        model_id="claude-test",
        provider="anthropic",
    )

    assert session.model_id == "claude-test"
    assert session.provider == "anthropic"
    assert len(session.messages) == 0


def test_session_manager_create():
    """Test creating a session via manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        session = manager.create_session(
            model_id="test-model",
            provider="test-provider",
        )

        assert session.model_id == "test-model"
        assert session.id is not None


def test_session_manager_save_and_load():
    """Test saving and loading a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        session = manager.create_session(
            model_id="test-model",
            provider="anthropic",
        )
        session.messages.append({"role": "user", "content": "Hello"})

        # Save
        path = manager.save_session(session)
        assert path.exists()

        # Load
        loaded = manager.load_session(session.id)
        assert loaded is not None
        assert loaded.model_id == "test-model"
        assert len(loaded.messages) == 1


def test_session_manager_list():
    """Test listing sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        # Create multiple sessions
        session1 = manager.create_session("model1", "provider1")
        session2 = manager.create_session("model2", "provider2")

        manager.save_session(session1)
        manager.save_session(session2)

        sessions = manager.list_sessions()
        assert len(sessions) == 2


def test_session_manager_delete():
    """Test deleting a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        session = manager.create_session("test", "test")
        manager.save_session(session)

        result = manager.delete_session(session.id)
        assert result is True

        loaded = manager.load_session(session.id)
        assert loaded is None


def test_session_add_message():
    """Test adding message to session."""
    manager = SessionManager()
    session = Session(model_id="test", provider="test")

    from pypi_ai.types import UserMessage
    msg = UserMessage(content="Hello")
    manager.add_message(session, msg)

    assert len(session.messages) == 1


def test_session_get_messages():
    """Test getting messages from session."""
    manager = SessionManager()
    session = Session(model_id="test", provider="test")

    from pypi_ai.types import UserMessage
    msg = UserMessage(content="Hello")
    manager.add_message(session, msg)

    messages = manager.get_messages(session)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello"


def test_session_manager_load_by_path():
    """Test loading session by path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        session = manager.create_session("test-model", "test-provider")
        path = manager.save_session(session)

        loaded = manager.load_session_by_path(path)
        assert loaded is not None
        assert loaded.id == session.id


def test_session_manager_load_by_path_not_exists():
    """Test loading session from nonexistent path."""
    manager = SessionManager()

    loaded = manager.load_session_by_path(Path("/nonexistent/path.json"))
    assert loaded is None


def test_session_manager_list_invalid_json():
    """Test listing sessions with invalid JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        # Create an invalid JSON file
        invalid_file = manager.sessions_dir / "session_invalid.json"
        invalid_file.write_text("not valid json")

        # Create a valid session
        session = manager.create_session("test", "test")
        manager.save_session(session)

        # Should only return the valid session
        sessions = manager.list_sessions()
        assert len(sessions) == 1


def test_session_delete_nonexistent():
    """Test deleting nonexistent session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(sessions_dir=Path(tmpdir))

        result = manager.delete_session("nonexistent")
        assert result is False