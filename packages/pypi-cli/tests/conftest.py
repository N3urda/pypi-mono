"""Pytest configuration for pypi-cli tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("Test content\nLine 2\nLine 3\n")
    yield file_path