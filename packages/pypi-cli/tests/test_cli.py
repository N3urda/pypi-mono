"""Tests for CLI."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from pypi_cli.cli import create_parser, get_tools, create_config


def test_create_parser():
    """Test argument parser creation."""
    parser = create_parser()

    # Test with no args
    args = parser.parse_args([])
    assert args.prompt is None
    assert args.model == "claude-sonnet-4-20250514"
    assert args.provider == "anthropic"


def test_parser_with_prompt():
    """Test parser with prompt argument."""
    parser = create_parser()
    args = parser.parse_args(["Hello world"])

    assert args.prompt == "Hello world"


def test_parser_with_model():
    """Test parser with model option."""
    parser = create_parser()
    args = parser.parse_args(["--model", "gpt-4", "test"])

    assert args.model == "gpt-4"


def test_parser_with_provider():
    """Test parser with provider option."""
    parser = create_parser()
    args = parser.parse_args(["--provider", "openai", "test"])

    assert args.provider == "openai"


def test_parser_with_session():
    """Test parser with session option."""
    parser = create_parser()
    args = parser.parse_args(["--session", "session_123"])

    assert args.session == "session_123"


def test_parser_version():
    """Test parser version."""
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_parser_help():
    """Test parser help."""
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_get_tools():
    """Test get_tools returns tool list."""
    tools = get_tools()

    assert len(tools) == 6
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    assert "read" in tool_names
    assert "write" in tool_names
    assert "edit" in tool_names
    assert "grep" in tool_names
    assert "find" in tool_names


def test_create_config():
    """Test create_config returns AgentLoopConfig."""
    config = create_config("claude-test", "anthropic")

    assert config.model is not None
    assert config.model.id == "claude-test"


def test_create_config_custom():
    """Test create_config with custom model."""
    config = create_config("gpt-4", "openai")

    assert config.model.id == "gpt-4"


def test_create_config_convert_to_llm():
    """Test convert_to_llm in create_config."""
    config = create_config("claude-test", "anthropic")

    # Test the convert_to_llm function
    result = config.convert_to_llm([{"role": "user", "content": "test"}])
    assert result == [{"role": "user", "content": "test"}]


# =============================================================================
# Main Function Tests
# =============================================================================


def test_main_returns_int():
    """Test main function returns int."""
    from pypi_cli.cli import main

    with patch("sys.argv", ["pypi-cli"]):
        with patch("pypi_cli.cli.run_interactive", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None
            result = main()
            assert result == 0


def test_main_with_prompt_argument():
    """Test main with prompt argument calls run_single_prompt."""
    from pypi_cli.cli import main

    with patch("sys.argv", ["pypi-cli", "Hello"]):
        with patch("pypi_cli.cli.run_single_prompt", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None
            result = main()
            assert result == 0


def test_main_interactive_mode():
    """Test main in interactive mode calls run_interactive."""
    from pypi_cli.cli import main

    with patch("sys.argv", ["pypi-cli"]):
        with patch("pypi_cli.cli.run_interactive", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None
            result = main()
            assert result == 0


def test_main_with_session():
    """Test main with session argument."""
    from pypi_cli.cli import main

    with patch("sys.argv", ["pypi-cli", "--session", "test_session"]):
        with patch("pypi_cli.cli.run_interactive", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None
            result = main()
            assert result == 0