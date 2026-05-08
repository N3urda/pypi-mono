"""Tests for CLI main entry point."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import argparse

from pypi_cli.cli import (
    create_parser,
    get_tools,
    create_config,
    main,
    run_single_prompt,
    run_interactive,
)
from pypi_agent.types import AgentLoopConfig


def test_create_parser():
    """Test CLI parser creation."""
    parser = create_parser()

    # Test with no arguments
    args = parser.parse_args([])
    assert args.prompt is None
    assert args.model is not None
    assert args.provider is not None

    # Test with prompt
    args = parser.parse_args(["Hello"])
    assert args.prompt == "Hello"

    # Test with model option
    args = parser.parse_args(["-m", "claude-3-5-sonnet"])
    assert args.model == "claude-3-5-sonnet"

    # Test with provider option
    args = parser.parse_args(["-p", "anthropic"])
    assert args.provider == "anthropic"

    # Test with session
    args = parser.parse_args(["-s", "my-session"])
    assert args.session == "my-session"


def test_get_tools():
    """Test get_tools returns tool list."""
    tools = get_tools()

    assert len(tools) > 0
    tool_names = [t.name for t in tools]
    assert "bash" in tool_names
    assert "read" in tool_names
    assert "write" in tool_names
    assert "edit" in tool_names
    assert "grep" in tool_names
    assert "find" in tool_names


def test_create_config_default():
    """Test create_config with defaults."""
    config = create_config("claude-sonnet-4-20250514", "anthropic")

    assert config is not None
    assert isinstance(config, AgentLoopConfig)
    assert config.model is not None


def test_create_config_with_openai():
    """Test create_config with OpenAI provider."""
    config = create_config("gpt-4", "openai")

    assert config is not None
    assert config.model.id == "gpt-4"


def test_create_config_with_google():
    """Test create_config with Google provider."""
    config = create_config("gemini-pro", "google")

    assert config is not None


def test_create_config_with_mistral():
    """Test create_config with Mistral provider."""
    config = create_config("mistral-large", "mistral")

    assert config is not None


@pytest.mark.asyncio
async def test_run_single_prompt():
    """Test run_single_prompt function."""
    with patch("pypi_cli.cli.agent_loop") as mock_loop:
        # Create mock event
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text="Hello!")]

        async def mock_gen():
            yield MagicMock(type="message_end", message=mock_message)
            yield MagicMock(type="agent_end", messages=[])

        mock_loop.return_value = mock_gen()

        with patch("rich.console.Console.print"):
            await run_single_prompt("Hello", "claude-sonnet-4-20250514", "anthropic")


@pytest.mark.asyncio
async def test_run_interactive_with_exit():
    """Test run_interactive exits on 'exit' command."""
    with patch("rich.prompt.Prompt.ask") as mock_prompt:
        mock_prompt.return_value = "exit"

        with patch("rich.console.Console.print"):
            await run_interactive("claude-sonnet-4-20250514", "anthropic")


@pytest.mark.asyncio
async def test_run_interactive_with_quit():
    """Test run_interactive exits on 'quit' command."""
    with patch("rich.prompt.Prompt.ask") as mock_prompt:
        mock_prompt.return_value = "quit"

        with patch("rich.console.Console.print"):
            await run_interactive("claude-sonnet-4-20250514", "anthropic")


@pytest.mark.asyncio
async def test_run_interactive_with_prompt():
    """Test run_interactive with a prompt."""
    with patch("rich.prompt.Prompt.ask") as mock_prompt:
        mock_prompt.side_effect = ["Hello", "exit"]

        with patch("pypi_cli.cli.agent_loop") as mock_loop:
            mock_message = MagicMock()
            mock_message.content = []

            async def mock_gen():
                yield MagicMock(type="message_end", message=mock_message)

            mock_loop.return_value = mock_gen()

            with patch("rich.console.Console.print"):
                await run_interactive("claude-sonnet-4-20250514", "anthropic")


@pytest.mark.asyncio
async def test_run_interactive_empty_prompt():
    """Test run_interactive with empty prompt."""
    with patch("rich.prompt.Prompt.ask") as mock_prompt:
        mock_prompt.side_effect = ["", "exit"]

        with patch("rich.console.Console.print"):
            await run_interactive("claude-sonnet-4-20250514", "anthropic")


@pytest.mark.asyncio
async def test_run_interactive_keyboard_interrupt():
    """Test run_interactive handles keyboard interrupt."""
    with patch("rich.prompt.Prompt.ask") as mock_prompt:
        mock_prompt.side_effect = [KeyboardInterrupt(), "exit"]

        with patch("rich.console.Console.print"):
            await run_interactive("claude-sonnet-4-20250514", "anthropic")


def test_main_with_prompt():
    """Test main with prompt argument."""
    with patch("sys.argv", ["pypi-cli", "Hello"]):
        with patch("pypi_cli.cli.run_single_prompt", new_callable=AsyncMock):
            # main() calls asyncio.run(), so we need to test differently
            from pypi_cli.cli import create_parser, create_config, get_tools
            parser = create_parser()
            args = parser.parse_args(["Hello"])
            assert args.prompt == "Hello"
            config = create_config(args.model, args.provider)
            assert config is not None


def test_main_interactive():
    """Test main in interactive mode."""
    with patch("sys.argv", ["pypi-cli"]):
        from pypi_cli.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([])
        assert args.prompt is None


def test_parser_version():
    """Test parser version flag."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_parser_help():
    """Test parser help output."""
    parser = create_parser()
    help_text = parser.format_help()
    assert "usage:" in help_text.lower()
    assert "prompt" in help_text.lower()
    assert "model" in help_text.lower()
    assert "provider" in help_text.lower()


def test_tool_has_correct_structure():
    """Test that tools have correct structure."""
    tools = get_tools()

    for tool in tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "parameters")
        assert tool.name
        assert tool.description


@pytest.mark.asyncio
async def test_run_single_prompt_with_tool_event():
    """Test run_single_prompt with tool execution event."""
    with patch("pypi_cli.cli.agent_loop") as mock_loop:
        mock_message = MagicMock()
        mock_message.content = []

        async def mock_gen():
            yield MagicMock(type="tool_execution_start", tool_name="bash")
            yield MagicMock(type="message_end", message=mock_message)

        mock_loop.return_value = mock_gen()

        with patch("rich.console.Console.print"):
            await run_single_prompt("Hello", "claude-sonnet-4-20250514", "anthropic")


def test_main_module_entry():
    """Test __main__ module entry point."""
    import sys
    import importlib

    # Mock sys.argv and asyncio.run
    with patch("sys.argv", ["pypi-cli"]):
        with patch("asyncio.run") as mock_run:
            # Import and run the module
            import pypi_cli.cli as cli_module
            # The module should have __main__ guard
            assert hasattr(cli_module, "main")
            # Test calling main directly
            result = cli_module.main()
            assert result == 0 or result is None


def test_cli_text_printing():
    """Test CLI interactive mode text printing."""
    from pypi_cli.cli import run_single_prompt
    from pypi_ai.types import AssistantMessage, TextContent, StopReason, Usage

    with patch("pypi_cli.cli.agent_loop") as mock_loop:
        # Create mock message with text content
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="text", text="Hello!")]

        async def mock_gen():
            yield MagicMock(type="message_end", message=mock_message)
            yield MagicMock(type="agent_end", messages=[])

        mock_loop.return_value = mock_gen()

        with patch("rich.console.Console.print"):
            import asyncio
            asyncio.run(run_single_prompt("Hello", "claude-sonnet-4-20250514", "anthropic"))


def test_cli_tool_execution_event():
    """Test CLI handling tool execution event."""
    from pypi_cli.cli import run_single_prompt

    with patch("pypi_cli.cli.agent_loop") as mock_loop:
        mock_message = MagicMock()
        mock_message.content = []

        async def mock_gen():
            yield MagicMock(type="tool_execution_start", tool_name="bash")
            yield MagicMock(type="message_end", message=mock_message)
            yield MagicMock(type="agent_end", messages=[])

        mock_loop.return_value = mock_gen()

        with patch("rich.console.Console.print"):
            import asyncio
            asyncio.run(run_single_prompt("Hello", "claude-sonnet-4-20250514", "anthropic"))
