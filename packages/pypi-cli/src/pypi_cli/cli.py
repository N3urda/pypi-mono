"""
CLI entry point for pypi coding agent.
"""

import argparse
import asyncio
import sys
from typing import Optional

from pypi_ai import get_model
from pypi_ai.types import Api, UserMessage

from pypi_agent import AgentState, AgentLoopConfig, AgentTool
from pypi_agent.types import AgentContext
from pypi_agent.loop import agent_loop


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="pypi",
        description="Python AI coding agent CLI",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single prompt to process (omit for interactive mode)",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--provider", "-p",
        default="anthropic",
        help="Provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--session", "-s",
        help="Resume from session file",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.0.1",
    )
    return parser


def get_tools() -> list[AgentTool]:
    """Get built-in tools."""
    from pypi_cli.tools.bash import bash_tool
    from pypi_cli.tools.read import read_tool
    from pypi_cli.tools.write import write_tool
    from pypi_cli.tools.edit import edit_tool
    from pypi_cli.tools.grep import grep_tool
    from pypi_cli.tools.find import find_tool

    return [bash_tool, read_tool, write_tool, edit_tool, grep_tool, find_tool]


def create_config(model_id: str, provider: str) -> AgentLoopConfig:
    """Create agent loop configuration."""
    model = get_model(provider, model_id)

    def convert_to_llm(messages):
        return messages

    return AgentLoopConfig(
        model=model,
        convert_to_llm=convert_to_llm,
    )


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Determine mode
    if args.prompt:
        # Single prompt mode
        asyncio.run(run_single_prompt(args.prompt, args.model, args.provider))
        return 0
    else:
        # Interactive mode
        asyncio.run(run_interactive(args.model, args.provider, args.session))
        return 0


async def run_single_prompt(prompt: str, model_id: str, provider: str) -> None:
    """Run a single prompt and exit."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    config = create_config(model_id, provider)
    tools = get_tools()

    state = AgentState(
        model=config.model,
        tools=tools,
    )

    from pypi_ai.types import UserMessage
    user_msg = UserMessage(content=prompt)
    context = AgentContext(
        system_prompt=state.system_prompt,
        messages=[user_msg],
        tools=tools,
    )

    console.print(f"[bold blue]User:[/] {prompt}")
    console.print("[bold green]Assistant:[/] ", end="")

    async for event in agent_loop([user_msg], context, config):
        if event.type == "message_end":
            msg = event.message
            if hasattr(msg, "content"):
                for c in msg.content:
                    if hasattr(c, "type") and c.type == "text":
                        console.print(c.text, end="")
        elif event.type == "tool_execution_start":
            console.print(f"\n[dim yellow]Tool: {event.tool_name}[/]")

    console.print()


async def run_interactive(
    model_id: str,
    provider: str,
    session_file: Optional[str] = None,
) -> None:
    """Run interactive REPL."""
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel

    console = Console()
    config = create_config(model_id, provider)
    tools = get_tools()

    state = AgentState(
        model=config.model,
        tools=tools,
    )

    console.print(Panel.fit(
        "[bold blue]pypi[/] - Python AI Coding Agent\n"
        "Type your prompt and press Enter. Ctrl+C to exit.",
        title="Welcome",
    ))

    while True:
        try:
            prompt = Prompt.ask("\n[bold blue]You[/]")
            if not prompt.strip():
                continue

            if prompt.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/]")
                break

            from pypi_ai.types import UserMessage
            user_msg = UserMessage(content=prompt)
            context = AgentContext(
                system_prompt=state.system_prompt,
                messages=[*state.messages, user_msg],
                tools=tools,
            )

            console.print("[bold green]Assistant:[/] ", end="")

            async for event in agent_loop([user_msg], context, config):
                if event.type == "message_end":
                    msg = event.message
                    if hasattr(msg, "content"):
                        for c in msg.content:
                            if hasattr(c, "type") and c.type == "text":
                                console.print(c.text, end="")
                elif event.type == "tool_execution_start":
                    console.print(f"\n[dim yellow]Tool: {event.tool_name}[/]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Press Ctrl+C again to exit.[/]")
            continue


if __name__ == "__main__":
    sys.exit(main())