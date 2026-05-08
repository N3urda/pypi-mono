"""
Bash tool for executing shell commands.
"""

import asyncio
from typing import Optional, Any

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class BashParameters(BaseModel):
    """Parameters for Bash tool."""

    command: str = Field(description="The shell command to execute")
    timeout: Optional[int] = Field(default=30000, description="Timeout in milliseconds")


async def execute_bash(
    tool_call_id: str,
    params: BashParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Execute a bash command."""
    try:
        proc = await asyncio.create_subprocess_shell(
            params.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=params.timeout / 1000 if params.timeout else 30,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Command timed out after {params.timeout}ms")],
                details={"timeout": True},
            )

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += stderr.decode("utf-8", errors="replace")

        return AgentToolResult(
            content=[TextContent(type="text", text=output or "(no output)")],
            details={
                "exit_code": proc.returncode,
                "command": params.command,
            },
        )

    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error executing command: {e}")],
            details={"error": str(e)},
        )


bash_tool = AgentTool(
    name="bash",
    description="Execute a shell command and return the output",
    parameters=BashParameters.model_json_schema(),
    label="Bash",
    execute=lambda tool_call_id, params, signal, **kw: execute_bash(
        tool_call_id,
        BashParameters(**params),
        signal,
    ),
)