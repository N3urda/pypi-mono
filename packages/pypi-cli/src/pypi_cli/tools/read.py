"""
Read tool for reading file contents.
"""

from typing import Optional, Any

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class ReadParameters(BaseModel):
    """Parameters for Read tool."""

    file_path: str = Field(description="The absolute path to the file to read")
    offset: Optional[int] = Field(default=None, description="Line number to start reading from")
    limit: Optional[int] = Field(default=None, description="Number of lines to read")


async def execute_read(
    tool_call_id: str,
    params: ReadParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Read file contents with line numbers."""
    try:
        with open(params.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Apply offset and limit
        start = params.offset or 1
        if start < 1:
            start = 1

        end = len(lines)
        if params.limit:
            end = min(start + params.limit, len(lines))

        # Format with line numbers
        result_lines = []
        for i in range(start - 1, end):
            result_lines.append(f"{i + 1}\t{lines[i]}")

        content = "".join(result_lines)

        return AgentToolResult(
            content=[TextContent(type="text", text=content)],
            details={
                "file_path": params.file_path,
                "total_lines": len(lines),
                "start_line": start,
                "end_line": end,
            },
        )

    except FileNotFoundError:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"File not found: {params.file_path}")],
            details={"error": "file_not_found"},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error reading file: {e}")],
            details={"error": str(e)},
        )


read_tool = AgentTool(
    name="read",
    description="Read contents of a file with line numbers",
    parameters=ReadParameters.model_json_schema(),
    label="Read File",
    execute=lambda tool_call_id, params, signal, **kw: execute_read(
        tool_call_id,
        ReadParameters(**params),
        signal,
    ),
)