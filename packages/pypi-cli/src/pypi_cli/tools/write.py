"""
Write tool for creating or overwriting files.
"""

from typing import Optional, Any

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class WriteParameters(BaseModel):
    """Parameters for Write tool."""

    file_path: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")


async def execute_write(
    tool_call_id: str,
    params: WriteParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Write content to a file."""
    try:
        with open(params.file_path, "w", encoding="utf-8") as f:
            f.write(params.content)

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Successfully wrote to {params.file_path}")],
            details={
                "file_path": params.file_path,
                "bytes_written": len(params.content.encode("utf-8")),
            },
        )

    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error writing file: {e}")],
            details={"error": str(e)},
        )


write_tool = AgentTool(
    name="write",
    description="Write content to a file, creating it if it doesn't exist",
    parameters=WriteParameters.model_json_schema(),
    label="Write File",
    execute=lambda tool_call_id, params, signal, **kw: execute_write(
        tool_call_id,
        WriteParameters(**params),
        signal,
    ),
)