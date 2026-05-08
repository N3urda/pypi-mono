"""
Edit tool for making precise string replacements in files.
"""

from typing import Optional, Any

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class EditParameters(BaseModel):
    """Parameters for Edit tool."""

    file_path: str = Field(description="The absolute path to the file to edit")
    old_string: str = Field(description="The exact string to replace (must be unique in file)")
    new_string: str = Field(description="The string to replace with")


async def execute_edit(
    tool_call_id: str,
    params: EditParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Edit a file by replacing an exact string match."""
    try:
        with open(params.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count occurrences
        count = content.count(params.old_string)

        if count == 0:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"String not found in file: {params.old_string[:50]}...")],
                details={"error": "string_not_found"},
            )

        if count > 1:
            return AgentToolResult(
                content=[TextContent(type="text", text=f"String appears {count} times - must be unique")],
                details={"error": "not_unique", "count": count},
            )

        # Perform replacement
        new_content = content.replace(params.old_string, params.new_string, 1)

        with open(params.file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return AgentToolResult(
            content=[TextContent(type="text", text=f"Successfully edited {params.file_path}")],
            details={
                "file_path": params.file_path,
                "old_length": len(params.old_string),
                "new_length": len(params.new_string),
            },
        )

    except FileNotFoundError:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"File not found: {params.file_path}")],
            details={"error": "file_not_found"},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error editing file: {e}")],
            details={"error": str(e)},
        )


edit_tool = AgentTool(
    name="edit",
    description="Edit a file by replacing an exact unique string with a new string",
    parameters=EditParameters.model_json_schema(),
    label="Edit File",
    execute=lambda tool_call_id, params, signal, **kw: execute_edit(
        tool_call_id,
        EditParameters(**params),
        signal,
    ),
)