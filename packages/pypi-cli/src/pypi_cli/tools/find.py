"""
Find tool for finding files by name or pattern.
"""

from typing import Optional, Any

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class FindParameters(BaseModel):
    """Parameters for Find tool."""

    pattern: str = Field(description="Glob pattern to match file names")
    path: str = Field(default=".", description="Directory to search in")
    type: Optional[str] = Field(default=None, description="File type: 'file' or 'dir'")


async def execute_find(
    tool_call_id: str,
    params: FindParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Find files matching a pattern."""
    try:
        from pathlib import Path

        root = Path(params.path)
        if not root.exists():
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Path not found: {params.path}")],
                details={"error": "path_not_found"},
            )

        results = []

        for item in root.rglob(params.pattern):
            # Filter by type
            if params.type == "file" and not item.is_file():
                continue
            if params.type == "dir" and not item.is_dir():
                continue

            rel_path = item.relative_to(root)
            results.append(str(rel_path))

        # Sort and limit
        results.sort()
        output = "\n".join(results[:200])
        if len(results) > 200:
            output += f"\n... ({len(results) - 200} more results)"

        return AgentToolResult(
            content=[TextContent(type="text", text=output or "No files found")],
            details={
                "pattern": params.pattern,
                "path": params.path,
                "count": len(results),
            },
        )

    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error finding files: {e}")],
            details={"error": str(e)},
        )


find_tool = AgentTool(
    name="find",
    description="Find files matching a glob pattern",
    parameters=FindParameters.model_json_schema(),
    label="Find",
    execute=lambda tool_call_id, params, signal, **kw: execute_find(
        tool_call_id,
        FindParameters(**params),
        signal,
    ),
)