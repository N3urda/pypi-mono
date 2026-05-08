"""
Grep tool for searching file contents.
"""

from typing import Optional, Any
import re

from pydantic import BaseModel, Field

from pypi_agent import AgentTool, AgentToolResult
from pypi_ai.types import TextContent


class GrepParameters(BaseModel):
    """Parameters for Grep tool."""

    pattern: str = Field(description="The regex pattern to search for")
    path: str = Field(default=".", description="The directory or file to search in")
    include: Optional[str] = Field(default=None, description="Glob pattern for files to include")
    ignore_case: bool = Field(default=False, description="Case insensitive search")


async def execute_grep(
    tool_call_id: str,
    params: GrepParameters,
    signal: Optional[Any] = None,
) -> AgentToolResult:
    """Search for pattern in files."""
    try:
        from pathlib import Path

        root = Path(params.path)
        if not root.exists():
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Path not found: {params.path}")],
                details={"error": "path_not_found"},
            )

        flags = re.IGNORECASE if params.ignore_case else 0
        pattern = re.compile(params.pattern, flags)

        results = []
        files_searched = 0
        matches_found = 0

        def search_file(file_path: Path) -> list[str]:
            nonlocal files_searched, matches_found
            file_results = []
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    files_searched += 1
                    for line_num, line in enumerate(f, 1):
                        if pattern.search(line):
                            matches_found += 1
                            rel_path = file_path.relative_to(root) if root.is_dir() else file_path.name
                            file_results.append(f"{rel_path}:{line_num}:{line.rstrip()}")
            except Exception:
                pass
            return file_results

        if root.is_file():
            results.extend(search_file(root))
        else:
            # Search directory
            include_pattern = params.include or "*"
            for file_path in root.rglob(include_pattern):
                if file_path.is_file():
                    results.extend(search_file(file_path))

        output = "\n".join(results[:100])  # Limit results
        if len(results) > 100:
            output += f"\n... ({len(results) - 100} more results)"

        return AgentToolResult(
            content=[TextContent(type="text", text=output or "No matches found")],
            details={
                "pattern": params.pattern,
                "files_searched": files_searched,
                "matches_found": matches_found,
            },
        )

    except Exception as e:
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Error searching: {e}")],
            details={"error": str(e)},
        )


grep_tool = AgentTool(
    name="grep",
    description="Search for a regex pattern in files",
    parameters=GrepParameters.model_json_schema(),
    label="Grep",
    execute=lambda tool_call_id, params, signal, **kw: execute_grep(
        tool_call_id,
        GrepParameters(**params),
        signal,
    ),
)