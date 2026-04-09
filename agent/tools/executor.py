from __future__ import annotations

from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry


ALLOWED_TOOL_NAMES = {
    "list_files",
    "read_file",
    "grep_code",
    "replace_in_file",
    "write_file",
}


def execute_tool(
    *,
    registry: ToolRegistry,
    tool_name: str,
    tool_input: dict,
) -> ToolResult:
    """
    tool_name / tool_input을 받아 실제 툴을 실행한다.

    Phase 6의 범위:
    - list_files
    - read_file
    - grep_code
    - replace_in_file
    - write_file

    설계 원칙:
    - 알 수 없는 툴 이름은 명확한 실패로 반환
    - 예외가 나도 ToolResult 형태로 감싸서 반환
    - 툴별 기본값을 적절히 보정
    """
    try:
        if tool_name not in ALLOWED_TOOL_NAMES:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name}",
            )

        if tool_name == "list_files":
            path = str(tool_input.get("path", "."))
            depth = int(tool_input.get("depth", 2))
            return registry.filesystem.list_files(path=path, depth=depth)

        if tool_name == "read_file":
            path = str(tool_input.get("path", "")).strip()
            if not path:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error="read_file requires 'path' in tool_input.",
                )
            return registry.filesystem.read_file(path=path)

        if tool_name == "grep_code":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error="grep_code requires 'query' in tool_input.",
                )
            path = str(tool_input.get("path", "."))
            return registry.filesystem.grep_code(query=query, path=path)

        if tool_name == "replace_in_file":
            path = str(tool_input.get("path", "")).strip()
            old_text = str(tool_input.get("old_text", ""))
            new_text = str(tool_input.get("new_text", ""))
            replace_all = bool(tool_input.get("replace_all", False))

            if not path:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error="replace_in_file requires 'path' in tool_input.",
                )

            if old_text == "":
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error="replace_in_file requires non-empty 'old_text'.",
                )

            return registry.filesystem.replace_in_file(
                path=path,
                old_text=old_text,
                new_text=new_text,
                replace_all=replace_all,
            )

        if tool_name == "write_file":
            path = str(tool_input.get("path", "")).strip()
            content = str(tool_input.get("content", ""))

            if not path:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error="write_file requires 'path' in tool_input.",
                )

            return registry.filesystem.write_file(
                path=path,
                content=content,
            )

    except Exception as exc:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=str(exc),
        )