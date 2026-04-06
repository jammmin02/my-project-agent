from __future__ import annotations

from pathlib import Path

from agent.tools.filesystem import FileSystemToolset


class ToolRegistry:
    """
    현재 프로젝트에서 사용할 툴 묶음을 관리한다.

    지금은 filesystem 툴만 담지만,
    이후 shell, patch, git, approval 툴도 여기에 확장 가능하다.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.filesystem = FileSystemToolset(project_root=project_root)