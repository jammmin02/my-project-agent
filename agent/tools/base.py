from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    모든 툴 실행 결과의 공통 형식.

    success:
        실행 성공 여부

    tool_name:
        어떤 툴이 실행되었는지 식별

    content:
        성공 시 사용자/모델에게 보여줄 핵심 텍스트 결과

    error:
        실패 시 오류 메시지

    metadata:
        파일 크기, 경로, 매칭 수 같은 부가 정보
    """
    success: bool
    tool_name: str
    content: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_display_text(self) -> str:
        """
        CLI나 디버그 출력용 간단 문자열.
        """
        if self.success:
            return self.content
        return f"[ERROR] {self.error}"