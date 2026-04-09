from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Anthropic messages API에 넣기 위한 기본 메시지 구조.

    role:
        - user
        - assistant

    content:
        실제 대화 문자열
    """
    role: Literal["user", "assistant"]
    content: str


class AgentDecision(BaseModel):
    """
    모델이 한 턴에서 반환해야 하는 구조화된 판단 결과.

    Phase 6부터는 읽기 툴뿐 아니라
    파일 수정 툴까지 선택할 수 있어야 한다.

    설계 원칙:
    - 툴이 필요 없으면 tool_name=None, tool_input={}
    - 툴이 필요하면 tool_name에 허용된 툴 이름을 넣는다
    - tool_input은 해당 툴에 넘길 인자 딕셔너리다
    """

    # 현재 턴의 응답 성격
    # answer     : 바로 답변 가능
    # clarify    : 사용자에게 추가 질문 필요
    # plan_only  : 계획/중간 판단 중심, 또는 툴 실행 전 상태
    mode: Literal["answer", "clarify", "plan_only"] = "answer"

    # 사용자의 요청을 한 줄 목표로 요약
    goal: str = Field(
        default="",
        description="현재 사용자의 요청을 한 줄로 요약한 목표"
    )

    # 문제 해결 계획
    plan: list[str] = Field(
        default_factory=list,
        description="문제를 해결하기 위한 구체적인 단계"
    )

    # 사용자에게 실제로 보여줄 응답
    response: str = Field(
        default="",
        description="최종 사용자 표시용 응답"
    )

    # 현재 턴 기준 완료 여부
    done: bool = Field(
        default=True,
        description="현재 턴 기준으로 응답이 완료되었는지 여부"
    )

    # 추가 질문 필요 여부
    needs_clarification: bool = Field(
        default=False,
        description="사용자에게 추가 질문이 필요한지 여부"
    )

    # 명확화 질문
    clarification_question: Optional[str] = Field(
        default=None,
        description="추가 질문이 필요할 때 사용자에게 물을 질문"
    )

    # 실행할 툴 이름
    # 툴이 필요 없으면 None
    tool_name: Optional[
        Literal[
            "list_files",
            "read_file",
            "grep_code",
            "replace_in_file",
            "write_file",
        ]
    ] = Field(
        default=None,
        description="실행할 툴 이름. 필요 없으면 null"
    )

    # 툴 입력 인자
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="툴 실행 인자 딕셔너리"
    )


class SessionState(BaseModel):
    """
    대화 메시지 기록을 관리한다.
    """

    messages: list[ChatMessage] = Field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """
        user 메시지를 대화 기록에 추가한다.
        """
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """
        assistant 메시지를 대화 기록에 추가한다.
        """
        self.messages.append(ChatMessage(role="assistant", content=content))

    def clear(self) -> None:
        """
        세션 메시지 기록 초기화.
        """
        self.messages.clear()