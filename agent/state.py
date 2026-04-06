from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.models import AgentDecision


@dataclass
class AgentState:
    """
    에이전트의 '작업 상태'를 저장하는 클래스.

    SessionState는 대화 자체의 기록이라면,
    AgentState는 '이번 작업을 어떻게 처리하고 있는가'에 대한 상태다.
    """

    # 현재까지 파악한 핵심 목표
    current_goal: str = ""

    # 직전 턴에서의 구조화 판단 결과
    last_decision: Optional[AgentDecision] = None

    # 몇 번째 턴인지 기록
    turn_count: int = 0

    # debug 출력 여부
    debug_mode: bool = False

    # 마지막 사용자 입력
    last_user_input: str = ""

    # 마지막으로 사용자에게 보여준 응답
    last_visible_response: str = ""

    # 최근 계획
    recent_plan: list[str] = field(default_factory=list)

    # 마지막 raw 모델 응답(JSON 파싱 전)
    last_raw_output: str = ""

    def update_from_decision(
        self,
        *,
        user_input: str,
        decision: AgentDecision,
        raw_output: str,
    ) -> None:
        """
        한 턴 처리 후 상태를 갱신한다.
        """
        self.turn_count += 1
        self.last_user_input = user_input
        self.last_decision = decision
        self.current_goal = decision.goal
        self.recent_plan = decision.plan
        self.last_visible_response = decision.response
        self.last_raw_output = raw_output

    def reset(self) -> None:
        """
        에이전트 상태를 초기화한다.
        /reset 처리 시 SessionState와 함께 초기화하는 용도.
        """
        self.current_goal = ""
        self.last_decision = None
        self.turn_count = 0
        self.last_user_input = ""
        self.last_visible_response = ""
        self.recent_plan = []
        self.last_raw_output = ""