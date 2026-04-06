from __future__ import annotations

import json
import sys

from agent.models import AgentDecision


def strip_code_fences(text: str) -> str:
    """
    모델이 규칙을 어기고 ```json ... ``` 형태로 감싸는 경우를 대비한 방어 코드.

    ideally JSON only가 와야 하지만,
    실제 LLM은 종종 fence를 붙이므로 최소한의 복구를 해준다.
    """
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def parse_agent_decision(raw_text: str) -> AgentDecision:
    """
    raw 모델 응답을 AgentDecision으로 파싱한다.

    1. code fence 제거
    2. JSON 파싱
    3. Pydantic 검증

    실패 시:
    - 프로그램이 죽지 않도록 fallback AgentDecision 반환
    """
    cleaned = strip_code_fences(raw_text)

    try:
        data = json.loads(cleaned)
        return AgentDecision.model_validate(data)
    except Exception as exc:
        print(f"[parser] AgentDecision 파싱 실패: {exc}", file=sys.stderr)
        # JSON 파싱 실패 시 fallback
        # 여기서 raw_text를 그대로 response로 내려주면
        # 최소한 사용자는 응답 텍스트를 볼 수 있다.
        return AgentDecision(
            mode="answer",
            goal="사용자 요청에 응답",
            plan=[
                "모델 응답을 구조화 형식으로 파싱 시도",
                "파싱 실패로 일반 응답 텍스트를 fallback 처리",
            ],
            response=cleaned or "응답을 생성했지만 구조화 파싱에 실패했습니다.",
            done=True,
            needs_clarification=False,
            clarification_question=None,
        )