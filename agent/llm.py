from __future__ import annotations

from typing import Iterable

from anthropic import Anthropic

from agent.models import ChatMessage


def create_client(api_key: str) -> Anthropic:
    """
    Anthropic 클라이언트 생성.
    """
    return Anthropic(api_key=api_key)


def to_anthropic_messages(messages: Iterable[ChatMessage]) -> list[dict]:
    """
    Pydantic ChatMessage 목록을 Anthropic API 형식으로 변환한다.

    Anthropic Messages API는 대체로 아래 형식을 받는다:
    [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요"}
    ]
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def extract_text_from_response(response) -> str:
    """
    Anthropic 응답 객체에서 text block만 추출해 하나의 문자열로 합친다.

    이유:
    - response.content가 block 리스트 형태일 수 있음
    - text 외 다른 타입(block)이 섞일 수도 있음
    """
    parts: list[str] = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


def call_model(
    *,
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """
    모델을 실제로 호출하고 문자열 응답을 반환한다.

    temperature:
    - 너무 높으면 JSON이 흔들릴 수 있어서 낮게 둔다.
    - Phase 3에서는 구조 안정성이 중요하므로 0.2 정도가 무난하다.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages, # type: ignore
    )

    return extract_text_from_response(response)