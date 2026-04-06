from __future__ import annotations

import json

from anthropic import Anthropic

from agent.llm import call_model, to_anthropic_messages
from agent.models import AgentDecision, SessionState
from agent.parser import parse_agent_decision
from agent.prompts import build_context_hint
from agent.state import AgentState
from agent.tools.base import ToolResult
from agent.tools.executor import execute_tool
from agent.tools.registry import ToolRegistry


def build_tool_result_message(
    *,
    tool_name: str,
    tool_input: dict,
    tool_result: ToolResult,
) -> str:
    """
    툴 실행 결과를 모델에게 다시 전달하기 위한 후속 메시지를 만든다.

    이 메시지는 '이제 실제 결과를 봤으니 최종 응답을 생성하라'는 역할을 한다.
    """
    return f"""
Tool execution result:

tool_name: {tool_name}
tool_input: {json.dumps(tool_input, ensure_ascii=False)}
success: {tool_result.success}

content:
{tool_result.content}

error:
{tool_result.error}

metadata:
{json.dumps(tool_result.metadata, ensure_ascii=False)}

Now produce the final user-facing answer as valid JSON only.

Rules for this follow-up:
- Do not request another tool in this turn.
- Use the tool result above as the source of truth.
- If the tool failed, explain the failure clearly and helpfully.
- Set "tool_name" to null.
- Set "tool_input" to {{}}
""".strip()


def run_agent_turn(
    *,
    client: Anthropic,
    model: str,
    system_prompt: str,
    session_state: SessionState,
    agent_state: AgentState,
    tool_registry: ToolRegistry,
    user_input: str,
) -> AgentDecision:
    """
    한 턴을 처리하고 AgentDecision을 반환한다.

    Phase 4 동작 순서:
    1. 세션에 사용자 입력 기록
    2. 1차 모델 호출
    3. 구조화 응답 파싱
    4. 필요 시 툴 실행
    5. 툴 결과를 넣어 2차 모델 호출
    6. 최종 상태 업데이트
    """
    # 1. 사용자 입력을 세션 기록에 저장한다.
    session_state.add_user_message(user_input)

    # 2. 현재 세션 메시지를 API 형식으로 변환한다.
    messages = to_anthropic_messages(session_state.messages)

    # 3. 멀티턴일 경우, 이전 턴의 목표/계획 힌트를 현재 user 메시지 앞에 붙인다.
    if agent_state.turn_count > 0:
        context_hint = build_context_hint(
            current_goal=agent_state.current_goal,
            recent_plan=agent_state.recent_plan,
            turn_count=agent_state.turn_count,
        )

        last_msg = messages[-1]
        messages = messages[:-1] + [
            {
                "role": last_msg["role"],
                "content": f"{context_hint}\n\n{last_msg['content']}",
            }
        ]

    # 4. 1차 모델 호출:
    #    - 바로 답변 가능한지
    #    - 명확화가 필요한지
    #    - 툴을 써야 하는지
    raw_output = call_model(
        client=client,
        model=model,
        system_prompt=system_prompt,
        messages=messages,
    )

    first_decision = parse_agent_decision(raw_output)

    # 5. 툴 호출이 필요 없는 경우:
    #    기존 Phase 3와 유사하게 바로 응답을 기록하고 종료한다.
    if not first_decision.tool_name:
        session_state.add_assistant_message(first_decision.response)
        agent_state.update_from_decision(
            user_input=user_input,
            decision=first_decision,
            raw_output=raw_output,
        )
        return first_decision

    # 6. 툴 호출이 필요한 경우:
    #    executor를 통해 실제 툴을 실행한다.
    tool_result = execute_tool(
        registry=tool_registry,
        tool_name=first_decision.tool_name,
        tool_input=first_decision.tool_input,
    )

    # 7. 툴 실행 결과를 다시 모델에 전달할 후속 메시지를 만든다.
    tool_result_message = build_tool_result_message(
        tool_name=first_decision.tool_name,
        tool_input=first_decision.tool_input,
        tool_result=tool_result,
    )

    # 8. 2차 모델 호출용 메시지를 구성한다.
    #    - 방금 assistant가 했던 1차 판단(raw_output)
    #    - 그에 대한 tool result를 user 메시지로 추가
    followup_messages = messages + [
        {"role": "assistant", "content": raw_output},
        {"role": "user", "content": tool_result_message},
    ]

    # 9. 2차 모델 호출:
    #    이제는 툴 결과를 바탕으로 최종 사용자 응답을 생성하게 한다.
    final_raw_output = call_model(
        client=client,
        model=model,
        system_prompt=system_prompt,
        messages=followup_messages,
    )

    final_decision = parse_agent_decision(final_raw_output)

    # 10. 후속 호출에서는 더 이상 툴을 부르지 않게 유도했지만,
    #     방어적으로 최종 decision의 tool 필드를 비운다.
    final_decision.tool_name = None
    final_decision.tool_input = {}

    # 11. 최종 사용자 응답을 세션과 상태에 기록한다.
    session_state.add_assistant_message(final_decision.response)
    agent_state.update_from_decision(
        user_input=user_input,
        decision=final_decision,
        raw_output=final_raw_output,
    )

    return final_decision