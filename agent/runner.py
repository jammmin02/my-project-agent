from __future__ import annotations

import json
import sys

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
    step_index: int,
    tool_name: str,
    tool_input: dict,
    tool_result: ToolResult,
) -> str:
    """
    툴 실행 결과를 다음 모델 호출에 전달하기 위한 메시지를 만든다.

    Phase 4에서는 툴 실행 후 곧바로 최종 답변을 만들게 했지만,
    Phase 5에서는 툴 결과를 보고 다음 행동을 다시 판단해야 한다.

    따라서 이 메시지는:
    - 방금 실행된 툴 결과를 구조적으로 전달하고
    - 모델이 다음 step에서 다시 tool_name / tool_input을 결정하게 만든다.
    """
    return f"""
Tool execution result for step {step_index}:

tool_name: {tool_name}
tool_input: {json.dumps(tool_input, ensure_ascii=False)}
success: {tool_result.success}

content:
{tool_result.content}

error:
{tool_result.error}

metadata:
{json.dumps(tool_result.metadata, ensure_ascii=False)}

Now decide the next best step and return valid JSON only.

Rules for this follow-up:
- Use the tool result above as the source of truth.
- If another tool is needed, choose exactly one tool and provide valid tool_input.
- If no more tool use is needed, set "tool_name" to null and answer the user.
- If the tool failed, explain the failure clearly and decide whether to stop or recover.
""".strip()


def should_finish(decision: AgentDecision) -> bool:
    """
    현재 decision 기준으로 loop를 종료할지 판단한다.

    종료 기준:
    - clarify 모드면 사용자 추가 입력이 필요하므로 종료
    - tool_name이 없으면 더 이상 실행할 툴이 없으므로 종료

    여기서는 done만으로 종료하지 않는다.
    이유는 모델이 done=True와 tool_name을 동시에 잘못 줄 수도 있기 때문이다.
    Phase 5에서는 실제 행동 필요 여부를 tool_name 기준으로 보는 편이 더 안전하다.
    """
    if decision.mode == "clarify":
        return True

    if decision.tool_name is None:
        return True

    return False


def debug_log(agent_state: AgentState, message: str) -> None:
    """
    debug_mode가 켜져 있을 때만 stderr로 디버그 로그를 출력한다.

    stdout은 사용자 응답용으로 남겨두고,
    내부 실행 흐름은 stderr에 보내는 편이 디버깅에 유리하다.
    """
    if agent_state.debug_mode:
        print(message, file=sys.stderr)


def run_agent_turn(
    *,
    client: Anthropic,
    model: str,
    system_prompt: str,
    session_state: SessionState,
    agent_state: AgentState,
    tool_registry: ToolRegistry,
    user_input: str,
    max_steps: int = 5,
) -> AgentDecision:
    """
    한 턴을 처리하고 AgentDecision을 반환한다.

    Phase 5 핵심 동작:
    1. 사용자 입력을 세션에 기록
    2. 현재 세션 메시지를 준비
    3. 필요하면 이전 목표/계획 힌트를 마지막 user 메시지에 붙임
    4. 각 step마다 모델이 다음 행동을 다시 판단
    5. tool_name이 있으면 툴 실행 후 결과를 messages에 누적
    6. tool_name이 없거나 clarify면 종료
    7. max_steps를 넘으면 fallback 종료
    """
    # 사용자 입력을 세션 기록에 추가한다.
    session_state.add_user_message(user_input)

    # 현재 세션 전체를 Anthropic 메시지 형식으로 변환한다.
    messages = to_anthropic_messages(session_state.messages)

    # 멀티턴일 경우, 직전 목표와 최근 계획을 마지막 user 메시지 앞에 붙여서
    # 모델이 이전 맥락을 자연스럽게 이어받도록 한다.
    if agent_state.turn_count > 0:
        context_hint = build_context_hint(
            current_goal=agent_state.current_goal,
            recent_plan=agent_state.recent_plan,
            turn_count=agent_state.turn_count,
        )

        last_msg = messages[-1]
        messages[-1] = {
            "role": last_msg["role"],
            "content": f"{context_hint}\n\n{last_msg['content']}",
        }

    last_decision: AgentDecision | None = None
    last_raw_output = ""

    # Phase 5의 핵심 loop.
    # 한 요청 안에서 여러 step을 수행하게 만든다.
    for step_index in range(1, max_steps + 1):
        debug_log(agent_state, f"[loop] step={step_index} model_call=start")

        # 현재까지 누적된 messages를 바탕으로 모델이 이번 step의 판단을 생성한다.
        raw_output = call_model(
            client=client,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
        )
        last_raw_output = raw_output

        debug_log(agent_state, f"[loop] step={step_index} raw_output={raw_output}")

        # 모델의 JSON 응답을 구조화된 AgentDecision으로 파싱한다.
        decision = parse_agent_decision(raw_output)
        last_decision = decision

        debug_log(
            agent_state,
            "[loop] step={step} decision mode={mode} done={done} tool_name={tool}".format(
                step=step_index,
                mode=decision.mode,
                done=decision.done,
                tool=decision.tool_name,
            ),
        )

        # 종료 가능한 경우:
        # - clarify 모드
        # - 더 이상 툴이 필요 없는 경우
        if should_finish(decision):
            session_state.add_assistant_message(decision.response)
            agent_state.update_from_decision(
                user_input=user_input,
                decision=decision,
                raw_output=raw_output,
            )
            return decision

        # 여기까지 왔으면 현재 step에서는 툴 실행이 필요하다는 뜻이다.
        tool_result = execute_tool(
            registry=tool_registry,
            tool_name=decision.tool_name, # type: ignore
            tool_input=decision.tool_input,
        )

        debug_log(
            agent_state,
            "[loop] step={step} tool_executed tool_name={tool} success={success}".format(
                step=step_index,
                tool=decision.tool_name,
                success=tool_result.success,
            ),
        )

        # 방금 모델이 내린 구조화 판단(raw JSON)을 assistant 메시지로 추가한다.
        # 다음 step에서 모델이 자신의 직전 판단을 맥락으로 볼 수 있게 해준다.
        messages.append(
            {
                "role": "assistant",
                "content": raw_output,
            }
        )

        # 툴 실행 결과를 user 메시지로 추가한다.
        # 다음 step에서는 이 결과를 바탕으로 또 다른 툴을 고를 수도 있고,
        # 혹은 tool_name=null로 종료할 수도 있다.
        tool_result_message = build_tool_result_message(
            step_index=step_index,
            tool_name=decision.tool_name, # type: ignore
            tool_input=decision.tool_input,
            tool_result=tool_result,
        )

        messages.append(
            {
                "role": "user",
                "content": tool_result_message,
            }
        )

    # max_steps를 넘기면 무한 반복을 피하기 위해 안전 종료한다.
    fallback = AgentDecision(
        mode="answer",
        goal=last_decision.goal if last_decision else "요청 처리",
        plan=last_decision.plan if last_decision else ["loop 제한으로 종료"],
        response=(
            "정해진 최대 step 수 안에서 작업을 마무리하지 못했습니다.\n\n"
            "가능한 원인:\n"
            "- 모델이 충분한 결론에 도달하지 못했습니다.\n"
            "- 더 구체적인 파일명, 경로, 함수명 정보가 필요할 수 있습니다.\n\n"
            "다음에는 더 구체적인 단서를 포함해 다시 요청해 주세요."
        ),
        done=True,
        needs_clarification=False,
        clarification_question=None,
        tool_name=None,
        tool_input={},
    )

    session_state.add_assistant_message(fallback.response)
    agent_state.update_from_decision(
        user_input=user_input,
        decision=fallback,
        raw_output=last_raw_output,
    )

    return fallback