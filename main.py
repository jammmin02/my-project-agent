from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent.config import get_env, load_agent_md
from agent.llm import create_client
from agent.models import SessionState
from agent.prompts import build_system_prompt
from agent.runner import run_agent_turn
from agent.state import AgentState
from agent.tools.registry import ToolRegistry


app = typer.Typer(help="CLI project agent")
console = Console()


def get_project_root() -> Path:
    """
    현재 프로젝트 루트 경로를 반환한다.

    지금 예제에서는 main.py 기준 상위 루트를 프로젝트 루트로 본다.
    필요하면 나중에 더 정교하게 바꿀 수 있다.
    """
    return Path(__file__).resolve().parent


def render_welcome() -> None:
    """
    chat 시작 시 안내 문구 출력.
    """
    welcome = Text()
    welcome.append("Phase 4 CLI Agent\n", style="bold green")
    welcome.append("명령어: ", style="bold")
    welcome.append("exit / quit", style="cyan")
    welcome.append(", ", style="")
    welcome.append("/reset", style="cyan")
    console.print(Panel(welcome, title="Welcome", expand=False))


def render_assistant_response(text: str) -> None:
    """
    사용자에게 보이는 assistant 응답 출력.
    """
    console.print(Panel(text, title="Assistant", expand=False))


def render_debug_panel(agent_state: AgentState) -> None:
    """
    debug 모드일 때 에이전트의 행동 흐름을 로그 형태로 출력한다.
    """
    decision = agent_state.last_decision

    if decision is None:
        console.print(Panel("No decision yet.", title="Agent Debug", expand=False))
        return

    lines: list[str] = []

    # 턴 헤더
    done_mark = "[green]✓[/green]" if decision.done else "[yellow]…[/yellow]"
    clarify_mark = " [yellow]· needs clarification[/yellow]" if decision.needs_clarification else ""
    lines.append(
        f"[bold]#{agent_state.turn_count}[/bold]  "
        f"[cyan]{decision.mode}[/cyan]  "
        f"{done_mark}{clarify_mark}"
    )

    # 목표
    if decision.goal:
        lines.append(f"[dim]goal[/dim]  {decision.goal}")

    # 툴 호출
    if decision.tool_name:
        import json as _json
        input_str = _json.dumps(decision.tool_input, ensure_ascii=False)
        lines.append(f"[yellow]→ {decision.tool_name}[/yellow]  [dim]{input_str}[/dim]")
    else:
        lines.append("[dim]→ no tool[/dim]")

    # 명확화 질문
    if decision.clarification_question:
        lines.append(f"[yellow]? {decision.clarification_question}[/yellow]")

    # 계획 (있을 때만)
    if decision.plan:
        lines.append("")
        for i, step in enumerate(decision.plan, 1):
            lines.append(f"  [dim]{i}.[/dim] {step}")

    console.print(Panel("\n".join(lines), title="Agent Debug", expand=False))


@app.command()
def run(prompt: str) -> None:
    """
    단일 프롬프트 실행 모드.

    예:
        python main.py run "이 프로젝트용 CLI 에이전트의 다음 단계 3개를 제안해줘"
    """
    api_key = get_env("ANTHROPIC_API_KEY", required=True)
    model = get_env("ANTHROPIC_MODEL", default="claude-sonnet-4-5")
    agent_md = load_agent_md()

    client = create_client(api_key)
    system_prompt = build_system_prompt(agent_md=agent_md)

    session_state = SessionState()
    agent_state = AgentState(debug_mode=False)
    registry = ToolRegistry(project_root=get_project_root())

    decision = run_agent_turn(
        client=client,
        model=model,
        system_prompt=system_prompt,
        session_state=session_state,
        agent_state=agent_state,
        tool_registry=registry,
        user_input=prompt,
    )

    render_assistant_response(decision.response)


@app.command()
def chat(
    debug: bool = typer.Option(False, help="에이전트 내부 상태를 함께 표시"),
) -> None:
    """
    멀티턴 chat 모드.

    예:
        python main.py chat
        python main.py chat --debug
    """
    api_key = get_env("ANTHROPIC_API_KEY", required=True)
    model = get_env("ANTHROPIC_MODEL", default="claude-sonnet-4-5")
    agent_md = load_agent_md()

    client = create_client(api_key)
    system_prompt = build_system_prompt(agent_md=agent_md)

    session_state = SessionState()
    agent_state = AgentState(debug_mode=debug)
    registry = ToolRegistry(project_root=get_project_root())

    render_welcome()

    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]종료합니다.[/yellow]")
            break

        normalized = user_input.strip().lower()

        if normalized in {"exit", "quit"}:
            console.print("[yellow]Bye![/yellow]")
            break

        if normalized == "/reset":
            session_state.clear()
            agent_state.reset()
            console.print("[cyan]세션과 에이전트 상태를 초기화했습니다.[/cyan]")
            continue

        if not user_input.strip():
            console.print("[red]빈 입력은 처리할 수 없습니다.[/red]")
            continue

        decision = run_agent_turn(
            client=client,
            model=model,
            system_prompt=system_prompt,
            session_state=session_state,
            agent_state=agent_state,
            tool_registry=registry,
            user_input=user_input,
        )

        render_assistant_response(decision.response)

        if debug:
            render_debug_panel(agent_state)


@app.command("list-files")
def list_files_command(
    path: str = typer.Argument(".", help="프로젝트 루트 기준 상대경로"),
    depth: int = typer.Option(2, help="탐색 깊이"),
) -> None:
    """
    프로젝트 내부 파일 트리를 조회한다.
    """
    registry = ToolRegistry(project_root=get_project_root())
    result = registry.filesystem.list_files(path=path, depth=depth)

    if result.success:
        console.print(Panel(result.content, title="list_files", expand=False))
    else:
        console.print(Panel(result.error, title="list_files error", expand=False))


@app.command("read-file")
def read_file_command(
    path: str = typer.Argument(..., help="읽을 파일 경로"),
) -> None:
    """
    프로젝트 내부 텍스트 파일을 읽는다.
    """
    registry = ToolRegistry(project_root=get_project_root())
    result = registry.filesystem.read_file(path=path)

    if result.success:
        console.print(Panel(result.content, title=f"read_file: {path}", expand=False))
    else:
        console.print(Panel(result.error, title="read_file error", expand=False))


@app.command("grep-code")
def grep_code_command(
    query: str = typer.Argument(..., help="검색할 문자열"),
    path: str = typer.Option(".", help="검색 시작 경로"),
) -> None:
    """
    프로젝트 내부 텍스트 파일에서 문자열 검색을 수행한다.
    """
    registry = ToolRegistry(project_root=get_project_root())
    result = registry.filesystem.grep_code(query=query, path=path)

    if result.success:
        console.print(Panel(result.content, title=f"grep_code: {query}", expand=False))
    else:
        console.print(Panel(result.error, title="grep_code error", expand=False))


if __name__ == "__main__":
    app()