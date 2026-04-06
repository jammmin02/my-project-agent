from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# 프로젝트 루트 기준 .env 로드
# main.py를 어디서 실행하든 최대한 일관적으로 환경변수를 불러오도록 한다.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    환경변수를 읽어오는 공통 함수.

    Args:
        name: 환경변수 이름
        default: 기본값
        required: 필수 여부

    Returns:
        환경변수 값 문자열

    Raises:
        ValueError: 필수 환경변수가 없을 때
    """
    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(f"Required environment variable is missing: {name}")

    return value or ""


def load_agent_md(path: str = "AGENT.md") -> str:
    """
    프로젝트 루트의 AGENT.md 파일을 읽는다.
    없으면 빈 문자열을 반환해서 전체 프로그램이 죽지 않게 한다.
    """
    agent_md_path = BASE_DIR / path

    if not agent_md_path.exists():
        return ""

    return agent_md_path.read_text(encoding="utf-8")