from __future__ import annotations

import fnmatch
from pathlib import Path

from agent.tools.base import ToolResult


# 텍스트로 읽지 않을 확률이 높은 확장자들
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".db", ".sqlite", ".sqlite3",
    ".pyc", ".class",
}

# 기본적으로 숨기거나 차단할 민감 파일 패턴
SENSITIVE_PATTERNS = {
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "*.p12",
    "*.pfx",
}


class FileSystemToolset:
    """
    프로젝트 루트 내부에서만 동작하는 안전한 파일 시스템 툴 모음.

    이 클래스는 아래 책임을 가진다:
    - 경로 검증
    - 프로젝트 루트 제한
    - 텍스트 파일 읽기
    - 파일 트리 조회
    - 간단한 코드 검색
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        max_read_bytes: int = 100_000,
        block_sensitive_files: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.max_read_bytes = max_read_bytes
        self.block_sensitive_files = block_sensitive_files

    def _is_sensitive(self, path: Path) -> bool:
        """
        민감 파일 패턴에 해당하는지 검사한다.
        """
        name = path.name
        rel = str(path.relative_to(self.project_root)) if path.exists() or path.is_absolute() else str(path)

        for pattern in SENSITIVE_PATTERNS:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                return True
        return False

    def _is_binary_by_extension(self, path: Path) -> bool:
        """
        확장자로 대략적인 바이너리 여부를 판별한다.
        """
        return path.suffix.lower() in BINARY_EXTENSIONS

    def _resolve_path(self, user_path: str) -> Path:
        """
        사용자 입력 경로를 프로젝트 루트 기준 절대경로로 변환하고,
        루트 밖 접근을 차단한다.

        Raises:
            ValueError: 프로젝트 루트 밖 접근 시도일 때
        """
        candidate = (self.project_root / user_path).resolve()

        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("Path escapes the project root.") from exc

        return candidate

    def _safe_read_text(self, path: Path) -> str:
        """
        텍스트 파일을 utf-8 기준으로 읽는다.
        디코딩 실패 시 errors='replace'로 최대한 복구한다.
        """
        return path.read_text(encoding="utf-8", errors="replace")

    def list_files(self, path: str = ".", depth: int = 2) -> ToolResult:
        """
        주어진 경로 아래 파일/폴더를 트리 형태 텍스트로 나열한다.

        Args:
            path: 프로젝트 루트 기준 상대경로
            depth: 몇 단계까지 내려갈지

        Returns:
            ToolResult
        """
        try:
            target = self._resolve_path(path)

            if not target.exists():
                return ToolResult(
                    success=False,
                    tool_name="list_files",
                    error=f"Path does not exist: {path}",
                )

            if not target.is_dir():
                return ToolResult(
                    success=False,
                    tool_name="list_files",
                    error=f"Path is not a directory: {path}",
                )

            lines: list[str] = [f"{target.relative_to(self.project_root) or '.'}/"]

            def walk(current: Path, level: int) -> None:
                if level > depth:
                    return

                try:
                    entries = sorted(
                        current.iterdir(),
                        key=lambda p: (not p.is_dir(), p.name.lower())
                    )
                except PermissionError:
                    lines.append("  " * level + "[Permission denied]")
                    return

                for entry in entries:
                    name = entry.name

                    # 너무 시끄러운 기본 제외 규칙
                    if name in {".git", "__pycache__", ".pytest_cache", "node_modules", ".next", ".idea", ".venv", "venv"}:
                        continue

                    prefix = "  " * level
                    display_name = f"{name}/" if entry.is_dir() else name
                    lines.append(prefix + display_name)

                    if entry.is_dir():
                        walk(entry, level + 1)

            walk(target, 1)

            return ToolResult(
                success=True,
                tool_name="list_files",
                content="\n".join(lines),
                metadata={
                    "path": path,
                    "resolved_path": str(target),
                    "depth": depth,
                },
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name="list_files",
                error=str(exc),
            )

    def read_file(self, path: str) -> ToolResult:
        """
        텍스트 파일을 읽는다.

        제한:
        - 프로젝트 루트 밖 접근 금지
        - 바이너리 확장자 차단
        - 민감 파일 차단(기본)
        - 너무 큰 파일 차단
        """
        try:
            target = self._resolve_path(path)

            if not target.exists():
                return ToolResult(
                    success=False,
                    tool_name="read_file",
                    error=f"File does not exist: {path}",
                )

            if not target.is_file():
                return ToolResult(
                    success=False,
                    tool_name="read_file",
                    error=f"Path is not a file: {path}",
                )

            if self.block_sensitive_files and self._is_sensitive(target):
                return ToolResult(
                    success=False,
                    tool_name="read_file",
                    error=f"Access denied for sensitive file: {path}",
                )

            if self._is_binary_by_extension(target):
                return ToolResult(
                    success=False,
                    tool_name="read_file",
                    error=f"Binary or non-text file is blocked: {path}",
                )

            size = target.stat().st_size
            if size > self.max_read_bytes:
                return ToolResult(
                    success=False,
                    tool_name="read_file",
                    error=f"File too large to read safely ({size} bytes): {path}",
                    metadata={"size_bytes": size},
                )

            text = self._safe_read_text(target)

            return ToolResult(
                success=True,
                tool_name="read_file",
                content=text,
                metadata={
                    "path": path,
                    "resolved_path": str(target),
                    "size_bytes": size,
                    "line_count": len(text.splitlines()),
                },
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name="read_file",
                error=str(exc),
            )

    def grep_code(self, query: str, path: str = ".") -> ToolResult:
        """
        프로젝트 내부 텍스트 파일에서 간단한 문자열 검색을 수행한다.

        현재는 가장 단순한 substring 검색만 구현한다.
        이후 정규식 검색이나 파일 필터를 추가할 수 있다.
        """
        try:
            target = self._resolve_path(path)

            if not target.exists():
                return ToolResult(
                    success=False,
                    tool_name="grep_code",
                    error=f"Path does not exist: {path}",
                )

            if not query.strip():
                return ToolResult(
                    success=False,
                    tool_name="grep_code",
                    error="Query must not be empty.",
                )

            files_to_search: list[Path] = []

            if target.is_file():
                files_to_search = [target]
            else:
                for p in target.rglob("*"):
                    if not p.is_file():
                        continue

                    if any(part in {".git", "__pycache__", ".pytest_cache", "node_modules", ".next", ".venv", "venv"} for part in p.parts):
                        continue

                    if self.block_sensitive_files and self._is_sensitive(p):
                        continue

                    if self._is_binary_by_extension(p):
                        continue

                    try:
                        if p.stat().st_size > self.max_read_bytes:
                            continue
                    except OSError:
                        continue

                    files_to_search.append(p)

            matches: list[str] = []
            total_matches = 0

            for file_path in files_to_search:
                try:
                    text = self._safe_read_text(file_path)
                except Exception:
                    continue

                rel_path = file_path.relative_to(self.project_root)

                for idx, line in enumerate(text.splitlines(), start=1):
                    if query in line:
                        total_matches += 1
                        matches.append(f"{rel_path}:{idx}: {line.strip()}")

            if not matches:
                return ToolResult(
                    success=True,
                    tool_name="grep_code",
                    content="No matches found.",
                    metadata={
                        "query": query,
                        "path": path,
                        "match_count": 0,
                    },
                )

            return ToolResult(
                success=True,
                tool_name="grep_code",
                content="\n".join(matches[:200]),
                metadata={
                    "query": query,
                    "path": path,
                    "match_count": total_matches,
                    "returned_count": min(len(matches), 200),
                },
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name="grep_code",
                error=str(exc),
            )