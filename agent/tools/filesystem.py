from __future__ import annotations

import difflib
import fnmatch
from pathlib import Path

from agent.tools.base import ToolResult


# 텍스트로 읽지 않을 확률이 높은 확장자들
# 읽기뿐 아니라 수정도 막아야 하는 대상이므로 공통으로 사용한다.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".db", ".sqlite", ".sqlite3",
    ".pyc", ".class",
}

# 기본적으로 숨기거나 차단할 민감 파일 패턴
# .env, 키 파일, 인증서 등은 읽기/수정 모두 매우 조심해야 하므로 기본 차단한다.
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
    - 안전한 파일 수정
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        max_read_bytes: int = 100_000,
        block_sensitive_files: bool = True,
    ) -> None:
        """
        Args:
            project_root:
                모든 파일 작업의 기준이 되는 프로젝트 루트 경로

            max_read_bytes:
                너무 큰 파일은 읽기/수정 과정에서 위험하므로 제한한다.
                현재는 단순성과 안전성을 위해 같은 제한을 수정에도 재사용한다.

            block_sensitive_files:
                민감 파일 접근을 기본 차단할지 여부
        """
        self.project_root = Path(project_root).resolve()
        self.max_read_bytes = max_read_bytes
        self.block_sensitive_files = block_sensitive_files

    def _is_sensitive(self, path: Path) -> bool:
        """
        민감 파일 패턴에 해당하는지 검사한다.

        파일명 자체와, 프로젝트 루트 기준 상대경로 둘 다 비교해서
        .env, *.pem 같은 패턴을 폭넓게 차단한다.
        """
        name = path.name
        rel = str(path.relative_to(self.project_root))

        for pattern in SENSITIVE_PATTERNS:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                return True
        return False

    def _is_binary_by_extension(self, path: Path) -> bool:
        """
        확장자로 대략적인 바이너리 여부를 판별한다.

        완벽한 바이너리 판별은 아니지만,
        현재 단계에서는 안전한 차단용 휴리스틱으로 충분하다.
        """
        return path.suffix.lower() in BINARY_EXTENSIONS

    def _resolve_path(self, user_path: str) -> Path:
        """
        사용자 입력 경로를 프로젝트 루트 기준 절대경로로 변환하고,
        루트 밖 접근을 차단한다.

        Raises:
            ValueError:
                프로젝트 루트 밖 접근 시도일 때
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
        에이전트 도구에서는 '완벽한 디코딩'보다
        '작업이 중간에 터지지 않는 것'이 더 중요할 때가 많다.
        """
        return path.read_text(encoding="utf-8", errors="replace")

    def _validate_editable_text_file(
        self,
        *,
        target: Path,
        tool_name: str,
        original_path: str,
        allow_create: bool = False,
    ) -> ToolResult | None:
        """
        수정 대상 파일이 안전하게 편집 가능한지 검사한다.

        이 helper를 분리한 이유:
        - write_file / replace_in_file가 같은 안전 규칙을 공유해야 함
        - 수정 로직 본문을 짧고 읽기 쉽게 유지하기 위함

        Args:
            target:
                _resolve_path를 거친 절대경로

            tool_name:
                실패 ToolResult에 어떤 툴이 문제였는지 넣기 위한 값

            original_path:
                사용자가 입력한 원래 path 문자열
                에러 메시지에는 이 값을 그대로 보여주는 편이 이해하기 쉽다.

            allow_create:
                새 파일 생성 허용 여부
                - replace_in_file: False
                - write_file: True

        Returns:
            - 문제가 없으면 None
            - 문제가 있으면 실패 ToolResult
        """
        if target.exists():
            if not target.is_file():
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"Path is not a file: {original_path}",
                )

            if self.block_sensitive_files and self._is_sensitive(target):
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"Access denied for sensitive file: {original_path}",
                )

            if self._is_binary_by_extension(target):
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"Binary or non-text file is blocked: {original_path}",
                )

            try:
                size = target.stat().st_size
            except OSError as exc:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=str(exc),
                )

            if size > self.max_read_bytes:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"File too large to edit safely ({size} bytes): {original_path}",
                    metadata={"size_bytes": size},
                )

            return None

        if not allow_create:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"File does not exist: {original_path}",
            )

        # 새 파일 생성 허용 상황이어도 민감/바이너리 경로는 미리 차단한다.
        # 예를 들어 .env를 새로 만드는 작업도 기본 정책상 막는 편이 안전하다.
        if self.block_sensitive_files and self._is_sensitive(target):
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Access denied for sensitive file: {original_path}",
            )

        if self._is_binary_by_extension(target):
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Binary or non-text file is blocked: {original_path}",
            )

        return None

    def _build_unified_diff(
        self,
        *,
        old_text: str,
        new_text: str,
        path: str,
    ) -> str:
        """
        수정 전/후 텍스트를 바탕으로 unified diff 문자열을 만든다.

        이 diff는:
        - 사용자에게 변경 내용을 설명할 때
        - 나중에 승인 레이어를 붙일 때
        - 디버깅할 때

        매우 유용하므로 metadata에 포함시키는 용도로 사용한다.
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        return "".join(diff_lines)

    def _snapshot_metadata(
        self,
        *,
        before_text: str,
        after_text: str,
    ) -> dict[str, int]:
        """
        수정 전후 텍스트에 대한 간단한 스냅샷 통계를 만든다.

        전체 원문을 metadata에 넣으면 너무 커질 수 있으므로,
        우선은 길이와 줄 수 같은 핵심 정보만 저장한다.
        """
        return {
            "before_chars": len(before_text),
            "after_chars": len(after_text),
            "before_lines": len(before_text.splitlines()),
            "after_lines": len(after_text.splitlines()),
        }

    def list_files(self, path: str = ".", depth: int = 2) -> ToolResult:
        """
        주어진 경로 아래 파일/폴더를 트리 형태 텍스트로 나열한다.

        Args:
            path:
                프로젝트 루트 기준 상대경로

            depth:
                몇 단계까지 내려갈지

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

    def replace_in_file(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> ToolResult:
        """
        기존 텍스트 파일 안에서 old_text를 new_text로 치환한다.

        특징:
        - 기존 파일이 반드시 존재해야 한다
        - 민감 파일 / 바이너리 파일 / 과대 파일은 차단한다
        - 수정 전후 diff를 생성한다
        - replace_all=False면 첫 번째 일치만 치환한다
        """
        try:
            target = self._resolve_path(path)

            validation_error = self._validate_editable_text_file(
                target=target,
                tool_name="replace_in_file",
                original_path=path,
                allow_create=False,
            )
            if validation_error is not None:
                return validation_error

            original_text = self._safe_read_text(target)

            # old_text가 실제로 존재하지 않으면 수정하지 않는다.
            # 이 경우 성공 처리하면 모델이 잘못된 수정 성공으로 오해할 수 있다.
            if old_text not in original_text:
                return ToolResult(
                    success=False,
                    tool_name="replace_in_file",
                    error="old_text was not found in the target file.",
                    metadata={
                        "path": path,
                        "resolved_path": str(target),
                    },
                )

            occurrence_count = original_text.count(old_text)

            if replace_all:
                updated_text = original_text.replace(old_text, new_text)
                replaced_count = occurrence_count
            else:
                updated_text = original_text.replace(old_text, new_text, 1)
                replaced_count = 1

            # 논리상 거의 드물지만, 방어적으로 실제 변화가 없으면 실패 처리한다.
            if updated_text == original_text:
                return ToolResult(
                    success=False,
                    tool_name="replace_in_file",
                    error="No effective change was made.",
                    metadata={
                        "path": path,
                        "resolved_path": str(target),
                    },
                )

            # 실제 파일 내용을 갱신한다.
            target.write_text(updated_text, encoding="utf-8")

            diff_text = self._build_unified_diff(
                old_text=original_text,
                new_text=updated_text,
                path=path,
            )

            metadata = {
                "path": path,
                "resolved_path": str(target),
                "replace_all": replace_all,
                "occurrence_count": occurrence_count,
                "replaced_count": replaced_count,
                "diff": diff_text,
                **self._snapshot_metadata(
                    before_text=original_text,
                    after_text=updated_text,
                ),
            }

            return ToolResult(
                success=True,
                tool_name="replace_in_file",
                content=(
                    f"Updated file successfully: {path}\n"
                    f"Replaced occurrences: {replaced_count}"
                ),
                metadata=metadata,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name="replace_in_file",
                error=str(exc),
            )

    def write_file(self, *, path: str, content: str) -> ToolResult:
        """
        텍스트 파일을 새로 생성하거나 전체 내용을 덮어쓴다.

        특징:
        - 프로젝트 루트 밖 접근 금지
        - 민감 파일 / 바이너리 파일 차단
        - 부모 디렉토리가 없으면 생성
        - 기존 파일이 있으면 diff 생성
        - 새 파일이면 creation metadata를 남긴다
        """
        try:
            target = self._resolve_path(path)

            validation_error = self._validate_editable_text_file(
                target=target,
                tool_name="write_file",
                original_path=path,
                allow_create=True,
            )
            if validation_error is not None:
                return validation_error

            existed_before = target.exists()
            before_text = ""

            if existed_before:
                before_text = self._safe_read_text(target)

            # 새 파일 생성 시 부모 디렉토리가 없으면 함께 만든다.
            # 단, _resolve_path를 이미 거쳤으므로 프로젝트 루트 밖으로 나가지는 않는다.
            target.parent.mkdir(parents=True, exist_ok=True)

            target.write_text(content, encoding="utf-8")

            diff_text = self._build_unified_diff(
                old_text=before_text,
                new_text=content,
                path=path,
            )

            metadata = {
                "path": path,
                "resolved_path": str(target),
                "created": not existed_before,
                "overwritten": existed_before,
                "diff": diff_text,
                **self._snapshot_metadata(
                    before_text=before_text,
                    after_text=content,
                ),
            }

            return ToolResult(
                success=True,
                tool_name="write_file",
                content=(
                    f"Wrote file successfully: {path}\n"
                    f"Created new file: {not existed_before}"
                ),
                metadata=metadata,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name="write_file",
                error=str(exc),
            )