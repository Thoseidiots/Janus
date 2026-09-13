"""
C / C++ adapter
Capabilities
  • detect:  *.c, *.cpp, *.cc, *.cxx, *.h, *.hpp files or language tags
  • static:  regex pitfall checks + optional gcc/clang -fsyntax-only
  • error normalisation: GCC / Clang error-line format → Issue
"""
from __future__ import annotations
import os, re, subprocess
from typing import Any, Dict, List, Optional
try:
    from ..core.adapter import LanguageAdapter
    from ..core.events import DebugEvent, Issue, StackFrame, Variable
except ImportError:  # pragma: no cover - top-level package fallback
    from core.adapter import LanguageAdapter
    from core.events import DebugEvent, Issue, StackFrame, Variable

# GCC/Clang diagnostic:  "file.c:10:5: error: ..."
_GCC_ERR_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<sev>error|warning|note):\s+(?P<msg>.+)$")

_PATTERNS = [
    (re.compile(r"\bgets\s*\("),            "error",   "gets() is insecure and removed in C11.",
     "Replace gets(buf) with fgets(buf, sizeof(buf), stdin)."),
    (re.compile(r"\bsprintf\s*\("),         "warning", "sprintf() can overflow the buffer.",
     "Use snprintf(buf, sizeof(buf), fmt, ...) instead."),
    (re.compile(r"\bstrcpy\s*\("),          "warning", "strcpy() can overflow; no bounds check.",
     "Use strlcpy() or strncpy() with explicit size."),
    (re.compile(r"\bmalloc\s*\([^)]+\)\s*;(?!\s*if)"), "warning",
     "malloc() return value not checked for NULL.",
     "Always check: ptr = malloc(...); if (!ptr) { handle_error(); }"),
    (re.compile(r"\bfree\s*\(\s*\w+\s*\)\s*;"),   "info",
     "After free(), set pointer to NULL to avoid use-after-free.",
     "Add: ptr = NULL; immediately after free(ptr);"),
    (re.compile(r"//\s*TODO|//\s*FIXME|//\s*HACK"), "info", "TODO/FIXME/HACK comment.",
     "Address the outstanding issue noted in the comment."),
    (re.compile(r"==\s*NULL\b|!=\s*NULL\b"),        "info", "Consider using explicit NULL checks.",
     "Ensure pointer is fully initialised before comparison."),
]


class CCppAdapter(LanguageAdapter):
    language = "c/c++"
    _EXTS  = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}
    _TAGS  = {"c", "c++", "cpp", "cc", "cxx"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: GDB `break file.c:10` or LLDB equivalent

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="C/C++ execution continued (GDB continue)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="C/C++ execution paused (GDB interrupt)")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: GDB `bt` / MI -stack-list-frames

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: GDB `info locals`

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _GCC_ERR_RE.match(raw.splitlines()[0]) if raw else None
        if m:
            frame = StackFrame(id="0", name="<top>",
                               source_path=m.group("path"), line=int(m.group("line")))
        return Issue(severity="error", language=self.language,
                     message=raw.splitlines()[0],
                     source_path=frame.source_path if frame else None,
                     line=frame.line if frame else None, frame=frame)

    def analyze_code(self, code: str, filename: str = "<code>") -> List[Issue]:
        issues: List[Issue] = []
        for lineno, text in enumerate(code.splitlines(), 1):
            for pat, sev, msg, fix in _PATTERNS:
                if pat.search(text):
                    issues.append(Issue(severity=sev, language=self.language,
                                        message=msg, source_path=filename,
                                        line=lineno, fix_suggestion=fix))
        return issues

    def analyze_file(self, source_path: str) -> List[Issue]:
        try:
            code = open(source_path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            return [Issue(severity="error", language=self.language,
                          message=str(e), source_path=source_path)]
        issues = self.analyze_code(code, source_path)
        issues.extend(_compiler_check(source_path))
        return issues


def _compiler_check(path: str) -> List[Issue]:
    """Try gcc then clang for syntax-only check."""
    for compiler in ("gcc", "g++", "clang", "clang++"):
        try:
            r = subprocess.run(
                [compiler, "-fsyntax-only", "-Wall", "-Wextra", path],
                capture_output=True, text=True, timeout=15
            )
            out = []
            for line in (r.stderr + r.stdout).splitlines():
                m = _GCC_ERR_RE.match(line)
                if m and m.group("sev") in ("error", "warning"):
                    out.append(Issue(
                        severity=m.group("sev"),
                        language="c/c++",
                        message=m.group("msg"),
                        source_path=m.group("path"),
                        line=int(m.group("line")),
                        column=int(m.group("col")),
                    ))
            return out
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            break
    return []
