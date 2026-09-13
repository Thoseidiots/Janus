"""
Zig adapter
Capabilities
  • detect:  *.zig files or 'zig' tag
  • static:  regex pitfall checks + optional `zig ast-check`
  • error normalisation: Zig error string format → Issue
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

# zig ast-check:  "file.zig:10:5: error: ..."
_ZIG_ERR_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<sev>error|warning|note):\s+(?P<msg>.+)$"
)

_PATTERNS = [
    (re.compile(r"\bunreachable\b"),       "warning", "unreachable is a safety hazard in release-safe builds.",
     "Replace with a proper error-return or assertion if the branch is truly unreachable."),
    (re.compile(r"@panic\s*\("),           "warning", "@panic will crash the program.",
     "Return an error union instead: return error.SomeError;"),
    (re.compile(r"\bundefined\b"),         "info",    "Use of 'undefined' – ensure value is set before use.",
     "Initialise the variable to a defined value before use to avoid undefined behaviour."),
    (re.compile(r"//\s*TODO|//\s*FIXME"), "info",    "TODO/FIXME comment.",
     "Address or file a ticket for the outstanding issue."),
    (re.compile(r"std\.debug\.print"),    "info",    "Debug std.debug.print in code.",
     "Remove before shipping or use a structured logging approach."),
]


class ZigAdapter(LanguageAdapter):
    language = "zig"
    _EXTS = {".zig"}
    _TAGS = {"zig"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: LLDB / GDB via zig build debug

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Zig execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Zig execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _ZIG_ERR_RE.match(raw.splitlines()[0]) if raw else None
        if m:
            frame = StackFrame(id="0", name="<zig>",
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
            code = open(source_path, encoding="utf-8").read()
        except OSError as e:
            return [Issue(severity="error", language=self.language,
                          message=str(e), source_path=source_path)]
        issues = self.analyze_code(code, source_path)
        issues.extend(_zig_ast_check(source_path))
        return issues


def _zig_ast_check(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["zig", "ast-check", path],
                           capture_output=True, text=True, timeout=15)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _ZIG_ERR_RE.match(line)
            if m:
                out.append(Issue(severity=m.group("sev"), language="zig",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line")), column=int(m.group("col"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
