"""
Swift adapter
Capabilities
  • detect:  *.swift files or 'swift' tag
  • static:  regex pitfall checks + optional `swiftc -typecheck`
  • error normalisation: Swift error format → Issue
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

# swiftc error:  "file.swift:10:5: error: ..."
_SWIFT_ERR_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<sev>error|warning|note):\s+(?P<msg>.+)$"
)

_PATTERNS = [
    (re.compile(r"try!\s+"),                "warning", "try! will crash on error.",
     "Use try? with optional handling, or try within a do-catch block."),
    (re.compile(r"!\s*\w+\s*\.\w+|!\s+as\s+"), "warning", "Force unwrap or force cast '!' can crash.",
     "Use guard let / if let / as? for safe unwrapping."),
    (re.compile(r"\bprint\s*\("),           "info",    "Debug print() statement.",
     "Remove or replace with os_log / Logger for production."),
    (re.compile(r"//\s*TODO|//\s*FIXME"),   "info",    "TODO/FIXME comment.",
     "Address or file a ticket for the outstanding issue."),
    (re.compile(r"DispatchQueue\.main\.sync"), "warning", "DispatchQueue.main.sync can deadlock.",
     "Use DispatchQueue.main.async or async/await instead."),
    (re.compile(r"@discardableResult\s*\npublic"), "info",
     "@discardableResult on a public API; callers may silently ignore errors.",
     "Consider returning a Result type instead."),
]


class SwiftAdapter(LanguageAdapter):
    language = "swift"
    _EXTS = {".swift"}
    _TAGS = {"swift"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: LLDB (lldb-vscode / codelldb DAP)

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Swift execution continued (LLDB continue)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Swift execution paused (LLDB pause)")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: LLDB thread backtrace

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: LLDB frame variable

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _SWIFT_ERR_RE.match(raw.splitlines()[0]) if raw else None
        if m:
            frame = StackFrame(id="0", name="<swift>",
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
        issues.extend(_swiftc_check(source_path))
        return issues


def _swiftc_check(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["swiftc", "-typecheck", path],
                           capture_output=True, text=True, timeout=30)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _SWIFT_ERR_RE.match(line)
            if m:
                out.append(Issue(severity=m.group("sev"), language="swift",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line")), column=int(m.group("col"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
