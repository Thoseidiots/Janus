"""
Go adapter
Capabilities
  • detect:  *.go files or 'go' tag
  • static:  regex pitfall checks + optional `go vet` / `go build -o /dev/null`
  • error normalisation: Go panic / runtime format → Issue
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

# goroutine panic frame:  "main.foo(...)  /path/to/file.go:42 +0x..."
_GO_FRAME_RE = re.compile(r"(?P<path>[^\s]+\.go):(?P<line>\d+)")
# go vet / build error:  "path/file.go:10:5: error text"
_GO_ERR_RE   = re.compile(r"^(?P<path>[^:]+\.go):(?P<line>\d+):(?P<col>\d+):\s+(?P<msg>.+)$")

_PATTERNS = [
    (re.compile(r"\bpanic\s*\("),           "warning", "Explicit panic() call.",
     "Return an error value instead: return fmt.Errorf(\"...\")"),
    (re.compile(r"err\s*!=\s*nil\s*\{\s*\}"), "warning", "Empty error-handling block.",
     "Handle or propagate the error: if err != nil { return err }"),
    (re.compile(r"\bfmt\.Println\b"),       "info",    "fmt.Println in production code.",
     "Use log.Println or a structured logger (zap, zerolog)."),
    (re.compile(r"_ ="),                    "info",    "Blank identifier discarding a value.",
     "Ensure you intentionally discard this value; handle errors if applicable."),
    (re.compile(r"\btime\.Sleep\b"),        "info",    "time.Sleep can block goroutines unexpectedly.",
     "Prefer context.WithTimeout or ticker channels for timed operations."),
    (re.compile(r"defer\s+\w+\.Close\s*\(\s*\)\s*$"), "info",
     "defer Close() without error check.",
     "Capture the error: defer func() { if err := f.Close(); err != nil { log.Error(err) } }()"),
]


class GoAdapter(LanguageAdapter):
    language = "go"
    _EXTS = {".go"}
    _TAGS = {"go", "golang"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: Delve (dlv) DAP

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Go execution continued (dlv continue)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Go execution paused (dlv pause)")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: dlv stacktrace

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: dlv locals

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _GO_FRAME_RE.search(raw)
        if m:
            frame = StackFrame(id="0", name="<goroutine>",
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
        issues.extend(_go_vet(source_path))
        return issues


def _go_vet(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["go", "vet", path], capture_output=True, text=True, timeout=30)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _GO_ERR_RE.match(line)
            if m:
                out.append(Issue(severity="warning", language="go",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line")), column=int(m.group("col"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
