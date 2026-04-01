"""
Ruby adapter
Capabilities
  • detect:  *.rb, *.rake, *.gemspec, Gemfile, Rakefile, or 'ruby' tag
  • static:  regex pitfall checks + optional `ruby -wc` syntax check
  • error normalisation: Ruby backtrace format → Issue
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

# Ruby backtrace:  "path/to/file.rb:42:in `method_name'"
_RB_FRAME_RE = re.compile(r"(?P<path>[^:]+\.rb):(?P<line>\d+):in\s+`(?P<method>[^']+)'")
_RB_WARN_RE  = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):\s+(?P<msg>.+)$")

_PATTERNS = [
    (re.compile(r"\brescue\s*$"),           "warning", "Bare rescue catches StandardError only; obscures origin.",
     "Use rescue SpecificError => e to target the right exception."),
    (re.compile(r"\beval\s*\("),            "warning", "eval() is dangerous and slow.",
     "Refactor to use a data structure, Proc, or method object."),
    (re.compile(r"puts\s+"),                "info",    "Debug puts statement.",
     "Remove or replace with Rails logger: Rails.logger.debug(...)"),
    (re.compile(r"send\s*\(\s*params"),     "warning", "Dynamic method dispatch with user input.",
     "Whitelist allowed methods before calling send()."),
    (re.compile(r"\.save\s*$"),             "info",    ".save without bang may silently fail.",
     "Use .save! to raise on failure, or check the return value explicitly."),
    (re.compile(r"//\s*TODO|#\s*TODO|#\s*FIXME"), "info", "TODO/FIXME comment.",
     "Address or file a ticket for the outstanding issue."),
]

_SPECIALS = {"Gemfile", "Rakefile", "Guardfile"}


class RubyAdapter(LanguageAdapter):
    language = "ruby"
    _EXTS = {".rb", ".rake", ".gemspec", ".ru"}
    _TAGS = {"ruby", "rb"}

    def detect(self, target: str) -> bool:
        base = os.path.basename(target)
        return (os.path.splitext(target)[1].lower() in self._EXTS
                or target.lower() in self._TAGS
                or base in _SPECIALS)

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: byebug / ruby-debug-ide / DAP via rdbg

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Ruby execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Ruby execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _RB_FRAME_RE.search(raw)
        if m:
            frame = StackFrame(id="0", name=m.group("method"),
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
        issues.extend(_ruby_wc(source_path))
        return issues


def _ruby_wc(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["ruby", "-wc", path], capture_output=True, text=True, timeout=10)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            if "Syntax OK" in line:
                continue
            m = _RB_WARN_RE.match(line)
            if m:
                out.append(Issue(severity="error" if "SyntaxError" in line else "warning",
                                 language="ruby", message=m.group("msg"),
                                 source_path=m.group("path"), line=int(m.group("line"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
