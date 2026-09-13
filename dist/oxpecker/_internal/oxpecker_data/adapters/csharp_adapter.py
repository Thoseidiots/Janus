"""
C# adapter
Capabilities
  • detect:  *.cs, *.csx files or 'csharp'/'c#' tags
  • static:  regex pitfall checks + optional `dotnet build` / Roslyn
  • error normalisation: .NET exception format → Issue
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

# .NET stack frame:  "   at Namespace.Class.Method(args) in file.cs:line 42"
_CS_FRAME_RE = re.compile(
    r"at\s+(?P<method>[^\(]+)\([^)]*\)\s+in\s+(?P<path>[^:]+):line\s+(?P<line>\d+)"
)
# MSBuild error:  "file.cs(10,5): error CS0001: ..."
_MSBUILD_RE = re.compile(
    r"^(?P<path>[^(]+)\((?P<line>\d+),(?P<col>\d+)\):\s+(?P<sev>error|warning)\s+CS\d+:\s+(?P<msg>.+)$"
)

_PATTERNS = [
    (re.compile(r"\.Result\b"),             "warning", ".Result on Task can deadlock.",
     "Use 'await' instead of .Result or .Wait() to avoid deadlocks."),
    (re.compile(r"\bThread\.Sleep\b"),      "warning", "Thread.Sleep blocks a thread pool thread.",
     "Use 'await Task.Delay(ms)' in async code."),
    (re.compile(r"catch\s*\(\s*Exception\s+\w+\s*\)\s*\{\s*\}"), "warning",
     "Empty catch block swallows exceptions.",
     "Log or rethrow: catch (Exception ex) { _logger.LogError(ex, \"...\"); throw; }"),
    (re.compile(r"\bConsole\.Write"),       "info",    "Console.Write in production code.",
     "Use ILogger<T> from Microsoft.Extensions.Logging."),
    (re.compile(r"\.ToString\(\)\s*=="),    "warning", "String comparison via ToString() may fail for nulls.",
     "Use string.Equals(a, b, StringComparison.Ordinal) for safe comparison."),
    (re.compile(r"//\s*TODO|//\s*FIXME"),   "info",    "TODO/FIXME comment.",
     "Address the outstanding issue or file a tracking ticket."),
]


class CSharpAdapter(LanguageAdapter):
    language = "csharp"
    _EXTS = {".cs", ".csx"}
    _TAGS = {"csharp", "c#", "cs", "dotnet", ".net"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: .NET debugger (vsdbg / netcoredbg) via DAP

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="C# execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="C# execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _CS_FRAME_RE.search(raw)
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
        return self.analyze_code(code, source_path)
        # real: run `dotnet build /path/to/project.csproj` and parse MSBuild output
