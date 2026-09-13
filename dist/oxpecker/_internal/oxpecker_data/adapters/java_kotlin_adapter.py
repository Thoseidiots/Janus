"""
Java / Kotlin adapter
Capabilities
  • detect:  *.java, *.kt, *.kts files or language tags
  • static:  regex pitfall checks + optional `javac` syntax check
  • error normalisation: Java stack-trace format → Issue
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

_JAVA_FRAME_RE = re.compile(
    r"at\s+(?P<cls>[\w.$]+)\.(?P<method>\w+)\((?P<file>[^:)]+):(?P<line>\d+)\)"
)
_JAVAC_ERR_RE = re.compile(r"^(?P<path>.+):(?P<line>\d+):\s+(?P<sev>error|warning):\s+(?P<msg>.+)$")

_PATTERNS = [
    (re.compile(r"\.equals\s*\(null\)"),        "warning", "Use obj == null instead of .equals(null).",
     "Replace 'obj.equals(null)' with 'obj == null'; calling equals on null throws NPE."),
    (re.compile(r"catch\s*\(\s*Exception\s+\w+\s*\)\s*\{\s*\}"), "warning",
     "Empty catch block silently swallows exceptions.",
     "Log the exception or handle it: catch (Exception e) { log.error(e); }"),
    (re.compile(r"\bSystem\.out\.print"),       "info",    "Remove System.out.print in production.",
     "Use a proper logger (java.util.logging, SLF4J, Log4j2)."),
    (re.compile(r"\.printStackTrace\(\)"),      "info",    "Avoid printStackTrace() in production.",
     "Use a logger: logger.error(\"message\", e);"),
    (re.compile(r"new\s+\w+\s*\[\s*\]\s*=\s*new\s+\w+\s*\[0\]"), "info",
     "Zero-length array allocation.",
     "Store a static empty-array constant and reuse it."),
    (re.compile(r"==\s*\""),                   "warning", "String comparison with == checks reference, not content.",
     "Use .equals() to compare String values: str.equals(\"literal\")"),
]


class JavaKotlinAdapter(LanguageAdapter):
    language = "java/kotlin"
    _EXTS  = {".java", ".kt", ".kts"}
    _TAGS  = {"java", "kotlin", "kt"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: JDWP EventRequest

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="JVM execution continued (JDWP resume)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="JVM execution paused (JDWP suspend)")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: JDWP StackFrame.getVariables

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _JAVA_FRAME_RE.search(raw)
        if m:
            frame = StackFrame(
                id="0",
                name=f"{m.group('cls')}.{m.group('method')}",
                source_path=m.group("file"),
                line=int(m.group("line")),
            )
        return Issue(severity="error", language=self.language,
                     message=raw.splitlines()[0], frame=frame,
                     source_path=frame.source_path if frame else None,
                     line=frame.line if frame else None)

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
        if source_path.endswith(".java"):
            issues.extend(_javac_check(source_path))
        return issues


def _javac_check(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["javac", "-Xlint:all", path],
                           capture_output=True, text=True, timeout=15)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _JAVAC_ERR_RE.match(line)
            if m:
                out.append(Issue(severity=m.group("sev"), language="java/kotlin",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
