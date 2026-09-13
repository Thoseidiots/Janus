"""
PHP adapter
Capabilities
  • detect:  *.php, *.phtml, *.php5 or 'php' tag
  • static:  regex pitfall checks + optional `php -l` lint
  • error normalisation: PHP error/exception format → Issue
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

_PHP_FRAME_RE = re.compile(r"#\d+\s+(?P<path>[^(]+)\((?P<line>\d+)\):\s+(?P<method>.+)")
_PHP_LINT_RE  = re.compile(r"PHP (?P<sev>Parse error|Fatal error|Warning):\s+(?P<msg>.+?)\s+in\s+(?P<path>.+)\s+on\s+line\s+(?P<line>\d+)")

_PATTERNS = [
    (re.compile(r"\beval\s*\("),               "error",   "eval() is a major security risk.",
     "Remove eval(); restructure code to avoid dynamic PHP execution."),
    (re.compile(r"\$_GET\[|\$_POST\[|\$_REQUEST\["), "warning",
     "Unsanitised user input.",
     "Filter/validate input: htmlspecialchars(), intval(), filter_input()."),
    (re.compile(r"mysql_\w+\s*\("),            "error",   "Deprecated mysql_* functions (removed in PHP 7).",
     "Migrate to PDO or MySQLi with prepared statements."),
    (re.compile(r"md5\s*\(\s*\$"),             "warning", "md5() is not suitable for password hashing.",
     "Use password_hash() and password_verify()."),
    (re.compile(r"die\s*\(|exit\s*\("),        "warning", "die()/exit() abruptly terminates script.",
     "Throw an exception or return an error code; avoid hard stops."),
    (re.compile(r"echo\s+\$"),                 "info",    "Echoing a variable directly.",
     "Escape output: echo htmlspecialchars($var, ENT_QUOTES, 'UTF-8');"),
    (re.compile(r"//\s*TODO|//\s*FIXME|#\s*TODO"), "info", "TODO/FIXME comment.",
     "Address or track the outstanding issue."),
]


class PhpAdapter(LanguageAdapter):
    language = "php"
    _EXTS = {".php", ".phtml", ".php3", ".php4", ".php5", ".phar"}
    _TAGS = {"php"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: Xdebug DBGp protocol

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="PHP execution continued (Xdebug run)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="PHP execution paused (Xdebug break)")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: Xdebug stack_get

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: Xdebug context_get

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _PHP_FRAME_RE.search(raw)
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
            code = open(source_path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            return [Issue(severity="error", language=self.language,
                          message=str(e), source_path=source_path)]
        issues = self.analyze_code(code, source_path)
        issues.extend(_php_lint(source_path))
        return issues


def _php_lint(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["php", "-l", path], capture_output=True, text=True, timeout=10)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _PHP_LINT_RE.search(line)
            if m:
                sev = "error" if "error" in m.group("sev").lower() else "warning"
                out.append(Issue(severity=sev, language="php",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
