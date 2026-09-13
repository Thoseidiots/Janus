"""
Lua adapter
Capabilities
  • detect:  *.lua files or 'lua' tag
  • static:  regex pitfall checks + optional `luac -p` syntax check
  • error normalisation: Lua error string format → Issue
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

# Lua error:  "file.lua:42: attempt to index..."
_LUA_ERR_RE = re.compile(r"(?P<path>[^:]+\.lua):(?P<line>\d+):\s+(?P<msg>.+)")
_LUAC_ERR_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):\s+(?P<msg>.+)$")

_PATTERNS = [
    (re.compile(r"\bloadstring\s*\(|\bload\s*\("), "warning", "Dynamic code loading (load/loadstring).",
     "Avoid load(); use a table dispatch instead."),
    (re.compile(r"\bpcall\s*\(\s*\)"),              "warning", "Empty pcall with no handler.",
     "Capture the status and error: local ok, err = pcall(fn); if not ok then ... end"),
    (re.compile(r"\bprint\s*\("),                   "info",    "Debug print() statement.",
     "Remove or replace with a proper logging library."),
    (re.compile(r"==\s*nil\b|~=\s*nil\b"),         "info",    "Nil comparison – ensure variable is declared.",
     "Use 'if var then' (truthy check) unless you specifically need nil vs false distinction."),
    (re.compile(r"\btostring\s*\(\s*nil\s*\)"),     "warning", "tostring(nil) returns 'nil'; likely a bug.",
     "Check the variable for nil before calling tostring()."),
    (re.compile(r"--\s*TODO|--\s*FIXME"),           "info",    "TODO/FIXME comment.",
     "Address or file a ticket for the outstanding issue."),
]


class LuaAdapter(LanguageAdapter):
    language = "lua"
    _EXTS = {".lua"}
    _TAGS = {"lua"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: MobDebug / local-lua-debugger-vscode

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Lua execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Lua execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []  # real: debug.getinfo()

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: debug.getlocal()

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _LUA_ERR_RE.search(raw)
        if m:
            frame = StackFrame(id="0", name="<chunk>",
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
        issues.extend(_luac_check(source_path))
        return issues


def _luac_check(path: str) -> List[Issue]:
    try:
        r = subprocess.run(["luac", "-p", path], capture_output=True, text=True, timeout=10)
        out = []
        for line in (r.stderr + r.stdout).splitlines():
            m = _LUAC_ERR_RE.match(line)
            if m:
                out.append(Issue(severity="error", language="lua",
                                 message=m.group("msg"), source_path=m.group("path"),
                                 line=int(m.group("line"))))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
