"""
JavaScript / TypeScript adapter
Capabilities
  • detect:          *.js, *.mjs, *.cjs, *.ts, *.tsx, *.jsx files or language tags
  • static analysis: regex + structural checks on the source text
                     optional: call `node --check` for JS syntax
                     optional: call `tsc --noEmit` for TypeScript
  • node inspector:  build Chrome DevTools Protocol commands (no live ws needed)
  • error normalisation: V8 stack-trace format → Issue
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    from ..core.adapter import LanguageAdapter
    from ..core.events import DebugEvent, Issue, StackFrame, Variable
except ImportError:  # pragma: no cover - top-level package fallback
    from core.adapter import LanguageAdapter
    from core.events import DebugEvent, Issue, StackFrame, Variable

# V8 stack frame pattern:  "    at funcName (file.js:10:5)"
_V8_FRAME_RE = re.compile(
    r"at\s+(?:(?P<func>[^\s(]+)\s+)?\(?(?P<path>[^):]+):(?P<line>\d+):(?P<col>\d+)\)?"
)

# TypeScript error:  "file.ts(10,5): error TS2345: ..."
_TSC_ERROR_RE = re.compile(
    r"^(?P<path>.+)\((?P<line>\d+),(?P<col>\d+)\):\s+(?P<sev>error|warning)\s+TS\d+:\s+(?P<msg>.+)$"
)

# common JS/TS pitfalls (regex-based)
_PATTERNS: List[tuple] = [
    (re.compile(r"\beval\s*\("),       "warning", "Avoid eval() – security and performance risk.",
     "Remove eval(); refactor to use a lookup table or JSON.parse() instead."),
    (re.compile(r"(?<![=!])==(?!=)"),  "warning", "Use strict equality '===' instead of '=='.",
     "Replace '==' with '===' (and '!=' with '!==') to prevent type-coercion bugs."),
    (re.compile(r"\bvar\s+\w"),        "info",    "Prefer 'const' or 'let' over 'var'.",
     "Replace 'var' with 'const' (preferred) or 'let' to get block-scoped variables."),
    (re.compile(r"console\.log\s*\("), "info",    "Remove debug console.log() before shipping.",
     "Delete or replace with a proper logger (e.g., winston, pino, structuredLog)."),
    (re.compile(r"(?<!\w)undefined\s*=="),  "warning", "Unsafe undefined comparison.",
     "Use 'typeof x === \"undefined\"' or 'x === undefined' for reliable checks."),
    (re.compile(r"catch\s*\(\s*\)\s*\{"), "warning", "Empty catch block swallows errors.",
     "At minimum log the error: catch (e) { console.error(e); }"),
    (re.compile(r"new\s+Promise\s*\(.*resolve.*reject.*\{.*new\s+Promise", re.DOTALL),
     "warning", "Promise constructor anti-pattern (nested new Promise).",
     "Flatten promise chains; avoid wrapping existing promises in new Promise()."),
]


class JavaScriptTypeScriptAdapter(LanguageAdapter):
    language = "javascript/typescript"

    _JS_EXTS  = {".js", ".mjs", ".cjs", ".jsx"}
    _TS_EXTS  = {".ts", ".tsx"}
    _LANG_TAGS = {"javascript", "typescript", "js", "ts", "jsx", "tsx", "node"}

    # ── detection ─────────────────────────────────────────────────────────
    def detect(self, target: str) -> bool:
        ext = os.path.splitext(target)[1].lower()
        return ext in self._JS_EXTS | self._TS_EXTS or target.lower() in self._LANG_TAGS

    def _is_typescript(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._TS_EXTS

    # ── session ────────────────────────────────────────────────────────────
    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        is_ts = self._is_typescript(target)
        return {
            "target": target,
            "language": "typescript" if is_ts else "javascript",
            "options": options or {},
            "inspector_url": "ws://127.0.0.1:9229",   # default node inspector port
        }

    # ── execution control (CDP stubs) ─────────────────────────────────────
    def set_breakpoint(self, source_path: str, line: int,
                       column: Optional[int] = None) -> bool:
        return True  # real: Debugger.setBreakpoint via CDP websocket

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="JS/TS execution continued (Debugger.resume)")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="JS/TS execution paused (Debugger.pause)")

    # ── state inspection ───────────────────────────────────────────────────
    def read_stack(self) -> List[StackFrame]:
        return []  # real: Debugger.getStackTrace via CDP

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []  # real: Runtime.getProperties via CDP

    # ── error normalisation ────────────────────────────────────────────────
    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        frame: Optional[StackFrame] = None
        m = _V8_FRAME_RE.search(raw)
        if m:
            frame = StackFrame(
                id="0",
                name=m.group("func") or "<anonymous>",
                source_path=m.group("path"),
                line=int(m.group("line")),
                column=int(m.group("col")),
            )
        return Issue(
            severity="error",
            language=self.language,
            message=raw.splitlines()[0] if raw else "Unknown JS/TS error",
            source_path=frame.source_path if frame else None,
            line=frame.line if frame else None,
            column=frame.column if frame else None,
            frame=frame,
        )

    # ── static analysis ────────────────────────────────────────────────────
    def analyze_code(self, code: str, filename: str = "<code>") -> List[Issue]:
        issues: List[Issue] = []
        lines = code.splitlines()
        for lineno, text in enumerate(lines, start=1):
            for pattern, sev, msg, fix in _PATTERNS:
                if pattern.search(text):
                    issues.append(Issue(
                        severity=sev,
                        language=self.language,
                        message=msg,
                        source_path=filename,
                        line=lineno,
                        fix_suggestion=fix,
                    ))
        return issues

    def analyze_file(self, source_path: str) -> List[Issue]:
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                code = fh.read()
        except OSError as e:
            return [Issue(severity="error", language=self.language,
                          message=str(e), source_path=source_path)]

        issues = self.analyze_code(code, source_path)

        # Optional: node --check for JS syntax errors
        if not self._is_typescript(source_path):
            issues.extend(_node_check(source_path))
        else:
            issues.extend(_tsc_check(source_path))

        return issues


# ─────────────────────────────────────────────────────────────────────────────
# External tool helpers
# ─────────────────────────────────────────────────────────────────────────────
def _node_check(source_path: str) -> List[Issue]:
    try:
        result = subprocess.run(
            ["node", "--check", source_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return []
        issues = []
        for line in (result.stderr or result.stdout).splitlines():
            # "path.js:10\nSyntaxError: ..."
            m = re.match(r".+:(\d+)$", line)
            lineno = int(m.group(1)) if m else None
            if "SyntaxError" in line or lineno:
                issues.append(Issue(
                    severity="error",
                    language="javascript/typescript",
                    message=line.strip(),
                    source_path=source_path,
                    line=lineno,
                ))
        return issues
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []


def _tsc_check(source_path: str) -> List[Issue]:
    """Run tsc --noEmit on a single TypeScript file if tsc is available."""
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--allowJs", "--checkJs",
             "--strict", "--target", "ES2020", source_path],
            capture_output=True, text=True, timeout=30
        )
        issues = []
        for line in result.stdout.splitlines():
            m = _TSC_ERROR_RE.match(line)
            if m:
                issues.append(Issue(
                    severity=m.group("sev"),
                    language="typescript",
                    message=m.group("msg"),
                    source_path=m.group("path"),
                    line=int(m.group("line")),
                    column=int(m.group("col")),
                ))
        return issues
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
