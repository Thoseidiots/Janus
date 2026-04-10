"""
Rust adapter
Capabilities
  • detect:  *.rs files or 'rust' tag
  • static:  regex pitfall checks + optional `rustc --edition 2021 --error-format=json`
  • error normalisation: rustc JSON error format → Issue
"""
from __future__ import annotations
import json, os, re, subprocess
from typing import Any, Dict, List, Optional
try:
    from ..core.adapter import LanguageAdapter
    from ..core.events import DebugEvent, Issue, StackFrame, Variable
except ImportError:  # pragma: no cover - top-level package fallback
    from core.adapter import LanguageAdapter
    from core.events import DebugEvent, Issue, StackFrame, Variable

_PATTERNS = [
    (re.compile(r"\bunwrap\s*\(\s*\)"),     "warning", "unwrap() will panic on None/Err.",
     "Use expect(\"message\") for debugging or handle with if let / match / ? operator."),
    (re.compile(r"\bpanic!\s*\("),          "warning", "Explicit panic!() in code.",
     "Replace panic! with a proper Result return or thiserror/anyhow error type."),
    (re.compile(r"\bclone\s*\(\s*\)"),      "info",    "clone() call detected – may be avoidable.",
     "Consider borrowing instead of cloning if lifetime permits."),
    (re.compile(r"unsafe\s*\{"),            "warning", "Unsafe block detected.",
     "Document invariants that make the unsafe block sound, or find a safe abstraction."),
    (re.compile(r"#\[allow\(unused"),       "info",    "#[allow(unused_...)] suppressor.",
     "Remove dead code instead of silencing the warning."),
    (re.compile(r"todo!\s*\(\s*\)"),        "info",    "todo!() placeholder.",
     "Implement the function body before shipping."),
    (re.compile(r"eprintln!\s*\("),         "info",    "Debug eprintln! in code.",
     "Remove or replace with a tracing/log macro."),
]


class RustAdapter(LanguageAdapter):
    language = "rust"
    _EXTS = {".rs"}
    _TAGS = {"rust", "rs"}

    def detect(self, target: str) -> bool:
        return os.path.splitext(target)[1].lower() in self._EXTS or target.lower() in self._TAGS

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self.language, "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True  # real: rust-gdb / lldb-mi or DAP via rust-analyzer

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Rust execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Rust execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        # Try to parse rustc JSON
        try:
            obj = json.loads(raw)
            spans = obj.get("spans", [])
            sp = spans[0] if spans else {}
            return Issue(
                severity=obj.get("level", "error"),
                language=self.language,
                message=obj.get("message", raw),
                source_path=sp.get("file_name"),
                line=sp.get("line_start"),
                column=sp.get("column_start"),
            )
        except (json.JSONDecodeError, AttributeError):
            return Issue(severity="error", language=self.language, message=raw.splitlines()[0])

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
        issues.extend(_rustc_check(source_path))
        return issues


def _rustc_check(path: str) -> List[Issue]:
    try:
        r = subprocess.run(
            ["rustc", "--edition", "2021", "--error-format=json",
             "--emit=metadata", "-o", "/dev/null", path],
            capture_output=True, text=True, timeout=30
        )
        out = []
        for line in r.stderr.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("level") not in ("error", "warning"):
                continue
            spans = obj.get("spans", [])
            sp = spans[0] if spans else {}
            out.append(Issue(
                severity=obj["level"],
                language="rust",
                message=obj.get("message", ""),
                source_path=sp.get("file_name"),
                line=sp.get("line_start"),
                column=sp.get("column_start"),
            ))
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
