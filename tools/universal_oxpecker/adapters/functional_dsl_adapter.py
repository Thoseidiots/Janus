"""
Functional / DSL adapter
Covers: Haskell, Erlang, Elixir, Clojure, F#, OCaml, Scala, SQL, HCL, YAML, TOML
Capabilities
  • detect:  file extensions for all listed languages
  • static:  per-language regex pitfall checks
  • error normalisation: best-effort parse → Issue
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

_EXT_MAP = {
    ".hs":    "haskell",
    ".lhs":   "haskell",
    ".erl":   "erlang",
    ".ex":    "elixir",
    ".exs":   "elixir",
    ".clj":   "clojure",
    ".cljs":  "clojure",
    ".cljc":  "clojure",
    ".fs":    "fsharp",
    ".fsx":   "fsharp",
    ".ml":    "ocaml",
    ".mli":   "ocaml",
    ".scala": "scala",
    ".sc":    "scala",
    ".sql":   "sql",
    ".tf":    "hcl",
    ".tfvars":"hcl",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".toml":  "toml",
    ".elm":   "elm",
    ".pl":    "prolog",
    ".pro":   "prolog",
}

_TAGS = {
    "haskell", "erlang", "elixir", "clojure", "fsharp", "f#",
    "ocaml", "scala", "sql", "hcl", "terraform", "yaml", "toml", "elm", "prolog",
    "functional", "dsl",
}

# Per-sub-language pattern banks
_HASKELL_PATTERNS = [
    (re.compile(r"\bhead\s+\[|\btail\s+\["),  "warning", "head/tail on empty list throws exception.",
     "Use pattern matching or safe variants (Data.Maybe) instead."),
    (re.compile(r"\bunsafePerformIO\b"),       "error",   "unsafePerformIO breaks referential transparency.",
     "Restructure the code to carry IO in the monad stack."),
]
_ELIXIR_PATTERNS = [
    (re.compile(r"\bIO\.puts\b"),              "info",    "Debug IO.puts in code.",
     "Remove or use Logger.debug/info for production."),
    (re.compile(r"rescue\s+e\s+in\s+_"),      "warning", "Catching all exceptions.",
     "Be specific: rescue e in [ArgumentError, RuntimeError]"),
]
_SQL_PATTERNS = [
    (re.compile(r"SELECT\s+\*", re.I),         "warning", "SELECT * fetches all columns.",
     "List only the columns you need to reduce I/O and improve query clarity."),
    (re.compile(r"DROP\s+TABLE", re.I),        "warning", "DROP TABLE is irreversible.",
     "Add 'IF EXISTS' and ensure a backup before executing."),
    (re.compile(r"DELETE\s+FROM\s+\w+\s*;", re.I), "warning", "DELETE without WHERE clause deletes all rows.",
     "Add a WHERE clause or wrap in a transaction with explicit ROLLBACK on error."),
]
_YAML_PATTERNS = [
    (re.compile(r":\s+yes$|:\s+no$|:\s+on$|:\s+off$", re.M), "warning",
     "YAML bare yes/no/on/off parsed as boolean.",
     "Quote the value: key: 'yes'  to keep it as a string."),
    (re.compile(r"\t"),                        "error",   "Tab character in YAML is not allowed.",
     "Replace all tabs with spaces."),
]


def _get_patterns(sublang: str) -> list:
    return {
        "haskell": _HASKELL_PATTERNS,
        "elixir":  _ELIXIR_PATTERNS,
        "sql":     _SQL_PATTERNS,
        "yaml":    _YAML_PATTERNS,
    }.get(sublang, [])


class FunctionalDslAdapter(LanguageAdapter):
    language = "functional/DSL"

    def detect(self, target: str) -> bool:
        ext = os.path.splitext(target)[1].lower()
        return ext in _EXT_MAP or target.lower() in _TAGS

    def _sublang(self, target: str) -> str:
        ext = os.path.splitext(target)[1].lower()
        return _EXT_MAP.get(ext, self.language)

    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "language": self._sublang(target), "options": options or {}}

    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        return True

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message=f"{self.language} execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message=f"{self.language} execution paused")

    def read_stack(self) -> List[StackFrame]:
        return []

    def read_variables(self, frame_id: str) -> List[Variable]:
        return []

    def normalize_error(self, error: Exception) -> Issue:
        raw = str(error)
        return Issue(severity="error", language=self.language, message=raw.splitlines()[0])

    def analyze_code(self, code: str, filename: str = "<code>") -> List[Issue]:
        sublang = self._sublang(filename)
        issues: List[Issue] = []
        for lineno, text in enumerate(code.splitlines(), 1):
            for pat, sev, msg, fix in _get_patterns(sublang):
                if pat.search(text):
                    issues.append(Issue(severity=sev, language=sublang,
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
        # YAML: validate with PyYAML if available
        sublang = self._sublang(source_path)
        if sublang == "yaml":
            issues.extend(_yaml_validate(source_path))
        return issues


def _yaml_validate(path: str) -> List[Issue]:
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            yaml.safe_load_all(fh)
    except Exception as e:
        return [Issue(severity="error", language="yaml", message=str(e), source_path=path)]
    return []
