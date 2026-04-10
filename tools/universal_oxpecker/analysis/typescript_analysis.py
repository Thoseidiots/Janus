"""
TypeScriptAnalyzer  ─  helper that complements the JS/TS adapter with
additional TypeScript-specific static checks that benefit from reading
the whole file as a unit (cross-line patterns, import analysis, etc.).
"""
from __future__ import annotations

import re
from typing import List

try:
    from ..core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from core.events import Issue

# Cross-line / file-level TS checks
_TS_CHECKS = [
    # any-type sprawl
    (re.compile(r":\s*any\b"),
     "warning",
     "Explicit 'any' type weakens TypeScript's type safety.",
     "Replace 'any' with a specific type, 'unknown', or a generic parameter."),

    # non-null assertion overuse
    (re.compile(r"!\s*[;,)\]]"),
     "warning",
     "Non-null assertion '!' bypasses null safety.",
     "Use optional chaining (?.) or add an explicit null check instead."),

    # @ts-ignore suppressor
    (re.compile(r"//\s*@ts-ignore"),
     "warning",
     "@ts-ignore suppresses all TypeScript errors on the next line.",
     "Fix the underlying type error instead of suppressing it, "
     "or use @ts-expect-error with a comment explaining why."),

    # require() in a TS file (should use import)
    (re.compile(r"\brequire\s*\("),
     "info",
     "require() in a TypeScript file.",
     "Use ES module syntax: import { X } from 'module';"),

    # Object spread on potentially undefined
    (re.compile(r"\.\.\.\w+\s*[,}]"),
     "info",
     "Object spread – ensure source is not undefined/null.",
     "Guard with: ...( obj ?? {} )"),
]


class TypeScriptAnalyzer:
    """
    Run TypeScript-specific analysis on source text.
    Intended to be called from JavaScriptTypeScriptAdapter.analyze_code()
    or directly for more detailed TS-only analysis.
    """

    def analyze(self, code: str, filename: str = "<code>") -> List[Issue]:
        issues: List[Issue] = []
        lines = code.splitlines()

        for lineno, text in enumerate(lines, start=1):
            for pattern, sev, msg, fix in _TS_CHECKS:
                if pattern.search(text):
                    issues.append(Issue(
                        severity=sev,
                        language="typescript",
                        message=msg,
                        source_path=filename,
                        line=lineno,
                        fix_suggestion=fix,
                    ))

        # File-level check: no explicit return type on exported functions
        _check_missing_return_types(lines, filename, issues)
        return issues


def _check_missing_return_types(lines: List[str], filename: str, issues: List[Issue]) -> None:
    """Flag exported functions/methods with no return-type annotation."""
    pattern = re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    code = "\n".join(lines)
    for m in pattern.finditer(code):
        lineno = code[: m.start()].count("\n") + 1
        func_name = m.group(1)
        # only flag if there's no ': ' before the opening brace
        sig = m.group(0)
        if ":" not in sig.split("(", 1)[-1]:
            issues.append(Issue(
                severity="info",
                language="typescript",
                message=f"Function '{func_name}' has no explicit return type annotation.",
                source_path=filename,
                line=lineno,
                fix_suggestion=(
                    f"Add a return type: function {func_name}(...): ReturnType {{ ... }}"
                ),
            ))
