"""
FixSuggester  ─  post-processing layer that enriches Issues with
human-readable fix suggestions based on known error patterns.

If the adapter already filled issue.fix_suggestion, this is a no-op.
Otherwise a rule-table lookup augments the issue in place.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    from ..core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from core.events import Issue

# Rule: (regex pattern to match issue.message, language or '*', fix text)
Rule = Tuple[re.Pattern, str, str]

_RULES: List[Rule] = [
    # ── Python ──────────────────────────────────────────────────────────
    (re.compile(r"NameError.*name '(\w+)' is not defined", re.I),
     "python",
     "Variable '{1}' is used before assignment. Check for typos or missing import."),

    (re.compile(r"AttributeError.*'(\w+)' object has no attribute '(\w+)'", re.I),
     "python",
     "Object of type '{1}' does not have attribute '{2}'. "
     "Check spelling, ensure the object is the expected type, and look for None."),

    (re.compile(r"TypeError.*takes (\d+) positional argument", re.I),
     "python",
     "Wrong number of arguments. Check the function signature and the call site."),

    (re.compile(r"IndentationError", re.I),
     "python",
     "Fix indentation: use 4 spaces per level consistently. Never mix tabs and spaces."),

    (re.compile(r"ImportError|ModuleNotFoundError", re.I),
     "python",
     "Module not found. Run: pip install <package-name>  "
     "or check the import path for typos."),

    (re.compile(r"KeyError:\s*'?(\w+)'?", re.I),
     "python",
     "Key '{1}' not found in dict. Use dict.get(key, default) for safe access."),

    (re.compile(r"ZeroDivisionError", re.I),
     "python",
     "Guard the denominator: if denominator != 0: result = numerator / denominator"),

    (re.compile(r"RecursionError", re.I),
     "python",
     "Stack overflow from infinite recursion. Add a base case or increase sys.setrecursionlimit()."),

    # ── JavaScript / TypeScript ─────────────────────────────────────────
    (re.compile(r"Cannot read propert(?:y|ies) of (undefined|null)", re.I),
     "javascript/typescript",
     "The object is {1} when accessed. Use optional chaining: obj?.prop  "
     "or add a null guard before access."),

    (re.compile(r"is not a function", re.I),
     "javascript/typescript",
     "The value is not callable. Verify the variable holds a function, "
     "and check for missing parentheses or incorrect import."),

    (re.compile(r"ReferenceError.*is not defined", re.I),
     "javascript/typescript",
     "Variable used before declaration. Move the declaration to the top, "
     "or check for a missing import/require."),

    (re.compile(r"SyntaxError.*Unexpected token", re.I),
     "javascript/typescript",
     "Unexpected token – usually a missing comma, bracket, or semicolon "
     "on or just before the reported line."),

    # ── Java / Kotlin ───────────────────────────────────────────────────
    (re.compile(r"NullPointerException", re.I),
     "java/kotlin",
     "A reference is null. Add a null check or use Optional<T> / Kotlin's ?. operator."),

    (re.compile(r"ClassCastException", re.I),
     "java/kotlin",
     "Invalid cast. Use instanceof / is before casting, or redesign with generics."),

    (re.compile(r"ArrayIndexOutOfBoundsException", re.I),
     "java/kotlin",
     "Array index out of range. Check loop bounds: use arr.length or indices."),

    (re.compile(r"StackOverflowError", re.I),
     "java/kotlin",
     "Infinite recursion. Add a base case to recursive method."),

    # ── C / C++ ─────────────────────────────────────────────────────────
    (re.compile(r"segmentation fault|SIGSEGV", re.I),
     "c/c++",
     "Segfault: check for NULL pointer dereference, out-of-bounds array access, "
     "or stack corruption."),

    (re.compile(r"double free|heap corruption", re.I),
     "c/c++",
     "Double-free or heap corruption: ensure each malloc'd pointer is free'd exactly once "
     "and set to NULL after freeing."),

    # ── Rust ────────────────────────────────────────────────────────────
    (re.compile(r"cannot borrow.*as mutable.*is also borrowed as immutable", re.I),
     "rust",
     "Borrow checker conflict. Restructure to end the immutable borrow before "
     "creating the mutable borrow."),

    (re.compile(r"does not live long enough", re.I),
     "rust",
     "Lifetime violation. Ensure the referenced data lives at least as long as the reference. "
     "Consider owning the data (use String instead of &str, or clone)."),

    # ── Go ───────────────────────────────────────────────────────────────
    (re.compile(r"nil pointer dereference", re.I),
     "go",
     "Nil pointer dereference. Check for nil before calling methods: "
     "if ptr != nil { ptr.Method() }"),

    (re.compile(r"index out of range", re.I),
     "go",
     "Slice/array index out of range. Verify loop bounds and slice lengths."),

    # ── Generic / cross-language ─────────────────────────────────────────
    (re.compile(r"timeout|timed out", re.I),
     "*",
     "Operation timed out. Increase the timeout threshold or add retry logic "
     "with exponential back-off."),

    (re.compile(r"connection refused|ECONNREFUSED", re.I),
     "*",
     "Connection refused. Verify the server is running, the host/port are correct, "
     "and that firewalls allow the connection."),

    (re.compile(r"permission denied|EACCES|EPERM", re.I),
     "*",
     "Permission denied. Check file/directory ownership and mode bits (chmod/chown)."),

    (re.compile(r"out of memory|OOM|MemoryError", re.I),
     "*",
     "Out of memory. Profile heap usage, stream large datasets instead of loading "
     "all at once, and add memory limits."),
]


class FixSuggester:
    """
    Attach fix suggestions and repair metadata to Issues.
    """

    def suggest(self, issue: Issue) -> Optional[str]:
        """Return a fix suggestion string, or None if no rule matched."""
        for pattern, lang, template in _RULES:
            if lang != "*" and lang not in (issue.language or ""):
                continue
            m = pattern.search(issue.message or "")
            if m:
                result = template
                for i, group in enumerate(m.groups(), start=1):
                    result = result.replace("{" + str(i) + "}", group or "?")
                return result
        return None

    def classify_complexity(self, issue: Issue) -> str:
        message = (issue.message or "").lower()
        suggestion = (issue.fix_suggestion or "").lower()
        combined = f"{message} {suggestion}"

        tier_1_tokens = [
            "strict equality", "===", "var", "console.log", "bare 'except:'",
            "use '==' instead of 'is'", "indentation", "syntaxerror",
        ]
        tier_3_tokens = [
            "segmentation fault", "sigsegv", "double free", "heap corruption",
            "unsafe", "eval", "out of memory", "recursionerror", "nullpointerexception",
        ]

        if any(token in combined for token in tier_3_tokens):
            return "Tier 3"
        if any(token in combined for token in tier_1_tokens):
            return "Tier 1"
        return "Tier 2"

    def enrich(self, issues: List[Issue]) -> List[Issue]:
        """Enrich a list of Issues in-place and return them."""
        for issue in issues:
            if not issue.fix_suggestion:
                issue.fix_suggestion = self.suggest(issue)
            if not issue.complexity_tier:
                issue.complexity_tier = self.classify_complexity(issue)
            if issue.repair_stage == "localized" and issue.fix_suggestion:
                issue.repair_stage = "proposed"
        return issues
