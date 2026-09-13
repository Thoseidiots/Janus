from __future__ import annotations

import re
from typing import List, Optional

try:
    from ..repair_models import CandidateDraft
    from .base import RepairPlugin
    from ...core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from analysis.repair_models import CandidateDraft
    from analysis.repair_plugins.base import RepairPlugin
    from core.events import Issue


class JavaScriptTypeScriptRepairPlugin(RepairPlugin):
    name = "javascript-typescript"
    language = "javascript/typescript"

    def build_candidates(
        self,
        source_path: str,
        original_text: str,
        issue: Issue,
        engine,
    ) -> List[CandidateDraft]:
        message = (issue.message or "").lower()
        candidates: List[CandidateDraft] = []

        if issue.line and ("strict equality" in message or "==='" in message):
            strict_text = _replace_on_line(original_text, issue.line, _replace_loose_equality)
            if strict_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="Replace loose equality with strict equality.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=strict_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="text",
                    )
                )

        if issue.line and "prefer 'const' or 'let' over 'var'" in message:
            const_text = _replace_on_line(original_text, issue.line, lambda line: re.sub(r"\bvar\b", "const", line, count=1))
            if const_text != original_text:
                metadata = {"binding": "const"}
                if not _is_reassigned(original_text, issue.line):
                    metadata["preferred"] = "true"
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="Replace var with const.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=const_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="text",
                        metadata=metadata,
                    )
                )
            let_text = _replace_on_line(original_text, issue.line, lambda line: re.sub(r"\bvar\b", "let", line, count=1))
            if let_text != original_text:
                metadata = {"binding": "let"}
                if _is_reassigned(original_text, issue.line):
                    metadata["preferred"] = "true"
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="Replace var with let.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=let_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="text",
                        metadata=metadata,
                    )
                )

        return _dedupe(candidates)

    def preflight_validate(
        self,
        candidate_text: str,
        source_path: str,
        issue: Optional[Issue] = None,
    ):
        return True, "JavaScript/TypeScript preflight delegated to static validation."


def _replace_on_line(text: str, line_no: int, transform) -> str:
    lines = text.splitlines(keepends=True)
    index = line_no - 1
    if index < 0 or index >= len(lines):
        return text
    updated = transform(lines[index])
    if updated == lines[index]:
        return text
    lines[index] = updated
    return "".join(lines)


def _replace_loose_equality(line: str) -> str:
    line = re.sub(r"!=(?!=)", "!==", line)
    line = re.sub(r"(?<!!)==(?!=)", "===", line)
    return line


def _extract_declared_name(line: str) -> Optional[str]:
    match = re.search(r"\bvar\s+([A-Za-z_$][\w$]*)", line)
    return match.group(1) if match else None


def _is_reassigned(text: str, declaration_line: int) -> bool:
    lines = text.splitlines()
    if declaration_line - 1 >= len(lines):
        return False
    name = _extract_declared_name(lines[declaration_line - 1])
    if not name:
        return False
    assignment_re = re.compile(rf"\b{name}\b\s*([+\-*/%]?=|\+\+|--)" )
    for line in lines[declaration_line:]:
        if assignment_re.search(line):
            return True
    return False


def _dedupe(candidates: List[CandidateDraft]) -> List[CandidateDraft]:
    seen = set()
    unique: List[CandidateDraft] = []
    for candidate in candidates:
        key = (candidate.description, candidate.candidate_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
