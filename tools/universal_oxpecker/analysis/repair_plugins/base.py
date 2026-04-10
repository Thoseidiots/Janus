from __future__ import annotations

from abc import ABC
from typing import List, Optional, Tuple

try:
    from ..repair_models import CandidateDraft
    from ...core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from analysis.repair_models import CandidateDraft
    from core.events import Issue


class RepairPlugin(ABC):
    name = "base"
    language = "*"

    def supports(self, language: str) -> bool:
        return self.language == "*" or self.language == language

    def build_candidates(
        self,
        source_path: str,
        original_text: str,
        issue: Issue,
        engine,
    ) -> List[CandidateDraft]:
        return []

    def preflight_validate(
        self,
        candidate_text: str,
        source_path: str,
        issue: Optional[Issue] = None,
    ) -> Tuple[bool, str]:
        return True, "Preflight validation passed."
