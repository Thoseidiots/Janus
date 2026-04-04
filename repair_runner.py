"""
Repair_Runner — orchestrates Oxpecker codebase repair for the Janus repository.

Calls OxpeckerOrchestrator.repair_project() and aggregates results into a
RepairReport. Priority files are processed first.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the real Oxpecker stack; fall back to minimal stubs so the
# module is importable even when tools/universal_oxpecker is not on sys.path.
# ---------------------------------------------------------------------------
try:
    _oxpecker_root = os.path.join(os.path.dirname(__file__), "tools", "universal_oxpecker")
    if _oxpecker_root not in sys.path:
        sys.path.insert(0, _oxpecker_root)

    from core.orchestrator import OxpeckerOrchestrator  # type: ignore
    from analysis.repair_workflow import RepairResult  # type: ignore

    _OXPECKER_AVAILABLE = True

except ImportError:
    _OXPECKER_AVAILABLE = False

    # ------------------------------------------------------------------
    # Minimal stubs — used only when Oxpecker cannot be imported.
    # ------------------------------------------------------------------
    @dataclass
    class RepairResult:  # type: ignore[no-redef]
        original_path: str = ""
        working_path: str = ""
        language: str = ""
        initial_issues: list = field(default_factory=list)
        applied_patches: list = field(default_factory=list)
        candidate_history: list = field(default_factory=list)
        rollback_history: list = field(default_factory=list)
        test_runs: list = field(default_factory=list)
        remaining_issues: list = field(default_factory=list)
        pending_manual_fixes: list = field(default_factory=list)

        @property
        def initial_issue_count(self) -> int:
            return len(self.initial_issues)

        @property
        def final_issue_count(self) -> int:
            return len(self.remaining_issues)

        @property
        def fixed_issue_count(self) -> int:
            return max(0, self.initial_issue_count - self.final_issue_count)

    class OxpeckerOrchestrator:  # type: ignore[no-redef]
        """Fallback stub — returns empty results when Oxpecker is unavailable."""

        def repair_project(
            self,
            target: str,
            recursive: bool = True,
            workers: int = 4,
            output_dir: Optional[str] = None,
            max_rounds: int = 10,
        ) -> List[RepairResult]:
            logger.warning(
                "OxpeckerOrchestrator stub: Oxpecker not available. "
                "Returning empty repair results for %s",
                target,
            )
            return []


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    """Summary produced by RepairRunner after a repair pass."""

    fixed_count: int
    remaining_count: int
    manual_fix_files: List[str]


# ---------------------------------------------------------------------------
# RepairRunner
# ---------------------------------------------------------------------------

class RepairRunner:
    """
    Orchestrates an Oxpecker repair pass over a repository root.

    Priority files (PRIORITY_FILES) are guaranteed to be processed before
    any other files in the project.
    """

    PRIORITY_FILES: List[str] = [
        "App.tsx",
        "JanusHub.tsx",
        "HyperparameterGenerator.tsx",
        "UnifiedStudio.tsx",
        "Chatbot.tsx",
        "actions_and_feedback.py",
        "advanced_3d_face_generator.py",
        "agent_loop.py",
        "analyzer_v5_full_system.py",
        # All files under archive/ are also priority — matched by basename prefix
        "archive/cognitive_loop.py",
        "archive/janus_capability_hub.py",
        "archive/janus_core.py",
        "archive/memory.py",
        "archive/updated_cognitive_loop.py",
        "archive/web_autonomy.py",
    ]

    def __init__(self) -> None:
        self._orchestrator = OxpeckerOrchestrator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, repo_root: str) -> RepairReport:
        """
        Invoke OxpeckerOrchestrator.repair_project() against *repo_root*,
        ordering results so PRIORITY_FILES are processed first.

        Returns a RepairReport summarising the outcome.
        """
        try:
            results: List[RepairResult] = self._orchestrator.repair_project(
                repo_root,
                output_dir=".oxpecker_work",
            )
        except Exception as exc:  # pragma: no cover
            logger.error("repair_project raised an exception: %s", exc)
            return RepairReport(fixed_count=0, remaining_count=0, manual_fix_files=[])

        ordered = self._sort_by_priority(results, repo_root)
        return self.report(ordered)

    def report(self, results: List[RepairResult]) -> RepairReport:
        """
        Aggregate a list of RepairResult objects into a RepairReport.

        - fixed_count    : number of results where fixed_issue_count > 0
        - remaining_count: total remaining issues across all results
        - manual_fix_files: paths of files that have pending_manual_fixes
        """
        fixed_count = sum(1 for r in results if r.fixed_issue_count > 0)
        remaining_count = sum(r.final_issue_count for r in results)
        manual_fix_files = [
            r.original_path
            for r in results
            if r.pending_manual_fixes
        ]
        return RepairReport(
            fixed_count=fixed_count,
            remaining_count=remaining_count,
            manual_fix_files=manual_fix_files,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_by_priority(
        self, results: List[RepairResult], repo_root: str
    ) -> List[RepairResult]:
        """
        Return *results* reordered so that PRIORITY_FILES come first (in the
        order they appear in PRIORITY_FILES), followed by all remaining files
        in their original scan order.
        """
        # Build a lookup: normalised relative path → priority index
        priority_index: dict[str, int] = {}
        for idx, pf in enumerate(self.PRIORITY_FILES):
            # Normalise to forward-slash, lower-case for comparison
            priority_index[pf.replace("\\", "/").lower()] = idx

        def _priority_key(result: RepairResult) -> tuple[int, str]:
            rel = _relative_path(result.original_path, repo_root)
            norm = rel.replace("\\", "/").lower()
            # Also check basename for simple filename matches
            basename = os.path.basename(result.original_path).lower()
            # Try full relative path first, then basename
            if norm in priority_index:
                return (priority_index[norm], norm)
            if basename in priority_index:
                return (priority_index[basename], norm)
            # Check if the path ends with any priority suffix (handles archive/ files)
            for pf, pidx in priority_index.items():
                if norm.endswith(pf):
                    return (pidx, norm)
            return (len(self.PRIORITY_FILES), norm)

        return sorted(results, key=_priority_key)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _relative_path(path: str, base: str) -> str:
    """Return *path* relative to *base*, or *path* unchanged if not under *base*."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        # On Windows, relpath raises ValueError for paths on different drives.
        return path
