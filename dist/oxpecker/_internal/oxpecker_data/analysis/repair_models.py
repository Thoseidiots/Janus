from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from ..core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from core.events import Issue


@dataclass
class CandidateDraft:
    source_path: str
    line: Optional[int]
    description: str
    complexity_tier: str
    candidate_text: str
    issue_message: str = ""
    plugin_name: str = "generic"
    patch_kind: str = "text"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatchProposal:
    source_path: str
    line: Optional[int]
    description: str
    complexity_tier: str
    diff: str
    candidate_text: str
    stage: str = "proposed"
    approval: str = "pending"
    validation_notes: str = ""
    score: int = 0
    issue_message: str = ""
    plugin_name: str = "generic"
    patch_kind: str = "text"
    issue_reduction: int = 0
    test_status: str = "not-run"
    test_command: Optional[str] = None
    test_output: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RollbackEntry:
    revision: int
    description: str
    snapshot_path: str
    created_from: str
    reverted: bool = False


@dataclass
class TestRunResult:
    command: str
    working_directory: str
    exit_code: int
    success: bool
    stdout: str = ""
    stderr: str = ""
    kind: str = "candidate"


@dataclass
class RepairResult:
    original_path: str
    working_path: str
    language: str
    initial_issues: List[Issue] = field(default_factory=list)
    applied_patches: List[PatchProposal] = field(default_factory=list)
    candidate_history: List[PatchProposal] = field(default_factory=list)
    rollback_history: List[RollbackEntry] = field(default_factory=list)
    test_runs: List[TestRunResult] = field(default_factory=list)
    remaining_issues: List[Issue] = field(default_factory=list)
    pending_manual_fixes: List[Issue] = field(default_factory=list)

    @property
    def initial_issue_count(self) -> int:
        return len(self.initial_issues)

    @property
    def final_issue_count(self) -> int:
        return len(self.remaining_issues)

    @property
    def fixed_issue_count(self) -> int:
        return max(0, self.initial_issue_count - self.final_issue_count)
