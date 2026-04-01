from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .fix_suggester import FixSuggester
    from .repair_models import (
        CandidateDraft,
        PatchProposal,
        RepairResult,
        RollbackEntry,
        TestRunResult,
    )
    from .repair_plugins import BUILTIN_REPAIR_PLUGINS
    from ..core.events import Issue
    from ..core.scanner import ProjectScanner
except ImportError:  # pragma: no cover - top-level package fallback
    from analysis.fix_suggester import FixSuggester
    from analysis.repair_models import (
        CandidateDraft,
        PatchProposal,
        RepairResult,
        RollbackEntry,
        TestRunResult,
    )
    from analysis.repair_plugins import BUILTIN_REPAIR_PLUGINS
    from core.events import Issue
    from core.scanner import ProjectScanner


class AutomatedProgramRepair:
    """
    Stronger APR workflow:
      localize -> plugin-driven multi-candidate synthesis -> AST-aware preflight
      -> static validation -> optional test-run validation -> ranking -> apply on a copy
      -> persistent rollback history.
    The original file is never modified.
    """

    def __init__(self, engine, plugins=None) -> None:
        self.engine = engine
        self.suggester = FixSuggester()
        self.scanner = ProjectScanner(engine)
        self.plugins = list(plugins or BUILTIN_REPAIR_PLUGINS)
        self._baseline_test_cache: Dict[Tuple[str, str, str], Optional[Dict[str, object]]] = {}

    def repair_file(
        self,
        source_path: str,
        output_dir: Optional[str] = None,
        max_rounds: int = 10,
    ) -> RepairResult:
        adapter = self.engine.detect_adapter(source_path)
        
        # APR-STATE-006: Check if a working copy already exists to resume state
        working_path = self._get_existing_working_copy(source_path, output_dir=output_dir)
        if not working_path:
            working_path = self._create_working_copy(source_path, output_dir=output_dir)
            self._ensure_history_manifest(working_path)
        
        initial_issues = self.engine.debug_file(working_path)
        result = RepairResult(
            original_path=source_path,
            working_path=working_path,
            language=adapter.language,
            initial_issues=initial_issues,
        )

        baseline_context = self._get_baseline_test_context(source_path, adapter.language)
        if baseline_context and baseline_context.get("baseline_result"):
            result.test_runs.append(baseline_context["baseline_result"])  # type: ignore[arg-type]

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            current_issues = self.engine.debug_file(working_path)
            if not current_issues:
                result.remaining_issues = []
                result.pending_manual_fixes = []
                result.rollback_history = self.list_rollback_history(working_path)
                return result

            proposals, manual_only = self._propose_for_file(working_path, current_issues)
            if not proposals:
                result.remaining_issues = current_issues
                result.pending_manual_fixes = manual_only
                result.rollback_history = self.list_rollback_history(working_path)
                return result

            ranked, test_runs = self._validate_and_rank(
                working_path,
                source_path,
                adapter.language,
                current_issues,
                proposals,
                baseline_context,
            )
            result.candidate_history.extend(ranked)
            result.test_runs.extend(test_runs)

            approved = [proposal for proposal in ranked if proposal.approval == "approved"]
            if not approved:
                result.remaining_issues = current_issues
                result.pending_manual_fixes = manual_only
                result.rollback_history = self.list_rollback_history(working_path)
                return result

            chosen = approved[0]
            self._record_rollback_snapshot(working_path, chosen.description)
            with open(working_path, "w", encoding="utf-8") as fh:
                fh.write(chosen.candidate_text)
            chosen.stage = "applied"
            result.applied_patches.append(chosen)

        result.remaining_issues = self.engine.debug_file(working_path)
        result.pending_manual_fixes = [
            issue for issue in result.remaining_issues
            if issue.complexity_tier in {"Tier 2", "Tier 3"}
        ]
        result.rollback_history = self.list_rollback_history(working_path)
        return result

    def repair_project(
        self,
        target: str,
        recursive: bool = True,
        workers: int = 4,
        output_dir: Optional[str] = None,
        max_rounds: int = 10,
    ) -> List[RepairResult]:
        scan_results = self.scanner.scan(target, recursive=recursive, workers=workers)
        repairs: List[RepairResult] = []
        for scan in scan_results:
            repairs.append(self.repair_file(scan.source_path, output_dir=output_dir, max_rounds=max_rounds))
        return repairs

    def rollback_last(self, working_path: str, steps: int = 1) -> RollbackEntry:
        manifest = self._load_history_manifest(working_path)
        candidates = [entry for entry in manifest["entries"] if not entry.get("reverted")]
        if not candidates:
            raise ValueError(f"No rollback snapshots available for {working_path}.")

        restored: Optional[Dict[str, object]] = None
        for _ in range(max(1, steps)):
            active = [entry for entry in manifest["entries"] if not entry.get("reverted")]
            if not active:
                break
            restored = active[-1]
            shutil.copy2(restored["snapshot_path"], working_path)
            restored["reverted"] = True
        self._save_history_manifest(working_path, manifest)
        if restored is None:
            raise ValueError(f"No rollback snapshots available for {working_path}.")
        return self._rollback_from_dict(restored)

    def list_rollback_history(self, working_path: str) -> List[RollbackEntry]:
        manifest = self._load_history_manifest(working_path)
        return [self._rollback_from_dict(entry) for entry in manifest["entries"]]

    def _get_existing_working_copy(self, source_path: str, output_dir: Optional[str]) -> Optional[str]:
        destination_root = output_dir or os.path.join(
            os.path.dirname(source_path),
            ".oxpecker_work",
        )
        destination = os.path.join(destination_root, os.path.basename(source_path))
        if os.path.exists(destination):
            return destination
        return None

    def _create_working_copy(self, source_path: str, output_dir: Optional[str]) -> str:
        destination_root = output_dir or os.path.join(
            os.path.dirname(source_path),
            ".oxpecker_work",
        )
        os.makedirs(destination_root, exist_ok=True)
        destination = os.path.join(destination_root, os.path.basename(source_path))
        shutil.copy2(source_path, destination)
        return destination

    def _propose_for_file(
        self,
        source_path: str,
        issues: List[Issue],
    ) -> Tuple[List[PatchProposal], List[Issue]]:
        with open(source_path, "r", encoding="utf-8") as fh:
            original_text = fh.read()

        proposals: List[PatchProposal] = []
        manual_only: List[Issue] = []
        seen = set()

        for issue in issues:
            if not issue.fix_suggestion:
                issue.fix_suggestion = self.suggester.suggest(issue)
            if not issue.complexity_tier:
                issue.complexity_tier = self.suggester.classify_complexity(issue)

            language = issue.language or self.engine.detect_adapter(source_path).language
            drafts: List[CandidateDraft] = []
            for plugin in self._plugins_for_language(language):
                drafts.extend(plugin.build_candidates(source_path, original_text, issue, self.engine))

            if not drafts:
                manual_only.append(issue)
                continue

            built_any = False
            for draft in drafts:
                key = (draft.description, draft.candidate_text)
                if key in seen:
                    continue
                seen.add(key)
                proposal = self._proposal_from_draft(original_text, issue, draft)
                if proposal is not None:
                    proposals.append(proposal)
                    built_any = True
            if not built_any:
                manual_only.append(issue)

        return proposals, manual_only

    def _proposal_from_draft(
        self,
        original_text: str,
        issue: Issue,
        draft: CandidateDraft,
    ) -> Optional[PatchProposal]:
        if draft.candidate_text == original_text:
            return None
        diff = "\n".join(
            difflib.unified_diff(
                original_text.splitlines(),
                draft.candidate_text.splitlines(),
                fromfile=draft.source_path,
                tofile=draft.source_path,
                lineterm="",
            )
        )
        score = _severity_score(issue.severity) + _tier_score(draft.complexity_tier)
        if draft.patch_kind == "text":
            score += 15
        if draft.metadata.get("preferred") == "true":
            score += 10
        return PatchProposal(
            source_path=draft.source_path,
            line=draft.line,
            description=draft.description,
            complexity_tier=draft.complexity_tier,
            diff=diff,
            candidate_text=draft.candidate_text,
            issue_message=draft.issue_message,
            score=score,
            plugin_name=draft.plugin_name,
            patch_kind=draft.patch_kind,
            metadata=draft.metadata,
        )

    def _validate_and_rank(
        self,
        working_path: str,
        source_path: str,
        language: str,
        current_issues: List[Issue],
        proposals: List[PatchProposal],
        baseline_context: Optional[Dict[str, object]],
    ) -> Tuple[List[PatchProposal], List[TestRunResult]]:
        current_count = len(current_issues)
        ranked: List[PatchProposal] = []
        test_runs: List[TestRunResult] = []
        current_signatures = {_issue_signature(issue) for issue in current_issues}

        for proposal in proposals:
            plugin = self._plugin_by_name(proposal.plugin_name)
            preflight_ok = True
            preflight_notes = "Plugin preflight skipped."
            if plugin is not None:
                preflight_ok, preflight_notes = plugin.preflight_validate(
                    proposal.candidate_text,
                    source_path,
                    None,
                )

            if not preflight_ok:
                proposal.approval = "needs-review"
                proposal.validation_notes = preflight_notes
                ranked.append(proposal)
                continue

            reduced_count, notes, candidate_issues, test_result = self._validate_candidate(
                source_path,
                proposal,
                current_count,
                language,
                baseline_context,
            )
            proposal.validation_notes = f"{preflight_notes} {notes}".strip()
            proposal.issue_reduction = current_count - reduced_count
            if proposal.issue_reduction > 0:
                proposal.score += proposal.issue_reduction * 20
            remaining_signatures = {_issue_signature(issue) for issue in candidate_issues}
            if proposal.issue_message and proposal.issue_message not in {sig[0] for sig in remaining_signatures}:
                proposal.score += 10

            if test_result is not None:
                test_runs.append(test_result)
                proposal.test_status = "passed" if test_result.success else "failed"
                proposal.test_command = test_result.command
                proposal.test_output = (test_result.stdout or test_result.stderr)[-1200:]
                if test_result.success:
                    proposal.score += 25
                else:
                    proposal.score -= 40
            else:
                proposal.test_status = "not-run"

            baseline_success = bool(baseline_context and baseline_context.get("baseline_success"))
            if proposal.issue_reduction > 0 and (not baseline_success or proposal.test_status != "failed"):
                proposal.approval = "approved"
            else:
                proposal.approval = "needs-review"
            ranked.append(proposal)


        ranked.sort(
            key=lambda item: (
                item.approval != "approved",
                -(item.score),
                item.patch_kind != "text",
                item.line or 0,
            )
        )
        return ranked, test_runs

    def _validate_candidate(
        self,
        source_path: str,
        proposal: PatchProposal,
        current_count: int,
        language: str,
        baseline_context: Optional[Dict[str, object]],
    ) -> Tuple[int, str, List[Issue], Optional[TestRunResult]]:
        test_result: Optional[TestRunResult] = None
        project_root = baseline_context.get("project_root") if baseline_context else None
        command = baseline_context.get("command") if baseline_context else None

        # APR-VAL-007: Validation Sandbox Isolation
        if project_root and command:
            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = os.path.join(tmpdir, "candidate_workspace")
                # Optimization: Only copy necessary files or use a lightweight approach if possible
                shutil.copytree(
                    project_root,
                    workspace,
                    ignore=shutil.ignore_patterns(
                        "__pycache__",
                        ".pytest_cache",
                        ".oxpecker_work",
                        ".oxpecker_history",
                        ".git",
                    ),
                )
                relative_path = os.path.relpath(os.path.abspath(source_path), os.path.abspath(project_root))
                candidate_path = os.path.join(workspace, relative_path)
                os.makedirs(os.path.dirname(candidate_path), exist_ok=True)
                with open(candidate_path, "w", encoding="utf-8") as fh:
                    fh.write(proposal.candidate_text)
                issues = self.engine.debug_file(candidate_path)
                reduced_count = len(issues)
                notes = [f"Static validation: issue count {current_count} -> {reduced_count}."]
                if reduced_count < current_count:
                    test_result = self._run_test_command(workspace, command, kind="candidate")
                    notes.append(
                        f"Test validation {'passed' if test_result.success else 'failed'} via {test_result.command}."
                    )
                else:
                    notes.append("Candidate skipped test execution because static validation did not improve issue count.")
                return reduced_count, " ".join(notes), issues, test_result

        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=os.path.splitext(source_path)[1],
            encoding="utf-8",
        ) as fh:
            fh.write(proposal.candidate_text)
            temp_path = fh.name
        try:
            issues = self.engine.debug_file(temp_path)
            reduced_count = len(issues)
            if reduced_count < current_count:
                return reduced_count, f"Static validation passed: issue count {current_count} -> {reduced_count}.", issues, None
            return reduced_count, f"Static validation did not improve issue count ({current_count} -> {reduced_count}).", issues, None
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _plugins_for_language(self, language: str):
        return [plugin for plugin in self.plugins if plugin.supports(language)]

    def _plugin_by_name(self, name: str):
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None

    def _get_baseline_test_context(self, source_path: str, language: str) -> Optional[Dict[str, object]]:
        project_root = self._find_project_root(source_path)
        # APR-EDGE-001: Cache by source_path too if needed, but for now we'll stick to project_root + language
        # but improve the command detection.
        cache_key = (project_root, language, source_path)
        if cache_key in self._baseline_test_cache:
            return self._baseline_test_cache[cache_key]

        command = self._detect_test_command(project_root, language)
        if not command:
            self._baseline_test_cache[cache_key] = None
            return None

        baseline = self._run_test_command(project_root, command, kind="baseline")
        context: Dict[str, object] = {
            "project_root": project_root,
            "command": command,
            "baseline_success": baseline.success,
            "baseline_result": baseline,
        }
        self._baseline_test_cache[cache_key] = context
        return context

    def _find_project_root(self, source_path: str) -> str:
        current = os.path.abspath(os.path.dirname(source_path))
        markers = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "pytest.ini",
            "package.json",
            ".git",
        }
        while True:
            # APR-ROOT-009: Prefer the closest directory that looks like a project root
            if any(os.path.exists(os.path.join(current, marker)) for marker in markers):
                return current
            if os.path.isdir(os.path.join(current, "tests")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return os.path.abspath(os.path.dirname(source_path))

    def _detect_test_command(self, project_root: str, language: str) -> Optional[List[str]]:
        if language == "python":
            if os.path.isdir(os.path.join(project_root, "tests")):
                if shutil.which("pytest"):
                    return ["pytest", "-q"]
                # Fallback to python -m pytest if pytest is not in PATH
                return [sys.executable, "-m", "pytest", "-q"]
        return None

    def _run_test_command(self, working_directory: str, command: Sequence[str], kind: str) -> TestRunResult:
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = working_directory if not existing_path else f"{working_directory}{os.pathsep}{existing_path}"
        try:
            result = subprocess.run(
                list(command),
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            return TestRunResult(
                command=" ".join(command),
                working_directory=working_directory,
                exit_code=result.returncode,
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                kind=kind,
            )
        except subprocess.TimeoutExpired as exc:
            return TestRunResult(
                command=" ".join(command),
                working_directory=working_directory,
                exit_code=124,
                success=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Timed out while running tests.",
                kind=kind,
            )

    def _history_dir(self, working_path: str) -> str:
        stem = os.path.basename(working_path)
        return os.path.join(os.path.dirname(working_path), ".oxpecker_history", stem)

    def _history_manifest_path(self, working_path: str) -> str:
        return os.path.join(self._history_dir(working_path), "history.json")

    def _ensure_history_manifest(self, working_path: str) -> None:
        history_dir = self._history_dir(working_path)
        os.makedirs(history_dir, exist_ok=True)
        manifest_path = self._history_manifest_path(working_path)
        if os.path.exists(manifest_path):
            return
        snapshot_path = os.path.join(history_dir, f"revision_0000{os.path.splitext(working_path)[1]}")
        shutil.copy2(working_path, snapshot_path)
        manifest = {
            "next_revision": 1,
            "entries": [
                {
                    "revision": 0,
                    "description": "Original working-copy snapshot.",
                    "snapshot_path": snapshot_path,
                    "created_from": working_path,
                    "reverted": False,
                }
            ],
        }
        self._save_history_manifest(working_path, manifest)

    def _record_rollback_snapshot(self, working_path: str, description: str) -> RollbackEntry:
        manifest = self._load_history_manifest(working_path)
        revision = int(manifest["next_revision"])
        snapshot_path = os.path.join(
            self._history_dir(working_path),
            f"revision_{revision:04d}{os.path.splitext(working_path)[1]}",
        )
        shutil.copy2(working_path, snapshot_path)
        entry = {
            "revision": revision,
            "description": description,
            "snapshot_path": snapshot_path,
            "created_from": working_path,
            "reverted": False,
        }
        manifest["entries"].append(entry)
        manifest["next_revision"] = revision + 1
        self._save_history_manifest(working_path, manifest)
        return self._rollback_from_dict(entry)

    def _load_history_manifest(self, working_path: str) -> Dict[str, object]:
        self._ensure_history_manifest(working_path)
        with open(self._history_manifest_path(working_path), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_history_manifest(self, working_path: str, manifest: Dict[str, object]) -> None:
        with open(self._history_manifest_path(working_path), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    def _rollback_from_dict(self, entry: Dict[str, object]) -> RollbackEntry:
        return RollbackEntry(
            revision=int(entry["revision"]),
            description=str(entry["description"]),
            snapshot_path=str(entry["snapshot_path"]),
            created_from=str(entry["created_from"]),
            reverted=bool(entry.get("reverted", False)),
        )


def _issue_signature(issue: Issue) -> Tuple[str, Optional[int], Optional[int]]:
    return (issue.message or "", issue.line, issue.column)


def _severity_score(severity: str) -> int:
    return {"error": 60, "warning": 35, "info": 15}.get(severity.lower(), 10)


def _tier_score(complexity_tier: str) -> int:
    return {"Tier 1": 30, "Tier 2": 20, "Tier 3": 10}.get(complexity_tier, 5)


__all__ = [
    "PatchProposal",
    "RepairResult",
    "RollbackEntry",
    "TestRunResult",
    "AutomatedProgramRepair",
]
