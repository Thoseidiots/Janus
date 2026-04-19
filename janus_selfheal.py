"""
janus_selfheal.py
==================
Self-healing system for Janus.

Runs the integration test suite on startup and after each work cycle.
When tests fail, uses the Oxpecker auto-repair engine to fix the broken
files, then re-runs the tests. Repeats up to MAX_REPAIR_ROUNDS times.

If all tests pass  → Janus starts normally.
If repair succeeds → Janus starts with patched files.
If repair fails    → Janus logs the issue, notifies the owner, and starts
                     anyway in degraded mode (non-critical failures only).

Usage (called by janus_daemon.py before starting the work cycle):
    from janus_selfheal import SelfHeal
    healer = SelfHeal()
    ok = await healer.run()   # True = healthy, False = degraded
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("janus.selfheal")

MAX_REPAIR_ROUNDS = 3          # how many Oxpecker repair attempts per failure
REPORT_PATH       = Path("janus_selfheal_report.json")

# Files the Oxpecker is allowed to repair (critical Janus modules only)
REPAIRABLE_FILES = [
    "janus_autonomous_worker.py",
    "janus_credits.py",
    "janus_checkpoint.py",
    "janus_worker_completion.py",
    "janus_inference_pipeline.py",
    "janus_notify.py",
    "janus_human_core.py",
]


class SelfHeal:
    """
    Runs integration tests and auto-repairs failures using the Oxpecker.

    Workflow
    --------
    1. Run janus_integration_test.py
    2. If all pass → return True immediately
    3. For each failing test, identify which file is likely responsible
    4. Run Oxpecker repair on that file (safe working copy)
    5. If repair improved the file, copy the working copy back
    6. Re-run tests
    7. Repeat up to MAX_REPAIR_ROUNDS
    8. Return True if all tests pass, False otherwise
    """

    def __init__(self) -> None:
        self._results: List[Dict] = []
        self._repair_log: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> bool:
        """
        Run the full self-heal cycle.

        Returns True if all tests pass (possibly after repair), False if
        the system is in degraded state.
        """
        log.info("[SelfHeal] Starting health check...")
        start = time.time()

        for attempt in range(1, MAX_REPAIR_ROUNDS + 2):
            log.info("[SelfHeal] Test run %d/%d", attempt, MAX_REPAIR_ROUNDS + 1)
            results = await self._run_tests()
            self._results = results

            failed = [r for r in results if not r.get("passed")]
            passed = [r for r in results if r.get("passed")]

            log.info(
                "[SelfHeal] %d/%d tests passed",
                len(passed), len(results),
            )

            if not failed:
                log.info("[SelfHeal] ✓ All tests passed — system healthy")
                self._save_report(healthy=True, elapsed=time.time() - start)
                return True

            if attempt > MAX_REPAIR_ROUNDS:
                log.warning(
                    "[SelfHeal] Max repair rounds reached. "
                    "Starting in degraded mode. Failures: %s",
                    [r["name"] for r in failed],
                )
                self._save_report(healthy=False, elapsed=time.time() - start)
                return False

            # Attempt repair
            log.info(
                "[SelfHeal] %d test(s) failed — attempting Oxpecker repair (round %d)",
                len(failed), attempt,
            )
            repaired_any = await self._repair_failures(failed)
            if not repaired_any:
                log.warning("[SelfHeal] Oxpecker could not improve any files")
                self._save_report(healthy=False, elapsed=time.time() - start)
                return False

        self._save_report(healthy=False, elapsed=time.time() - start)
        return False

    def get_report(self) -> Dict:
        """Return the last self-heal report."""
        if REPORT_PATH.exists():
            try:
                return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    # ── Test runner ───────────────────────────────────────────────────────────

    async def _run_tests(self) -> List[Dict]:
        """Run janus_integration_test.py and return the results list."""
        try:
            from janus_integration_test import JanusIntegrationTest
            suite = JanusIntegrationTest()
            summary = await suite.run_all()
            return summary.get("results", [])
        except ImportError as e:
            log.error("[SelfHeal] Could not import integration test: %s", e)
            return [{"name": "import_check", "passed": False, "details": str(e), "duration_ms": 0}]
        except Exception as e:
            log.error("[SelfHeal] Test run failed: %s", e)
            return [{"name": "test_run", "passed": False, "details": str(e), "duration_ms": 0}]

    # ── Repair engine ─────────────────────────────────────────────────────────

    async def _repair_failures(self, failed_tests: List[Dict]) -> bool:
        """
        Use the Oxpecker to repair files implicated in test failures.
        Returns True if at least one file was improved.
        """
        try:
            import sys as _sys
            tools_path = str(Path("tools").resolve())
            if tools_path not in _sys.path:
                _sys.path.insert(0, tools_path)
            from universal_oxpecker.core.orchestrator import OxpeckerOrchestrator
        except ImportError as e:
            log.warning("[SelfHeal] Oxpecker not available: %s", e)
            return False

        orchestrator = OxpeckerOrchestrator()
        repaired_any = False

        # Map test names to likely responsible files
        test_file_map = {
            "test_full_jc_loop":         ["janus_credits.py", "janus_worker_completion.py"],
            "test_work_generation":      ["janus_autonomous_worker.py", "janus_inference_pipeline.py"],
            "test_checkpoint_lifecycle": ["janus_checkpoint.py"],
        }

        files_to_repair: List[str] = []
        for test in failed_tests:
            name = test.get("name", "")
            files_to_repair.extend(test_file_map.get(name, []))

        # Also scan all repairable files for static issues
        files_to_repair = list(dict.fromkeys(files_to_repair + REPAIRABLE_FILES))

        for filepath in files_to_repair:
            if not Path(filepath).exists():
                continue
            try:
                log.info("[SelfHeal] Scanning %s for issues...", filepath)
                issues = orchestrator.debug_file(filepath)
                errors = [i for i in issues if i.severity == "error"]

                if not errors:
                    log.debug("[SelfHeal] %s has no errors — skipping repair", filepath)
                    continue

                log.info(
                    "[SelfHeal] %s has %d error(s) — running Oxpecker repair",
                    filepath, len(errors),
                )
                repair = await asyncio.to_thread(
                    orchestrator.repair_file,
                    filepath,
                    ".oxpecker_work",
                    MAX_REPAIR_ROUNDS,
                )

                if repair.fixed_issue_count > 0:
                    # Copy the repaired working copy back over the original
                    import shutil
                    shutil.copy2(repair.working_path, filepath)
                    msg = (
                        f"Repaired {filepath}: fixed {repair.fixed_issue_count} issue(s) "
                        f"via {[p.description for p in repair.applied_patches]}"
                    )
                    log.info("[SelfHeal] ✓ %s", msg)
                    self._repair_log.append(msg)
                    repaired_any = True
                else:
                    log.info(
                        "[SelfHeal] Oxpecker found no auto-fixable issues in %s",
                        filepath,
                    )

            except Exception as e:
                log.warning("[SelfHeal] Repair failed for %s: %s", filepath, e)

        return repaired_any

    # ── Report ────────────────────────────────────────────────────────────────

    def _save_report(self, healthy: bool, elapsed: float) -> None:
        """Persist the self-heal report to disk."""
        report = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "healthy":     healthy,
            "elapsed_s":   round(elapsed, 2),
            "test_results": self._results,
            "repair_log":  self._repair_log,
        }
        try:
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("[SelfHeal] Could not save report: %s", e)


# ── Standalone runner ─────────────────────────────────────────────────────────

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    healer = SelfHeal()
    healthy = await healer.run()
    report = healer.get_report()
    print(json.dumps(report, indent=2))
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    asyncio.run(_main())
