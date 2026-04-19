"""
janus_integration_test.py
==========================
Full end-to-end integration test for the Janus loop.

Proves that:
  1. A requester can post a task to JCTaskMarket (JC held in escrow).
  2. A worker can claim the task.
  3. WorkGenerator produces non-empty output for the task.
  4. Completing the task transfers JC from escrow to the worker.
  5. Checkpointer correctly saves, loads, and clears state.

Run:
    python janus_integration_test.py
    python janus_integration_test.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Graceful imports — each dependency is optional so the test can report a
# clear ImportError rather than crashing with a traceback.
# ---------------------------------------------------------------------------

try:
    from janus_credits import JCLedger, JCTaskMarket, ComputeProofValidator
    _HAS_CREDITS = True
    _CREDITS_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_CREDITS = False
    _CREDITS_ERR = str(_e)

try:
    from janus_checkpoint import Checkpointer
    _HAS_CHECKPOINT = True
    _CHECKPOINT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_CHECKPOINT = False
    _CHECKPOINT_ERR = str(_e)

try:
    from janus_autonomous_worker import WorkGenerator, Job, JobStatus
    from datetime import datetime, timedelta
    _HAS_WORKER = True
    _WORKER_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_WORKER = False
    _WORKER_ERR = str(_e)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUESTER_ID = "test_requester"
WORKER_ID = "test_worker"
INITIAL_BALANCE = 50.0
TASK_REWARD = 5.0
TASK_DESCRIPTION = "Write a 200-word blog post about Python"
TASK_TYPE = "writing"

# Databases created by the modules under test
_CREDITS_DB = Path("janus_credits.db")
_CHECKPOINTS_DB = Path("janus_checkpoints.db")

VERBOSE = False


def _log(msg: str) -> None:
    """Print a message when verbose mode is active."""
    if VERBOSE:
        print(f"    {msg}")


# ---------------------------------------------------------------------------
# JanusIntegrationTest
# ---------------------------------------------------------------------------


class JanusIntegrationTest:
    """
    Self-contained integration test suite for the Janus JC loop.

    All state is written to the real SQLite databases used by the modules
    under test.  ``teardown`` removes only the rows created by this test
    suite so that pre-existing data is not disturbed.
    """

    def __init__(self) -> None:
        """Initialise instance variables; actual setup happens in ``setup``."""
        self._ledger: Optional[Any] = None
        self._market: Optional[Any] = None
        self._validator: Optional[Any] = None
        self._checkpointer: Optional[Any] = None
        self._task_id: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Create test accounts and initialise all subsystems.

        Creates ``test_requester`` and ``test_worker`` accounts, each topped
        up to ``INITIAL_BALANCE`` JC (welcome bonus is 10 JC; the remainder
        is added via ``earn_jc``).
        """
        if not _HAS_CREDITS:
            raise ImportError(f"janus_credits not available: {_CREDITS_ERR}")
        if not _HAS_CHECKPOINT:
            raise ImportError(f"janus_checkpoint not available: {_CHECKPOINT_ERR}")
        if not _HAS_WORKER:
            raise ImportError(f"janus_autonomous_worker not available: {_WORKER_ERR}")

        self._ledger = JCLedger()
        self._market = JCTaskMarket(self._ledger)
        self._validator = ComputeProofValidator()
        self._checkpointer = Checkpointer()

        # Ensure clean state — remove any leftover rows from a previous run
        self._delete_test_accounts()

        # Create accounts (welcome bonus = 10 JC each)
        self._ledger.create_account(REQUESTER_ID)
        self._ledger.create_account(WORKER_ID)

        # Top up to INITIAL_BALANCE
        top_up = INITIAL_BALANCE - 10.0  # 10 JC already from welcome bonus
        if top_up > 0:
            proof = self._validator.generate_proof(REQUESTER_ID, "inference", top_up)
            self._ledger.earn_jc(
                REQUESTER_ID, top_up, proof, "Test setup top-up"
            )
            proof = self._validator.generate_proof(WORKER_ID, "inference", top_up)
            self._ledger.earn_jc(
                WORKER_ID, top_up, proof, "Test setup top-up"
            )

        _log(
            f"Accounts created: {REQUESTER_ID}={self._ledger.get_balance(REQUESTER_ID)} JC, "
            f"{WORKER_ID}={self._ledger.get_balance(WORKER_ID)} JC"
        )

    def teardown(self) -> None:
        """
        Remove all test data from SQLite.

        Deletes the test accounts, their transactions, and the task created
        during the test.  Also clears any checkpoint left by the test.
        """
        self._delete_test_accounts()
        if self._task_id:
            self._delete_test_task(self._task_id)
        if self._checkpointer and self._task_id:
            try:
                self._checkpointer.complete(self._task_id)
            except Exception:
                pass
        _log("Teardown complete.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _delete_test_accounts(self) -> None:
        """Remove test accounts and their transactions from the credits DB."""
        if not _CREDITS_DB.exists():
            return
        conn = sqlite3.connect(_CREDITS_DB)
        try:
            conn.execute(
                "DELETE FROM accounts WHERE user_id IN (?, ?)",
                (REQUESTER_ID, WORKER_ID),
            )
            conn.execute(
                "DELETE FROM transactions WHERE from_user IN (?, ?) OR to_user IN (?, ?)",
                (REQUESTER_ID, WORKER_ID, REQUESTER_ID, WORKER_ID),
            )
            conn.commit()
        except Exception as exc:
            _log(f"Warning during account cleanup: {exc}")
        finally:
            conn.close()

    def _delete_test_task(self, task_id: str) -> None:
        """Remove a specific task row from the credits DB."""
        if not _CREDITS_DB.exists():
            return
        conn = sqlite3.connect(_CREDITS_DB)
        try:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()
        except Exception as exc:
            _log(f"Warning during task cleanup: {exc}")
        finally:
            conn.close()

    @staticmethod
    def _result(
        name: str,
        passed: bool,
        details: str,
        duration_ms: float,
    ) -> Dict[str, Any]:
        """Build a standardised test-result dict."""
        return {
            "name": name,
            "passed": passed,
            "details": details,
            "duration_ms": round(duration_ms, 2),
        }

    # ── Test methods ──────────────────────────────────────────────────────────

    async def test_full_jc_loop(self) -> Dict[str, Any]:
        """
        End-to-end test of the complete Janus JC loop.

        Steps:
          1. Post a task (requester balance decreases by TASK_REWARD via escrow).
          2. Worker claims the task.
          3. Worker completes the task (JC released from escrow to worker).
          4. Assert requester balance == INITIAL_BALANCE - TASK_REWARD.
          5. Assert worker balance == INITIAL_BALANCE + TASK_REWARD.
          6. Assert task status == "completed".

        Returns:
            A result dict with keys: name, passed, details, duration_ms.
        """
        name = "test_full_jc_loop"
        t0 = time.perf_counter()
        steps: list[str] = []

        try:
            # ── Step 1: post task ──────────────────────────────────────────
            req_before = self._ledger.get_balance(REQUESTER_ID)
            _log(f"Requester balance before post: {req_before} JC")

            self._task_id = self._market.post_task(
                requester_id=REQUESTER_ID,
                task_description=TASK_DESCRIPTION,
                jc_reward=TASK_REWARD,
                task_type=TASK_TYPE,
            )
            steps.append("✓ Task posted")
            _log(f"Task posted: {self._task_id}")

            req_after_post = self._ledger.get_balance(REQUESTER_ID)
            assert req_after_post == req_before - TASK_REWARD, (
                f"Expected requester balance {req_before - TASK_REWARD}, "
                f"got {req_after_post}"
            )
            steps.append("✓ Requester balance decreased by escrow amount")

            # ── Step 2: claim task ─────────────────────────────────────────
            claimed = self._market.claim_task(WORKER_ID, self._task_id)
            assert claimed, "claim_task returned False"
            steps.append("✓ Task claimed by worker")

            task_info = self._market.get_task_status(self._task_id)
            assert task_info.get("status") == "claimed", (
                f"Expected status 'claimed', got {task_info.get('status')!r}"
            )
            steps.append("✓ Task status is 'claimed'")

            # ── Step 3: complete task ──────────────────────────────────────
            worker_before = self._ledger.get_balance(WORKER_ID)
            _log(f"Worker balance before completion: {worker_before} JC")

            completed = self._market.complete_task(
                self._task_id,
                result="Blog post about Python completed by test_worker.",
            )
            assert completed, "complete_task returned False"
            steps.append("✓ Task completed")

            # ── Step 4 & 5: balance assertions ─────────────────────────────
            req_final = self._ledger.get_balance(REQUESTER_ID)
            worker_final = self._ledger.get_balance(WORKER_ID)
            _log(f"Requester final balance: {req_final} JC")
            _log(f"Worker final balance: {worker_final} JC")

            expected_req = INITIAL_BALANCE - TASK_REWARD
            expected_worker = INITIAL_BALANCE + TASK_REWARD

            assert req_final == expected_req, (
                f"Requester balance: expected {expected_req}, got {req_final}"
            )
            steps.append(f"✓ Requester balance = {expected_req} JC")

            assert worker_final == expected_worker, (
                f"Worker balance: expected {expected_worker}, got {worker_final}"
            )
            steps.append(f"✓ Worker balance = {expected_worker} JC")

            # ── Step 6: task status ────────────────────────────────────────
            task_final = self._market.get_task_status(self._task_id)
            assert task_final.get("status") == "completed", (
                f"Expected status 'completed', got {task_final.get('status')!r}"
            )
            steps.append("✓ Task status is 'completed'")

            duration_ms = (time.perf_counter() - t0) * 1000
            return self._result(name, True, "; ".join(steps), duration_ms)

        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            failed_steps = "; ".join(steps) + f"; ✗ FAILED: {exc}"
            return self._result(name, False, failed_steps, duration_ms)

    async def test_work_generation(self) -> Dict[str, Any]:
        """
        Test that WorkGenerator produces non-empty output for the task.

        Constructs a ``Job`` object matching the integration task and calls
        ``generate_work``.  Asserts the result is a non-empty string.

        Returns:
            A result dict with keys: name, passed, details, duration_ms.
        """
        name = "test_work_generation"
        t0 = time.perf_counter()
        steps: list[str] = []

        try:
            if not _HAS_WORKER:
                raise ImportError(f"janus_autonomous_worker not available: {_WORKER_ERR}")

            generator = WorkGenerator()
            steps.append("✓ WorkGenerator instantiated")

            job = Job(
                id="test-job-work-gen",
                title=TASK_DESCRIPTION,
                description=TASK_DESCRIPTION,
                required_skills=["writing", "python"],
                budget=TASK_REWARD,
                deadline=datetime.now() + timedelta(days=1),
                platform="janus",
                status=JobStatus.CLAIMED,
            )
            steps.append("✓ Job object created")

            work = await generator.generate_work(job)

            assert work is not None, "generate_work returned None"
            assert isinstance(work, str), f"Expected str, got {type(work).__name__}"
            assert len(work.strip()) > 0, "generate_work returned empty string"
            steps.append(f"✓ Work generated ({len(work)} chars)")

            duration_ms = (time.perf_counter() - t0) * 1000
            return self._result(name, True, "; ".join(steps), duration_ms)

        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            failed_steps = "; ".join(steps) + f"; ✗ FAILED: {exc}"
            return self._result(name, False, failed_steps, duration_ms)

    async def test_checkpoint_lifecycle(self) -> Dict[str, Any]:
        """
        Test the full checkpoint save → load → complete lifecycle.

        Steps:
          1. Save a checkpoint for a synthetic job_id.
          2. Load it back and verify stage and data round-trip correctly.
          3. Call ``complete`` and verify the checkpoint is removed.

        Returns:
            A result dict with keys: name, passed, details, duration_ms.
        """
        name = "test_checkpoint_lifecycle"
        t0 = time.perf_counter()
        steps: list[str] = []
        cp_job_id = "test-checkpoint-job-001"

        try:
            if not _HAS_CHECKPOINT:
                raise ImportError(f"janus_checkpoint not available: {_CHECKPOINT_ERR}")

            cp = self._checkpointer

            # ── Save ──────────────────────────────────────────────────────
            payload = {"prompt": TASK_DESCRIPTION, "partial_work": "intro written"}
            cp.save(cp_job_id, stage="generating", data=payload)
            steps.append("✓ Checkpoint saved")

            # ── Load ──────────────────────────────────────────────────────
            state = cp.load(cp_job_id)
            assert state is not None, "load returned None after save"
            assert state["stage"] == "generating", (
                f"Expected stage 'generating', got {state['stage']!r}"
            )
            assert state["data"] == payload, (
                f"Data mismatch: {state['data']!r} != {payload!r}"
            )
            assert state["attempt"] >= 1, "attempt should be >= 1"
            steps.append("✓ Checkpoint loaded with correct stage and data")

            # ── Verify it appears in incomplete list ───────────────────────
            incomplete = cp.get_incomplete()
            ids = [c["job_id"] for c in incomplete]
            assert cp_job_id in ids, (
                f"{cp_job_id!r} not found in get_incomplete()"
            )
            steps.append("✓ Checkpoint appears in get_incomplete()")

            # ── Complete ──────────────────────────────────────────────────
            cp.complete(cp_job_id)
            steps.append("✓ Checkpoint cleared via complete()")

            # ── Verify removal ────────────────────────────────────────────
            state_after = cp.load(cp_job_id)
            assert state_after is None, (
                f"Expected None after complete(), got {state_after!r}"
            )
            steps.append("✓ Checkpoint no longer exists after complete()")

            duration_ms = (time.perf_counter() - t0) * 1000
            return self._result(name, True, "; ".join(steps), duration_ms)

        except Exception as exc:
            # Best-effort cleanup
            try:
                if self._checkpointer:
                    self._checkpointer.complete(cp_job_id)
            except Exception:
                pass
            duration_ms = (time.perf_counter() - t0) * 1000
            failed_steps = "; ".join(steps) + f"; ✗ FAILED: {exc}"
            return self._result(name, False, failed_steps, duration_ms)

    async def run_all(self) -> Dict[str, Any]:
        """
        Run all test methods and return a summary dict.

        Calls ``setup`` before the suite and ``teardown`` after (even on
        failure).  Prints a pass/fail table to stdout and exits with code 0
        if all tests pass, or code 1 if any fail.

        Returns:
            A dict with keys: total, passed, failed, results (list of result
            dicts), and all_passed (bool).
        """
        print("\n" + "═" * 60)
        print("  Janus Integration Test Suite")
        print("═" * 60)

        results: list[Dict[str, Any]] = []

        # ── Setup ──────────────────────────────────────────────────────────
        print("\n[Setup]")
        try:
            self.setup()
            print("  ✓ Setup complete")
        except Exception as exc:
            print(f"  ✗ Setup FAILED: {exc}")
            print("\nCannot run tests without successful setup.")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "results": [],
                "all_passed": False,
            }

        # ── Run tests ──────────────────────────────────────────────────────
        print("\n[Tests]")
        tests = [
            self.test_full_jc_loop,
            self.test_work_generation,
            self.test_checkpoint_lifecycle,
        ]

        for test_fn in tests:
            print(f"\n  Running {test_fn.__name__} …")
            result = await test_fn()
            results.append(result)
            icon = "✓" if result["passed"] else "✗"
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {icon} {status}  ({result['duration_ms']:.1f} ms)")
            if VERBOSE or not result["passed"]:
                for step in result["details"].split("; "):
                    print(f"      {step}")

        # ── Teardown ───────────────────────────────────────────────────────
        print("\n[Teardown]")
        try:
            self.teardown()
            print("  ✓ Teardown complete")
        except Exception as exc:
            print(f"  ✗ Teardown warning: {exc}")

        # ── Summary table ──────────────────────────────────────────────────
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed

        print("\n" + "─" * 60)
        print(f"  {'Test':<40} {'Result':<8} {'ms':>8}")
        print("─" * 60)
        for r in results:
            icon = "✓" if r["passed"] else "✗"
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {icon} {r['name']:<38} {status:<8} {r['duration_ms']:>8.1f}")
        print("─" * 60)
        print(f"  Total: {total}  Passed: {passed}  Failed: {failed}")
        print("═" * 60 + "\n")

        all_passed = failed == 0
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
            "all_passed": all_passed,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI entry point.

    Parses ``--verbose`` flag, runs the full test suite, and exits with
    code 0 on success or 1 if any test fails.
    """
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Janus end-to-end integration test suite."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step output for every test.",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    suite = JanusIntegrationTest()
    summary = asyncio.run(suite.run_all())

    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
