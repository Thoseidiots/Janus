"""
janus_self_improvement.py
==========================
Janus looks at its own failures and rewrites its own code to fix them.

This is the self-improvement loop:
  1. Scan work_log.jsonl for failed tasks
  2. Analyze what went wrong using JanusBrain
  3. Identify which file/function caused the failure
  4. Generate a fix
  5. Run the oxpecker to validate the fix
  6. Apply the fix if it passes validation
  7. Record what was learned

No API keys. Uses JanusBrain + oxpecker for validation.

Usage:
    from janus_self_improvement import JanusSelfImprovement
    si = JanusSelfImprovement()
    si.run_improvement_cycle()
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMPROVEMENT_LOG = Path("improvement_log.jsonl")
_REPO_ROOT       = Path(__file__).parent


# ── Failure record ────────────────────────────────────────────────────────────

@dataclass
class FailureRecord:
    task:        str
    error:       str
    category:    str
    file_hint:   Optional[str]   # which file likely caused it
    func_hint:   Optional[str]   # which function
    timestamp:   str


@dataclass
class ImprovementRecord:
    improvement_id: str
    failure:        FailureRecord
    diagnosis:      str
    proposed_fix:   str
    target_file:    str
    applied:        bool          = False
    validated:      bool          = False
    applied_at:     Optional[str] = None
    result:         str           = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure"] = asdict(self.failure)
        return d


# ── Failure analyzer ──────────────────────────────────────────────────────────

class FailureAnalyzer:
    """Reads work logs and extracts failure patterns."""

    def get_recent_failures(self, n: int = 20) -> List[FailureRecord]:
        work_log = _REPO_ROOT / "work_log.jsonl"
        if not work_log.exists():
            return []

        failures = []
        lines    = work_log.read_text().strip().splitlines()
        for line in lines[-100:]:
            try:
                entry = json.loads(line)
                if entry.get("status") == "failed":
                    error = entry.get("notes", entry.get("error", "unknown error"))
                    failures.append(FailureRecord(
                        task      = entry.get("title", "unknown task"),
                        error     = error,
                        category  = entry.get("category", "unknown"),
                        file_hint = self._extract_file_hint(error),
                        func_hint = self._extract_func_hint(error),
                        timestamp = entry.get("created_at", ""),
                    ))
            except Exception:
                pass

        return failures[-n:]

    def get_escalated_tasks(self) -> List[FailureRecord]:
        """Get tasks that were escalated due to repeated failures."""
        esc_file = _REPO_ROOT / "escalations.json"
        if not esc_file.exists():
            return []
        try:
            data = json.loads(esc_file.read_text())
            escs = data.get("escalations", [])
            failures = []
            for e in escs:
                if e.get("trigger") == "task_failure":
                    ctx = e.get("context", {})
                    failures.append(FailureRecord(
                        task      = ctx.get("task", "unknown"),
                        error     = ctx.get("error", ""),
                        category  = "escalated",
                        file_hint = self._extract_file_hint(ctx.get("error", "")),
                        func_hint = None,
                        timestamp = e.get("created_at", ""),
                    ))
            return failures
        except Exception:
            return []

    def _extract_file_hint(self, error: str) -> Optional[str]:
        """Try to find a filename in an error message."""
        m = re.search(r'File "([^"]+\.py)"', error)
        if m:
            return m.group(1)
        m = re.search(r'([\w/\\.-]+\.py)', error)
        if m:
            return m.group(1)
        return None

    def _extract_func_hint(self, error: str) -> Optional[str]:
        """Try to find a function name in an error message."""
        m = re.search(r'in (\w+)\n', error)
        if m:
            return m.group(1)
        return None


# ── Code fixer ────────────────────────────────────────────────────────────────

class CodeFixer:
    """Uses JanusBrain to diagnose and fix code issues."""

    def diagnose(self, failure: FailureRecord) -> str:
        """Diagnose why a task failed."""
        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.solve(
                f"Task '{failure.task}' failed with error: {failure.error[:300]}\n"
                f"Category: {failure.category}\n"
                f"What is the root cause and how should it be fixed?"
            )
        except Exception:
            return f"Error in {failure.category} task: {failure.error[:200]}"

    def generate_fix(
        self,
        failure:   FailureRecord,
        diagnosis: str,
        file_path: Path,
    ) -> Optional[str]:
        """Generate a code fix for a specific file."""
        if not file_path.exists():
            return None

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        try:
            from avus_brain import get_brain
            brain = get_brain()
            prompt = (
                f"Fix this Python code.\n\n"
                f"File: {file_path.name}\n"
                f"Error: {failure.error[:200]}\n"
                f"Diagnosis: {diagnosis[:200]}\n\n"
                f"Current code (relevant section):\n"
                f"{content[:3000]}\n\n"
                f"Write ONLY the fixed version of the relevant function or section. "
                f"Do not include explanations, just the corrected code."
            )
            return brain.ask(prompt, max_tokens=500)
        except Exception:
            return None

    def validate_fix(self, file_path: Path, fixed_code: str) -> Tuple[bool, str]:
        """
        Validate a proposed fix using the oxpecker.
        Returns (is_valid, message).
        """
        # First check Python syntax
        try:
            ast.parse(fixed_code)
        except SyntaxError as e:
            return False, f"Syntax error in fix: {e}"

        # Write to temp file and run oxpecker
        tmp = file_path.with_suffix(".fix_candidate.py")
        try:
            tmp.write_text(fixed_code)
            result = subprocess.run(
                [sys.executable, "-m", "tools.universal_oxpecker.cli",
                 "scan", str(tmp), "--verbose"],
                capture_output=True, text=True, timeout=30,
                cwd=str(_REPO_ROOT),
            )
            output = result.stdout + result.stderr
            errors = output.count("[ERROR]")
            if errors == 0:
                return True, "Fix validated — no errors detected"
            return False, f"Fix has {errors} error(s): {output[:300]}"
        except Exception as e:
            return False, str(e)
        finally:
            tmp.unlink(missing_ok=True)

    def apply_fix(
        self,
        file_path:  Path,
        fixed_code: str,
        backup:     bool = True,
    ) -> bool:
        """Apply a validated fix to a file."""
        if backup:
            backup_path = file_path.with_suffix(f".backup_{int(time.time())}.py")
            backup_path.write_text(file_path.read_text())

        try:
            # Try to apply as a patch (replace matching function)
            original = file_path.read_text()
            patched  = self._smart_patch(original, fixed_code)
            file_path.write_text(patched)
            return True
        except Exception as e:
            print(f"[SelfImprovement] Apply failed: {e}")
            return False

    def _smart_patch(self, original: str, fix: str) -> str:
        """
        Try to apply fix intelligently:
        - If fix contains a function def, replace that function in original
        - Otherwise append as a comment with the suggestion
        """
        # Find function name in fix
        m = re.search(r'def (\w+)\s*\(', fix)
        if not m:
            return original  # can't patch safely

        func_name = m.group(1)

        # Find and replace the function in original
        pattern = rf'(def {re.escape(func_name)}\s*\([^)]*\)[^:]*:.*?)(?=\ndef |\nclass |\Z)'
        if re.search(pattern, original, re.DOTALL):
            return re.sub(pattern, fix.strip(), original, flags=re.DOTALL)

        return original


# ── Main self-improvement engine ──────────────────────────────────────────────

class JanusSelfImprovement:
    """
    Janus looks at its own failures and fixes its own code.
    """

    def __init__(self):
        self._analyzer = FailureAnalyzer()
        self._fixer    = CodeFixer()
        self._history: List[ImprovementRecord] = []

    def run_improvement_cycle(self, max_fixes: int = 3) -> List[ImprovementRecord]:
        """
        Run one improvement cycle.
        Returns list of improvements attempted.
        """
        print("[SelfImprovement] Starting improvement cycle...")

        # Collect failures
        failures = self._analyzer.get_recent_failures(20)
        failures += self._analyzer.get_escalated_tasks()

        if not failures:
            print("[SelfImprovement] No failures to analyze")
            return []

        print(f"[SelfImprovement] Analyzing {len(failures)} failure(s)")

        improvements = []
        for failure in failures[:max_fixes]:
            imp = self._attempt_fix(failure)
            if imp:
                improvements.append(imp)
                self._log(imp)

        # Learn from what worked
        self._update_heuristics(improvements)

        print(f"[SelfImprovement] Cycle complete: "
              f"{sum(1 for i in improvements if i.applied)} fix(es) applied")
        return improvements

    def _attempt_fix(self, failure: FailureRecord) -> Optional[ImprovementRecord]:
        """Attempt to fix a single failure."""
        import uuid

        print(f"  Analyzing: {failure.task[:50]}")

        # Diagnose
        diagnosis = self._fixer.diagnose(failure)

        # Find target file
        target_file = self._find_target_file(failure)
        if not target_file:
            print(f"    Could not identify target file")
            return ImprovementRecord(
                improvement_id = str(uuid.uuid4())[:8],
                failure        = failure,
                diagnosis      = diagnosis,
                proposed_fix   = "",
                target_file    = "unknown",
                result         = "no target file identified",
            )

        print(f"    Target: {target_file.name}")

        # Generate fix
        fix = self._fixer.generate_fix(failure, diagnosis, target_file)
        if not fix:
            return None

        # Validate
        valid, msg = self._fixer.validate_fix(target_file, fix)
        print(f"    Validation: {'✓' if valid else '✗'} {msg[:60]}")

        imp = ImprovementRecord(
            improvement_id = str(uuid.uuid4())[:8],
            failure        = failure,
            diagnosis      = diagnosis,
            proposed_fix   = fix[:500],
            target_file    = str(target_file),
            validated      = valid,
        )

        # Apply if valid
        if valid:
            applied = self._fixer.apply_fix(target_file, fix)
            imp.applied    = applied
            imp.applied_at = datetime.now().isoformat() if applied else None
            imp.result     = "fix applied" if applied else "apply failed"
            if applied:
                print(f"    ✓ Fix applied to {target_file.name}")
        else:
            imp.result = msg

        self._history.append(imp)
        return imp

    def _find_target_file(self, failure: FailureRecord) -> Optional[Path]:
        """Find the most likely file to fix."""
        # Use hint from error message
        if failure.file_hint:
            p = Path(failure.file_hint)
            if p.exists():
                return p
            # Try relative to repo root
            p = _REPO_ROOT / failure.file_hint
            if p.exists():
                return p

        # Map category to likely file
        category_files = {
            "code":       "janus_worker.py",
            "data":       "janus_worker.py",
            "automation": "janus_autonomous_loop.py",
            "review":     "janus_worker.py",
            "finance":    "janus_finance.py",
        }
        fname = category_files.get(failure.category)
        if fname:
            p = _REPO_ROOT / fname
            if p.exists():
                return p

        return None

    def _update_heuristics(self, improvements: List[ImprovementRecord]):
        """Record what was learned from this cycle."""
        try:
            from janus_identity import get_identity
            ident = get_identity()
            for imp in improvements:
                if imp.applied:
                    ident.learn(
                        f"Fixed bug in {Path(imp.target_file).name}: {imp.diagnosis[:80]}",
                        context  = imp.failure.error[:100],
                        confidence = 0.75,
                    )
        except Exception:
            pass

    def _log(self, imp: ImprovementRecord):
        with _IMPROVEMENT_LOG.open("a") as f:
            f.write(json.dumps(imp.to_dict()) + "\n")

    def get_history(self) -> List[dict]:
        if not _IMPROVEMENT_LOG.exists():
            return []
        results = []
        for line in _IMPROVEMENT_LOG.read_text().strip().splitlines():
            try:
                results.append(json.loads(line))
            except Exception:
                pass
        return results

    def stats(self) -> dict:
        history = self.get_history()
        return {
            "total_attempts": len(history),
            "applied":        sum(1 for h in history if h.get("applied")),
            "validated":      sum(1 for h in history if h.get("validated")),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_si: Optional[JanusSelfImprovement] = None

def get_self_improvement() -> JanusSelfImprovement:
    global _si
    if _si is None:
        _si = JanusSelfImprovement()
    return _si


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Janus Self-Improvement")
    parser.add_argument("--run",   action="store_true", help="Run improvement cycle")
    parser.add_argument("--stats", action="store_true", help="Show improvement stats")
    parser.add_argument("--history", action="store_true", help="Show improvement history")
    args = parser.parse_args()

    si = JanusSelfImprovement()

    if args.run:
        improvements = si.run_improvement_cycle()
        print(f"\nApplied {sum(1 for i in improvements if i.applied)} fix(es)")
    elif args.stats:
        print(json.dumps(si.stats(), indent=2))
    elif args.history:
        for h in si.get_history()[-10:]:
            status = "✓" if h.get("applied") else "✗"
            print(f"{status} {h['failure']['task'][:50]} → {h['result'][:50]}")
    else:
        parser.print_help()
