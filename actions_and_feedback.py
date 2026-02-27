# actions_and_feedback.py

“””
Janus Real World Actions + Feedback Loop — gaps 4 and 5.

Gap 4: Real world actions that matter

- Form submission (job applications, contact forms)
- Account-aware browsing (session cookies, login state)
- File downloads and uploads
- Email draft generation

Gap 5: Feedback loop

- Did the action actually work?
- What changed in the world after Janus acted?
- Feed outcomes back into valence and memory
- Build a success/failure model over time

These two are built together because feedback is meaningless
without real actions, and real actions are dangerous without feedback.

Usage:
from actions_and_feedback import ActionEngine, FeedbackLoop
engine  = ActionEngine()
outcome = engine.submit_form(“https://example.com/contact”, {
“name”: “Ishmael Sears”, “message”: “I’m interested in…”
})
feedback = FeedbackLoop()
feedback.record(outcome)
“””

from **future** import annotations

import json
import time
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ── Action result types ───────────────────────────────────────────────────────

class ActionStatus(str, Enum):
SUCCESS        = “success”
PARTIAL        = “partial”       # action ran but outcome uncertain
FAILED         = “failed”
NEEDS_AUTH     = “needs_auth”
NEEDS_CAPTCHA  = “needs_captcha”
BLOCKED        = “blocked”

@dataclass
class ActionOutcome:
action_id:   str
action_type: str
target:      str                  # URL or description
status:      ActionStatus
evidence:    dict                 # what Janus observed after acting
timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
notes:       str = “”

# ── Action Engine ─────────────────────────────────────────────────────────────

class ActionEngine:
“””
Executes real-world actions via Playwright with outcome verification.

```
Every action:
  1. Takes a before-snapshot (page state before acting)
  2. Executes the action
  3. Takes an after-snapshot (page state after acting)
  4. Compares snapshots to verify something actually changed
  5. Returns an ActionOutcome with evidence
"""

def __init__(self, headless: bool = True, slow_mo: int = 50):
    self._web        = None
    self._headless   = headless
    self._slow_mo    = slow_mo
    self._session    = SessionManager()

def _ensure_web(self):
    if self._web is None:
        from playwright.sync_api import sync_playwright
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
        )
        # Load saved session cookies if available
        storage = self._session.load_storage()
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            ),
            storage_state=storage,
        )
        self._web = self._context.new_page()
    return self._web

# ── Core actions ──────────────────────────────────────────────────────────

def navigate(self, url: str) -> ActionOutcome:
    """Navigate to a URL and verify page loaded."""
    page = self._ensure_web()
    action_id = self._new_id("nav")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        title = page.title()
        current_url = page.url

        return ActionOutcome(
            action_id   = action_id,
            action_type = "navigate",
            target      = url,
            status      = ActionStatus.SUCCESS,
            evidence    = {
                "title":       title,
                "final_url":   current_url,
                "redirected":  current_url != url,
            },
        )
    except Exception as e:
        return ActionOutcome(
            action_id   = action_id,
            action_type = "navigate",
            target      = url,
            status      = ActionStatus.FAILED,
            evidence    = {"error": str(e)},
        )

def submit_form(
    self,
    url: str,
    fields: dict[str, str],
    submit_selector: str = "button[type='submit'], input[type='submit']",
    success_signals: list[str] = None,
) -> ActionOutcome:
    """
    Navigate to a form, fill it, submit it, verify success.

    success_signals: list of text strings that indicate success
                     e.g. ["thank you", "message sent", "application received"]
    """
    page       = self._ensure_web()
    action_id  = self._new_id("form")
    signals    = success_signals or [
        "thank you", "thanks", "success", "sent", "received",
        "submitted", "application", "we'll be in touch", "confirmation"
    ]

    try:
        # Navigate
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        before_url = page.url

        # Fill fields
        filled = []
        for selector, value in fields.items():
            try:
                page.fill(selector, value)
                filled.append(selector)
                time.sleep(0.3)  # human-like pacing
            except Exception as e:
                pass  # try remaining fields

        # Take screenshot before submit
        before_shot = f"before_submit_{action_id}.png"
        page.screenshot(path=before_shot)

        # Submit
        submitted = False
        try:
            page.click(submit_selector, timeout=5000)
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            submitted = True
        except Exception as e:
            pass

        # After state
        after_url  = page.url
        after_text = page.inner_text("body")[:2000].lower()
        after_shot = f"after_submit_{action_id}.png"
        page.screenshot(path=after_shot)

        # Save session (cookies may have updated)
        self._session.save_storage(self._context)

        # Detect success
        url_changed    = after_url != before_url
        signal_matched = any(sig in after_text for sig in signals)
        captcha_found  = any(w in after_text for w in
                             ["captcha", "robot", "verify you are human"])
        auth_required  = any(w in after_text for w in
                             ["log in", "sign in", "login required"])

        if captcha_found:
            status = ActionStatus.NEEDS_CAPTCHA
        elif auth_required and not submitted:
            status = ActionStatus.NEEDS_AUTH
        elif signal_matched or url_changed:
            status = ActionStatus.SUCCESS
        elif submitted:
            status = ActionStatus.PARTIAL
        else:
            status = ActionStatus.FAILED

        return ActionOutcome(
            action_id   = action_id,
            action_type = "submit_form",
            target      = url,
            status      = status,
            evidence    = {
                "fields_filled":   filled,
                "submitted":       submitted,
                "url_changed":     url_changed,
                "signal_matched":  signal_matched,
                "before_url":      before_url,
                "after_url":       after_url,
                "before_shot":     before_shot,
                "after_shot":      after_shot,
                "captcha":         captcha_found,
                "auth_required":   auth_required,
            },
        )

    except Exception as e:
        return ActionOutcome(
            action_id   = action_id,
            action_type = "submit_form",
            target      = url,
            status      = ActionStatus.FAILED,
            evidence    = {"error": str(e)},
        )

def extract_structured(
    self,
    url: str,
    patterns: dict[str, str],
) -> ActionOutcome:
    """
    Navigate to a page and extract structured data using CSS selectors.

    patterns: {label: css_selector}
    e.g. {"title": "h1", "price": ".price", "description": ".desc"}
    """
    page      = self._ensure_web()
    action_id = self._new_id("extract")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        extracted = {}

        for label, selector in patterns.items():
            try:
                elements = page.query_selector_all(selector)
                if elements:
                    extracted[label] = [
                        el.inner_text()[:300] for el in elements[:10]
                    ]
                else:
                    extracted[label] = []
            except Exception:
                extracted[label] = []

        success = any(bool(v) for v in extracted.values())

        return ActionOutcome(
            action_id   = action_id,
            action_type = "extract_structured",
            target      = url,
            status      = ActionStatus.SUCCESS if success else ActionStatus.PARTIAL,
            evidence    = {"extracted": extracted},
        )

    except Exception as e:
        return ActionOutcome(
            action_id   = action_id,
            action_type = "extract_structured",
            target      = url,
            status      = ActionStatus.FAILED,
            evidence    = {"error": str(e)},
        )

def download_file(self, url: str, save_path: str) -> ActionOutcome:
    """Download a file and verify it saved correctly."""
    page      = self._ensure_web()
    action_id = self._new_id("dl")

    try:
        with page.expect_download() as dl_info:
            page.goto(url)
        download = dl_info.value
        download.save_as(save_path)
        size = Path(save_path).stat().st_size if Path(save_path).exists() else 0

        return ActionOutcome(
            action_id   = action_id,
            action_type = "download",
            target      = url,
            status      = ActionStatus.SUCCESS if size > 0 else ActionStatus.FAILED,
            evidence    = {"path": save_path, "size_bytes": size},
        )
    except Exception as e:
        return ActionOutcome(
            action_id   = action_id,
            action_type = "download",
            target      = url,
            status      = ActionStatus.FAILED,
            evidence    = {"error": str(e)},
        )

def draft_email(
    self,
    to: str,
    subject: str,
    body: str,
    save_path: str = "email_drafts.jsonl",
) -> ActionOutcome:
    """
    Saves an email draft to disk for human review before sending.
    Janus never sends email autonomously — human approves first.
    """
    action_id = self._new_id("email")
    draft = {
        "action_id": action_id,
        "to":        to,
        "subject":   subject,
        "body":      body,
        "created":   datetime.now().isoformat(),
        "status":    "pending_human_review",
    }
    with open(save_path, "a") as f:
        f.write(json.dumps(draft) + "\n")

    return ActionOutcome(
        action_id   = action_id,
        action_type = "draft_email",
        target      = to,
        status      = ActionStatus.SUCCESS,
        evidence    = {"saved_to": save_path, "subject": subject},
        notes       = "Awaiting human review before sending.",
    )

def stop(self):
    if self._web:
        try:
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass
        self._web = None

@staticmethod
def _new_id(prefix: str = "act") -> str:
    ts = str(time.time()).encode()
    return f"{prefix}_{hashlib.md5(ts).hexdigest()[:8]}"
```

# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
“””
Persists browser cookies and localStorage between runs.
This means Janus stays logged in across sessions without
storing raw passwords anywhere.
“””

```
SESSION_FILE = "janus_session.json"

def save_storage(self, context) -> bool:
    try:
        storage = context.storage_state()
        Path(self.SESSION_FILE).write_text(json.dumps(storage, indent=2))
        return True
    except Exception:
        return False

def load_storage(self) -> Optional[dict]:
    p = Path(self.SESSION_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None

def clear(self):
    Path(self.SESSION_FILE).unlink(missing_ok=True)
```

# ── Feedback Loop ─────────────────────────────────────────────────────────────

class FeedbackLoop:
“””
Records action outcomes and feeds them back into:
- Valence (did this feel good/bad?)
- Memory (what worked?)
- Goal scorer (what should we try next?)
- Success model (what’s our actual track record?)
“””

```
FEEDBACK_FILE = "action_feedback.jsonl"

def __init__(self):
    self._path = Path(self.FEEDBACK_FILE)
    self._core = None
    self._try_load_core()

def _try_load_core(self):
    try:
        from core import AutonomousCore
        self._core = AutonomousCore()
    except Exception:
        pass

def record(self, outcome: ActionOutcome) -> dict:
    """
    Record an outcome and compute its valence impact.
    Returns the feedback signal that was applied.
    """
    # Compute valence impact
    valence_delta = self._outcome_to_valence(outcome)

    # Build feedback record
    record = {
        **asdict(outcome),
        "valence_delta": valence_delta,
        "recorded_at":   datetime.now().isoformat(),
    }

    # Persist
    with self._path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    # Update AutonomousCore valence if available
    if self._core:
        stimulus = self._outcome_to_text(outcome)
        self._core.perceive(stimulus)

    return valence_delta

def _outcome_to_valence(self, outcome: ActionOutcome) -> dict:
    """Map action outcome to valence changes."""
    if outcome.status == ActionStatus.SUCCESS:
        return {
            "pleasure":   +0.15,
            "competence": +0.10,
            "autonomy":   +0.08,
            "arousal":    -0.05,  # calm after success
        }
    elif outcome.status == ActionStatus.PARTIAL:
        return {
            "pleasure":   +0.05,
            "competence": +0.03,
            "curiosity":  +0.08,  # partial = interesting, keep going
        }
    elif outcome.status == ActionStatus.NEEDS_AUTH:
        return {
            "pleasure":   -0.05,
            "autonomy":   -0.10,  # blocked by external gate
            "competence": -0.03,
        }
    elif outcome.status == ActionStatus.NEEDS_CAPTCHA:
        return {
            "pleasure":   -0.08,
            "autonomy":   -0.12,
            "arousal":    +0.10,  # frustrating
        }
    elif outcome.status == ActionStatus.BLOCKED:
        return {
            "pleasure":   -0.10,
            "autonomy":   -0.15,
            "competence": -0.05,
        }
    else:  # FAILED
        return {
            "pleasure":   -0.12,
            "competence": -0.08,
            "arousal":    +0.05,
        }

def _outcome_to_text(self, outcome: ActionOutcome) -> str:
    """Convert outcome to natural language for JanusGPT perception."""
    status_text = {
        ActionStatus.SUCCESS:       "successfully completed",
        ActionStatus.PARTIAL:       "partially completed",
        ActionStatus.NEEDS_AUTH:    "blocked — authentication required",
        ActionStatus.NEEDS_CAPTCHA: "blocked by CAPTCHA",
        ActionStatus.BLOCKED:       "blocked by the target site",
        ActionStatus.FAILED:        "failed",
    }.get(outcome.status, "completed with unknown status")

    return (
        f"Action '{outcome.action_type}' targeting '{outcome.target}' "
        f"{status_text}. Evidence: {json.dumps(outcome.evidence)[:200]}"
    )

def summary(self, last_n: int = 50) -> dict:
    """Return success statistics over last N actions."""
    if not self._path.exists():
        return {"total": 0, "success_rate": 0.0, "by_type": {}}

    lines = self._path.read_text().strip().splitlines()
    records = [json.loads(l) for l in lines[-last_n:]]

    total    = len(records)
    success  = sum(1 for r in records if r["status"] == ActionStatus.SUCCESS)
    by_type: dict[str, dict] = {}

    for r in records:
        t = r["action_type"]
        if t not in by_type:
            by_type[t] = {"total": 0, "success": 0}
        by_type[t]["total"] += 1
        if r["status"] == ActionStatus.SUCCESS:
            by_type[t]["success"] += 1

    return {
        "total":        total,
        "success_rate": round(success / total, 3) if total else 0.0,
        "by_type":      by_type,
        "recent":       records[-5:],
    }

def what_works(self) -> list[str]:
    """
    Returns action types with >60% success rate.
    Used by GoalScorer to prefer proven approaches.
    """
    stats = self.summary()
    return [
        action_type
        for action_type, s in stats["by_type"].items()
        if s["total"] >= 3 and (s["success"] / s["total"]) >= 0.6
    ]

def what_fails(self) -> list[str]:
    """Returns action types with >60% failure rate."""
    stats = self.summary()
    return [
        action_type
        for action_type, s in stats["by_type"].items()
        if s["total"] >= 3 and (s["success"] / s["total"]) < 0.4
    ]
```

# ── Integration helper ────────────────────────────────────────────────────────

class ActionFeedbackBridge:
“””
Connects ActionEngine + FeedbackLoop to the ReplanningAgent.
Registers action tools into the agent’s ToolExecutor so the
agent loop can call real-world actions like any other tool.
“””

```
def __init__(self, agent):
    self.engine   = ActionEngine()
    self.feedback = FeedbackLoop()
    self.agent    = agent
    self._register_tools()

def _register_tools(self):
    from tool_executor import ToolSpec, RiskTier

    def _wrap(fn):
        """Wraps an ActionEngine method to record feedback automatically."""
        def handler(args: dict):
            outcome = fn(**args)
            self.feedback.record(outcome)
            return asdict(outcome)
        return handler

    tools = [
        ToolSpec(
            name        = "action_navigate",
            description = "Navigate to a URL and verify page loaded",
            risk        = RiskTier.HIGH,
            parameters  = {"url": "str"},
            handler     = _wrap(lambda url: self.engine.navigate(url)),
        ),
        ToolSpec(
            name        = "action_submit_form",
            description = "Fill and submit a web form with outcome verification",
            risk        = RiskTier.HIGH,
            parameters  = {"url": "str", "fields": "dict"},
            handler     = _wrap(
                lambda url, fields, **kw:
                self.engine.submit_form(url, fields, **kw)
            ),
        ),
        ToolSpec(
            name        = "action_extract",
            description = "Extract structured data from a webpage",
            risk        = RiskTier.HIGH,
            parameters  = {"url": "str", "patterns": "dict"},
            handler     = _wrap(
                lambda url, patterns:
                self.engine.extract_structured(url, patterns)
            ),
        ),
        ToolSpec(
            name        = "action_draft_email",
            description = "Draft an email for human review (never sends autonomously)",
            risk        = RiskTier.MEDIUM,
            parameters  = {"to": "str", "subject": "str", "body": "str"},
            handler     = _wrap(
                lambda to, subject, body:
                self.engine.draft_email(to, subject, body)
            ),
        ),
    ]

    for spec in tools:
        self.agent.executor.registry.register(spec)

def feedback_summary(self) -> str:
    """Plain text summary for Janus to reason about its own performance."""
    stats    = self.feedback.summary()
    works    = self.feedback.what_works()
    fails    = self.feedback.what_fails()

    lines = [
        f"Action feedback summary ({stats['total']} actions recorded):",
        f"  Overall success rate: {stats['success_rate']:.0%}",
    ]
    if works:
        lines.append(f"  What works well: {', '.join(works)}")
    if fails:
        lines.append(f"  What struggles:  {', '.join(fails)}")

    return "\n".join(lines)

def stop(self):
    self.engine.stop()
```

# ── CLI / quick test ──────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys

```
engine   = ActionEngine(headless=True)
feedback = FeedbackLoop()

if "--test-nav" in sys.argv:
    print("Testing navigation...")
    outcome = engine.navigate("https://www.duckduckgo.com")
    print(f"Status: {outcome.status}")
    print(f"Evidence: {outcome.evidence}")
    fb = feedback.record(outcome)
    print(f"Valence delta: {fb}")

elif "--test-extract" in sys.argv:
    print("Testing extraction...")
    outcome = engine.extract_structured(
        "https://www.duckduckgo.com",
        {"title": "h1, h2", "links": "a"}
    )
    print(f"Status: {outcome.status}")
    print(f"Extracted: {json.dumps(outcome.evidence, indent=2)[:500]}")

elif "--summary" in sys.argv:
    print(feedback.summary())

else:
    print("Usage:")
    print("  python actions_and_feedback.py --test-nav")
    print("  python actions_and_feedback.py --test-extract")
    print("  python actions_and_feedback.py --summary")

engine.stop()
```