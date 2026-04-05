"""
janus_escalation.py
====================
Human-in-the-loop escalation for Janus.
When Janus doesn't know what to do, it stops and asks you.

This is the active circuit breaker that:
  1. Detects situations Janus can't handle autonomously
  2. Pauses the task and saves its state
  3. Notifies you (desktop + email)
  4. Waits for your response (email reply or CLI)
  5. Resumes with your guidance

No API keys. Uses janus_comms.py for notifications.

Escalation triggers:
  - Task fails N times in a row
  - Confidence score below threshold
  - Action involves money above a limit
  - Ethical flag raised
  - Unknown situation type
  - Explicit "I don't know" from JanusBrain
  - You can also manually escalate anything

Usage:
    from janus_escalation import EscalationManager, EscalationTrigger
    esc = EscalationManager()

    # Check before acting
    if esc.should_escalate(action="send_invoice", amount=5000):
        esc.escalate("Invoice amount exceeds auto-approve limit", context={...})
        return  # wait for human

    # Register a response handler
    esc.on_response("invoice_5000", lambda resp: process_invoice(resp))
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ESCALATION_FILE = Path("escalations.json")
_RESPONSE_FILE   = Path("escalation_responses.json")


# ── Trigger types ─────────────────────────────────────────────────────────────

class EscalationTrigger(str, Enum):
    TASK_FAILURE      = "task_failure"       # too many retries
    LOW_CONFIDENCE    = "low_confidence"     # brain not sure
    HIGH_VALUE        = "high_value"         # money above limit
    ETHICAL_FLAG      = "ethical_flag"       # moral concern raised
    UNKNOWN_SITUATION = "unknown_situation"  # no playbook for this
    MANUAL            = "manual"             # explicitly requested
    TIMEOUT           = "timeout"            # task running too long
    EXTERNAL_CHANGE   = "external_change"    # something unexpected happened


class EscalationStatus(str, Enum):
    OPEN     = "open"      # waiting for response
    ANSWERED = "answered"  # you responded
    EXPIRED  = "expired"   # timed out, Janus took safe default
    RESOLVED = "resolved"  # fully handled


# ── Escalation record ─────────────────────────────────────────────────────────

@dataclass
class Escalation:
    escalation_id: str
    trigger:       EscalationTrigger
    title:         str
    description:   str
    context:       Dict[str, Any]
    status:        EscalationStatus
    created_at:    str
    expires_at:    str                    # auto-resolve after this
    options:       List[str]              # choices you can pick
    default_option: Optional[str]         # what Janus does if you don't respond
    response:      Optional[str]  = None
    responded_at:  Optional[str]  = None
    resolved_at:   Optional[str]  = None
    resolution:    Optional[str]  = None
    task_paused:   bool           = True  # is the related task paused?

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trigger"] = self.trigger.value
        d["status"]  = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Escalation":
        d = dict(d)
        d["trigger"] = EscalationTrigger(d["trigger"])
        d["status"]  = EscalationStatus(d["status"])
        return cls(**d)

    @property
    def is_expired(self) -> bool:
        return datetime.fromisoformat(self.expires_at) < datetime.now()

    @property
    def age_hours(self) -> float:
        return (datetime.now() - datetime.fromisoformat(self.created_at)).total_seconds() / 3600


# ── Confidence evaluator ──────────────────────────────────────────────────────

class ConfidenceEvaluator:
    """
    Evaluates whether Janus is confident enough to act autonomously.
    Checks the brain's response for uncertainty signals.
    """

    # Phrases that indicate low confidence
    _UNCERTAINTY_PHRASES = [
        "i'm not sure", "i don't know", "unclear", "uncertain",
        "might be", "could be", "possibly", "perhaps", "not certain",
        "hard to say", "difficult to determine", "need more information",
        "unable to", "can't determine", "insufficient",
    ]

    # Phrases that indicate high confidence
    _CONFIDENCE_PHRASES = [
        "clearly", "definitely", "certainly", "the best approach",
        "recommend", "should", "will", "the answer is",
    ]

    def score(self, brain_response: str) -> float:
        """
        Returns 0.0 (no confidence) to 1.0 (full confidence).
        """
        if not brain_response:
            return 0.3  # no response = uncertain

        text = brain_response.lower()
        uncertainty_count = sum(1 for p in self._UNCERTAINTY_PHRASES if p in text)
        confidence_count  = sum(1 for p in self._CONFIDENCE_PHRASES  if p in text)

        # Base score
        score = 0.7
        score -= uncertainty_count * 0.15
        score += confidence_count  * 0.05
        return max(0.0, min(1.0, score))

    def is_confident(self, brain_response: str, threshold: float = 0.5) -> bool:
        return self.score(brain_response) >= threshold


# ── Escalation rules ──────────────────────────────────────────────────────────

@dataclass
class EscalationRules:
    """Configurable thresholds for when to escalate."""
    max_task_failures:      int   = 3       # escalate after N failures
    confidence_threshold:   float = 0.45    # escalate if brain score < this
    auto_approve_limit:     float = 100.0   # escalate money actions above $X
    task_timeout_minutes:   int   = 60      # escalate if task runs > N min
    expiry_hours:           int   = 24      # auto-resolve after N hours
    notify_email:           bool  = True
    notify_desktop:         bool  = True

    @classmethod
    def load(cls, path: Path = Path("escalation_rules.json")) -> "EscalationRules":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self, path: Path = Path("escalation_rules.json")):
        path.write_text(json.dumps(asdict(self), indent=2))


# ── Email response parser ─────────────────────────────────────────────────────

class ResponseParser:
    """
    Parses your email reply to extract a decision.
    You can reply with: "1", "approve", "option 2", "do nothing", etc.
    """

    def parse(self, email_body: str, options: List[str]) -> Optional[str]:
        text = email_body.strip().lower()

        # Direct number match: "1", "2", "3"
        m = re.match(r"^\s*(\d+)\s*$", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(options):
                return options[idx]

        # Direct option text match
        for opt in options:
            if opt.lower() in text or text in opt.lower():
                return opt

        # Common keywords
        if any(w in text for w in ("yes", "approve", "proceed", "go ahead", "do it")):
            return options[0] if options else "approve"
        if any(w in text for w in ("no", "reject", "cancel", "stop", "don't")):
            return options[-1] if options else "reject"
        if any(w in text for w in ("skip", "ignore", "later", "not now")):
            return "skip"

        # Return raw text if no match
        return text[:200] if text else None


# ── Main escalation manager ───────────────────────────────────────────────────

class EscalationManager:
    """
    The human-in-the-loop circuit breaker for Janus.

    Janus calls this before taking uncertain or high-stakes actions.
    If escalation is needed, the task pauses and you get notified.
    You respond via email or CLI, and Janus resumes.
    """

    def __init__(
        self,
        rules:          Optional[EscalationRules] = None,
        escalation_file: Path = _ESCALATION_FILE,
        response_file:   Path = _RESPONSE_FILE,
    ):
        self.rules          = rules or EscalationRules.load()
        self.esc_file       = escalation_file
        self.resp_file      = response_file
        self._escalations:  Dict[str, Escalation]         = {}
        self._callbacks:    Dict[str, Callable[[str], None]] = {}
        self._evaluator     = ConfidenceEvaluator()
        self._parser        = ResponseParser()
        self._lock          = threading.Lock()
        self._poll_thread:  Optional[threading.Thread] = None

        self._load()

    # ── Decision gate ─────────────────────────────────────────────────────────

    def should_escalate(
        self,
        action:          str,
        amount:          float           = 0.0,
        brain_response:  Optional[str]   = None,
        fail_count:      int             = 0,
        ethical_flags:   Optional[List[str]] = None,
    ) -> tuple[bool, str]:
        """
        Check whether an action should be escalated.
        Returns (should_escalate: bool, reason: str).
        """
        # Too many failures
        if fail_count >= self.rules.max_task_failures:
            return True, f"Task failed {fail_count} times"

        # Money above auto-approve limit
        if amount > self.rules.auto_approve_limit:
            return True, f"Amount ${amount:.2f} exceeds auto-approve limit ${self.rules.auto_approve_limit:.2f}"

        # Brain not confident
        if brain_response is not None:
            score = self._evaluator.score(brain_response)
            if score < self.rules.confidence_threshold:
                return True, f"Low confidence ({score:.0%}) in brain response"

        # Ethical flags
        if ethical_flags:
            return True, f"Ethical concern: {ethical_flags[0]}"

        return False, ""

    # ── Create escalation ─────────────────────────────────────────────────────

    def escalate(
        self,
        title:          str,
        description:    str,
        trigger:        EscalationTrigger = EscalationTrigger.UNKNOWN_SITUATION,
        context:        Optional[dict]    = None,
        options:        Optional[List[str]] = None,
        default_option: Optional[str]     = None,
        callback:       Optional[Callable[[str], None]] = None,
    ) -> Escalation:
        """
        Create an escalation, notify you, and pause the related task.
        Returns the escalation record.
        """
        esc_id  = str(uuid.uuid4())[:8]
        expires = (datetime.now() + timedelta(hours=self.rules.expiry_hours)).isoformat()

        default_options = ["Approve and continue", "Skip this task", "Cancel and review"]
        chosen_options  = options or default_options
        chosen_default  = default_option or chosen_options[-1]

        esc = Escalation(
            escalation_id = esc_id,
            trigger       = trigger,
            title         = title,
            description   = description,
            context       = context or {},
            status        = EscalationStatus.OPEN,
            created_at    = datetime.now().isoformat(),
            expires_at    = expires,
            options       = chosen_options,
            default_option= chosen_default,
        )

        with self._lock:
            self._escalations[esc_id] = esc
            if callback:
                self._callbacks[esc_id] = callback

        self._save()
        self._notify(esc)

        print(f"\n[Escalation] ⚠ {title}")
        print(f"  ID: {esc_id}")
        print(f"  Trigger: {trigger.value}")
        print(f"  Options: {', '.join(chosen_options)}")
        print(f"  Default (if no response in {self.rules.expiry_hours}h): {chosen_default}")
        print(f"  Respond: python janus_escalation.py --respond {esc_id} --choice 1\n")

        return esc

    # ── Respond to escalation ─────────────────────────────────────────────────

    def respond(self, escalation_id: str, response: str) -> bool:
        """
        Record your response to an escalation.
        Fires the callback if registered.
        """
        esc = self._escalations.get(escalation_id)
        if not esc:
            print(f"[Escalation] Not found: {escalation_id}")
            return False
        if esc.status != EscalationStatus.OPEN:
            print(f"[Escalation] Already {esc.status.value}: {escalation_id}")
            return False

        # Parse response against options
        parsed = self._parser.parse(response, esc.options)
        if not parsed:
            parsed = response

        esc.response     = parsed
        esc.responded_at = datetime.now().isoformat()
        esc.status       = EscalationStatus.ANSWERED
        esc.task_paused  = False

        self._save()

        print(f"[Escalation] ✅ Response recorded: '{parsed}' for {escalation_id}")

        # Fire callback
        cb = self._callbacks.get(escalation_id)
        if cb:
            try:
                cb(parsed)
            except Exception as e:
                print(f"[Escalation] Callback error: {e}")

        # Notify via comms
        try:
            from janus_comms import get_comms
            get_comms().post_update(
                "escalation_resolved",
                f"Escalation {escalation_id} resolved: {parsed[:60]}"
            )
        except Exception:
            pass

        return True

    # ── Auto-resolve expired escalations ─────────────────────────────────────

    def process_expired(self) -> List[Escalation]:
        """
        Auto-resolve escalations that have passed their expiry.
        Uses the default_option as the response.
        """
        resolved = []
        with self._lock:
            for esc in list(self._escalations.values()):
                if esc.status == EscalationStatus.OPEN and esc.is_expired:
                    esc.response     = esc.default_option
                    esc.responded_at = datetime.now().isoformat()
                    esc.status       = EscalationStatus.EXPIRED
                    esc.task_paused  = False
                    resolved.append(esc)

                    print(f"[Escalation] ⏰ Auto-resolved {esc.escalation_id}: "
                          f"'{esc.default_option}'")

                    cb = self._callbacks.get(esc.escalation_id)
                    if cb:
                        try:
                            cb(esc.default_option)
                        except Exception:
                            pass

        if resolved:
            self._save()
        return resolved

    # ── Inbox polling for email responses ─────────────────────────────────────

    def start_inbox_polling(self, interval_seconds: int = 120):
        """
        Poll your inbox for replies to escalation emails.
        Parses replies and auto-responds to matching escalations.
        """
        def _poll():
            while True:
                try:
                    self._check_inbox_for_responses()
                    self.process_expired()
                except Exception as e:
                    print(f"[Escalation] Poll error: {e}")
                time.sleep(interval_seconds)

        self._poll_thread = threading.Thread(
            target=_poll, daemon=True, name="janus-escalation-poll"
        )
        self._poll_thread.start()

    def _check_inbox_for_responses(self):
        """Check inbox for replies to escalation emails."""
        try:
            from janus_comms import get_comms
            emails = get_comms().check_inbox()
        except Exception:
            return

        for email_msg in emails:
            subject = email_msg.get("subject", "")
            body    = email_msg.get("body", "")

            # Look for escalation ID in subject or body
            m = re.search(r"\b([0-9a-f]{8})\b", subject + " " + body)
            if not m:
                continue

            esc_id = m.group(1)
            esc    = self._escalations.get(esc_id)
            if not esc or esc.status != EscalationStatus.OPEN:
                continue

            # Parse the response
            parsed = self._parser.parse(body, esc.options)
            if parsed:
                self.respond(esc_id, parsed)

    # ── Status & reporting ────────────────────────────────────────────────────

    def get_open_escalations(self) -> List[Escalation]:
        return [e for e in self._escalations.values()
                if e.status == EscalationStatus.OPEN]

    def get_all(self) -> List[Escalation]:
        return list(self._escalations.values())

    def print_status(self):
        open_escs = self.get_open_escalations()
        all_escs  = self.get_all()

        print(f"\n── Escalation Status ───────────────────────────────")
        print(f"  Open:     {len(open_escs)}")
        print(f"  Total:    {len(all_escs)}")

        if open_escs:
            print(f"\n  Open escalations:")
            for e in open_escs:
                age = f"{e.age_hours:.1f}h ago"
                print(f"    [{e.escalation_id}] {e.title[:45]} ({age})")
                print(f"      Options: {' | '.join(e.options)}")
                print(f"      Default: {e.default_option} (in {self.rules.expiry_hours}h)")
        print("────────────────────────────────────────────────────\n")

    # ── Notification ──────────────────────────────────────────────────────────

    def _notify(self, esc: Escalation):
        """Send desktop + email notification for a new escalation."""
        options_text = "\n".join(
            f"  {i+1}. {opt}" for i, opt in enumerate(esc.options)
        )
        body = (
            f"Janus needs your input.\n\n"
            f"Escalation ID: {esc.escalation_id}\n"
            f"Trigger: {esc.trigger.value}\n\n"
            f"{esc.description}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Reply to this email with the option number or text.\n"
            f"If no response in {self.rules.expiry_hours}h, Janus will: {esc.default_option}\n\n"
            f"Or respond via CLI:\n"
            f"  python janus_escalation.py --respond {esc.escalation_id} --choice 1"
        )

        try:
            from janus_comms import get_comms
            comms = get_comms()

            if self.rules.notify_desktop:
                comms.notify(f"⚠ {esc.title[:50]}", "Janus needs input")

            if self.rules.notify_email:
                comms.send_email(
                    f"[Janus] Action needed: {esc.title} [{esc.escalation_id}]",
                    body,
                )

            comms.post_update("escalation", f"⚠ {esc.title}", {
                "id": esc.escalation_id, "trigger": esc.trigger.value
            })
        except Exception as e:
            print(f"[Escalation] Notification failed: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "escalations": [e.to_dict() for e in self._escalations.values()],
                "saved_at":    datetime.now().isoformat(),
            }
            self.esc_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Escalation] Save failed: {e}")

    def _load(self):
        if not self.esc_file.exists():
            return
        try:
            data = json.loads(self.esc_file.read_text())
            for d in data.get("escalations", []):
                e = Escalation.from_dict(d)
                self._escalations[e.escalation_id] = e
            print(f"[Escalation] Loaded {len(self._escalations)} escalations "
                  f"({len(self.get_open_escalations())} open)")
        except Exception as e:
            print(f"[Escalation] Load failed: {e}")


# ── Convenience decorator ─────────────────────────────────────────────────────

def requires_confidence(threshold: float = 0.5, escalation_manager=None):
    """
    Decorator: escalates if JanusBrain response is below confidence threshold.

    @requires_confidence(threshold=0.6)
    def make_decision(situation: str) -> str:
        return brain.ask(situation)
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            mgr = escalation_manager or get_escalation_manager()
            evaluator = ConfidenceEvaluator()
            if isinstance(result, str) and evaluator.score(result) < threshold:
                mgr.escalate(
                    title       = f"Low confidence in: {fn.__name__}",
                    description = f"Brain response scored below {threshold:.0%}.\n\nResponse: {result[:300]}",
                    trigger     = EscalationTrigger.LOW_CONFIDENCE,
                    context     = {"function": fn.__name__, "response": result},
                    options     = ["Accept response anyway", "Skip action", "Provide manual answer"],
                    default_option = "Skip action",
                )
            return result
        return wrapper
    return decorator


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[EscalationManager] = None

def get_escalation_manager() -> EscalationManager:
    global _manager
    if _manager is None:
        _manager = EscalationManager()
    return _manager


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Escalation Manager")
    parser.add_argument("--status",  action="store_true", help="Show open escalations")
    parser.add_argument("--respond", type=str, metavar="ID", help="Respond to an escalation")
    parser.add_argument("--choice",  type=str, metavar="CHOICE",
                        help="Your choice (number or text)")
    parser.add_argument("--expire",  action="store_true",
                        help="Process expired escalations now")
    parser.add_argument("--rules",   action="store_true",
                        help="Show current escalation rules")
    parser.add_argument("--set-limit", type=float, metavar="AMOUNT",
                        help="Set auto-approve money limit")
    parser.add_argument("--set-confidence", type=float, metavar="SCORE",
                        help="Set confidence threshold (0.0-1.0)")
    args = parser.parse_args()

    mgr = EscalationManager()

    if args.status:
        mgr.print_status()

    elif args.respond and args.choice:
        mgr.respond(args.respond, args.choice)

    elif args.expire:
        resolved = mgr.process_expired()
        print(f"Auto-resolved {len(resolved)} expired escalation(s)")

    elif args.rules:
        r = mgr.rules
        print(f"\nEscalation Rules:")
        print(f"  Max task failures before escalate: {r.max_task_failures}")
        print(f"  Confidence threshold:              {r.confidence_threshold:.0%}")
        print(f"  Auto-approve money limit:          ${r.auto_approve_limit:.2f}")
        print(f"  Task timeout:                      {r.task_timeout_minutes} min")
        print(f"  Escalation expiry:                 {r.expiry_hours}h")

    elif args.set_limit:
        mgr.rules.auto_approve_limit = args.set_limit
        mgr.rules.save()
        print(f"Auto-approve limit set to ${args.set_limit:.2f}")

    elif args.set_confidence:
        mgr.rules.confidence_threshold = args.set_confidence
        mgr.rules.save()
        print(f"Confidence threshold set to {args.set_confidence:.0%}")

    else:
        parser.print_help()
