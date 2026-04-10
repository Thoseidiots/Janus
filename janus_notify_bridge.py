"""
janus_notify_bridge.py
=======================
Python bridge that lets Janus send SMS/email notifications
through the MeshISP dashboard's notify endpoint.

No API keys. Uses your own email via the dashboard's SMTP config.

Usage:
    from janus_notify_bridge import notify, alert, warn

    notify("Task completed: financial snapshot done")
    alert("Escalation needs your attention!")
    warn("Revenue goal is behind pace")

The bridge calls POST http://localhost:3000/api/notify
which the dashboard converts to SMS + email.

Falls back to janus_comms.py if the dashboard isn't running.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Literal

# ── Config ────────────────────────────────────────────────────────────────────

_DASHBOARD_URL = os.environ.get("MESH_DASHBOARD_URL", "http://localhost:3000")
_NOTIFY_SECRET = os.environ.get("NOTIFY_SECRET", "")
_TIMEOUT       = 5  # seconds


# ── Core send ─────────────────────────────────────────────────────────────────

def _send_to_dashboard(
    message: str,
    level:   Literal["info", "warning", "alert"] = "info",
    sms:     bool = True,
    email:   bool = True,
) -> bool:
    """
    POST to the dashboard's /api/notify endpoint.
    Returns True on success.
    """
    url     = f"{_DASHBOARD_URL}/api/notify"
    payload = json.dumps({
        "message": message[:500],
        "level":   level,
        "sms":     sms,
        "email":   email,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
    }
    if _NOTIFY_SECRET:
        headers["Authorization"] = f"Bearer {_NOTIFY_SECRET}"

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def _fallback_notify(message: str, level: str) -> bool:
    """Fall back to janus_comms if dashboard isn't reachable."""
    try:
        from janus_comms import get_comms
        comms = get_comms()
        if level == "alert":
            comms.notify(message, "Janus 🚨")
            comms.send_email(f"[JANUS ALERT] {message[:60]}", message)
        elif level == "warning":
            comms.notify(message, "Janus ⚠")
        else:
            comms.post_update("notification", message)
        return True
    except Exception:
        print(f"[Notify] {level.upper()}: {message}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def notify(message: str, sms: bool = False, email: bool = True) -> bool:
    """
    Send an info-level notification.
    SMS off by default for routine updates — use alert() for urgent ones.
    """
    ok = _send_to_dashboard(message, level="info", sms=sms, email=email)
    if not ok:
        ok = _fallback_notify(message, "info")
    return ok


def warn(message: str, sms: bool = True, email: bool = True) -> bool:
    """Send a warning — SMS on by default."""
    ok = _send_to_dashboard(message, level="warning", sms=sms, email=email)
    if not ok:
        ok = _fallback_notify(message, "warning")
    return ok


def alert(message: str, sms: bool = True, email: bool = True) -> bool:
    """
    Send an urgent alert — SMS + email both on.
    Use for escalations, failures, and things needing immediate attention.
    """
    ok = _send_to_dashboard(message, level="alert", sms=sms, email=email)
    if not ok:
        ok = _fallback_notify(message, "alert")
    return ok


def ping_dashboard() -> bool:
    """Check if the dashboard is reachable."""
    try:
        url = f"{_DASHBOARD_URL}/api/notify/ping"
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Wire into Janus systems ───────────────────────────────────────────────────

def patch_janus_comms():
    """
    Monkey-patch janus_comms to route notifications through the dashboard.
    Call this once at startup if you want all Janus notifications to go
    through the ISP dashboard instead of direct SMTP.
    """
    try:
        import janus_comms as _comms

        _orig_notify = _comms.JanusComms.notify

        def _patched_notify(self, message: str, title: str = "Janus") -> bool:
            # Try dashboard first
            level = "alert" if "🚨" in title or "alert" in title.lower() else \
                    "warning" if "⚠" in title else "info"
            if ping_dashboard():
                return _send_to_dashboard(message, level=level, sms=(level != "info"))
            return _orig_notify(self, message, title)

        _comms.JanusComms.notify = _patched_notify
        print("[NotifyBridge] Patched janus_comms to route through dashboard")
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Notification Bridge")
    parser.add_argument("--ping",    action="store_true", help="Check dashboard connectivity")
    parser.add_argument("--test",    action="store_true", help="Send a test notification")
    parser.add_argument("--message", type=str, help="Send a custom message")
    parser.add_argument("--level",   type=str, default="info",
                        choices=["info", "warning", "alert"])
    args = parser.parse_args()

    if args.ping:
        ok = ping_dashboard()
        print(f"Dashboard {'reachable ✓' if ok else 'not reachable ✗'} at {_DASHBOARD_URL}")

    elif args.test:
        print("Sending test notification...")
        ok = alert("Janus notification test — if you got this, it works!")
        print("Sent ✓" if ok else "Failed ✗ (check dashboard is running)")

    elif args.message:
        fn = alert if args.level == "alert" else warn if args.level == "warning" else notify
        ok = fn(args.message)
        print("Sent ✓" if ok else "Failed ✗")

    else:
        parser.print_help()
