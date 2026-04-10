"""
janus_relay_client.py
======================
Python client that lets Janus push messages to your phone
through the self-hosted relay server.

No carriers. No third parties. Your EliteDesk → your phone.

Usage:
    from janus_relay_client import relay_text, relay_alert, relay_voice

    relay_text("Task completed: financial snapshot done")
    relay_alert("Escalation needs your attention!")
    relay_voice("Hello, this is Janus. Revenue goal is 45 percent complete.")
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Literal

# ── Config ────────────────────────────────────────────────────────────────────

_RELAY_URL    = os.environ.get("RELAY_HTTP_URL", "http://localhost:3000")
_RELAY_SECRET = os.environ.get("RELAY_SECRET", "change-this-secret")
_TIMEOUT      = 5


# ── HTTP send ─────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> bool:
    url  = f"{_RELAY_URL}/api/relay/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_RELAY_SECRET}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[Relay] Send failed: {e}")
        return False


# ── TTS → base64 audio ────────────────────────────────────────────────────────

def _text_to_audio_b64(text: str) -> str | None:
    """Convert text to base64-encoded WAV using pyttsx3."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 145)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        if Path(tmp_path).exists():
            with open(tmp_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            Path(tmp_path).unlink(missing_ok=True)
            return audio_b64
    except Exception as e:
        print(f"[Relay] TTS failed: {e}")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def relay_text(message: str) -> bool:
    """Send a text message to your phone."""
    return _post("send", {"type": "text", "content": message})


def relay_alert(message: str) -> bool:
    """
    Send an urgent alert — vibrates your phone and shows a notification.
    Use for escalations and things needing immediate attention.
    """
    return _post("send", {"type": "alert", "content": message})


def relay_voice(message: str) -> bool:
    """
    Convert text to speech and send audio to your phone.
    Your phone plays it like a voice message.
    """
    audio_b64 = _text_to_audio_b64(message)
    if audio_b64:
        return _post("send", {"type": "voice", "content": audio_b64})
    # Fallback to text if TTS fails
    return relay_text(f"[Voice] {message}")


def relay_status(summary: str) -> bool:
    """Send a status update (shown with a checkmark icon)."""
    return _post("send", {"type": "status", "content": summary})


def is_phone_connected() -> bool:
    """Check if your phone's PWA is currently connected."""
    try:
        url = f"{_RELAY_URL}/api/relay/status"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {_RELAY_SECRET}"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data.get("connected", False)
    except Exception:
        return False


def read_inbox() -> list[dict]:
    """
    Read messages sent from your phone to Janus.
    Janus should call this periodically to check for your replies.
    """
    inbox_file = Path("janus_inbox.jsonl")
    if not inbox_file.exists():
        return []

    messages = []
    lines    = inbox_file.read_text().strip().splitlines()
    for line in lines:
        try:
            messages.append(json.loads(line))
        except Exception:
            pass

    # Clear after reading
    if messages:
        inbox_file.write_text("")

    return messages


# ── Wire into Janus systems ───────────────────────────────────────────────────

def patch_janus_comms():
    """
    Route all Janus notifications through the relay when phone is connected.
    Call once at startup.
    """
    try:
        import janus_comms as _comms

        _orig_notify = _comms.JanusComms.notify

        def _patched(self, message: str, title: str = "Janus") -> bool:
            if is_phone_connected():
                level = "alert" if "🚨" in title or "alert" in title.lower() else "text"
                fn    = relay_alert if level == "alert" else relay_text
                return fn(message)
            return _orig_notify(self, message, title)

        _comms.JanusComms.notify = _patched
        print("[RelayClient] Patched janus_comms to use relay")
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Relay Client")
    parser.add_argument("--status",  action="store_true", help="Check phone connection")
    parser.add_argument("--text",    type=str, help="Send a text message")
    parser.add_argument("--alert",   type=str, help="Send an alert")
    parser.add_argument("--voice",   type=str, help="Send a voice message")
    parser.add_argument("--inbox",   action="store_true", help="Read inbox messages")
    args = parser.parse_args()

    if args.status:
        connected = is_phone_connected()
        print(f"Phone {'connected ✓' if connected else 'not connected ✗'}")

    elif args.text:
        ok = relay_text(args.text)
        print("Sent ✓" if ok else "Failed ✗")

    elif args.alert:
        ok = relay_alert(args.alert)
        print("Sent ✓" if ok else "Failed ✗")

    elif args.voice:
        ok = relay_voice(args.voice)
        print("Sent ✓" if ok else "Failed ✗")

    elif args.inbox:
        messages = read_inbox()
        if not messages:
            print("No messages from you.")
        for m in messages:
            print(f"[{m['timestamp'][:19]}] {m['content']}")

    else:
        parser.print_help()
