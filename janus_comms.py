"""
janus_comms.py
===============
Real communication channels for Janus. Zero API keys.
Uses standard protocols your email provider already supports.

Channels:
  1. Email (SMTP send + IMAP receive) -- works with Gmail, Outlook, any provider
  2. Windows desktop notifications    -- ctypes, no install needed
  3. Local message queue              -- file-based, works with any app
  4. Daily digest report              -- summarizes what Janus did

Setup (one time):
  Create comms_config.json with your email credentials.
  Gmail: enable "App Passwords" in Google Account settings.
  Outlook: use your normal password with smtp.office365.com.

Usage:
    from janus_comms import JanusComms
    comms = JanusComms()
    comms.send_email("subject", "body")
    comms.notify("Janus completed a task")
    comms.post_update("Revenue goal: 45% complete")
"""

from __future__ import annotations

import email
import imaplib
import json
import logging
import os
import platform
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("janus.comms")

_CONFIG_FILE  = Path("comms_config.json")
_QUEUE_FILE   = Path("janus_messages.jsonl")
_DIGEST_FILE  = Path("janus_digest.json")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class CommsConfig:
    # Email send (SMTP)
    smtp_host:     str  = "smtp.gmail.com"
    smtp_port:     int  = 587
    smtp_user:     str  = ""          # your email address
    smtp_password: str  = ""          # app password (not your main password)
    from_name:     str  = "Janus"

    # Email receive (IMAP) -- for reading replies
    imap_host:     str  = "imap.gmail.com"
    imap_port:     int  = 993

    # Who to notify
    owner_email:   str  = ""          # your email address

    # Notification preferences
    notify_on_task_complete: bool = True
    notify_on_escalation:    bool = True
    notify_on_revenue:       bool = True
    daily_digest_hour:       int  = 20  # 8pm

    @classmethod
    def load(cls, path: Path = _CONFIG_FILE) -> "CommsConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
            except Exception as e:
                logger.warning(f"Could not load comms config: {e}")
        return cls()

    def save(self, path: Path = _CONFIG_FILE):
        path.write_text(json.dumps(asdict(self), indent=2))

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password and self.owner_email)


# ── Message record ────────────────────────────────────────────────────────────

@dataclass
class CommsMessage:
    message_id:  str
    channel:     str        # "email" | "notification" | "queue"
    subject:     str
    body:        str
    sent_at:     str
    delivered:   bool = False
    error:       Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Email channel ─────────────────────────────────────────────────────────────

class EmailChannel:
    """
    Sends and receives email using standard SMTP/IMAP.
    No API keys. Works with any email provider.

    Gmail setup:
      1. Go to myaccount.google.com → Security → App Passwords
      2. Create an app password for "Mail"
      3. Use that 16-char password as smtp_password

    Outlook setup:
      smtp_host = "smtp.office365.com", smtp_port = 587
      imap_host = "outlook.office365.com", imap_port = 993
    """

    def __init__(self, config: CommsConfig):
        self.config = config

    def send(
        self,
        subject: str,
        body: str,
        to: Optional[str] = None,
        html: bool = False,
    ) -> bool:
        """Send an email. Returns True on success."""
        if not self.config.is_configured:
            logger.warning("Email not configured — message logged locally only")
            self._log_locally(subject, body)
            return False

        recipient = to or self.config.owner_email

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"{self.config.from_name} <{self.config.smtp_user}>"
            msg["To"]      = recipient

            content_type = "html" if html else "plain"
            msg.attach(MIMEText(body, content_type, "utf-8"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(self.config.smtp_user, recipient, msg.as_string())

            logger.info(f"Email sent to {recipient}: {subject}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("Email auth failed — check smtp_user and smtp_password in comms_config.json")
            return False
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def check_inbox(self, limit: int = 10) -> List[dict]:
        """
        Check inbox for new messages addressed to Janus.
        Returns list of message dicts with subject, sender, body, date.
        """
        if not self.config.is_configured:
            return []

        messages = []
        try:
            context = ssl.create_default_context()
            with imaplib.IMAP4_SSL(
                self.config.imap_host, self.config.imap_port, ssl_context=context
            ) as mail:
                mail.login(self.config.smtp_user, self.config.smtp_password)
                mail.select("INBOX")

                # Search for unread messages
                _, data = mail.search(None, "UNSEEN")
                ids = data[0].split()[-limit:]  # most recent N

                for uid in ids:
                    _, msg_data = mail.fetch(uid, "(RFC822)")
                    raw = msg_data[0][1]
                    msg = email.message_from_bytes(raw)

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

                    messages.append({
                        "uid":     uid.decode(),
                        "subject": msg.get("Subject", ""),
                        "sender":  msg.get("From", ""),
                        "date":    msg.get("Date", ""),
                        "body":    body[:2000],
                    })

        except Exception as e:
            logger.error(f"IMAP check failed: {e}")

        return messages

    def _log_locally(self, subject: str, body: str):
        """Fallback: write to local queue when email isn't configured."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "channel":   "email_fallback",
            "subject":   subject,
            "body":      body[:500],
        }
        with _QUEUE_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Desktop notification channel ─────────────────────────────────────────────

class NotificationChannel:
    """
    Windows desktop toast notifications via ctypes.
    No third-party packages needed.
    Falls back to console print on non-Windows.
    """

    def __init__(self):
        self._is_windows = platform.system() == "Windows"

    def notify(self, title: str, message: str, duration: int = 5) -> bool:
        """Show a desktop notification."""
        if self._is_windows:
            return self._windows_toast(title, message, duration)
        else:
            # Fallback for Linux/Mac
            print(f"\n🔔 [{title}] {message}")
            return True

    def _windows_toast(self, title: str, message: str, duration: int) -> bool:
        """
        Windows 10/11 toast notification via PowerShell.
        No COM registration needed, no third-party packages.
        """
        try:
            import subprocess
            # Escape single quotes in title/message
            t = title.replace("'", "''")
            m = message.replace("'", "''")
            script = (
                f"[Windows.UI.Notifications.ToastNotificationManager, "
                f"Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
                f"$template = [Windows.UI.Notifications.ToastNotificationManager]"
                f"::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]"
                f"::ToastText02); "
                f"$template.SelectSingleNode('//text[@id=1]').InnerText = '{t}'; "
                f"$template.SelectSingleNode('//text[@id=2]').InnerText = '{m}'; "
                f"$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                f"[Windows.UI.Notifications.ToastNotificationManager]"
                f"::CreateToastNotifier('Janus').Show($toast)"
            )
            subprocess.run(
                ["powershell", "-WindowStyle", "Hidden", "-Command", script],
                capture_output=True, timeout=5
            )
            return True
        except Exception as e:
            logger.debug(f"Toast notification failed: {e}")
            print(f"\n🔔 [{title}] {message}")
            return False


# ── Local message queue ───────────────────────────────────────────────────────

class MessageQueue:
    """
    File-based message queue. Janus writes updates here.
    You can read them from any app, script, or the daily digest.
    """

    def __init__(self, path: Path = _QUEUE_FILE):
        self.path = path

    def post(self, category: str, message: str, data: Optional[dict] = None):
        """Post a message to the queue."""
        entry = {
            "id":        f"{int(time.time())}_{category[:8]}",
            "timestamp": datetime.now().isoformat(),
            "category":  category,
            "message":   message,
            "data":      data or {},
            "read":      False,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_unread(self, limit: int = 50) -> List[dict]:
        """Read unread messages."""
        if not self.path.exists():
            return []
        entries = []
        lines   = self.path.read_text().strip().splitlines()
        for line in lines[-limit:]:
            try:
                e = json.loads(line)
                if not e.get("read"):
                    entries.append(e)
            except Exception:
                pass
        return entries

    def mark_all_read(self):
        """Mark all messages as read."""
        if not self.path.exists():
            return
        lines = self.path.read_text().strip().splitlines()
        updated = []
        for line in lines:
            try:
                e = json.loads(line)
                e["read"] = True
                updated.append(json.dumps(e))
            except Exception:
                updated.append(line)
        self.path.write_text("\n".join(updated) + "\n")

    def tail(self, n: int = 20) -> List[dict]:
        """Get the last N messages regardless of read status."""
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().splitlines()
        result = []
        for line in lines[-n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result


# ── Daily digest ──────────────────────────────────────────────────────────────

class DigestBuilder:
    """
    Builds a daily summary email from the message queue and CEO state.
    Sent automatically at the configured hour.
    """

    def __init__(self, queue: MessageQueue):
        self.queue = queue

    def build(self) -> tuple[str, str]:
        """Returns (subject, html_body)."""
        now      = datetime.now()
        messages = self.queue.tail(100)
        today    = [m for m in messages
                    if m["timestamp"][:10] == now.strftime("%Y-%m-%d")]

        # Group by category
        by_cat: Dict[str, List[dict]] = {}
        for m in today:
            by_cat.setdefault(m["category"], []).append(m)

        # CEO state
        ceo_summary = self._get_ceo_summary()

        subject = f"Janus Daily Report — {now.strftime('%B %d, %Y')}"

        rows = ""
        for cat, msgs in sorted(by_cat.items()):
            rows += f"<tr><td colspan='2' style='background:#f0f0f0;padding:6px;font-weight:bold'>{cat.upper()}</td></tr>"
            for m in msgs[-5:]:  # last 5 per category
                ts = m["timestamp"][11:16]
                rows += (f"<tr><td style='padding:4px 8px;color:#666;white-space:nowrap'>{ts}</td>"
                         f"<td style='padding:4px 8px'>{m['message'][:120]}</td></tr>")

        html = f"""
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
<h2 style="color:#333">Janus Daily Report</h2>
<p style="color:#666">{now.strftime('%A, %B %d, %Y')}</p>

<h3>Financial Snapshot</h3>
<pre style="background:#f8f8f8;padding:12px;border-radius:4px">{ceo_summary}</pre>

<h3>Activity Log ({len(today)} events today)</h3>
<table style="width:100%;border-collapse:collapse;font-size:14px">
{rows if rows else '<tr><td style="padding:8px;color:#999">No activity logged today.</td></tr>'}
</table>

<hr style="margin:24px 0;border:none;border-top:1px solid #eee">
<p style="color:#999;font-size:12px">
  Janus Autonomous System &bull; {now.strftime('%Y-%m-%d %H:%M')}
</p>
</body></html>
"""
        return subject, html

    def _get_ceo_summary(self) -> str:
        try:
            state_file = Path("ceo_state.json")
            if not state_file.exists():
                return "No CEO state available."
            state    = json.loads(state_file.read_text())
            fin      = state.get("financial", {})
            goals    = state.get("goals", {})
            active   = sum(1 for g in goals.values()
                           if g.get("status") not in ("completed", "failed"))
            complete = sum(1 for g in goals.values()
                           if g.get("status") == "completed")
            return (
                f"Cash:     ${fin.get('cash', 0):,.2f}\n"
                f"Revenue:  ${fin.get('revenue', 0):,.2f}\n"
                f"Expenses: ${fin.get('expenses', 0):,.2f}\n"
                f"Goals:    {active} active, {complete} completed"
            )
        except Exception as e:
            return f"Could not load CEO state: {e}"


# ── Main comms hub ────────────────────────────────────────────────────────────

class JanusComms:
    """
    Single entry point for all Janus communication.
    Wire this into the scheduler and CEO loop.

    Quick start (no email):
        comms = JanusComms()
        comms.notify("Task done")
        comms.post_update("revenue", "Earned $500 today")

    With email:
        comms = JanusComms()
        comms.setup_email("you@gmail.com", "app_password_here")
        comms.send_email("Daily report", "Everything is running.")
    """

    def __init__(self, config: Optional[CommsConfig] = None):
        self.config       = config or CommsConfig.load()
        self.email        = EmailChannel(self.config)
        self.notifications= NotificationChannel()
        self.queue        = MessageQueue()
        self.digest       = DigestBuilder(self.queue)
        self._inbox_callbacks: List[Callable[[dict], None]] = []
        self._digest_thread: Optional[threading.Thread] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup_email(
        self,
        email_address: str,
        app_password: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        imap_host: str = "imap.gmail.com",
    ):
        """
        Configure email credentials.
        Saves to comms_config.json for future runs.
        """
        self.config.smtp_user     = email_address
        self.config.smtp_password = app_password
        self.config.owner_email   = email_address
        self.config.smtp_host     = smtp_host
        self.config.smtp_port     = smtp_port
        self.config.imap_host     = imap_host
        self.config.save()
        logger.info(f"Email configured for {email_address}")

    # ── Send ──────────────────────────────────────────────────────────────────

    def send_email(self, subject: str, body: str,
                   to: Optional[str] = None, html: bool = False) -> bool:
        """Send an email to the owner (or a specific address)."""
        ok = self.email.send(subject, body, to=to, html=html)
        self.queue.post("email_sent", f"Email: {subject}", {"to": to or self.config.owner_email})
        return ok

    def notify(self, message: str, title: str = "Janus") -> bool:
        """Show a desktop notification."""
        self.queue.post("notification", message)
        return self.notifications.notify(title, message)

    def post_update(self, category: str, message: str, data: Optional[dict] = None):
        """Post an update to the local message queue."""
        self.queue.post(category, message, data)
        logger.info(f"[{category}] {message}")

    # ── Receive ───────────────────────────────────────────────────────────────

    def check_inbox(self) -> List[dict]:
        """Check email inbox for new messages."""
        return self.email.check_inbox()

    def on_inbox_message(self, callback: Callable[[dict], None]):
        """Register a callback for incoming email messages."""
        self._inbox_callbacks.append(callback)

    def start_inbox_polling(self, interval_seconds: int = 300):
        """Poll inbox every N seconds and fire callbacks."""
        def _poll():
            while True:
                try:
                    messages = self.check_inbox()
                    for msg in messages:
                        for cb in self._inbox_callbacks:
                            try:
                                cb(msg)
                            except Exception as e:
                                logger.error(f"Inbox callback error: {e}")
                except Exception as e:
                    logger.error(f"Inbox poll error: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=_poll, daemon=True, name="janus-inbox")
        t.start()
        logger.info(f"Inbox polling started (every {interval_seconds}s)")

    # ── Digest ────────────────────────────────────────────────────────────────

    def send_daily_digest(self) -> bool:
        """Build and send the daily digest email."""
        subject, html = self.digest.build()
        ok = self.send_email(subject, html, html=True)
        if ok:
            self.queue.mark_all_read()
            logger.info("Daily digest sent and queue marked as read")
        return ok

    def start_digest_scheduler(self):
        """
        Start a background thread that sends the daily digest
        at the configured hour (default 8pm).
        """
        def _loop():
            while True:
                now = datetime.now()
                target = now.replace(
                    hour=self.config.daily_digest_hour,
                    minute=0, second=0, microsecond=0
                )
                if target <= now:
                    target += timedelta(days=1)
                wait = (target - now).total_seconds()
                logger.info(f"Next digest in {wait/3600:.1f}h at {target.strftime('%H:%M')}")
                time.sleep(wait)
                self.send_daily_digest()

        self._digest_thread = threading.Thread(
            target=_loop, daemon=True, name="janus-digest"
        )
        self._digest_thread.start()

    # ── Convenience methods for CEO/scheduler integration ─────────────────────

    def report_task_complete(self, task_name: str, result: str):
        self.post_update("task_complete", f"✓ {task_name}: {result[:100]}")
        if self.config.notify_on_task_complete:
            self.notify(f"Task done: {task_name}", "Janus")

    def report_escalation(self, task_name: str, error: str):
        self.post_update("escalation", f"⚠ {task_name} needs attention: {error[:100]}")
        if self.config.notify_on_escalation:
            self.notify(f"Needs attention: {task_name}", "Janus ⚠")
            self.send_email(
                f"[Janus] Task escalated: {task_name}",
                f"Task '{task_name}' failed too many times and needs your review.\n\n"
                f"Last error: {error}\n\nCheck scheduler_state.json for details."
            )

    def report_revenue(self, amount: float, source: str):
        self.post_update("revenue", f"💰 ${amount:.2f} from {source}")
        if self.config.notify_on_revenue:
            self.notify(f"${amount:.2f} from {source}", "Janus 💰")

    def get_unread_summary(self) -> str:
        """Get a text summary of unread messages for the brain to read."""
        msgs = self.queue.read_unread(20)
        if not msgs:
            return "No unread messages."
        lines = [f"[{m['category']}] {m['message']}" for m in msgs]
        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_comms: Optional[JanusComms] = None

def get_comms() -> JanusComms:
    global _comms
    if _comms is None:
        _comms = JanusComms()
    return _comms


# ── CLI / setup wizard ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Communications Setup")
    parser.add_argument("--setup",  action="store_true", help="Interactive email setup")
    parser.add_argument("--test",   action="store_true", help="Send a test email")
    parser.add_argument("--digest", action="store_true", help="Send daily digest now")
    parser.add_argument("--inbox",  action="store_true", help="Check inbox")
    parser.add_argument("--queue",  action="store_true", help="Show message queue")
    args = parser.parse_args()

    comms = JanusComms()

    if args.setup:
        print("Janus Email Setup")
        print("-" * 40)
        print("For Gmail: enable App Passwords at myaccount.google.com → Security")
        print("For Outlook: use smtp.office365.com / outlook.office365.com")
        print()
        addr = input("Your email address: ").strip()
        pwd  = input("App password (not your main password): ").strip()
        host = input("SMTP host [smtp.gmail.com]: ").strip() or "smtp.gmail.com"
        comms.setup_email(addr, pwd, smtp_host=host)
        print(f"\nConfig saved to {_CONFIG_FILE}")

    elif args.test:
        print("Sending test email...")
        ok = comms.send_email(
            "Janus Test Email",
            "If you received this, Janus email is working correctly.\n\n"
            f"Sent at: {datetime.now().isoformat()}"
        )
        print("Sent!" if ok else "Failed — check comms_config.json")

    elif args.digest:
        print("Building and sending daily digest...")
        ok = comms.send_daily_digest()
        print("Sent!" if ok else "Failed or not configured")

    elif args.inbox:
        print("Checking inbox...")
        msgs = comms.check_inbox()
        if not msgs:
            print("No unread messages.")
        for m in msgs:
            print(f"\nFrom: {m['sender']}")
            print(f"Subject: {m['subject']}")
            print(f"Date: {m['date']}")
            print(f"Body: {m['body'][:200]}...")

    elif args.queue:
        msgs = comms.queue.tail(20)
        if not msgs:
            print("Queue is empty.")
        else:
            print(f"{'Time':<20} {'Category':<18} {'Message'}")
            print("-" * 70)
            for m in msgs:
                ts  = m["timestamp"][11:19]
                cat = m["category"][:16]
                msg = m["message"][:50]
                print(f"{ts:<20} {cat:<18} {msg}")

    else:
        parser.print_help()
