"""
janus_notify.py — Notification system for the Janus autonomous worker.

Supports multiple optional channels:
- Email (smtplib + TLS)
- Webhook (Discord / Slack / custom HTTP POST)
- Desktop toast (win10toast → plyer → log fallback)

All channels are optional and fail gracefully.
Rate limiting prevents the same subject from being sent more than once per 30 minutes.
"""

import json
import logging
import time
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone

log = logging.getLogger("janus_notify")

# ---------------------------------------------------------------------------
# Default config structure
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "email": {
        "enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_password": "",
        "from_addr": "",
        "to_addr": "",
    },
    "webhook": {
        "enabled": False,
        "url": "",
    },
    "desktop": {
        "enabled": True,
    },
}

RATE_LIMIT_SECONDS = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class NotificationChannel(ABC):
    """Abstract base class for all notification channels."""

    @abstractmethod
    def send(self, subject: str, body: str, level: str) -> bool:
        """
        Send a notification.

        Args:
            subject: Short title / subject line.
            body: Full message body.
            level: Severity — "info", "warning", or "critical".

        Returns:
            True if the notification was sent successfully, False otherwise.
        """


# ---------------------------------------------------------------------------
# Email channel
# ---------------------------------------------------------------------------

class EmailNotifier(NotificationChannel):
    """
    Sends notifications via SMTP with STARTTLS.

    Falls back gracefully if smtplib is unavailable or credentials are missing.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_addr: str,
        to_addr: str,
    ) -> None:
        """Initialise the email notifier with SMTP credentials."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr
        self.to_addr = to_addr

    def send(self, subject: str, body: str, level: str) -> bool:
        """Send an email notification via SMTP/TLS."""
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.from_addr, self.to_addr]):
            log.warning("EmailNotifier: incomplete configuration, skipping.")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = self.to_addr
            msg["Subject"] = f"[Janus/{level.upper()}] {subject}"
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_addr, self.to_addr, msg.as_string())

            log.info("EmailNotifier: sent '%s' to %s", subject, self.to_addr)
            return True

        except Exception as exc:
            log.error("EmailNotifier: failed to send '%s': %s", subject, exc)
            return False


# ---------------------------------------------------------------------------
# Webhook channel
# ---------------------------------------------------------------------------

class WebhookNotifier(NotificationChannel):
    """
    Sends notifications via HTTP POST to a Discord/Slack/custom webhook URL.

    Payload format is compatible with Discord's webhook API.
    """

    def __init__(self, webhook_url: str) -> None:
        """Initialise with the target webhook URL."""
        self.webhook_url = webhook_url

    def send(self, subject: str, body: str, level: str) -> bool:
        """POST a JSON payload to the configured webhook URL."""
        if not self.webhook_url:
            log.warning("WebhookNotifier: no URL configured, skipping.")
            return False

        try:
            import requests  # type: ignore

            level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "📢")
            payload = {
                "content": f"{level_emoji} **{subject}**\n{body}",
                "username": "Janus",
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            log.info("WebhookNotifier: sent '%s' (HTTP %d)", subject, response.status_code)
            return True

        except Exception as exc:
            log.error("WebhookNotifier: failed to send '%s': %s", subject, exc)
            return False


# ---------------------------------------------------------------------------
# Desktop notification channel
# ---------------------------------------------------------------------------

class DesktopNotifier(NotificationChannel):
    """
    Shows a Windows desktop toast notification.

    Tries win10toast first, then plyer, then falls back to logging only.
    """

    def send(self, subject: str, body: str, level: str) -> bool:
        """Display a desktop toast notification."""
        message = body[:200]

        # Attempt win10toast
        try:
            from win10toast import ToastNotifier  # type: ignore
            toaster = ToastNotifier()
            toaster.show_toast(subject, message, duration=8, threaded=True)
            log.info("DesktopNotifier (win10toast): '%s'", subject)
            return True
        except Exception:
            pass

        # Attempt plyer
        try:
            from plyer import notification  # type: ignore
            notification.notify(
                title=subject,
                message=message,
                app_name="Janus",
                timeout=8,
            )
            log.info("DesktopNotifier (plyer): '%s'", subject)
            return True
        except Exception:
            pass

        # Final fallback: log only
        log.info("DesktopNotifier (log fallback): [%s] %s — %s", level.upper(), subject, message)
        return True  # Considered "sent" via log


# ---------------------------------------------------------------------------
# Main notifier
# ---------------------------------------------------------------------------

class JanusNotifier:
    """
    Central notification dispatcher for the Janus autonomous worker.

    Loads channel configuration from a JSON file, registers enabled channels,
    and provides convenience methods for common notification events.
    Rate-limits repeated notifications to once per 30 minutes per subject.
    """

    def __init__(self, config_file: str = "janus_notify_config.json") -> None:
        """
        Initialise the notifier.

        Args:
            config_file: Path to the JSON configuration file.
                         A default config is created if the file does not exist.
        """
        self.config_file = pathlib.Path(config_file)
        self.channels: list[NotificationChannel] = []
        self._rate_limit: dict[str, float] = {}  # subject -> last sent timestamp

        self._load_config()
        self._register_channels()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load configuration from JSON, creating defaults if necessary."""
        if not self.config_file.exists():
            try:
                self.config_file.write_text(
                    json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8"
                )
                log.info("JanusNotifier: created default config at %s", self.config_file)
            except Exception as exc:
                log.warning("JanusNotifier: could not write default config: %s", exc)
            self.config = dict(DEFAULT_CONFIG)
            return

        try:
            self.config = json.loads(self.config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("JanusNotifier: failed to read config, using defaults: %s", exc)
            self.config = dict(DEFAULT_CONFIG)

    def _register_channels(self) -> None:
        """Instantiate and register all enabled notification channels."""
        # Email
        try:
            email_cfg = self.config.get("email", {})
            if email_cfg.get("enabled", False):
                self.channels.append(
                    EmailNotifier(
                        smtp_host=email_cfg.get("smtp_host", ""),
                        smtp_port=int(email_cfg.get("smtp_port", 587)),
                        smtp_user=email_cfg.get("smtp_user", ""),
                        smtp_password=email_cfg.get("smtp_password", ""),
                        from_addr=email_cfg.get("from_addr", ""),
                        to_addr=email_cfg.get("to_addr", ""),
                    )
                )
                log.info("JanusNotifier: EmailNotifier registered.")
        except Exception as exc:
            log.warning("JanusNotifier: could not register EmailNotifier: %s", exc)

        # Webhook
        try:
            webhook_cfg = self.config.get("webhook", {})
            if webhook_cfg.get("enabled", False):
                self.channels.append(WebhookNotifier(webhook_url=webhook_cfg.get("url", "")))
                log.info("JanusNotifier: WebhookNotifier registered.")
        except Exception as exc:
            log.warning("JanusNotifier: could not register WebhookNotifier: %s", exc)

        # Desktop
        try:
            desktop_cfg = self.config.get("desktop", {})
            if desktop_cfg.get("enabled", True):
                self.channels.append(DesktopNotifier())
                log.info("JanusNotifier: DesktopNotifier registered.")
        except Exception as exc:
            log.warning("JanusNotifier: could not register DesktopNotifier: %s", exc)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, subject: str) -> bool:
        """Return True if this subject was sent within the last 30 minutes."""
        last_sent = self._rate_limit.get(subject, 0.0)
        return (time.time() - last_sent) < RATE_LIMIT_SECONDS

    def _record_sent(self, subject: str) -> None:
        """Record the current time as the last-sent timestamp for this subject."""
        self._rate_limit[subject] = time.time()

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def notify(self, subject: str, body: str, level: str = "info") -> None:
        """
        Send a notification to all registered channels.

        Skips sending if the same subject was sent within the last 30 minutes.

        Args:
            subject: Short title / subject line.
            body: Full message body.
            level: Severity — "info", "warning", or "critical".
        """
        if self._is_rate_limited(subject):
            log.debug("JanusNotifier: rate-limited, skipping '%s'", subject)
            return

        self._record_sent(subject)

        if not self.channels:
            log.info("[%s] %s — %s", level.upper(), subject, body)
            return

        for channel in self.channels:
            try:
                channel.send(subject, body, level)
            except Exception as exc:
                log.error(
                    "JanusNotifier: channel %s raised an error for '%s': %s",
                    type(channel).__name__,
                    subject,
                    exc,
                )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def notify_payment(self, amount: float, job_title: str) -> None:
        """Notify that a payment has been received."""
        subject = f"Payment received: ${amount:.2f} for '{job_title}'"
        body = (
            f"Janus received a payment of ${amount:.2f} for the job '{job_title}'.\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="info")

    def notify_job_complete(self, job_title: str, quality: float) -> None:
        """Notify that a job has been completed."""
        subject = f"Job completed: '{job_title}' quality={quality:.0%}"
        body = (
            f"Job '{job_title}' finished successfully.\n"
            f"Quality score: {quality:.0%}\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="info")

    def notify_job_failed(self, job_title: str, reason: str) -> None:
        """Notify that a job has failed."""
        subject = f"Job failed: '{job_title}' — {reason}"
        body = (
            f"Job '{job_title}' failed.\n"
            f"Reason: {reason}\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="warning")

    def notify_low_balance(self, balance: float) -> None:
        """Notify that the JC balance is running low."""
        subject = f"Low JC balance: {balance:.2f} JC"
        body = (
            f"Janus JC balance has dropped to {balance:.2f} JC.\n"
            f"Consider topping up to avoid service interruptions.\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="warning")

    def notify_error(self, error: str, context: str) -> None:
        """Notify of a critical error."""
        subject = f"Error: {error}"
        body = (
            f"A critical error occurred in Janus.\n"
            f"Error: {error}\n"
            f"Context: {context}\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="critical")

    def notify_cycle_start(self, cycle_num: int, mood: str, energy: float) -> None:
        """
        Notify at the start of a work cycle — only when energy is low or mood is negative.

        Sends only if energy < 0.3 or mood is "stressed" or "sad".
        """
        if energy >= 0.3 and mood not in ("stressed", "sad"):
            return

        subject = f"Cycle #{cycle_num} starting — mood={mood}, energy={energy:.0%}"
        body = (
            f"Work cycle #{cycle_num} is starting with low resources.\n"
            f"Current mood: {mood}\n"
            f"Current energy: {energy:.0%}\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )
        self.notify(subject, body, level="info")
