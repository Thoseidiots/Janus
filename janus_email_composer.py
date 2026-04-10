"""
janus_email_composer.py
========================
Janus composes and sends professional emails on your behalf.

Janus can now:
  - Draft emails from a brief instruction
  - Send emails to third parties (clients, vendors, contacts)
  - Reply to emails in your inbox
  - Follow up automatically after a set time
  - Maintain a contact book

No API keys. Uses smtplib (stdlib) + JanusBrain for drafting.
Builds on janus_comms.py SMTP config.

Usage:
    from janus_email_composer import JanusEmailComposer
    composer = JanusEmailComposer()

    # Draft and send
    composer.send_to(
        to="client@example.com",
        instruction="Follow up on the invoice I sent last week. Be professional but friendly.",
        subject="Following up on Invoice #1234"
    )

    # Reply to an inbox message
    composer.reply_to(email_msg, instruction="Accept the project and ask for a start date")
"""

from __future__ import annotations

import json
import re
import smtplib
import ssl
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

_CONTACTS_FILE  = Path("contacts.json")
_DRAFTS_FILE    = Path("email_drafts.jsonl")
_FOLLOWUP_FILE  = Path("email_followups.json")


# ── Contact book ──────────────────────────────────────────────────────────────

@dataclass
class Contact:
    email:        str
    name:         str
    relationship: str   = "contact"   # client | vendor | partner | contact
    notes:        str   = ""
    last_emailed: Optional[str] = None
    email_count:  int   = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ContactBook:
    def __init__(self):
        self._contacts: Dict[str, Contact] = {}
        self._load()

    def add(self, email: str, name: str, relationship: str = "contact", notes: str = ""):
        self._contacts[email.lower()] = Contact(email, name, relationship, notes)
        self._save()

    def get(self, email: str) -> Optional[Contact]:
        return self._contacts.get(email.lower())

    def get_name(self, email: str) -> str:
        c = self.get(email)
        return c.name if c else email.split("@")[0].title()

    def record_email(self, email: str):
        c = self.get(email)
        if c:
            c.last_emailed = datetime.now().isoformat()
            c.email_count += 1
            self._save()

    def list_all(self) -> List[Contact]:
        return list(self._contacts.values())

    def _save(self):
        data = {k: v.to_dict() for k, v in self._contacts.items()}
        _CONTACTS_FILE.write_text(json.dumps(data, indent=2))

    def _load(self):
        if _CONTACTS_FILE.exists():
            try:
                data = json.loads(_CONTACTS_FILE.read_text())
                for k, v in data.items():
                    self._contacts[k] = Contact(**v)
            except Exception:
                pass


# ── Email draft ───────────────────────────────────────────────────────────────

@dataclass
class EmailDraft:
    draft_id:   str
    to:         str
    subject:    str
    body:       str
    created_at: str
    sent:       bool          = False
    sent_at:    Optional[str] = None
    instruction: str          = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Composer ──────────────────────────────────────────────────────────────────

class JanusEmailComposer:
    """
    Drafts and sends professional emails using JanusBrain.
    Builds on the SMTP config from janus_comms.py.
    """

    def __init__(self):
        self.contacts = ContactBook()
        self._drafts: List[EmailDraft] = []

    # ── Drafting ──────────────────────────────────────────────────────────────

    def draft(
        self,
        to:          str,
        instruction: str,
        subject:     Optional[str] = None,
        context:     str           = "",
        tone:        str           = "professional",
    ) -> EmailDraft:
        """
        Draft an email from a brief instruction.
        Uses JanusBrain to write the actual email.
        """
        recipient_name = self.contacts.get_name(to)

        prompt = (
            f"Write a {tone} email.\n\n"
            f"To: {recipient_name} ({to})\n"
            f"Instruction: {instruction}\n"
        )
        if context:
            prompt += f"Context: {context}\n"
        prompt += (
            f"\nWrite ONLY the email body (no subject line, no 'To:' header).\n"
            f"Start with a greeting. End with a professional sign-off.\n"
            f"Keep it concise — under 200 words unless the task requires more."
        )

        # Generate subject if not provided
        if not subject:
            subject = self._generate_subject(instruction)

        # Generate body
        body = self._generate_body(prompt)

        import uuid
        draft = EmailDraft(
            draft_id    = str(uuid.uuid4())[:8],
            to          = to,
            subject     = subject,
            body        = body,
            created_at  = datetime.now().isoformat(),
            instruction = instruction,
        )
        self._drafts.append(draft)
        self._save_draft(draft)
        return draft

    def reply_to(
        self,
        original_email: dict,
        instruction:    str,
        tone:           str = "professional",
    ) -> EmailDraft:
        """
        Draft a reply to an email from your inbox.
        original_email: dict with 'sender', 'subject', 'body' keys.
        """
        sender  = original_email.get("sender", "")
        subject = original_email.get("subject", "")
        body    = original_email.get("body", "")[:500]

        # Extract email address from "Name <email>" format
        email_match = re.search(r'<([^>]+)>', sender)
        to_email    = email_match.group(1) if email_match else sender

        context = f"Original email from {sender}:\nSubject: {subject}\n{body}"
        reply_subject = subject if subject.startswith("Re:") else f"Re: {subject}"

        return self.draft(
            to          = to_email,
            instruction = instruction,
            subject     = reply_subject,
            context     = context,
            tone        = tone,
        )

    # ── Sending ───────────────────────────────────────────────────────────────

    def send(self, draft: EmailDraft) -> bool:
        """Send a drafted email. Returns True on success."""
        try:
            from janus_comms import JanusComms
            comms = JanusComms()

            if not comms.config.is_configured:
                print(f"[EmailComposer] SMTP not configured — draft saved but not sent")
                print(f"  To: {draft.to}")
                print(f"  Subject: {draft.subject}")
                print(f"  Body preview: {draft.body[:100]}...")
                return False

            ok = comms.email.send(
                subject = draft.subject,
                body    = draft.body,
                to      = draft.to,
            )

            if ok:
                draft.sent    = True
                draft.sent_at = datetime.now().isoformat()
                self.contacts.record_email(draft.to)
                self._save_draft(draft)
                print(f"[EmailComposer] Sent to {draft.to}: {draft.subject}")

            return ok

        except Exception as e:
            print(f"[EmailComposer] Send failed: {e}")
            return False

    def send_to(
        self,
        to:          str,
        instruction: str,
        subject:     Optional[str] = None,
        context:     str           = "",
        tone:        str           = "professional",
        preview:     bool          = True,
    ) -> bool:
        """Draft and send in one call."""
        draft = self.draft(to, instruction, subject, context, tone)

        if preview:
            print(f"\n── Email Draft ──────────────────────────────────")
            print(f"To:      {draft.to}")
            print(f"Subject: {draft.subject}")
            print(f"Body:\n{draft.body}")
            print(f"────────────────────────────────────────────────\n")

        return self.send(draft)

    # ── Follow-up scheduler ───────────────────────────────────────────────────

    def schedule_followup(
        self,
        to:          str,
        instruction: str,
        days:        int = 3,
        subject:     Optional[str] = None,
    ):
        """Schedule a follow-up email to be sent in N days."""
        followup = {
            "to":          to,
            "instruction": instruction,
            "subject":     subject,
            "send_after":  (datetime.now() + timedelta(days=days)).isoformat(),
            "sent":        False,
        }
        followups = self._load_followups()
        followups.append(followup)
        _FOLLOWUP_FILE.write_text(json.dumps(followups, indent=2))
        print(f"[EmailComposer] Follow-up scheduled in {days} days to {to}")

    def process_followups(self) -> int:
        """Send any due follow-up emails. Returns count sent."""
        followups = self._load_followups()
        sent = 0
        now  = datetime.now().isoformat()

        for fu in followups:
            if not fu["sent"] and fu["send_after"] <= now:
                ok = self.send_to(
                    to          = fu["to"],
                    instruction = fu["instruction"],
                    subject     = fu.get("subject"),
                    preview     = False,
                )
                if ok:
                    fu["sent"] = True
                    sent += 1

        _FOLLOWUP_FILE.write_text(json.dumps(followups, indent=2))
        return sent

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_subject(self, instruction: str) -> str:
        try:
            from avus_brain import get_brain
            brain = get_brain()
            subject = brain.ask(
                f"Write a concise email subject line (max 8 words) for this email: {instruction}",
                max_tokens=20,
            )
            # Clean up
            subject = subject.strip().strip('"').strip("'")
            return subject[:80] if subject else instruction[:60]
        except Exception:
            return instruction[:60]

    def _generate_body(self, prompt: str) -> str:
        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.ask(prompt, max_tokens=300)
        except Exception:
            return (
                f"Hello,\n\n"
                f"I am writing regarding: {prompt[:100]}\n\n"
                f"Please let me know if you have any questions.\n\n"
                f"Best regards,\nJanus"
            )

    def _save_draft(self, draft: EmailDraft):
        with _DRAFTS_FILE.open("a") as f:
            f.write(json.dumps(draft.to_dict()) + "\n")

    def _load_followups(self) -> List[dict]:
        if not _FOLLOWUP_FILE.exists():
            return []
        try:
            return json.loads(_FOLLOWUP_FILE.read_text())
        except Exception:
            return []

    def get_drafts(self, unsent_only: bool = False) -> List[EmailDraft]:
        if not _DRAFTS_FILE.exists():
            return []
        drafts = []
        for line in _DRAFTS_FILE.read_text().strip().splitlines():
            try:
                d = json.loads(line)
                draft = EmailDraft(**d)
                if not unsent_only or not draft.sent:
                    drafts.append(draft)
            except Exception:
                pass
        return drafts


# ── Module-level singleton ────────────────────────────────────────────────────

_composer: Optional[JanusEmailComposer] = None

def get_composer() -> JanusEmailComposer:
    global _composer
    if _composer is None:
        _composer = JanusEmailComposer()
    return _composer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Janus Email Composer")
    parser.add_argument("--draft",    nargs=2, metavar=("TO", "INSTRUCTION"))
    parser.add_argument("--send",     nargs=2, metavar=("TO", "INSTRUCTION"))
    parser.add_argument("--followup", nargs=3, metavar=("TO", "INSTRUCTION", "DAYS"))
    parser.add_argument("--process",  action="store_true", help="Process due follow-ups")
    parser.add_argument("--drafts",   action="store_true", help="Show unsent drafts")
    parser.add_argument("--contacts", action="store_true", help="List contacts")
    parser.add_argument("--add-contact", nargs=3, metavar=("EMAIL", "NAME", "RELATIONSHIP"))
    args = parser.parse_args()

    composer = JanusEmailComposer()

    if args.draft:
        draft = composer.draft(args.draft[0], args.draft[1])
        print(f"Subject: {draft.subject}\n\n{draft.body}")

    elif args.send:
        composer.send_to(args.send[0], args.send[1])

    elif args.followup:
        composer.schedule_followup(args.followup[0], args.followup[1], int(args.followup[2]))

    elif args.process:
        n = composer.process_followups()
        print(f"Sent {n} follow-up(s)")

    elif args.drafts:
        for d in composer.get_drafts(unsent_only=True):
            print(f"[{d.draft_id}] To: {d.to} | {d.subject}")

    elif args.contacts:
        for c in composer.contacts.list_all():
            print(f"{c.name} <{c.email}> [{c.relationship}]")

    elif args.add_contact:
        composer.contacts.add(*args.add_contact)
        print(f"Added: {args.add_contact[1]} <{args.add_contact[0]}>")

    else:
        parser.print_help()
