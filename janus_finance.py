"""
janus_finance.py
=================
Real money movement for Janus. Zero API keys.

How it works:
  1. Janus creates an invoice and emails it to the client automatically
  2. Janus monitors your inbox for payment confirmation emails
  3. Janus maintains a double-entry ledger (every dollar tracked)
  4. Outgoing transfers queue for your one-tap approval
  5. You approve via CLI or email reply -- Janus executes nothing without you

No bank API. No Stripe. No Plaid.
Uses: smtplib/imaplib (stdlib), payment_pipeline.py, janus_comms.py

Usage:
    from janus_finance import JanusFinance
    fin = JanusFinance()

    # Create and send an invoice to a client
    inv = fin.invoice_client("Alex", 250.00, "Logo design", "alex@example.com")

    # Check for payments received
    fin.scan_for_payments()

    # Queue a transfer for your approval
    fin.queue_transfer(100.00, "Savings account", "Monthly savings")

    # See what needs approval
    fin.show_approval_queue()
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# ── Ledger paths ──────────────────────────────────────────────────────────────

_LEDGER_FILE   = Path("finance_ledger.json")
_APPROVAL_FILE = Path("transfer_approvals.json")


# ── Enums ─────────────────────────────────────────────────────────────────────

class EntryType(str, Enum):
    INCOME   = "income"
    EXPENSE  = "expense"
    TRANSFER = "transfer"
    REFUND   = "refund"

class EntryStatus(str, Enum):
    PENDING   = "pending"    # created, not yet received/paid
    CONFIRMED = "confirmed"  # money actually received/paid
    CANCELLED = "cancelled"
    QUEUED    = "queued"     # waiting for your approval (outgoing only)

class PaymentMethod(str, Enum):
    REVOLUT  = "revolut"
    PAYPAL   = "paypal"
    FIVERR   = "fiverr"
    BANK     = "bank"
    CASH     = "cash"
    OTHER    = "other"


# ── Double-entry ledger record ────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    entry_id:    str
    type:        EntryType
    status:      EntryStatus
    amount:      float
    currency:    str
    description: str
    method:      PaymentMethod
    counterparty: str          # client name, vendor, etc.
    created_at:  str
    confirmed_at: Optional[str] = None
    invoice_sent: bool          = False
    invoice_email: Optional[str] = None
    payment_link:  Optional[str] = None
    notes:         str           = ""
    tags:          List[str]     = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"]   = self.type.value
        d["status"] = self.status.value
        d["method"] = self.method.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        d = dict(d)
        d["type"]   = EntryType(d["type"])
        d["status"] = EntryStatus(d["status"])
        d["method"] = PaymentMethod(d["method"])
        return cls(**d)

    @property
    def net_amount(self) -> float:
        """Amount after platform fees."""
        if self.method == PaymentMethod.FIVERR:
            return round(self.amount * 0.80, 2)
        return self.amount


# ── Transfer approval record ──────────────────────────────────────────────────

@dataclass
class TransferApproval:
    approval_id:  str
    amount:       float
    currency:     str
    destination:  str
    description:  str
    created_at:   str
    approved:     Optional[bool] = None
    approved_at:  Optional[str]  = None
    executed_at:  Optional[str]  = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransferApproval":
        return cls(**d)


# ── Invoice HTML template ─────────────────────────────────────────────────────

def _build_invoice_html(
    entry: LedgerEntry,
    owner_name: str,
    owner_email: str,
    payment_link: str,
) -> str:
    due_date = (datetime.now() + timedelta(days=7)).strftime("%B %d, %Y")
    return f"""
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;color:#333">
<div style="background:#1a1a2e;padding:24px;border-radius:8px 8px 0 0">
  <h1 style="color:#fff;margin:0;font-size:24px">Invoice</h1>
  <p style="color:#aaa;margin:4px 0 0">#{entry.entry_id[:8].upper()}</p>
</div>
<div style="border:1px solid #eee;border-top:none;padding:24px;border-radius:0 0 8px 8px">
  <table style="width:100%;margin-bottom:24px">
    <tr>
      <td><strong>From:</strong><br>{owner_name}<br>{owner_email}</td>
      <td style="text-align:right"><strong>To:</strong><br>{entry.counterparty}<br>
        <span style="color:#666">{entry.invoice_email or ''}</span></td>
    </tr>
  </table>

  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#f8f8f8">
      <th style="padding:10px;text-align:left;border-bottom:2px solid #eee">Description</th>
      <th style="padding:10px;text-align:right;border-bottom:2px solid #eee">Amount</th>
    </tr>
    <tr>
      <td style="padding:12px 10px">{entry.description}</td>
      <td style="padding:12px 10px;text-align:right">{entry.currency} {entry.amount:.2f}</td>
    </tr>
    <tr style="background:#f8f8f8;font-weight:bold">
      <td style="padding:10px;border-top:2px solid #eee">Total Due</td>
      <td style="padding:10px;text-align:right;border-top:2px solid #eee;color:#1a1a2e;font-size:18px">
        {entry.currency} {entry.amount:.2f}
      </td>
    </tr>
  </table>

  <div style="text-align:center;margin:24px 0">
    <a href="{payment_link}"
       style="background:#1a1a2e;color:#fff;padding:14px 32px;border-radius:6px;
              text-decoration:none;font-size:16px;display:inline-block">
      Pay Now
    </a>
  </div>

  <p style="color:#666;font-size:13px;text-align:center">
    Payment due by {due_date} &bull; {entry.method.value.title()}
  </p>
  <p style="color:#999;font-size:12px;text-align:center">
    Invoice #{entry.entry_id[:8].upper()} &bull; {datetime.now().strftime('%Y-%m-%d')}
  </p>
</div>
</body></html>
"""


# ── Payment confirmation detector ─────────────────────────────────────────────

class PaymentDetector:
    """
    Scans your inbox for payment confirmation emails.
    Matches against pending invoices by amount and counterparty name.
    No API — uses IMAP via janus_comms.
    """

    # Patterns that appear in PayPal/Revolut confirmation emails
    _PAYPAL_PATTERNS = [
        r"you received a payment of \$?([\d,]+\.?\d*)",
        r"payment of \$?([\d,]+\.?\d*) from (.+)",
        r"([\d,]+\.?\d*) USD from (.+)",
    ]
    _REVOLUT_PATTERNS = [
        r"received ([\d,]+\.?\d*) (USD|GBP|EUR) from (.+)",
        r"payment of ([\d,]+\.?\d*)",
    ]
    _FIVERR_PATTERNS = [
        r"order #\w+ has been completed",
        r"you've earned \$?([\d,]+\.?\d*)",
        r"funds are now available",
    ]

    def detect_payment(self, email_subject: str, email_body: str) -> Optional[dict]:
        """
        Try to extract payment info from an email.
        Returns dict with amount, sender, method or None.
        """
        text = (email_subject + " " + email_body).lower()

        # PayPal
        if "paypal" in text or "payment received" in text:
            for pat in self._PAYPAL_PATTERNS:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        amount = float(m.group(1).replace(",", ""))
                        sender = m.group(2).strip() if len(m.groups()) > 1 else "unknown"
                        return {"amount": amount, "sender": sender, "method": "paypal"}
                    except (ValueError, IndexError):
                        pass

        # Revolut
        if "revolut" in text:
            for pat in self._REVOLUT_PATTERNS:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        amount = float(m.group(1).replace(",", ""))
                        return {"amount": amount, "sender": "revolut", "method": "revolut"}
                    except (ValueError, IndexError):
                        pass

        # Fiverr
        if "fiverr" in text:
            for pat in self._FIVERR_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    m = re.search(r"\$?([\d,]+\.?\d*)", text)
                    amount = float(m.group(1).replace(",", "")) if m else 0
                    return {"amount": amount, "sender": "fiverr", "method": "fiverr"}

        return None


# ── Main finance engine ───────────────────────────────────────────────────────

class JanusFinance:
    """
    Real money tracking and movement for Janus.

    Janus can:
      - Create invoices and email them to clients
      - Detect incoming payments from your inbox
      - Track every dollar in a double-entry ledger
      - Queue outgoing transfers for your approval
      - Report financial position at any time

    You stay in control of outgoing money.
    Janus handles all the paperwork and chasing.
    """

    def __init__(
        self,
        ledger_path:   Path = _LEDGER_FILE,
        approval_path: Path = _APPROVAL_FILE,
        owner_name:    str  = "Janus Operations",
        owner_email:   str  = "",
    ):
        self.ledger_path   = ledger_path
        self.approval_path = approval_path
        self.owner_name    = owner_name
        self.owner_email   = owner_email

        self._entries:   Dict[str, LedgerEntry]      = {}
        self._approvals: Dict[str, TransferApproval] = {}
        self._detector   = PaymentDetector()

        self._load()

    # ── Invoice & income ──────────────────────────────────────────────────────

    def invoice_client(
        self,
        client_name:   str,
        amount:        float,
        description:   str,
        client_email:  Optional[str] = None,
        method:        PaymentMethod = PaymentMethod.REVOLUT,
        currency:      str           = "USD",
        send_email:    bool          = True,
    ) -> LedgerEntry:
        """
        Create an invoice, generate a payment link, and email it to the client.
        Returns the ledger entry.
        """
        entry_id = str(uuid.uuid4())

        # Build payment link
        payment_link = self._build_payment_link(method, amount, currency, description)

        entry = LedgerEntry(
            entry_id      = entry_id,
            type          = EntryType.INCOME,
            status        = EntryStatus.PENDING,
            amount        = amount,
            currency      = currency,
            description   = description,
            method        = method,
            counterparty  = client_name,
            created_at    = datetime.now().isoformat(),
            invoice_email = client_email,
            payment_link  = payment_link,
        )

        self._entries[entry_id] = entry

        # Send invoice email
        if send_email and client_email:
            self._send_invoice_email(entry)

        self._save()
        print(f"[Finance] Invoice created: {entry_id[:8]} | "
              f"{client_name} | {currency} {amount:.2f}")
        return entry

    def record_income(
        self,
        amount:       float,
        description:  str,
        counterparty: str,
        method:       PaymentMethod = PaymentMethod.OTHER,
        currency:     str           = "USD",
        notes:        str           = "",
    ) -> LedgerEntry:
        """Manually record confirmed income."""
        entry = LedgerEntry(
            entry_id     = str(uuid.uuid4()),
            type         = EntryType.INCOME,
            status       = EntryStatus.CONFIRMED,
            amount       = amount,
            currency     = currency,
            description  = description,
            method       = method,
            counterparty = counterparty,
            created_at   = datetime.now().isoformat(),
            confirmed_at = datetime.now().isoformat(),
            notes        = notes,
        )
        self._entries[entry.entry_id] = entry
        self._save()
        self._notify_income(entry)
        return entry

    def record_expense(
        self,
        amount:       float,
        description:  str,
        counterparty: str,
        method:       PaymentMethod = PaymentMethod.OTHER,
        currency:     str           = "USD",
        notes:        str           = "",
    ) -> LedgerEntry:
        """Record a confirmed expense."""
        entry = LedgerEntry(
            entry_id     = str(uuid.uuid4()),
            type         = EntryType.EXPENSE,
            status       = EntryStatus.CONFIRMED,
            amount       = amount,
            currency     = currency,
            description  = description,
            method       = method,
            counterparty = counterparty,
            created_at   = datetime.now().isoformat(),
            confirmed_at = datetime.now().isoformat(),
            notes        = notes,
        )
        self._entries[entry.entry_id] = entry
        self._save()
        return entry

    def confirm_payment(self, entry_id: str, notes: str = "") -> bool:
        """Mark a pending invoice as paid."""
        entry = self._get_entry(entry_id)
        if not entry:
            return False
        entry.status       = EntryStatus.CONFIRMED
        entry.confirmed_at = datetime.now().isoformat()
        if notes:
            entry.notes = notes
        self._save()
        self._notify_income(entry)
        print(f"[Finance] ✅ Payment confirmed: {entry_id[:8]} | "
              f"{entry.counterparty} | {entry.currency} {entry.amount:.2f}")
        return True

    # ── Inbox scanning ────────────────────────────────────────────────────────

    def scan_for_payments(self) -> List[LedgerEntry]:
        """
        Check email inbox for payment confirmations.
        Auto-confirms matching pending invoices.
        Returns list of newly confirmed entries.
        """
        confirmed = []
        try:
            from janus_comms import get_comms
            comms = get_comms()
            emails = comms.check_inbox()
        except Exception as e:
            print(f"[Finance] Inbox scan failed: {e}")
            return []

        for email_msg in emails:
            detected = self._detector.detect_payment(
                email_msg.get("subject", ""),
                email_msg.get("body", ""),
            )
            if not detected:
                continue

            # Try to match to a pending invoice
            matched = self._match_payment(
                detected["amount"],
                detected["sender"],
                detected["method"],
            )
            if matched:
                self.confirm_payment(matched.entry_id,
                                     notes=f"Auto-confirmed from inbox: {email_msg['subject'][:60]}")
                confirmed.append(matched)
                print(f"[Finance] Auto-confirmed payment: {matched.counterparty} "
                      f"${matched.amount:.2f}")

        return confirmed

    def _match_payment(
        self, amount: float, sender: str, method: str
    ) -> Optional[LedgerEntry]:
        """Find a pending invoice that matches this payment."""
        for entry in self._entries.values():
            if entry.status != EntryStatus.PENDING:
                continue
            if entry.type != EntryType.INCOME:
                continue
            # Match on amount (within $0.01) and method
            if abs(entry.amount - amount) < 0.02:
                if method in entry.method.value or entry.method.value in method:
                    return entry
            # Also match on counterparty name similarity
            if (sender.lower() in entry.counterparty.lower() or
                    entry.counterparty.lower() in sender.lower()):
                if abs(entry.amount - amount) < 1.00:  # within $1
                    return entry
        return None

    # ── Outgoing transfers (approval queue) ───────────────────────────────────

    def queue_transfer(
        self,
        amount:      float,
        destination: str,
        description: str,
        currency:    str = "USD",
    ) -> TransferApproval:
        """
        Queue an outgoing transfer for your approval.
        Janus will NOT execute this until you approve it.
        """
        approval = TransferApproval(
            approval_id = str(uuid.uuid4()),
            amount      = amount,
            currency    = currency,
            destination = destination,
            description = description,
            created_at  = datetime.now().isoformat(),
        )
        self._approvals[approval.approval_id] = approval
        self._save()

        # Notify owner
        try:
            from janus_comms import get_comms
            get_comms().notify(
                f"Transfer queued: ${amount:.2f} to {destination}",
                "Janus — Approval Needed"
            )
        except Exception:
            pass

        print(f"[Finance] Transfer queued for approval: {approval.approval_id[:8]} | "
              f"{destination} | {currency} {amount:.2f}")
        return approval

    def approve_transfer(self, approval_id: str) -> bool:
        """Approve a queued transfer. Records it as an expense."""
        approval = self._approvals.get(approval_id)
        if not approval:
            print(f"[Finance] Approval not found: {approval_id}")
            return False
        if approval.approved is not None:
            print(f"[Finance] Already processed: {approval_id[:8]}")
            return False

        approval.approved    = True
        approval.approved_at = datetime.now().isoformat()
        approval.executed_at = datetime.now().isoformat()

        # Record as expense
        self.record_expense(
            amount       = approval.amount,
            description  = approval.description,
            counterparty = approval.destination,
            currency     = approval.currency,
            notes        = f"Approved transfer: {approval.approval_id[:8]}",
        )

        self._save()
        print(f"[Finance] ✅ Transfer approved and recorded: "
              f"{approval.destination} | {approval.currency} {approval.amount:.2f}")
        return True

    def reject_transfer(self, approval_id: str) -> bool:
        """Reject a queued transfer."""
        approval = self._approvals.get(approval_id)
        if not approval:
            return False
        approval.approved    = False
        approval.approved_at = datetime.now().isoformat()
        self._save()
        print(f"[Finance] ❌ Transfer rejected: {approval_id[:8]}")
        return True

    def show_approval_queue(self) -> List[TransferApproval]:
        """Show all pending transfers awaiting approval."""
        pending = [a for a in self._approvals.values() if a.approved is None]
        if not pending:
            print("[Finance] No transfers pending approval.")
        else:
            print(f"\n{'ID':<10} {'Amount':>10} {'Destination':<25} {'Description':<30}")
            print("-" * 80)
            for a in pending:
                print(f"{a.approval_id[:8]:<10} "
                      f"${a.amount:>9.2f} "
                      f"{a.destination:<25} "
                      f"{a.description[:28]:<30}")
        return pending

    # ── Financial position ────────────────────────────────────────────────────

    def get_position(self) -> dict:
        """Current financial position."""
        confirmed = [e for e in self._entries.values()
                     if e.status == EntryStatus.CONFIRMED]
        pending   = [e for e in self._entries.values()
                     if e.status == EntryStatus.PENDING
                     and e.type == EntryType.INCOME]

        income   = sum(e.net_amount for e in confirmed if e.type == EntryType.INCOME)
        expenses = sum(e.amount     for e in confirmed if e.type == EntryType.EXPENSE)
        profit   = income - expenses
        pending_total = sum(e.amount for e in pending)

        return {
            "income":          round(income,        2),
            "expenses":        round(expenses,      2),
            "profit":          round(profit,        2),
            "pending_income":  round(pending_total, 2),
            "total_entries":   len(self._entries),
            "pending_approvals": len([a for a in self._approvals.values()
                                      if a.approved is None]),
        }

    def print_position(self):
        p = self.get_position()
        print("\n── Financial Position ──────────────────────────────")
        print(f"  Income (confirmed):  ${p['income']:>10,.2f}")
        print(f"  Expenses:            ${p['expenses']:>10,.2f}")
        print(f"  Net Profit:          ${p['profit']:>10,.2f}")
        print(f"  Pending income:      ${p['pending_income']:>10,.2f}")
        if p["pending_approvals"]:
            print(f"  Transfers awaiting approval: {p['pending_approvals']}")
        print("────────────────────────────────────────────────────\n")

    def recent_transactions(self, n: int = 10) -> List[LedgerEntry]:
        entries = sorted(self._entries.values(),
                         key=lambda e: e.created_at, reverse=True)
        return entries[:n]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_payment_link(
        self, method: PaymentMethod, amount: float, currency: str, description: str
    ) -> str:
        desc = description.replace(" ", "%20")[:50]
        if method == PaymentMethod.REVOLUT:
            from payment_pipeline import REVOLUT_LINK
            return f"{REVOLUT_LINK}?amount={amount:.2f}&currency={currency}&note={desc}"
        if method == PaymentMethod.PAYPAL:
            from payment_pipeline import PAYPAL_LINK
            return f"{PAYPAL_LINK}/{amount:.2f}{currency}"
        if method == PaymentMethod.FIVERR:
            return f"[Fiverr order] {description} -- ${amount:.2f}"
        return f"Payment of {currency} {amount:.2f} for: {description}"

    def _send_invoice_email(self, entry: LedgerEntry):
        try:
            from janus_comms import get_comms
            comms = get_comms()
            subject = f"Invoice #{entry.entry_id[:8].upper()} — {entry.currency} {entry.amount:.2f}"
            html    = _build_invoice_html(
                entry,
                self.owner_name,
                self.owner_email or comms.config.owner_email,
                entry.payment_link or "",
            )
            ok = comms.email.send(subject, html, to=entry.invoice_email, html=True)
            if ok:
                entry.invoice_sent = True
                self._save()
                print(f"[Finance] Invoice emailed to {entry.invoice_email}")
        except Exception as e:
            print(f"[Finance] Invoice email failed: {e}")

    def _notify_income(self, entry: LedgerEntry):
        try:
            from janus_comms import get_comms
            get_comms().report_revenue(entry.net_amount, entry.counterparty)
        except Exception:
            pass
        # Persist to identity
        try:
            from janus_identity import on_revenue
            on_revenue(entry.net_amount, entry.counterparty)
        except Exception:
            pass

    def _get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        # Support short IDs
        for eid, entry in self._entries.items():
            if eid == entry_id or eid.startswith(entry_id):
                return entry
        return None

    def _save(self):
        try:
            data = {
                "entries":   [e.to_dict() for e in self._entries.values()],
                "approvals": [a.to_dict() for a in self._approvals.values()],
                "saved_at":  datetime.now().isoformat(),
            }
            self.ledger_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Finance] Save failed: {e}")

    def _load(self):
        if not self.ledger_path.exists():
            return
        try:
            data = json.loads(self.ledger_path.read_text())
            for d in data.get("entries", []):
                e = LedgerEntry.from_dict(d)
                self._entries[e.entry_id] = e
            for d in data.get("approvals", []):
                a = TransferApproval.from_dict(d)
                self._approvals[a.approval_id] = a
            print(f"[Finance] Loaded {len(self._entries)} entries, "
                  f"{len(self._approvals)} approvals")
        except Exception as e:
            print(f"[Finance] Load failed: {e}")


# ── Module-level singleton ────────────────────────────────────────────────────

_finance: Optional[JanusFinance] = None

def get_finance() -> JanusFinance:
    global _finance
    if _finance is None:
        _finance = JanusFinance()
    return _finance


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Finance")
    parser.add_argument("--position",  action="store_true", help="Show financial position")
    parser.add_argument("--approvals", action="store_true", help="Show pending approvals")
    parser.add_argument("--approve",   type=str, metavar="ID", help="Approve a transfer")
    parser.add_argument("--reject",    type=str, metavar="ID", help="Reject a transfer")
    parser.add_argument("--scan",      action="store_true", help="Scan inbox for payments")
    parser.add_argument("--recent",    action="store_true", help="Show recent transactions")
    parser.add_argument("--invoice",   nargs=4,
                        metavar=("CLIENT", "AMOUNT", "DESC", "EMAIL"),
                        help="Create and send an invoice")
    args = parser.parse_args()

    fin = JanusFinance()

    if args.position:
        fin.print_position()

    elif args.approvals:
        fin.show_approval_queue()

    elif args.approve:
        fin.approve_transfer(args.approve)

    elif args.reject:
        fin.reject_transfer(args.reject)

    elif args.scan:
        print("Scanning inbox for payments...")
        confirmed = fin.scan_for_payments()
        print(f"Auto-confirmed {len(confirmed)} payment(s)")

    elif args.recent:
        entries = fin.recent_transactions(15)
        print(f"\n{'Date':<12} {'Type':<10} {'Status':<12} {'Amount':>10} {'Counterparty'}")
        print("-" * 65)
        for e in entries:
            sign = "+" if e.type == EntryType.INCOME else "-"
            print(f"{e.created_at[:10]:<12} {e.type.value:<10} {e.status.value:<12} "
                  f"{sign}${e.amount:>9.2f} {e.counterparty}")

    elif args.invoice:
        client, amount_str, desc, email = args.invoice
        inv = fin.invoice_client(
            client_name  = client,
            amount       = float(amount_str),
            description  = desc,
            client_email = email,
        )
        print(f"\nInvoice created: #{inv.entry_id[:8].upper()}")
        print(f"Payment link: {inv.payment_link}")

    else:
        parser.print_help()
