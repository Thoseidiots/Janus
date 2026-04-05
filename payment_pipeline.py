# """
payment_pipeline.py

Janus payment pipeline -- handles invoicing, tracking, and payment
confirmation across Revolut and PayPal.

Janus uses this to:

- Generate payment requests for clients
- Track outstanding and completed payments
- Confirm receipt and update revenue ledger
- Produce payment links to share with clients

Usage:
from payment_pipeline import PaymentPipeline
pipeline = PaymentPipeline()

```
# Create a payment request for a commission
request = pipeline.create_request(
    amount=25.00,
    currency="USD",
    description="Character concept art -- 2 characters",
    client_name="John"
)
print(request.payment_link)   # share this with the client
print(request.summary())      # full details
```

"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────

# Config -- your payment details

# ─────────────────────────────────────────────────────────────────────────────

REVOLUT_TAG     = "@i_sears"
REVOLUT_LINK    = "https://revolut.me/i_sears"
PAYPAL_NAME     = "Ishmael Sears"
PAYPAL_LINK     = "https://www.paypal.com/paypalme/ishmael_sears"  # update if different
LEDGER_FILE     = "payment_ledger.json"

# Fiverr commission rates (platform takes 20%)

FIVERR_FEE_RATE = 0.20

# ─────────────────────────────────────────────────────────────────────────────

# Enums

# ─────────────────────────────────────────────────────────────────────────────

class PaymentMethod(str, Enum):
REVOLUT = "revolut"
PAYPAL  = "paypal"
FIVERR  = "fiverr"   # handled by platform, tracked only
CASH    = "cash"

class PaymentStatus(str, Enum):
PENDING   = "pending"
SENT      = "sent"       # link sent to client
CONFIRMED = "confirmed"  # manually marked as received
CANCELLED = "cancelled"

class ServiceType(str, Enum):
COMMISSION      = "commission"
ASSET_PACK      = "asset_pack"
CONCEPT_ART     = "concept_art"
CHARACTER_SHEET = "character_sheet"
GAME_ASSET      = "game_asset"
OTHER           = "other"

# ─────────────────────────────────────────────────────────────────────────────

# Data models

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaymentRequest:
id:           str
amount:       float
currency:     str
description:  str
client_name:  str
method:       PaymentMethod
service_type: ServiceType
status:       PaymentStatus
created_at:   float
confirmed_at: Optional[float]
notes:        str
payment_link: str

```
def net_amount(self) -> float:
    """Amount after platform fees."""
    if self.method == PaymentMethod.FIVERR:
        return round(self.amount * (1 - FIVERR_FEE_RATE), 2)
    return self.amount

def summary(self) -> str:
    status_icon = {
        PaymentStatus.PENDING:   "⏳",
        PaymentStatus.SENT:      "📤",
        PaymentStatus.CONFIRMED: "✅",
        PaymentStatus.CANCELLED: "❌",
    }.get(self.status, "?")

    lines = [
        f"── Payment Request {self.id[:8]} ──",
        f"  Client:      {self.client_name}",
        f"  Service:     {self.service_type.value}",
        f"  Description: {self.description}",
        f"  Amount:      {self.currency} {self.amount:.2f}",
    ]
    if self.method == PaymentMethod.FIVERR:
        lines.append(f"  Net (after Fiverr 20%): {self.currency} {self.net_amount():.2f}")
    lines += [
        f"  Method:      {self.method.value}",
        f"  Status:      {status_icon} {self.status.value}",
        f"  Link:        {self.payment_link}",
    ]
    if self.notes:
        lines.append(f"  Notes:       {self.notes}")
    return "\n".join(lines)

def to_dict(self) -> dict:
    d = asdict(self)
    d["method"]       = self.method.value
    d["status"]       = self.status.value
    d["service_type"] = self.service_type.value
    return d

@staticmethod
def from_dict(d: dict) -> "PaymentRequest":
    d["method"]       = PaymentMethod(d["method"])
    d["status"]       = PaymentStatus(d["status"])
    d["service_type"] = ServiceType(d.get("service_type", "other"))
    return PaymentRequest(**d)
```

@dataclass
class RevenueSummary:
total_earned:     float = 0.0
total_pending:    float = 0.0
total_cancelled:  float = 0.0
request_count:    int   = 0
confirmed_count:  int   = 0
currency:         str   = "USD"
by_method:        Dict[str, float] = field(default_factory=dict)
by_service:       Dict[str, float] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────

# Payment link generators

# ─────────────────────────────────────────────────────────────────────────────

def _revolut_link(amount: float, currency: str, description: str) -> str:
"""Revolut payment link -- pre-fills amount and note."""
desc_encoded = description.replace(" ", "%20")[:50]
return (f"{REVOLUT_LINK}?amount={amount:.2f}"
f"&currency={currency}&note={desc_encoded}")

def _paypal_link(amount: float, currency: str, description: str) -> str:
"""PayPal.me link with amount."""
return f"{PAYPAL_LINK}/{amount:.2f}{currency}"

def _fiverr_note(amount: float, description: str) -> str:
"""Fiverr is handled by the platform -- just a tracking note."""
return f"[Fiverr order] {description} -- ${amount:.2f} gross"

# ─────────────────────────────────────────────────────────────────────────────

# Main pipeline

# ─────────────────────────────────────────────────────────────────────────────

class PaymentPipeline:
"""
Janus’s payment management system.

```
Tracks all payment requests, generates shareable links,
and maintains a persistent revenue ledger on disk.
"""

def __init__(self, ledger_path: str = LEDGER_FILE):
    self._ledger_path = Path(ledger_path)
    self._requests: Dict[str, PaymentRequest] = {}
    self._load_ledger()

# ── Create ────────────────────────────────────────────────────────────────

def create_request(
    self,
    amount:       float,
    description:  str,
    client_name:  str,
    method:       PaymentMethod  = PaymentMethod.REVOLUT,
    service_type: ServiceType    = ServiceType.COMMISSION,
    currency:     str            = "USD",
    notes:        str            = "",
) -> PaymentRequest:
    """Create a new payment request and generate a shareable link."""

    request_id = str(uuid.uuid4())

    # Generate the right link for each method
    if method == PaymentMethod.REVOLUT:
        link = _revolut_link(amount, currency, description)
    elif method == PaymentMethod.PAYPAL:
        link = _paypal_link(amount, currency, description)
    elif method == PaymentMethod.FIVERR:
        link = _fiverr_note(amount, description)
    else:
        link = f"Cash payment -- {currency} {amount:.2f}"

    req = PaymentRequest(
        id           = request_id,
        amount       = amount,
        currency     = currency,
        description  = description,
        client_name  = client_name,
        method       = method,
        service_type = service_type,
        status       = PaymentStatus.PENDING,
        created_at   = time.time(),
        confirmed_at = None,
        notes        = notes,
        payment_link = link,
    )

    self._requests[request_id] = req
    self._save_ledger()
    print(f"[PaymentPipeline] Created request {request_id[:8]} "
          f"-- {currency} {amount:.2f} via {method.value}")
    return req

# ── Status updates ────────────────────────────────────────────────────────

def mark_sent(self, request_id: str) -> bool:
    """Mark that the payment link was sent to the client."""
    req = self._get(request_id)
    if not req:
        return False
    req.status = PaymentStatus.SENT
    self._save_ledger()
    print(f"[PaymentPipeline] {request_id[:8]} marked as SENT")
    return True

def confirm_payment(self, request_id: str,
                    notes: str = "") -> bool:
    """Mark a payment as received/confirmed."""
    req = self._get(request_id)
    if not req:
        return False
    req.status       = PaymentStatus.CONFIRMED
    req.confirmed_at = time.time()
    if notes:
        req.notes = notes
    self._save_ledger()
    print(f"[PaymentPipeline] ✅ {request_id[:8]} CONFIRMED -- "
          f"{req.currency} {req.amount:.2f} from {req.client_name}")
    return True

def cancel_request(self, request_id: str) -> bool:
    req = self._get(request_id)
    if not req:
        return False
    req.status = PaymentStatus.CANCELLED
    self._save_ledger()
    return True

# ── Queries ───────────────────────────────────────────────────────────────

def get_pending(self) -> List[PaymentRequest]:
    return [r for r in self._requests.values()
            if r.status in (PaymentStatus.PENDING, PaymentStatus.SENT)]

def get_confirmed(self) -> List[PaymentRequest]:
    return [r for r in self._requests.values()
            if r.status == PaymentStatus.CONFIRMED]

def get_all(self) -> List[PaymentRequest]:
    return list(self._requests.values())

def get_by_id(self, request_id: str) -> Optional[PaymentRequest]:
    # Support short IDs (first 8 chars)
    for rid, req in self._requests.items():
        if rid == request_id or rid.startswith(request_id):
            return req
    return None

# ── Revenue summary ───────────────────────────────────────────────────────

def revenue_summary(self) -> RevenueSummary:
    summary = RevenueSummary()
    summary.request_count = len(self._requests)

    for req in self._requests.values():
        net = req.net_amount()
        if req.status == PaymentStatus.CONFIRMED:
            summary.total_earned   += net
            summary.confirmed_count += 1
            # By method
            m = req.method.value
            summary.by_method[m] = summary.by_method.get(m, 0.0) + net
            # By service
            s = req.service_type.value
            summary.by_service[s] = summary.by_service.get(s, 0.0) + net
        elif req.status in (PaymentStatus.PENDING, PaymentStatus.SENT):
            summary.total_pending += req.amount
        elif req.status == PaymentStatus.CANCELLED:
            summary.total_cancelled += req.amount

    summary.total_earned   = round(summary.total_earned,   2)
    summary.total_pending  = round(summary.total_pending,  2)
    return summary

def print_summary(self) -> None:
    s = self.revenue_summary()
    print("\n── Revenue Summary ─────────────────────────────────")
    print(f"  Total earned:    ${s.total_earned:.2f}")
    print(f"  Pending:         ${s.total_pending:.2f}")
    print(f"  Total requests:  {s.request_count}")
    print(f"  Confirmed:       {s.confirmed_count}")
    if s.by_method:
        print("  By method:")
        for method, amt in s.by_method.items():
            print(f"    {method:12s}  ${amt:.2f}")
    if s.by_service:
        print("  By service:")
        for svc, amt in s.by_service.items():
            print(f"    {svc:16s}  ${amt:.2f}")
    print("────────────────────────────────────────────────────\n")

# ── Message templates (for Janus to send to clients) ──────────────────────

def generate_payment_message(self, request_id: str) -> str:
    """Generate a ready-to-send message Janus can deliver to a client."""
    req = self.get_by_id(request_id)
    if not req:
        return "Payment request not found."

    if req.method == PaymentMethod.REVOLUT:
        return (
            f"Hi {req.client_name}! Here's the payment link for your order:\n\n"
            f"💳 Revolut: {req.payment_link}\n\n"
            f"Amount: {req.currency} {req.amount:.2f}\n"
            f"For: {req.description}\n\n"
            f"You don't need a Revolut account -- the link works for anyone. "
            f"Please let me know once sent and I'll get started right away!"
        )
    elif req.method == PaymentMethod.PAYPAL:
        return (
            f"Hi {req.client_name}! Here's the payment link for your order:\n\n"
            f"💙 PayPal: {req.payment_link}\n\n"
            f"Amount: {req.currency} {req.amount:.2f}\n"
            f"For: {req.description}\n\n"
            f"Please use 'Friends & Family' to avoid extra fees. "
            f"Let me know once sent and I'll begin immediately!"
        )
    elif req.method == PaymentMethod.FIVERR:
        return (
            f"Hi {req.client_name}! Please complete payment through the "
            f"Fiverr order page. Once confirmed I'll start your order.\n\n"
            f"Order: {req.description}\n"
            f"Amount: {req.currency} {req.amount:.2f}"
        )
    return f"Payment of {req.currency} {req.amount:.2f} for: {req.description}"

# ── Persistence ───────────────────────────────────────────────────────────

def _save_ledger(self) -> None:
    try:
        data = [r.to_dict() for r in self._requests.values()]
        with open(self._ledger_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[PaymentPipeline] Ledger save failed: {e}")

def _load_ledger(self) -> None:
    if not self._ledger_path.exists():
        return
    try:
        with open(self._ledger_path) as f:
            data = json.load(f)
        for d in data:
            req = PaymentRequest.from_dict(d)
            self._requests[req.id] = req
        print(f"[PaymentPipeline] Loaded {len(self._requests)} "
              f"records from ledger")
    except Exception as e:
        print(f"[PaymentPipeline] Ledger load failed: {e}")

def _get(self, request_id: str) -> Optional[PaymentRequest]:
    req = self.get_by_id(request_id)
    if not req:
        print(f"[PaymentPipeline] Request not found: {request_id}")
    return req

def __repr__(self) -> str:
    s = self.revenue_summary()
    return (f"PaymentPipeline(requests={s.request_count}, "
            f"earned=${s.total_earned:.2f}, "
            f"pending=${s.total_pending:.2f})")
```

# ─────────────────────────────────────────────────────────────────────────────

# Quick test

# ─────────────────────────────────────────────────────────────────────────────

if **name** == "**main**":
pipeline = PaymentPipeline(ledger_path="test_ledger.json")

```
# Simulate a Fiverr commission
r1 = pipeline.create_request(
    amount=30.00,
    description="Character concept art -- full body",
    client_name="Alex",
    method=PaymentMethod.FIVERR,
    service_type=ServiceType.CHARACTER_SHEET,
)
print(r1.summary())
print(f"\nNet after Fiverr fee: ${r1.net_amount():.2f}")
pipeline.confirm_payment(r1.id)

# Simulate a direct Revolut commission
r2 = pipeline.create_request(
    amount=25.00,
    description="2 character portraits",
    client_name="Maya",
    method=PaymentMethod.REVOLUT,
    service_type=ServiceType.COMMISSION,
)
print("\n" + r2.summary())
print("\n── Client message ──")
print(pipeline.generate_payment_message(r2.id))
pipeline.mark_sent(r2.id)

# Simulate PayPal asset pack sale
r3 = pipeline.create_request(
    amount=15.00,
    description="Dungeon asset pack -- 15 assets",
    client_name="Tom",
    method=PaymentMethod.PAYPAL,
    service_type=ServiceType.ASSET_PACK,
)
pipeline.confirm_payment(r3.id)

pipeline.print_summary()

# Clean up test file
import os
if os.path.exists("test_ledger.json"):
    os.remove("test_ledger.json")
```