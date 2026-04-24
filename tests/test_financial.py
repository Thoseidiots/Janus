"""
tests/test_financial.py
Property-based tests for financial aggregation correctness.

# Feature: janus-autonomous-worker-completion, Property 12
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janus_wallet import JanusWallet, WalletAnalytics, TransactionType


# ── Composite strategy for transactions ──────────────────────────────────────

@st.composite
def transaction_strategy(draw) -> Dict[str, Any]:
    """Generate arbitrary transaction dicts with amount (Decimal), tx_type, description."""
    amount = draw(
        st.decimals(
            min_value=Decimal("0.01"),
            max_value=Decimal("100000.00"),
            allow_nan=False,
            allow_infinity=False,
            places=2,
        )
    )
    tx_type = draw(st.sampled_from(["income", "expense"]))
    description = draw(st.text(max_size=200))
    return {
        "amount": amount,
        "tx_type": tx_type,
        "description": description,
    }


# ── Property 12: Financial aggregation is arithmetically correct ──────────────
# Feature: janus-autonomous-worker-completion, Property 12
# Validates: Requirements 15.3

@given(st.lists(transaction_strategy()))
@settings(max_examples=100)
def test_financial_aggregation_correctness(transactions: List[Dict[str, Any]]) -> None:
    """
    Property 12: For any list of Transaction objects, the values computed by
    JanusWallet SHALL satisfy:
      total_earned = sum(tx.amount for tx if tx.tx_type == INCOME)
      total_spent  = sum(tx.amount for tx if tx.tx_type == EXPENSE)
      current_balance = total_earned - total_spent

    Validates: Requirements 15.3
    """
    wallet = JanusWallet(db_path=":memory:")

    # Record all transactions into the wallet
    for tx in transactions:
        if tx["tx_type"] == "income":
            wallet.record_income(
                amount=tx["amount"],
                source="test_source",
                description=tx["description"],
            )
        else:
            wallet.record_expense(
                amount=tx["amount"],
                category="misc",
                description=tx["description"],
            )

    # Compute expected values from the raw transaction list
    expected_earned = sum(
        tx["amount"] for tx in transactions if tx["tx_type"] == "income"
    )
    expected_spent = sum(
        tx["amount"] for tx in transactions if tx["tx_type"] == "expense"
    )
    expected_balance = expected_earned - expected_spent

    # Verify via JanusWallet.get_balance()
    actual_balance = wallet.get_balance()
    assert actual_balance == expected_balance, (
        f"Balance mismatch: expected {expected_balance}, got {actual_balance}"
    )

    # Verify via WalletAnalytics.build_report()
    analytics = WalletAnalytics()
    all_txs = wallet._ledger.list_transactions()
    now = datetime.utcnow()
    report = analytics.build_report(all_txs, period_start=now, period_end=now)

    assert report.total_income == expected_earned, (
        f"total_income mismatch: expected {expected_earned}, got {report.total_income}"
    )
    assert report.total_expenses == expected_spent, (
        f"total_expenses mismatch: expected {expected_spent}, got {report.total_expenses}"
    )
    assert report.net == expected_balance, (
        f"net mismatch: expected {expected_balance}, got {report.net}"
    )
