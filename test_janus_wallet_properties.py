"""
Property-based tests for janus_wallet.py using Hypothesis.

Feature: janus-autonomous-wallet
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janus_wallet import JanusWallet, Transaction, TransactionType, WalletLedger


# ── Strategies ────────────────────────────────────────────────────────────────

def positive_decimal_strategy():
    """Generate positive Decimal amounts in the range [0.01, 10000.00]."""
    return st.decimals(
        min_value=Decimal("0.01"),
        max_value=Decimal("10000.00"),
        places=2,
        allow_nan=False,
        allow_infinity=False,
    )


def transaction_strategy():
    """Generate arbitrary INCOME or EXPENSE Transaction objects."""
    return st.builds(
        Transaction,
        id=st.just(""),
        tx_type=st.sampled_from([TransactionType.INCOME, TransactionType.EXPENSE]),
        amount=positive_decimal_strategy(),
        currency=st.just("USD"),
        source_or_category=st.just("test"),
        description=st.just(""),
        external_tx_id=st.just(""),
        timestamp=st.just(datetime.utcnow()),
        metadata=st.just({}),
        verified=st.just(False),
    )


# ── Property 1 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 1: balance equals sum of signed transactions
@given(transactions=st.lists(transaction_strategy(), min_size=0, max_size=50))
@settings(max_examples=200)
def test_balance_equals_sum_of_transactions(transactions):
    """
    **Validates: Requirements REQ-1.1, REQ-1.3, REQ-2.2**

    For any sequence of income and expense transactions, get_balance("USD")
    must equal sum(income amounts) - sum(expense amounts).
    """
    ledger = WalletLedger(db_path=":memory:")

    expected_income = Decimal("0")
    expected_expenses = Decimal("0")

    for tx in transactions:
        ledger.insert_transaction(tx)
        if tx.tx_type == TransactionType.INCOME:
            expected_income += tx.amount
        elif tx.tx_type == TransactionType.EXPENSE:
            expected_expenses += tx.amount

    expected_balance = expected_income - expected_expenses
    actual_balance = ledger.get_balance("USD")

    assert actual_balance == expected_balance, (
        f"Balance mismatch: expected {expected_balance}, got {actual_balance}. "
        f"income={expected_income}, expenses={expected_expenses}"
    )


# ── Property 2 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 2: transaction round-trip preserves all fields
@given(tx=transaction_strategy())
@settings(max_examples=200)
def test_transaction_round_trip(tx):
    """
    **Validates: Requirements REQ-1.2, REQ-1.3**

    For any Transaction inserted into the ledger, reading it back by ID
    must produce an object with all fields equal to the original.
    """
    ledger = WalletLedger(db_path=":memory:")

    saved = ledger.insert_transaction(tx)

    # Read back by filtering on the assigned ID
    all_txs = ledger.list_transactions(limit=1000)
    matches = [t for t in all_txs if t.id == saved.id]

    assert len(matches) == 1, f"Expected exactly 1 transaction with id={saved.id!r}, got {len(matches)}"

    retrieved = matches[0]

    assert retrieved.id == saved.id
    assert retrieved.tx_type == saved.tx_type
    assert retrieved.amount == saved.amount
    assert retrieved.currency == saved.currency
    assert retrieved.source_or_category == saved.source_or_category
    assert retrieved.description == saved.description
    assert retrieved.external_tx_id == saved.external_tx_id
    assert retrieved.timestamp == saved.timestamp
    assert retrieved.metadata == saved.metadata
    assert retrieved.verified == saved.verified

# ── Property 6 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 6: financial report totals are consistent with ledger
@given(transactions=st.lists(transaction_strategy(), min_size=0, max_size=100))
@settings(max_examples=200)
def test_report_totals_consistent(transactions):
    """
    **Validates: Requirements REQ-5.1**

    For any list of transactions, the FinancialReport produced by
    WalletAnalytics.build_report must satisfy:
      1. report.net == report.total_income - report.total_expenses
      2. sum(report.income_by_source.values()) == report.total_income
    """
    from janus_wallet import WalletAnalytics

    analytics = WalletAnalytics()
    period_start = datetime(2024, 1, 1)
    period_end = datetime(2024, 12, 31)

    report = analytics.build_report(transactions, period_start, period_end)

    assert report.net == report.total_income - report.total_expenses, (
        f"net mismatch: {report.net} != {report.total_income} - {report.total_expenses}"
    )

    income_sum = sum(report.income_by_source.values(), Decimal("0"))
    assert income_sum == report.total_income, (
        f"income_by_source sum {income_sum} != total_income {report.total_income}"
    )


# ── Property 7 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 7: budget allocation sums to total
@given(total=positive_decimal_strategy())
@settings(max_examples=200)
def test_budget_allocation_sums_to_total(total):
    """
    **Validates: Requirements REQ-5.2**

    For any positive Decimal total, the sum of all BudgetAllocation fields
    must equal the total exactly.
    """
    from janus_wallet import WalletAnalytics

    allocation = WalletAnalytics().allocate_budget(total)

    allocation_sum = (
        allocation.compute
        + allocation.api_costs
        + allocation.training
        + allocation.savings
        + allocation.emergency_reserve
    )

    assert allocation_sum == total, (
        f"Budget allocation sum {allocation_sum} != total {total}. "
        f"compute={allocation.compute}, api_costs={allocation.api_costs}, "
        f"training={allocation.training}, savings={allocation.savings}, "
        f"emergency_reserve={allocation.emergency_reserve}"
    )


# ── Property 3 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 3: recording income never decreases balance
@given(
    initial=st.lists(transaction_strategy(), min_size=0, max_size=50),
    amount=positive_decimal_strategy(),
)
@settings(max_examples=200)
def test_income_increases_balance(initial, amount):
    """
    **Validates: Requirements REQ-2.2, REQ-1.1**

    For any wallet state (arbitrary seed transactions) and any positive income
    amount, calling record_income must result in a balance strictly greater
    than the balance before the call.
    """
    wallet = JanusWallet(db_path=":memory:")

    # Seed the ledger with arbitrary transactions
    for tx in initial:
        wallet._ledger.insert_transaction(tx)

    balance_before = wallet.get_balance("USD")

    wallet.record_income(amount=amount, source="test", currency="USD")

    balance_after = wallet.get_balance("USD")

    assert balance_after > balance_before, (
        f"Balance did not increase after recording income of {amount}. "
        f"Before: {balance_before}, After: {balance_after}"
    )


# ── Property 4 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 4: recording expense never increases balance
@given(
    initial=st.lists(transaction_strategy(), min_size=0, max_size=50),
    amount=positive_decimal_strategy(),
)
@settings(max_examples=200)
def test_expense_decreases_balance(initial, amount):
    """
    **Validates: Requirements REQ-3.2, REQ-1.1**

    For any wallet state (arbitrary seed transactions) and any positive expense
    amount, calling record_expense must result in a balance less than or equal
    to the balance before the call.
    """
    from janus_wallet import ExpenseCategory

    wallet = JanusWallet(db_path=":memory:")

    # Seed the ledger with arbitrary transactions
    for tx in initial:
        wallet._ledger.insert_transaction(tx)

    balance_before = wallet.get_balance("USD")

    wallet.record_expense(amount=amount, category=ExpenseCategory.COMPUTE, currency="USD")

    balance_after = wallet.get_balance("USD")

    assert balance_after <= balance_before, (
        f"Balance increased after recording expense of {amount}. "
        f"Before: {balance_before}, After: {balance_after}"
    )


# ── Property 5 ────────────────────────────────────────────────────────────────

# Feature: janus-autonomous-wallet, Property 5: duplicate external IDs are rejected
@given(
    external_tx_id=st.text(min_size=1, max_size=50),
    amount=positive_decimal_strategy(),
)
@settings(max_examples=100)
def test_duplicate_external_id_rejected(external_tx_id, amount):
    """
    **Validates: Requirements REQ-6.1**

    For any non-empty external_tx_id, inserting a second transaction with the
    same external_tx_id must raise DuplicateTransactionError and leave the
    balance unchanged.
    """
    from janus_wallet import DuplicateTransactionError

    ledger = WalletLedger(db_path=":memory:")

    first_tx = Transaction(
        id="",
        tx_type=TransactionType.INCOME,
        amount=amount,
        currency="USD",
        source_or_category="test",
        description="",
        external_tx_id=external_tx_id,
        timestamp=datetime.utcnow(),
        metadata={},
        verified=False,
    )

    # First insert must succeed
    ledger.insert_transaction(first_tx)

    balance_after_first = ledger.get_balance("USD")

    # Second insert with same external_tx_id must raise DuplicateTransactionError
    second_tx = Transaction(
        id="",
        tx_type=TransactionType.INCOME,
        amount=amount,
        currency="USD",
        source_or_category="test",
        description="duplicate",
        external_tx_id=external_tx_id,
        timestamp=datetime.utcnow(),
        metadata={},
        verified=False,
    )

    with pytest.raises(DuplicateTransactionError):
        ledger.insert_transaction(second_tx)

    balance_after_second = ledger.get_balance("USD")

    assert balance_after_second == balance_after_first, (
        f"Balance changed after rejected duplicate insert. "
        f"Before: {balance_after_first}, After: {balance_after_second}"
    )
