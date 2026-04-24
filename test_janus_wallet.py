"""
test_janus_wallet.py
====================
Unit tests and smoke tests for janus_wallet.py

Tests concrete example-based scenarios (not property tests).
"""

import tempfile
from decimal import Decimal
from datetime import datetime

import pytest

from janus_wallet import (
    JanusWallet,
    Category,
    DuplicateTransactionError,
    ApprovalRequiredError,
    ReportPeriod,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8.1: Unit tests for core wallet operations
# ═══════════════════════════════════════════════════════════════════════════════


def test_record_income_updates_balance():
    """
    Test that recording income updates the balance by exactly the income amount.
    Requirements: REQ-1.1, REQ-1.3
    """
    wallet = JanusWallet(db_path=":memory:")
    
    # Initial balance should be zero
    initial_balance = wallet.get_balance("USD")
    assert initial_balance == Decimal("0")
    
    # Record income
    income_amount = Decimal("100.00")
    wallet.record_income(
        amount=income_amount,
        source="test_source",
        description="Test income",
    )
    
    # Check balance increased by exactly the income amount
    new_balance = wallet.get_balance("USD")
    assert new_balance == initial_balance + income_amount
    assert new_balance == Decimal("100.00")


def test_record_expense_updates_balance():
    """
    Test that recording expense updates the balance by exactly the expense amount (negative).
    Requirements: REQ-1.1, REQ-1.3
    """
    wallet = JanusWallet(db_path=":memory:")
    
    # Start with some income
    wallet.record_income(
        amount=Decimal("200.00"),
        source="test_source",
        description="Initial income",
    )
    
    initial_balance = wallet.get_balance("USD")
    assert initial_balance == Decimal("200.00")
    
    # Record expense
    expense_amount = Decimal("75.50")
    wallet.record_expense(
        amount=expense_amount,
        category=Category.COMPUTE_RESOURCES,
        description="Test expense",
    )
    
    # Check balance decreased by exactly the expense amount
    new_balance = wallet.get_balance("USD")
    assert new_balance == initial_balance - expense_amount
    assert new_balance == Decimal("124.50")


def test_duplicate_external_id_rejected():
    """
    Test that inserting the same external_tx_id twice raises DuplicateTransactionError.
    Requirements: REQ-6.1
    """
    wallet = JanusWallet(db_path=":memory:")
    
    external_id = "test_external_tx_12345"
    
    # First insert should succeed
    wallet.record_income(
        amount=Decimal("50.00"),
        source="test_source",
        description="First transaction",
        external_tx_id=external_id,
    )
    
    # Second insert with same external_tx_id should raise DuplicateTransactionError
    with pytest.raises(DuplicateTransactionError):
        wallet.record_income(
            amount=Decimal("50.00"),
            source="test_source",
            description="Duplicate transaction",
            external_tx_id=external_id,
        )


def test_send_payment_above_threshold_raises():
    """
    Test that send_payment raises ApprovalRequiredError when amount exceeds threshold.
    Requirements: REQ-6.2
    """
    wallet = JanusWallet(db_path=":memory:")
    
    # Test with require_approval_above=None (default threshold is 500)
    # Any large payment should trigger approval
    with pytest.raises(ApprovalRequiredError):
        wallet.send_payment(
            recipient_email="test@example.com",
            amount=Decimal("600.00"),
            note="Large payment",
            require_approval_above=None,
        )
    
    # Test with a small threshold
    with pytest.raises(ApprovalRequiredError):
        wallet.send_payment(
            recipient_email="test@example.com",
            amount=Decimal("100.00"),
            note="Payment above small threshold",
            require_approval_above=Decimal("50.00"),
        )


def test_report_empty_period():
    """
    Test that a report over a period with no transactions returns zeros for all numeric fields.
    Requirements: REQ-5.1
    """
    wallet = JanusWallet(db_path=":memory:")
    
    # Get report for daily period (no transactions)
    report = wallet.get_report(ReportPeriod.DAILY)
    
    # All numeric fields should be zero
    assert report.total_income == Decimal("0")
    assert report.total_expenses == Decimal("0")
    assert report.net == Decimal("0")
    assert report.transaction_count == 0
    assert report.savings_rate == 0.0
    assert len(report.income_by_source) == 0
    assert len(report.expenses_by_category) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8.2: Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_wallet_db_created_on_init():
    """
    Test that instantiating JanusWallet with a temp file path creates the db file.
    Requirements: REQ-7.1
    """
    import os
    
    # Create a temp file path
    temp_path = tempfile.mktemp(suffix=".db")
    
    try:
        # File should not exist yet
        assert not os.path.exists(temp_path)
        
        # Instantiate wallet
        wallet = JanusWallet(db_path=temp_path)
        
        # File should now exist
        assert os.path.exists(temp_path)
        
        # Close the connection to allow cleanup
        wallet._ledger._conn.close()
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_import_from_worker_context():
    """
    Test that importing JanusWallet and Category from janus_wallet succeeds.
    Requirements: REQ-7.1
    """
    # This test verifies the imports work
    from janus_wallet import JanusWallet, Category
    
    # Assert they are not None
    assert JanusWallet is not None
    assert Category is not None
    
    # Verify Category has expected attributes
    assert hasattr(Category, "FREELANCE_EARNINGS")
    assert hasattr(Category, "COMPUTE_RESOURCES")
    assert hasattr(Category, "TRAINING_DATA")
