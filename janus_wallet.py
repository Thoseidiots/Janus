"""
janus_wallet.py
===============
Janus Autonomous Wallet — financial backbone for the Janus worker system.

Provides real, persistent money tracking via SQLite, PayPal integration,
and financial-intelligence helpers so janus_autonomous_worker.py can
record every dollar earned and spent.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionType(str, Enum):
    """Type of wallet transaction."""
    INCOME   = "income"
    EXPENSE  = "expense"
    TRANSFER = "transfer"   # internal moves between currency buckets


class ExpenseCategory(str, Enum):
    """Categories for expense transactions."""
    COMPUTE      = "compute"         # GPU, cloud
    API_COSTS    = "api_costs"       # OpenAI, Anthropic, etc.
    TRAINING     = "training"        # datasets, fine-tuning runs
    INFRA        = "infrastructure"
    TOOLS        = "tools"
    MISC         = "misc"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """Represents a single financial transaction in the ledger."""
    id: str                        # UUID, generated on insert
    tx_type: TransactionType       # INCOME | EXPENSE | TRANSFER
    amount: Decimal                # Always positive
    currency: str                  # "USD", "BTC", "ETH", "USDC"
    source_or_category: str        # income source OR ExpenseCategory value
    description: str
    external_tx_id: str            # PayPal transaction ID, if any
    timestamp: datetime            # UTC, set on insert
    metadata: dict                 # arbitrary JSON blob
    verified: bool                 # True once confirmed by payment processor


@dataclass
class FinancialReport:
    """Summary of financial activity over a time period."""
    period_start: datetime
    period_end: datetime
    total_income: Decimal
    total_expenses: Decimal
    net: Decimal                          # income - expenses
    income_by_source: Dict[str, Decimal]
    expenses_by_category: Dict[str, Decimal]
    transaction_count: int
    savings_rate: float                   # net / income, 0–1


@dataclass
class BudgetAllocation:
    """Recommended budget split for a given total amount."""
    compute: Decimal
    api_costs: Decimal
    training: Decimal
    savings: Decimal
    emergency_reserve: Decimal


@dataclass
class BudgetStatus:
    """Current budget health snapshot."""
    allocation: BudgetAllocation
    current_balance: Decimal
    report: FinancialReport
    should_reinvest: bool


@dataclass
class PaymentResult:
    """Result of an outbound payment attempt."""
    success: bool
    batch_id: Optional[str]           # PayPal payout batch ID
    transaction: Optional[Transaction]
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class WalletError(Exception):
    """Base exception for all wallet errors."""


class DuplicateTransactionError(WalletError):
    """Raised when a transaction with the same external_tx_id already exists."""


class LedgerWriteError(WalletError):
    """Raised when a SQLite write operation fails."""


class ApprovalRequiredError(WalletError):
    """Raised when a payment exceeds the approval threshold."""


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PERIOD HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class ReportPeriod(str, Enum):
    """Predefined reporting periods."""
    DAILY   = "daily"
    WEEKLY  = "weekly"
    MONTHLY = "monthly"
    ALL     = "all"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY ALIAS
# ═══════════════════════════════════════════════════════════════════════════════

class _CategoryNamespace:
    """
    Namespace that maps the symbolic category names used by
    janus_autonomous_worker.py to their canonical values.

    Usage (already present in the worker):
        from janus_wallet import Category
        category=Category.FREELANCE_EARNINGS
        category=Category.COMPUTE_RESOURCES
        category=Category.TRAINING_DATA
    """
    # Income source — stored as a plain string in source_or_category
    FREELANCE_EARNINGS: str = "freelance_earnings"

    # Expense categories — stored as ExpenseCategory enum values
    COMPUTE_RESOURCES: ExpenseCategory = ExpenseCategory.COMPUTE
    TRAINING_DATA: ExpenseCategory = ExpenseCategory.TRAINING


Category = _CategoryNamespace()


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET LEDGER (SQLite persistence)
# ═══════════════════════════════════════════════════════════════════════════════

import sqlite3


class WalletLedger:
    """
    Owns the SQLite connection to janus_wallet.db.
    All public methods are thread-safe via self._lock.
    Amounts are stored as TEXT (string repr of Decimal) to preserve precision.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS transactions (
            id                  TEXT PRIMARY KEY,
            tx_type             TEXT NOT NULL,
            amount              TEXT NOT NULL,
            currency            TEXT NOT NULL DEFAULT 'USD',
            source_or_category  TEXT NOT NULL,
            description         TEXT NOT NULL DEFAULT '',
            external_tx_id      TEXT NOT NULL DEFAULT '',
            timestamp           TEXT NOT NULL,
            metadata            TEXT NOT NULL DEFAULT '{}',
            verified            INTEGER NOT NULL DEFAULT 0
        );
    """

    _CREATE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_tx_type     ON transactions(tx_type);",
        "CREATE INDEX IF NOT EXISTS idx_currency    ON transactions(currency);",
        "CREATE INDEX IF NOT EXISTS idx_timestamp   ON transactions(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_external_id ON transactions(external_tx_id);",
    ]

    # Partial unique index: only enforce uniqueness when external_tx_id is non-empty.
    _CREATE_UNIQUE_INDEX = (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_external_tx_id "
        "ON transactions(external_tx_id) "
        "WHERE external_tx_id != '';"
    )

    def __init__(self, db_path: str = "janus_wallet.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                self._CREATE_TABLE
                + "\n".join(self._CREATE_INDEXES)
                + "\n"
                + self._CREATE_UNIQUE_INDEX
            )
            self._conn.commit()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_transaction(row: sqlite3.Row) -> Transaction:
        return Transaction(
            id=row["id"],
            tx_type=TransactionType(row["tx_type"]),
            amount=Decimal(row["amount"]),
            currency=row["currency"],
            source_or_category=row["source_or_category"],
            description=row["description"],
            external_tx_id=row["external_tx_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]),
            verified=bool(row["verified"]),
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def insert_transaction(self, tx: Transaction) -> Transaction:
        """
        Persist *tx* to the database.

        - Assigns a new UUID if tx.id is empty.
        - Sets tx.timestamp to datetime.utcnow() if not already set.
        - Raises DuplicateTransactionError on UNIQUE constraint violation
          (only when external_tx_id is non-empty).
        - Raises LedgerWriteError on any other SQLite error.
        - Returns the saved Transaction with id and timestamp filled in.
        """
        if not tx.id:
            tx.id = str(uuid.uuid4())
        if tx.timestamp is None:
            tx.timestamp = datetime.utcnow()

        sql = """
            INSERT INTO transactions
                (id, tx_type, amount, currency, source_or_category,
                 description, external_tx_id, timestamp, metadata, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            tx.id,
            tx.tx_type.value if isinstance(tx.tx_type, TransactionType) else tx.tx_type,
            str(tx.amount),
            tx.currency,
            tx.source_or_category.value
            if isinstance(tx.source_or_category, ExpenseCategory)
            else tx.source_or_category,
            tx.description,
            tx.external_tx_id,
            tx.timestamp.isoformat(),
            json.dumps(tx.metadata),
            int(tx.verified),
        )

        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                if tx.external_tx_id:
                    raise DuplicateTransactionError(
                        f"Transaction with external_tx_id={tx.external_tx_id!r} already exists."
                    ) from exc
                raise LedgerWriteError(f"Integrity error on insert: {exc}") from exc
            except sqlite3.Error as exc:
                raise LedgerWriteError(f"SQLite error on insert: {exc}") from exc

        return tx

    # ── read ──────────────────────────────────────────────────────────────────

    def get_balance(self, currency: str = "USD") -> Decimal:
        """
        Return the net balance for *currency*:
            SUM(amount WHERE tx_type='income') - SUM(amount WHERE tx_type='expense')

        Returns Decimal("0") if no rows exist for the currency.
        """
        sql = """
            SELECT amount FROM transactions
            WHERE currency = ? AND tx_type = ?
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, (currency, TransactionType.INCOME.value))
            income = sum((Decimal(r[0]) for r in cur.fetchall()), Decimal("0"))

            cur.execute(sql, (currency, TransactionType.EXPENSE.value))
            expenses = sum((Decimal(r[0]) for r in cur.fetchall()), Decimal("0"))

        return income - expenses

    def list_transactions(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        tx_type: Optional[TransactionType] = None,
        category: Optional[str] = None,
        limit: int = 500,
    ) -> List[Transaction]:
        """
        Return up to *limit* transactions matching the optional filters,
        ordered by timestamp ascending.
        """
        clauses: List[str] = []
        params: List = []

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())
        if tx_type is not None:
            clauses.append("tx_type = ?")
            params.append(
                tx_type.value if isinstance(tx_type, TransactionType) else tx_type
            )
        if category is not None:
            clauses.append("source_or_category = ?")
            params.append(
                category.value if isinstance(category, ExpenseCategory) else category
            )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT * FROM transactions
            {where}
            ORDER BY timestamp ASC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

        result: List[Transaction] = []
        for row in rows:
            try:
                result.append(self._row_to_transaction(row))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unparseable ledger row %s: %s", dict(row), exc)
        return result

    def get_totals_by_category(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Decimal]:
        """
        Return a dict of category → total amount for EXPENSE rows in the period.
        """
        clauses: List[str] = ["tx_type = ?"]
        params: List = [TransactionType.EXPENSE.value]

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())

        where = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT source_or_category, amount
            FROM transactions
            {where}
        """

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

        totals: Dict[str, Decimal] = {}
        for row in rows:
            cat = row[0]
            amt = Decimal(row[1])
            totals[cat] = totals.get(cat, Decimal("0")) + amt
        return totals


# ═══════════════════════════════════════════════════════════════════════════════
# PAYPAL ADAPTER (wraps PayPalWallet from janus_paypal_integration.py)
# ═══════════════════════════════════════════════════════════════════════════════

import os


class PayPalAdapter:
    """
    Thin wrapper around PayPalWallet.

    If the integration module is unavailable or credentials are missing the
    adapter degrades gracefully: all methods return None / empty list so the
    rest of the wallet still works for local ledger operations.
    """

    def __init__(self) -> None:
        self._paypal = None
        try:
            from janus_paypal_integration import PayPalWallet  # type: ignore

            client_id = os.getenv("PAYPAL_CLIENT_ID", "")
            client_secret = os.getenv("PAYPAL_CLIENT_SECRET", "")
            if not client_id or not client_secret:
                logger.warning(
                    "PayPalAdapter: PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET not set. "
                    "PayPal features disabled; local ledger still works."
                )
            else:
                self._paypal = PayPalWallet()
                logger.info("PayPalAdapter: PayPalWallet initialised successfully.")
        except ImportError:
            logger.warning(
                "PayPalAdapter: janus_paypal_integration not found. "
                "PayPal features disabled; local ledger still works."
            )

    # ── outbound ──────────────────────────────────────────────────────────────

    def send_payment(
        self,
        recipient_email: str,
        amount: Decimal,
        note: str = "",
    ) -> Optional[str]:
        """
        Send *amount* USD to *recipient_email* via PayPal.

        Returns the payout batch ID on success, ``None`` on failure.
        Never raises — the caller decides what to do with a None result.
        """
        if self._paypal is None:
            logger.warning("PayPalAdapter.send_payment: PayPal unavailable, skipping.")
            return None
        try:
            return self._paypal.send_payment(recipient_email, amount, note)
        except Exception as exc:  # noqa: BLE001
            logger.error("PayPalAdapter.send_payment failed: %s", exc)
            return None

    # ── inbound ───────────────────────────────────────────────────────────────

    def create_payment_link(
        self,
        amount: Decimal,
        description: str = "Payment to Janus",
    ) -> Optional[str]:
        """
        Create a PayPal payment-request link for *amount* USD.

        Returns the URL string or ``None`` if PayPal is unavailable.
        """
        if self._paypal is None:
            logger.warning("PayPalAdapter.create_payment_link: PayPal unavailable.")
            return None
        try:
            return self._paypal.create_payment_request(amount, description)
        except Exception as exc:  # noqa: BLE001
            logger.error("PayPalAdapter.create_payment_link failed: %s", exc)
            return None

    # ── sync ──────────────────────────────────────────────────────────────────

    def fetch_recent_transactions(self, days: int = 7) -> List[Transaction]:
        """
        Fetch recent PayPal transactions and map them to ``Transaction`` objects.

        Each PayPalTransaction is mapped with:
        - tx_type  = TransactionType.INCOME
        - verified = True
        - external_tx_id = paypal_tx.transaction_id

        Returns an empty list if PayPal is unavailable or an error occurs.
        """
        if self._paypal is None:
            return []
        try:
            paypal_txs = self._paypal.get_recent_transactions(days=days)
        except Exception as exc:  # noqa: BLE001
            logger.error("PayPalAdapter.fetch_recent_transactions failed: %s", exc)
            return []

        result: List[Transaction] = []
        for ptx in paypal_txs:
            result.append(
                Transaction(
                    id="",
                    tx_type=TransactionType.INCOME,
                    amount=ptx.amount,
                    currency=ptx.currency,
                    source_or_category="paypal",
                    description=ptx.description,
                    external_tx_id=ptx.transaction_id,
                    timestamp=ptx.timestamp,
                    metadata={
                        "payer_email": ptx.payer_email,
                        "payee_email": ptx.payee_email,
                        "status": ptx.status,
                        "fee": str(ptx.fee),
                    },
                    verified=True,
                )
            )
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET ANALYTICS (pure functions, no I/O)
# ═══════════════════════════════════════════════════════════════════════════════


class WalletAnalytics:
    """
    Stateless financial-intelligence helpers.

    All methods are pure functions over ``list[Transaction]`` or simple
    ``Decimal`` values — no database access, no network calls.
    """

    # Budget split ratios (must sum to 1.00)
    _RATIOS: Dict[str, Decimal] = {
        "compute":          Decimal("0.20"),
        "api_costs":        Decimal("0.25"),
        "training":         Decimal("0.15"),
        "savings":          Decimal("0.30"),
        "emergency_reserve": Decimal("0.10"),
    }

    # ── reports ───────────────────────────────────────────────────────────────

    def build_report(
        self,
        transactions: List[Transaction],
        period_start: datetime,
        period_end: datetime,
    ) -> FinancialReport:
        """
        Compute a ``FinancialReport`` from *transactions* for the given period.

        - ``total_income``  — sum of all INCOME amounts
        - ``total_expenses`` — sum of all EXPENSE amounts
        - ``net``           — income − expenses
        - ``income_by_source`` — dict of source → total income
        - ``expenses_by_category`` — dict of category → total expense
        - ``transaction_count`` — number of transactions
        - ``savings_rate``  — net / income (0.0 when income is 0)
        """
        total_income = Decimal("0")
        total_expenses = Decimal("0")
        income_by_source: Dict[str, Decimal] = {}
        expenses_by_category: Dict[str, Decimal] = {}

        for tx in transactions:
            if tx.tx_type == TransactionType.INCOME:
                total_income += tx.amount
                income_by_source[tx.source_or_category] = (
                    income_by_source.get(tx.source_or_category, Decimal("0")) + tx.amount
                )
            elif tx.tx_type == TransactionType.EXPENSE:
                total_expenses += tx.amount
                expenses_by_category[tx.source_or_category] = (
                    expenses_by_category.get(tx.source_or_category, Decimal("0")) + tx.amount
                )

        net = total_income - total_expenses
        savings_rate = float(net / total_income) if total_income > 0 else 0.0

        return FinancialReport(
            period_start=period_start,
            period_end=period_end,
            total_income=total_income,
            total_expenses=total_expenses,
            net=net,
            income_by_source=income_by_source,
            expenses_by_category=expenses_by_category,
            transaction_count=len(transactions),
            savings_rate=savings_rate,
        )

    # ── budget ────────────────────────────────────────────────────────────────

    def allocate_budget(self, total: Decimal) -> BudgetAllocation:
        """
        Split *total* into the five budget buckets using fixed ratios.

        Uses ``Decimal`` arithmetic throughout.  Any rounding remainder is
        assigned to ``savings`` so that the five fields always sum to *total*.
        """
        compute          = (total * self._RATIOS["compute"]).quantize(Decimal("0.01"))
        api_costs        = (total * self._RATIOS["api_costs"]).quantize(Decimal("0.01"))
        training         = (total * self._RATIOS["training"]).quantize(Decimal("0.01"))
        emergency_reserve = (total * self._RATIOS["emergency_reserve"]).quantize(Decimal("0.01"))

        # Assign remainder to savings so the sum is exact
        savings = total - compute - api_costs - training - emergency_reserve

        return BudgetAllocation(
            compute=compute,
            api_costs=api_costs,
            training=training,
            savings=savings,
            emergency_reserve=emergency_reserve,
        )

    # ── decisions ─────────────────────────────────────────────────────────────

    def should_reinvest(self, report: FinancialReport) -> bool:
        """
        Return ``True`` when the savings rate is ≥ 30 % and net is positive.
        """
        return report.savings_rate >= 0.3 and report.net > Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════════
# JANUS WALLET (public façade)
# ═══════════════════════════════════════════════════════════════════════════════


class JanusWallet:
    """
    Public API consumed by ``janus_autonomous_worker.py``.

    Composes ``WalletLedger``, ``PayPalAdapter``, and ``WalletAnalytics``
    into a single, easy-to-use object.
    """

    BUSINESS_NAME: str = "Janus AI Worker"

    def __init__(self, db_path: str = "janus_wallet.db") -> None:
        self._ledger    = WalletLedger(db_path)
        self._paypal    = PayPalAdapter()
        self._analytics = WalletAnalytics()
        logger.info("JanusWallet initialised (db=%s)", db_path)

    @staticmethod
    def get_paypal_email() -> str:
        """Return the configured PayPal email (from PAYPAL_EMAIL env var)."""
        return os.getenv("PAYPAL_EMAIL", "")

    # ── income ────────────────────────────────────────────────────────────────

    def record_income(
        self,
        amount: Decimal,
        source: str,
        description: str = "",
        external_tx_id: str = "",
        currency: str = "USD",
        metadata: Optional[dict] = None,
    ) -> Transaction:
        """Record an income transaction and return the saved ``Transaction``."""
        if amount <= Decimal("0"):
            raise ValueError(f"Income amount must be > 0, got {amount!r}")

        tx = Transaction(
            id="",
            tx_type=TransactionType.INCOME,
            amount=amount,
            currency=currency,
            source_or_category=source,
            description=description,
            external_tx_id=external_tx_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            verified=False,
        )
        return self._ledger.insert_transaction(tx)

    # ── expenses ──────────────────────────────────────────────────────────────

    def record_expense(
        self,
        amount: Decimal,
        category,
        description: str = "",
        external_tx_id: str = "",
        currency: str = "USD",
        metadata: Optional[dict] = None,
    ) -> Transaction:
        """Record an expense transaction and return the saved ``Transaction``."""
        if amount <= Decimal("0"):
            raise ValueError(f"Expense amount must be > 0, got {amount!r}")

        if isinstance(category, ExpenseCategory):
            cat_str = category.value
        else:
            cat_str = str(category)

        tx = Transaction(
            id="",
            tx_type=TransactionType.EXPENSE,
            amount=amount,
            currency=currency,
            source_or_category=cat_str,
            description=description,
            external_tx_id=external_tx_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            verified=False,
        )
        return self._ledger.insert_transaction(tx)

    # ── balances ──────────────────────────────────────────────────────────────

    def get_balance(self, currency: str = "USD") -> Decimal:
        """Return the net balance for *currency*."""
        return self._ledger.get_balance(currency)

    def get_all_balances(self) -> Dict[str, Decimal]:
        """Return balances for USD, BTC, ETH, and USDC."""
        return {
            currency: self._ledger.get_balance(currency)
            for currency in ("USD", "BTC", "ETH", "USDC")
        }

    # ── outbound payments ─────────────────────────────────────────────────────

    def send_payment(
        self,
        recipient_email: str,
        amount: Decimal,
        note: str = "",
        require_approval_above: Optional[Decimal] = None,
    ) -> "PaymentResult":
        """
        Send *amount* USD to *recipient_email* via PayPal.

        Raises ``ApprovalRequiredError`` when:
        - ``require_approval_above`` is ``None`` and ``amount > 500``
        - ``require_approval_above`` is set and ``amount > require_approval_above``
        """
        threshold = require_approval_above if require_approval_above is not None else Decimal("500")
        if amount > threshold:
            raise ApprovalRequiredError(
                f"Payment of {amount} exceeds approval threshold of {threshold}. "
                "Set require_approval_above explicitly to override."
            )

        batch_id = self._paypal.send_payment(recipient_email, amount, note)
        success = batch_id is not None

        tx: Optional[Transaction] = None
        if success:
            try:
                tx = self.record_expense(
                    amount=amount,
                    category=ExpenseCategory.MISC,
                    description=f"PayPal payment to {recipient_email}: {note}",
                    external_tx_id=batch_id or "",
                    currency="USD",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("send_payment: failed to record expense in ledger: %s", exc)

        return PaymentResult(
            success=success,
            batch_id=batch_id,
            transaction=tx,
            error=None if success else "PayPal send_payment returned None",
        )

    # ── inbound payment links ─────────────────────────────────────────────────

    def create_payment_link(
        self,
        amount: Decimal,
        description: str = "Payment to Janus",
    ) -> Optional[str]:
        """Delegate to ``PayPalAdapter.create_payment_link``."""
        return self._paypal.create_payment_link(amount, description)

    # ── analytics ─────────────────────────────────────────────────────────────

    def get_report(self, period: ReportPeriod) -> FinancialReport:
        """
        Build a ``FinancialReport`` for the requested *period*.

        Period windows (relative to now):
        - DAILY   → last 1 day
        - WEEKLY  → last 7 days
        - MONTHLY → last 30 days
        - ALL     → no date filter
        """
        now = datetime.utcnow()
        _PERIOD_DAYS: Dict[ReportPeriod, Optional[int]] = {
            ReportPeriod.DAILY:   1,
            ReportPeriod.WEEKLY:  7,
            ReportPeriod.MONTHLY: 30,
            ReportPeriod.ALL:     None,
        }
        days = _PERIOD_DAYS[period]

        if days is not None:
            from datetime import timedelta
            since = now - timedelta(days=days)
        else:
            since = None

        transactions = self._ledger.list_transactions(since=since, until=now)
        period_start = since if since is not None else (
            transactions[0].timestamp if transactions else now
        )
        return self._analytics.build_report(transactions, period_start, now)

    def get_budget_status(self) -> BudgetStatus:
        """Return a ``BudgetStatus`` snapshot based on the current monthly report."""
        balance = self.get_balance("USD")
        monthly_report = self.get_report(ReportPeriod.MONTHLY)
        allocation = self._analytics.allocate_budget(balance)
        reinvest = self._analytics.should_reinvest(monthly_report)
        return BudgetStatus(
            allocation=allocation,
            current_balance=balance,
            report=monthly_report,
            should_reinvest=reinvest,
        )

    # ── autonomous decisions ──────────────────────────────────────────────────

    def should_reinvest(self) -> bool:
        """Return ``True`` when the monthly report meets the reinvestment threshold."""
        report = self.get_report(ReportPeriod.MONTHLY)
        return self._analytics.should_reinvest(report)

    def allocate_budget(self, total: Decimal) -> BudgetAllocation:
        """Delegate to ``WalletAnalytics.allocate_budget``."""
        return self._analytics.allocate_budget(total)

    # ── PayPal sync ───────────────────────────────────────────────────────────

    def sync_paypal_transactions(self, days: int = 7) -> int:
        """
        Fetch recent PayPal transactions and insert any that are new.

        Skips transactions that are already in the ledger
        (``DuplicateTransactionError``).  Returns the count of newly inserted
        rows.
        """
        fetched = self._paypal.fetch_recent_transactions(days)
        inserted = 0
        for tx in fetched:
            try:
                self._ledger.insert_transaction(tx)
                inserted += 1
            except DuplicateTransactionError:
                pass  # already recorded — skip silently
            except Exception as exc:  # noqa: BLE001
                logger.error("sync_paypal_transactions: failed to insert tx %s: %s", tx.external_tx_id, exc)
        return inserted

    # ── crypto stubs ──────────────────────────────────────────────────────────

    def get_crypto_addresses(self) -> dict:
        """Return crypto wallet addresses (not yet implemented)."""
        return {}

    def setup_crypto_wallets(self) -> None:
        """Stub — crypto wallets not yet implemented."""
        logger.warning("Crypto wallets not yet implemented")
