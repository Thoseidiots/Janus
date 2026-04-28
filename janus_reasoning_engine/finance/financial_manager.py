"""
FinancialManager — integrates subscription and financial intelligence systems.

Optional imports: janus_subscription_system, janus_financial_intelligence,
janus_wallet.  All degrade gracefully when unavailable.

Requirements: REQ-10.3, REQ-10.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from janus_wallet import JanusWallet, BudgetStatus  # type: ignore
    _WALLET_AVAILABLE = True
except ImportError:
    _WALLET_AVAILABLE = False
    BudgetStatus = None  # type: ignore[assignment,misc]
    logger.debug("janus_wallet not available")

try:
    from janus_financial_intelligence import FinancialIntelligence  # type: ignore
    _FI_AVAILABLE = True
except ImportError:
    _FI_AVAILABLE = False
    logger.debug("janus_financial_intelligence not available")

try:
    from janus_subscription_system import SubscriptionManager  # type: ignore
    _SUB_AVAILABLE = True
except ImportError:
    _SUB_AVAILABLE = False
    logger.debug("janus_subscription_system not available")


# ---------------------------------------------------------------------------
# Fallback BudgetStatus when janus_wallet is absent
# ---------------------------------------------------------------------------

@dataclass
class _FallbackBudgetStatus:
    """Minimal budget status used when janus_wallet is unavailable."""
    current_balance: float = 0.0
    allocated: Dict[str, float] = field(default_factory=dict)
    available: float = 0.0
    healthy: bool = True
    note: str = "wallet unavailable"


# ---------------------------------------------------------------------------
# FinancialManager
# ---------------------------------------------------------------------------

class FinancialManager:
    """
    High-level financial management: budget status, allocation, cash-flow
    prediction, and subscription health.

    Delegates to janus_wallet, janus_financial_intelligence, and
    janus_subscription_system when available.
    """

    def __init__(self) -> None:
        self._wallet: Optional[Any] = None
        self._fi: Optional[Any] = None
        self._sub: Optional[Any] = None
        self._budget: Dict[str, float] = {}
        self._init_backends()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backends(self) -> None:
        if _WALLET_AVAILABLE:
            try:
                self._wallet = JanusWallet()
                logger.info("JanusWallet backend initialised")
            except Exception as exc:
                logger.warning("JanusWallet init failed: %s", exc)

        if _FI_AVAILABLE and self._wallet is not None:
            try:
                self._fi = FinancialIntelligence(self._wallet)
                logger.info("FinancialIntelligence backend initialised")
            except Exception as exc:
                logger.warning("FinancialIntelligence init failed: %s", exc)

        if _SUB_AVAILABLE:
            try:
                self._sub = SubscriptionManager()  # type: ignore[name-defined]
                logger.info("SubscriptionManager backend initialised")
            except Exception as exc:
                logger.warning("SubscriptionManager init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_budget_status(self) -> Any:
        """
        Return current budget health.

        Uses JanusWallet.get_budget_status() when available, otherwise
        returns a lightweight fallback object.
        """
        if self._wallet is not None:
            try:
                return self._wallet.get_budget_status()
            except Exception as exc:
                logger.warning("get_budget_status failed: %s", exc)

        # Fallback
        allocated_total = sum(self._budget.values())
        return _FallbackBudgetStatus(
            current_balance=0.0,
            allocated=dict(self._budget),
            available=max(0.0, -allocated_total),
            healthy=True,
        )

    def allocate_budget(self, category: str, amount: float) -> bool:
        """
        Allocate *amount* USD to *category*.

        Delegates to FinancialIntelligence when available; otherwise
        maintains an in-memory allocation map.

        Returns True on success, False on failure.
        """
        if amount < 0:
            logger.warning("allocate_budget: negative amount %.2f rejected", amount)
            return False

        if self._fi is not None:
            try:
                from decimal import Decimal
                result = self._fi.should_spend(Decimal(str(amount)), category)
                if result:
                    self._budget[category] = self._budget.get(category, 0.0) + amount
                    logger.info("Allocated %.2f to %s via FinancialIntelligence", amount, category)
                return bool(result)
            except Exception as exc:
                logger.warning("FinancialIntelligence allocate failed: %s", exc)

        # Fallback: always allow allocation
        self._budget[category] = self._budget.get(category, 0.0) + amount
        logger.info("Allocated %.2f to %s (local)", amount, category)
        return True

    def predict_cash_flow(self, days: int = 30) -> float:
        """
        Predict net cash flow over the next *days* days.

        Uses FinancialIntelligence.predict_cash_flow() when available.
        Returns 0.0 when no data is available.
        """
        if self._fi is not None:
            try:
                result = self._fi.predict_cash_flow(days=days)
                # predict_cash_flow returns Dict[str, Decimal]; sum net values
                if isinstance(result, dict):
                    net = result.get("net", result.get("projected_net", 0))
                    return float(net)
                return float(result)
            except Exception as exc:
                logger.warning("predict_cash_flow failed: %s", exc)

        return 0.0

    def check_subscription_health(self) -> Dict[str, Any]:
        """
        Return a health summary for all active subscriptions.

        Uses SubscriptionManager when available; returns a minimal
        status dict otherwise.
        """
        if self._sub is not None:
            try:
                # Try common method names
                for method in ("get_health", "health_check", "get_status", "list_subscriptions"):
                    fn = getattr(self._sub, method, None)
                    if fn is not None:
                        raw = fn()
                        if isinstance(raw, dict):
                            return raw
                        return {"status": "ok", "data": str(raw)}
            except Exception as exc:
                logger.warning("check_subscription_health failed: %s", exc)
                return {"status": "error", "error": str(exc)}

        return {
            "status": "unavailable",
            "message": "janus_subscription_system not installed",
            "active_subscriptions": 0,
        }
