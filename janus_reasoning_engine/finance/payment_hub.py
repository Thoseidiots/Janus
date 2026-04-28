"""
PaymentHub — wraps existing payment modules (PayPal, Revolut, wallet).

All external integrations are optional; the hub degrades gracefully when
modules are unavailable.

Requirements: REQ-10.1, REQ-8.3
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_paypal_integration import PayPalClient, PayPalWallet  # type: ignore
    _PAYPAL_AVAILABLE = True
except ImportError:
    _PAYPAL_AVAILABLE = False
    logger.debug("janus_paypal_integration not available — PayPal disabled")

try:
    from janus_revolut_payments import RevolutClient  # type: ignore
    _REVOLUT_AVAILABLE = True
except ImportError:
    _REVOLUT_AVAILABLE = False
    logger.debug("janus_revolut_payments not available — Revolut disabled")

try:
    from janus_wallet import JanusWallet  # type: ignore
    _WALLET_AVAILABLE = True
except ImportError:
    _WALLET_AVAILABLE = False
    logger.debug("janus_wallet not available — wallet balance checks disabled")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PaymentResult:
    """Result of a payment attempt."""
    success: bool
    payment_id: str
    method: str
    amount: float
    recipient: str
    error: Optional[str] = None
    raw: Optional[object] = None


# ---------------------------------------------------------------------------
# PaymentHub
# ---------------------------------------------------------------------------

class PaymentHub:
    """
    Unified interface for all payment operations.

    Wraps PayPal, Revolut, and the Janus wallet.  Each backend is loaded
    lazily and only used when available.
    """

    _SUPPORTED_METHODS_BASE: List[str] = ["manual"]

    def __init__(self) -> None:
        self._paypal: Optional[object] = None
        self._revolut: Optional[object] = None
        self._wallet: Optional[object] = None
        self._init_backends()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_backends(self) -> None:
        if _PAYPAL_AVAILABLE:
            try:
                self._paypal = PayPalWallet()
                logger.info("PayPal backend initialised")
            except Exception as exc:
                logger.warning("PayPal init failed: %s", exc)

        if _REVOLUT_AVAILABLE:
            try:
                self._revolut = RevolutClient()  # type: ignore[name-defined]
                logger.info("Revolut backend initialised")
            except Exception as exc:
                logger.warning("Revolut init failed: %s", exc)

        if _WALLET_AVAILABLE:
            try:
                self._wallet = JanusWallet()
                logger.info("JanusWallet backend initialised")
            except Exception as exc:
                logger.warning("JanusWallet init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_supported_methods(self) -> List[str]:
        """Return list of currently available payment methods."""
        methods = list(self._SUPPORTED_METHODS_BASE)
        if self._paypal is not None:
            methods.append("paypal")
        if self._revolut is not None:
            methods.append("revolut")
        if self._wallet is not None:
            methods.append("wallet")
        return methods

    def process_payment(
        self,
        amount: float,
        method: str,
        recipient: str,
    ) -> PaymentResult:
        """
        Process a payment via the specified method.

        Parameters
        ----------
        amount:    Amount in USD (or native currency for the method).
        method:    One of the values returned by get_supported_methods().
        recipient: Email address, wallet address, or identifier.

        Returns
        -------
        PaymentResult with success flag and payment_id.
        """
        method = method.lower()
        payment_id = str(uuid.uuid4())

        if method == "paypal" and self._paypal is not None:
            return self._process_paypal(amount, recipient, payment_id)

        if method == "revolut" and self._revolut is not None:
            return self._process_revolut(amount, recipient, payment_id)

        if method == "wallet" and self._wallet is not None:
            return self._process_wallet(amount, recipient, payment_id)

        if method == "manual":
            logger.info("Manual payment of %.2f to %s recorded", amount, recipient)
            return PaymentResult(
                success=True,
                payment_id=payment_id,
                method="manual",
                amount=amount,
                recipient=recipient,
            )

        return PaymentResult(
            success=False,
            payment_id=payment_id,
            method=method,
            amount=amount,
            recipient=recipient,
            error=f"Payment method '{method}' not available",
        )

    def verify_payment(self, payment_id: str) -> bool:
        """
        Verify that a payment with the given ID was completed.

        Tries each available backend.  Returns True if any backend
        confirms the payment.
        """
        if not payment_id:
            return False

        # Try PayPal transaction history
        if self._paypal is not None:
            try:
                txns = self._paypal.get_recent_transactions(days=90)  # type: ignore[union-attr]
                for tx in txns:
                    tx_id = getattr(tx, "transaction_id", None) or getattr(tx, "id", None)
                    if tx_id == payment_id:
                        return True
            except Exception as exc:
                logger.debug("PayPal verify error: %s", exc)

        # Try wallet ledger
        if self._wallet is not None:
            try:
                from janus_wallet import ReportPeriod  # type: ignore
                report = self._wallet.get_report(ReportPeriod.MONTHLY)  # type: ignore[union-attr]
                # If we got a report without error, treat as best-effort
                _ = report
            except Exception as exc:
                logger.debug("Wallet verify error: %s", exc)

        # Fallback: unknown payment IDs cannot be verified without a DB
        logger.debug("Cannot verify payment_id=%s — no matching record found", payment_id)
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_paypal(
        self, amount: float, recipient: str, payment_id: str
    ) -> PaymentResult:
        try:
            from decimal import Decimal
            result = self._paypal.send_payment(  # type: ignore[union-attr]
                recipient_email=recipient,
                amount=Decimal(str(amount)),
                note=f"Janus payment {payment_id}",
            )
            success = result is not None
            return PaymentResult(
                success=success,
                payment_id=result or payment_id,
                method="paypal",
                amount=amount,
                recipient=recipient,
                raw=result,
            )
        except Exception as exc:
            logger.error("PayPal payment failed: %s", exc)
            return PaymentResult(
                success=False,
                payment_id=payment_id,
                method="paypal",
                amount=amount,
                recipient=recipient,
                error=str(exc),
            )

    def _process_revolut(
        self, amount: float, recipient: str, payment_id: str
    ) -> PaymentResult:
        try:
            result = self._revolut.send_payment(  # type: ignore[union-attr]
                recipient=recipient,
                amount=amount,
                reference=payment_id,
            )
            return PaymentResult(
                success=True,
                payment_id=payment_id,
                method="revolut",
                amount=amount,
                recipient=recipient,
                raw=result,
            )
        except Exception as exc:
            logger.error("Revolut payment failed: %s", exc)
            return PaymentResult(
                success=False,
                payment_id=payment_id,
                method="revolut",
                amount=amount,
                recipient=recipient,
                error=str(exc),
            )

    def _process_wallet(
        self, amount: float, recipient: str, payment_id: str
    ) -> PaymentResult:
        try:
            from decimal import Decimal
            result = self._wallet.send_payment(  # type: ignore[union-attr]
                recipient_email=recipient,
                amount=Decimal(str(amount)),
                note=f"Janus payment {payment_id}",
            )
            success = getattr(result, "success", result is not None)
            return PaymentResult(
                success=bool(success),
                payment_id=payment_id,
                method="wallet",
                amount=amount,
                recipient=recipient,
                raw=result,
            )
        except Exception as exc:
            logger.error("Wallet payment failed: %s", exc)
            return PaymentResult(
                success=False,
                payment_id=payment_id,
                method="wallet",
                amount=amount,
                recipient=recipient,
                error=str(exc),
            )
