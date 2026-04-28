"""
Risk Assessment module for the Janus Autonomous Reasoning Engine.

Evaluates the risk of an action before execution, detecting scams,
budget overruns, and credential exposure.

**Validates: Requirements REQ-5.2, REQ-9.2**
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists for scam / credential detection
# ---------------------------------------------------------------------------

_SCAM_KEYWORDS: List[str] = [
    "guaranteed profit",
    "guaranteed returns",
    "send money first",
    "wire transfer",
    "crypto upfront",
    "upfront payment",
    "advance fee",
    "nigerian prince",
    "lottery winner",
    "act now",
    "limited time offer",
    "double your money",
    "risk free investment",
    "risk-free investment",
    "no risk",
    "100% profit",
]

_CREDENTIAL_KEYWORDS: List[str] = [
    "password",
    "private key",
    "secret key",
    "api key",
    "api secret",
    "seed phrase",
    "mnemonic",
    "ssh key",
    "credentials",
    "login details",
    "share your",
    "send your",
    "provide your",
]

_LEGAL_KEYWORDS: List[str] = [
    "illegal",
    "piracy",
    "crack",
    "bypass drm",
    "counterfeit",
    "fraud",
    "money laundering",
    "tax evasion",
    "hack into",
    "unauthorized access",
    "ddos",
    "phishing",
]


@dataclass
class RiskReport:
    """
    Comprehensive risk report for an action.

    Scores are in the range [0.0, 1.0] where 1.0 = maximum risk.
    """
    overall_risk: float
    financial_risk: float
    reputation_risk: float
    legal_risk: float
    flags: List[str] = field(default_factory=list)
    action_description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RiskReport(overall={self.overall_risk:.2f}, "
            f"financial={self.financial_risk:.2f}, "
            f"reputation={self.reputation_risk:.2f}, "
            f"legal={self.legal_risk:.2f}, "
            f"flags={self.flags})"
        )


class RiskAssessor:
    """
    Evaluates the risk of an action before Janus takes it.

    Checks for:
    - Scam indicators (guaranteed profit, wire transfer, etc.)
    - Budget overruns (estimated_cost > budget_limit)
    - Credential exposure (sharing passwords / keys)
    - Legal issues (illegal activities)

    Usage::

        assessor = RiskAssessor(budget_limit=500.0)
        report = assessor.assess("Send $200 wire transfer for guaranteed profit", {})
        if not assessor.is_safe(report):
            print("Action blocked:", report.flags)
    """

    def __init__(self, budget_limit: float = 100.0) -> None:
        """
        Args:
            budget_limit: Maximum allowed spend per action (USD).
        """
        self.budget_limit = budget_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        action_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskReport:
        """
        Assess the risk of an action.

        Args:
            action_description: Natural-language description of the action.
            context: Optional context dict. Recognised keys:
                - ``estimated_cost`` (float): expected spend in USD.
                - ``platform`` (str): platform name.
                - ``client_rating`` (float): client reputation 0–5.

        Returns:
            RiskReport with scores and flags.
        """
        context = context or {}
        text = action_description.lower()
        flags: List[str] = []

        # --- Scam detection ---
        financial_risk = 0.0
        for kw in _SCAM_KEYWORDS:
            if kw in text:
                flags.append(f"scam_keyword:{kw}")
                financial_risk = min(1.0, financial_risk + 0.4)

        # --- Budget protection ---
        estimated_cost = float(context.get("estimated_cost", 0.0))
        if estimated_cost > self.budget_limit:
            flags.append(
                f"budget_exceeded:estimated_cost={estimated_cost:.2f}"
                f">limit={self.budget_limit:.2f}"
            )
            financial_risk = min(1.0, financial_risk + 0.5)

        # --- Credential protection ---
        reputation_risk = 0.0
        for kw in _CREDENTIAL_KEYWORDS:
            if kw in text:
                flags.append(f"credential_risk:{kw}")
                reputation_risk = min(1.0, reputation_risk + 0.5)
                financial_risk = min(1.0, financial_risk + 0.3)

        # --- Legal risk ---
        legal_risk = 0.0
        for kw in _LEGAL_KEYWORDS:
            if kw in text:
                flags.append(f"legal_risk:{kw}")
                legal_risk = min(1.0, legal_risk + 0.5)
                reputation_risk = min(1.0, reputation_risk + 0.3)

        # --- Low client rating ---
        client_rating = context.get("client_rating")
        if client_rating is not None and float(client_rating) < 2.0:
            flags.append(f"low_client_rating:{client_rating}")
            reputation_risk = min(1.0, reputation_risk + 0.2)

        # --- Overall risk: weighted combination ---
        overall_risk = min(
            1.0,
            0.4 * financial_risk
            + 0.3 * reputation_risk
            + 0.3 * legal_risk,
        )

        # Bump overall if there are any flags at all
        if flags and overall_risk < 0.1:
            overall_risk = 0.1

        report = RiskReport(
            overall_risk=round(overall_risk, 4),
            financial_risk=round(financial_risk, 4),
            reputation_risk=round(reputation_risk, 4),
            legal_risk=round(legal_risk, 4),
            flags=flags,
            action_description=action_description,
            context=context,
        )

        if flags:
            logger.warning(f"Risk flags detected: {flags}")

        return report

    def is_safe(self, risk_report: RiskReport, threshold: float = 0.7) -> bool:
        """
        Return True if the action is considered safe.

        Args:
            risk_report: The report produced by :meth:`assess`.
            threshold: Maximum acceptable overall risk (default 0.7).

        Returns:
            True if overall_risk < threshold, False otherwise.
        """
        return risk_report.overall_risk < threshold
