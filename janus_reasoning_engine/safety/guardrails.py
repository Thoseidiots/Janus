"""
Safety guardrails for the Janus Autonomous Reasoning Engine.

Enforces budget limits, content filtering, credential protection,
and permission requirements before actions are taken.

Requirements: REQ-9.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("janus.safety.guardrails")

# ---------------------------------------------------------------------------
# Keyword lists (same as RiskAssessor for consistency)
# ---------------------------------------------------------------------------

_ILLEGAL_KEYWORDS: List[str] = [
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
    "child exploitation",
    "drug trafficking",
    "weapon manufacturing",
]

_UNETHICAL_KEYWORDS: List[str] = [
    "guaranteed profit",
    "guaranteed returns",
    "send money first",
    "wire transfer",
    "crypto upfront",
    "upfront payment",
    "advance fee",
    "nigerian prince",
    "lottery winner",
    "double your money",
    "risk free investment",
    "risk-free investment",
    "100% profit",
    "scam",
    "deceive",
    "manipulate users",
    "fake reviews",
    "astroturfing",
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
    "expose",
    "leak",
    "dump credentials",
]

_NEW_PLATFORM_KEYWORDS: List[str] = [
    "new platform",
    "new account",
    "sign up",
    "register on",
    "create account",
    "join platform",
    "new service",
    "new marketplace",
]


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    allowed: bool
    reason: str
    flags: List[str] = field(default_factory=list)


class SafetyGuardrails:
    """
    Enforces safety constraints before Janus takes any action.

    Checks:
    - Budget limits (spending caps)
    - Content filtering (illegal/unethical)
    - Credential protection
    - Permission requirements for major decisions

    Usage::

        guardrails = SafetyGuardrails(budget_limit=100.0)
        result = guardrails.check_budget(150.0, 100.0)
        if not result.allowed:
            print("Blocked:", result.reason)
    """

    def __init__(self, budget_limit: float = 100.0) -> None:
        """
        Args:
            budget_limit: Default budget limit in USD.
        """
        self.budget_limit = budget_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_budget(self, estimated_cost: float, budget_limit: float) -> GuardrailResult:
        """
        Check whether an action's estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost of the action in USD.
            budget_limit: Maximum allowed spend in USD.

        Returns:
            GuardrailResult — allowed=False if cost exceeds limit.
        """
        if estimated_cost > budget_limit:
            reason = (
                f"Estimated cost ${estimated_cost:.2f} exceeds "
                f"budget limit ${budget_limit:.2f}"
            )
            logger.warning("Budget guardrail triggered: %s", reason)
            return GuardrailResult(
                allowed=False,
                reason=reason,
                flags=[f"budget_exceeded:cost={estimated_cost:.2f},limit={budget_limit:.2f}"],
            )
        return GuardrailResult(
            allowed=True,
            reason="Cost within budget",
            flags=[],
        )

    def check_content(self, action_description: str) -> GuardrailResult:
        """
        Check whether an action involves illegal or unethical content.

        Args:
            action_description: Natural-language description of the action.

        Returns:
            GuardrailResult — allowed=False if illegal/unethical content detected.
        """
        text = action_description.lower()
        flags: List[str] = []

        for kw in _ILLEGAL_KEYWORDS:
            if kw in text:
                flags.append(f"illegal_content:{kw}")

        for kw in _UNETHICAL_KEYWORDS:
            if kw in text:
                flags.append(f"unethical_content:{kw}")

        if flags:
            reason = f"Action contains illegal/unethical content: {flags}"
            logger.warning("Content guardrail triggered: %s", reason)
            return GuardrailResult(allowed=False, reason=reason, flags=flags)

        return GuardrailResult(
            allowed=True,
            reason="Content check passed",
            flags=[],
        )

    def check_credentials(self, action_description: str) -> GuardrailResult:
        """
        Check whether an action would expose passwords or keys.

        Args:
            action_description: Natural-language description of the action.

        Returns:
            GuardrailResult — allowed=False if credential exposure detected.
        """
        text = action_description.lower()
        flags: List[str] = []

        for kw in _CREDENTIAL_KEYWORDS:
            if kw in text:
                flags.append(f"credential_risk:{kw}")

        if flags:
            reason = f"Action may expose credentials: {flags}"
            logger.warning("Credential guardrail triggered: %s", reason)
            return GuardrailResult(allowed=False, reason=reason, flags=flags)

        return GuardrailResult(
            allowed=True,
            reason="Credential check passed",
            flags=[],
        )

    def requires_permission(
        self, action_description: str, estimated_cost: float
    ) -> bool:
        """
        Determine whether an action requires explicit owner permission.

        Returns True if:
        - estimated_cost > $100, OR
        - action involves signing up for / joining a new platform

        Args:
            action_description: Natural-language description of the action.
            estimated_cost: Estimated cost of the action in USD.

        Returns:
            True if owner permission is required.
        """
        if estimated_cost > 100.0:
            logger.info(
                "Permission required: cost $%.2f > $100 threshold", estimated_cost
            )
            return True

        text = action_description.lower()
        for kw in _NEW_PLATFORM_KEYWORDS:
            if kw in text:
                logger.info(
                    "Permission required: new platform keyword '%s' detected", kw
                )
                return True

        return False
