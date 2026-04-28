"""CEO multi-business management bridge.

Requirements: REQ-14.4
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

# Optional imports
try:
    import janus_ai_ceo as _janus_ai_ceo  # type: ignore
    _HAS_CEO = True
except (ImportError, SyntaxError, Exception):
    _janus_ai_ceo = None
    _HAS_CEO = False

try:
    import ceo_agent as _ceo_agent  # type: ignore
    _HAS_CEO_AGENT = True
except (ImportError, SyntaxError, Exception):
    _ceo_agent = None
    _HAS_CEO_AGENT = False


class CEOBridge:
    """Bridge to CEO multi-business management.

    Wraps janus_ai_ceo / ceo_agent when available; falls back to stubs.
    """

    _BUSINESS_TYPES = {
        "content_agency",
        "ecommerce",
        "marketing_agency",
        "data_analysis",
        "web_development",
    }

    def __init__(self) -> None:
        self._businesses: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_business(
        self,
        business_type: str,
        name: str,
        target_revenue: float,
    ) -> str:
        """Start a new business entity and return its business_id."""
        if _HAS_CEO and hasattr(_janus_ai_ceo, "start_business"):
            result = _janus_ai_ceo.start_business(business_type, name, target_revenue)
            business_id = str(result) if result else str(uuid.uuid4())
        else:
            business_id = str(uuid.uuid4())

        self._businesses[business_id] = {
            "business_id": business_id,
            "type": business_type,
            "name": name,
            "target_revenue": target_revenue,
            "revenue": 0.0,
            "status": "active",
        }
        return business_id

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Return a summary of all managed businesses."""
        if _HAS_CEO and hasattr(_janus_ai_ceo, "get_portfolio_summary"):
            return _janus_ai_ceo.get_portfolio_summary()

        total_revenue = sum(b["revenue"] for b in self._businesses.values())
        return {
            "total_businesses": len(self._businesses),
            "total_revenue": total_revenue,
            "businesses": list(self._businesses.values()),
        }

    def make_strategic_decision(self, context: str, options: List[str]) -> str:
        """Return the best option description given context and options."""
        if _HAS_CEO and hasattr(_janus_ai_ceo, "make_strategic_decision"):
            return str(_janus_ai_ceo.make_strategic_decision(context, options))

        if _HAS_CEO_AGENT and hasattr(_ceo_agent, "make_decision"):
            return str(_ceo_agent.make_decision(context, options))

        # Stub: pick the first option as the "best"
        if not options:
            return "No options provided."
        return f"Strategic choice for '{context}': {options[0]}"
