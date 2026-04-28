"""
MarketplaceHub — job posting, browsing, and active listing management.

All external integrations are optional; degrades gracefully when modules
are unavailable.

Requirements: REQ-11.2, REQ-2.1
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_marketplace import JanusMarketplace  # type: ignore
    _MARKETPLACE_AVAILABLE = True
except ImportError:
    _MARKETPLACE_AVAILABLE = False
    logger.debug("janus_marketplace not available — using in-memory stub")

try:
    from janus_platform_browser import PlatformBrowser  # type: ignore
    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False
    logger.debug("janus_platform_browser not available — browse_jobs returns []")


# ---------------------------------------------------------------------------
# MarketplaceHub
# ---------------------------------------------------------------------------

class MarketplaceHub:
    """
    Unified interface for marketplace operations: posting jobs, browsing
    available work, and tracking active listings.
    """

    def __init__(self) -> None:
        self._listings: Dict[str, Dict] = {}
        self._marketplace: Optional[object] = None
        self._browser: Optional[object] = None

        if _MARKETPLACE_AVAILABLE:
            try:
                self._marketplace = JanusMarketplace()
                logger.info("JanusMarketplace backend initialised")
            except Exception as exc:
                logger.warning("JanusMarketplace init failed: %s", exc)

        if _BROWSER_AVAILABLE:
            try:
                self._browser = PlatformBrowser()
                logger.info("PlatformBrowser backend initialised")
            except Exception as exc:
                logger.warning("PlatformBrowser init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_job(self, title: str, description: str, budget: float) -> str:
        """
        Post a new job listing.

        Returns a unique job_id string.  Delegates to janus_marketplace
        when available; otherwise stores the listing in memory.
        """
        job_id = str(uuid.uuid4())

        if self._marketplace is not None:
            try:
                result = self._marketplace.post_job(  # type: ignore[union-attr]
                    title=title,
                    description=description,
                    budget=budget,
                )
                job_id = str(getattr(result, "id", job_id) or job_id)
                logger.info("Job posted via marketplace: job_id=%s", job_id)
            except Exception as exc:
                logger.warning("marketplace.post_job failed: %s", exc)

        self._listings[job_id] = {
            "id": job_id,
            "title": title,
            "description": description,
            "budget": budget,
            "status": "active",
        }
        return job_id

    def browse_jobs(
        self,
        skills: List[str],
        platforms: List[str],
    ) -> List[Dict]:
        """
        Browse available jobs matching the given skills and platforms.

        Delegates to janus_platform_browser when available; otherwise
        returns an empty list.
        """
        if self._browser is not None:
            try:
                raw = self._browser.browse(skills=skills, platforms=platforms)  # type: ignore[union-attr]
                jobs: List[Dict] = []
                for item in raw or []:
                    if isinstance(item, dict):
                        jobs.append(item)
                    else:
                        jobs.append(
                            {
                                "id": str(getattr(item, "id", "")),
                                "title": str(getattr(item, "title", "")),
                                "description": str(getattr(item, "description", "")),
                                "budget": float(getattr(item, "budget", 0.0)),
                                "platform": str(getattr(item, "platform", "")),
                            }
                        )
                return jobs
            except Exception as exc:
                logger.warning("browse_jobs via browser failed: %s", exc)

        return []

    def get_active_listings(self) -> List[Dict]:
        """Return all currently active job listings managed by this hub."""
        return [
            listing
            for listing in self._listings.values()
            if listing.get("status") == "active"
        ]
