"""
opportunity_monitor.py
======================
Multi-source opportunity monitor for the Janus Reasoning Engine.

Monitors job boards, social media, GitHub bounties, and hackathons for
earning opportunities. Integrates with PlatformBrowser for human-like
browsing of Upwork/Fiverr. All external integrations are optional and
fail gracefully.

**Validates: Requirements REQ-2.1, REQ-8.1, REQ-8.4**
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.discovery.monitor")


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

class JobBoardAdapter:
    """
    Adapter for job board platforms (Upwork, Fiverr) via PlatformBrowser.

    Uses PlatformBrowser + ComputerUseEngine to browse like a human.
    All calls are wrapped in try/except — returns [] on any failure.
    """

    def __init__(self, engine: Any = None):
        """
        Args:
            engine: ComputerUseEngine instance (optional). If None, returns [].
        """
        self._engine = engine
        self._browser: Any = None

        if engine is not None:
            try:
                from janus_platform_browser import PlatformBrowser
                self._browser = PlatformBrowser(engine)
            except Exception as exc:
                logger.warning(f"[JobBoardAdapter] PlatformBrowser unavailable: {exc}")

    async def fetch(self, skills: List[str], platforms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Browse job boards and return raw opportunity dicts.

        Args:
            skills: Skills to search for.
            platforms: Platforms to search (default: upwork + fiverr).

        Returns:
            List of raw opportunity dicts.
        """
        if self._browser is None:
            logger.debug("[JobBoardAdapter] No browser available — skipping job board scan")
            return []

        try:
            jobs = await self._browser.find_jobs(skills, platforms=platforms)
            results = []
            for job in jobs:
                results.append({
                    "id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "platform": job.platform,
                    "url": job.url,
                    "budget": job.budget,
                    "required_skills": job.required_skills,
                    "raw_data": {"browser_job": True, "raw_text": job.raw_text},
                })
            logger.info(f"[JobBoardAdapter] Found {len(results)} jobs")
            return results
        except Exception as exc:
            logger.warning(f"[JobBoardAdapter] fetch failed: {exc}")
            return []


class SocialMediaAdapter:
    """
    Stub adapter for social media opportunity discovery (Twitter, Reddit, Discord).

    Real implementation would use computer use engine to browse social platforms.
    Returns empty list until implemented.
    """

    def __init__(self, engine: Any = None):
        self._engine = engine

    async def fetch(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Scan social media for opportunity signals.

        Args:
            keywords: Keywords/topics to monitor.

        Returns:
            List of raw opportunity dicts (stub: always []).
        """
        try:
            # Stub: real implementation would use computer use engine to
            # browse Twitter/Reddit/Discord for freelance requests, job posts,
            # and collaboration opportunities.
            logger.debug("[SocialMediaAdapter] Social media scan (stub) — returning []")
            return []
        except Exception as exc:
            logger.warning(f"[SocialMediaAdapter] fetch failed: {exc}")
            return []


class GitHubBountiesAdapter:
    """
    Stub adapter for GitHub issues/bounties discovery.

    Real implementation would browse GitHub issues tagged with bounty labels
    or use the GitHub API to find paid issues.
    """

    def __init__(self, engine: Any = None):
        self._engine = engine

    async def fetch(self, topics: List[str]) -> List[Dict[str, Any]]:
        """
        Scan GitHub for bounty-tagged issues.

        Args:
            topics: Topics/languages to search for.

        Returns:
            List of raw opportunity dicts (stub: always []).
        """
        try:
            # Stub: real implementation would query GitHub search API or
            # browse github.com/issues with bounty labels.
            logger.debug("[GitHubBountiesAdapter] GitHub bounties scan (stub) — returning []")
            return []
        except Exception as exc:
            logger.warning(f"[GitHubBountiesAdapter] fetch failed: {exc}")
            return []


class HackathonAdapter:
    """
    Stub adapter for hackathon and competition discovery.

    Real implementation would browse Devpost, HackerEarth, Kaggle, etc.
    """

    def __init__(self, engine: Any = None):
        self._engine = engine

    async def fetch(self, skills: List[str]) -> List[Dict[str, Any]]:
        """
        Discover active hackathons and competitions.

        Args:
            skills: Skills to match against competition requirements.

        Returns:
            List of raw opportunity dicts (stub: always []).
        """
        try:
            # Stub: real implementation would browse Devpost, Kaggle, etc.
            logger.debug("[HackathonAdapter] Hackathon scan (stub) — returning []")
            return []
        except Exception as exc:
            logger.warning(f"[HackathonAdapter] fetch failed: {exc}")
            return []


class ScreenVisionAdapter:
    """
    Stub adapter for screen recorder + vision-based opportunity detection.

    Uses screen recording to observe what's on screen and detect opportunities
    "like a human would notice them" while browsing.
    """

    def __init__(self, screen_recorder: Any = None, vision: Any = None):
        self._recorder = screen_recorder
        self._vision = vision

    async def capture_opportunities(self) -> List[Dict[str, Any]]:
        """
        Capture and analyse current screen for opportunity signals.

        Returns:
            List of raw opportunity dicts (stub: always []).
        """
        try:
            if self._recorder is None or self._vision is None:
                logger.debug("[ScreenVisionAdapter] No recorder/vision — skipping screen scan")
                return []

            # Stub: real implementation would:
            # 1. Capture screenshot via screen recorder
            # 2. Run vision model to detect job listings, notifications, etc.
            # 3. Extract structured opportunity data
            logger.debug("[ScreenVisionAdapter] Screen vision scan (stub) — returning []")
            return []
        except Exception as exc:
            logger.warning(f"[ScreenVisionAdapter] capture failed: {exc}")
            return []


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

class OpportunityMonitor:
    """
    Multi-source opportunity monitor.

    Aggregates opportunities from:
    - Job boards (Upwork/Fiverr via PlatformBrowser)
    - Social media (stub)
    - GitHub bounties (stub)
    - Hackathons (stub)
    - Screen vision (stub)

    All sources are optional — failures return empty lists so the monitor
    always produces a result even when external services are unavailable.

    **Validates: Requirements REQ-2.1, REQ-8.1, REQ-8.4**
    """

    def __init__(
        self,
        engine: Any = None,
        screen_recorder: Any = None,
        vision: Any = None,
        enabled_sources: Optional[List[str]] = None,
    ):
        """
        Args:
            engine: ComputerUseEngine instance (optional).
            screen_recorder: Screen recorder instance (optional).
            vision: Vision model instance (optional).
            enabled_sources: Which sources to enable. Defaults to all.
                Options: "job_boards", "social_media", "github_bounties",
                         "hackathons", "screen_vision".
        """
        self._enabled = set(
            enabled_sources
            if enabled_sources is not None
            else ["job_boards", "social_media", "github_bounties", "hackathons", "screen_vision"]
        )

        self.job_boards = JobBoardAdapter(engine)
        self.social_media = SocialMediaAdapter(engine)
        self.github_bounties = GitHubBountiesAdapter(engine)
        self.hackathons = HackathonAdapter(engine)
        self.screen_vision = ScreenVisionAdapter(screen_recorder, vision)

        self._last_scan: Optional[datetime] = None

    async def scan(
        self,
        skills: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        platforms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan all enabled sources for opportunities.

        Args:
            skills: Skills to search for (used by job boards, hackathons, GitHub).
            keywords: Keywords for social media monitoring.
            platforms: Specific job board platforms to target.

        Returns:
            Deduplicated list of raw opportunity dicts from all sources.
        """
        skills = skills or []
        keywords = keywords or skills  # fall back to skills as keywords
        all_raw: List[Dict[str, Any]] = []

        tasks = []

        if "job_boards" in self._enabled:
            tasks.append(("job_boards", self.job_boards.fetch(skills, platforms)))

        if "social_media" in self._enabled:
            tasks.append(("social_media", self.social_media.fetch(keywords)))

        if "github_bounties" in self._enabled:
            tasks.append(("github_bounties", self.github_bounties.fetch(skills)))

        if "hackathons" in self._enabled:
            tasks.append(("hackathons", self.hackathons.fetch(skills)))

        if "screen_vision" in self._enabled:
            tasks.append(("screen_vision", self.screen_vision.capture_opportunities()))

        # Run all sources concurrently
        if tasks:
            names, coros = zip(*tasks)
            results = await asyncio.gather(*coros, return_exceptions=True)
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    logger.warning(f"[OpportunityMonitor] Source '{name}' raised: {result}")
                elif isinstance(result, list):
                    all_raw.extend(result)
                    logger.debug(f"[OpportunityMonitor] '{name}' returned {len(result)} items")

        self._last_scan = datetime.utcnow()
        logger.info(
            f"[OpportunityMonitor] Scan complete — {len(all_raw)} raw opportunities "
            f"from {len(tasks)} sources"
        )

        return self._deduplicate(all_raw)

    def _deduplicate(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate opportunities by URL or ID."""
        seen: set = set()
        unique = []
        for item in raw:
            key = item.get("url") or item.get("id") or str(item.get("title", ""))
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
            elif not key:
                unique.append(item)
        return unique

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """Timestamp of the most recent scan."""
        return self._last_scan
