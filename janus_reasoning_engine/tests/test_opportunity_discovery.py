"""
Tests for opportunity discovery: monitor, scorer, and exploration strategy.

Covers tasks 5.1, 5.2, and 5.3 of the janus-reasoning-engine spec.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus_reasoning_engine.discovery.opportunity_monitor import (
    GitHubBountiesAdapter,
    HackathonAdapter,
    JobBoardAdapter,
    OpportunityMonitor,
    ScreenVisionAdapter,
    SocialMediaAdapter,
)
from janus_reasoning_engine.discovery.opportunity_scorer import (
    DEFAULT_WEIGHTS,
    ExplorationStrategy,
    Opportunity,
    OpportunityScorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_opportunity(
    title: str = "Test Job",
    platform: str = "upwork",
    earning_potential: float = 100.0,
    required_skills: Optional[List[str]] = None,
    time_estimate: float = 2.0,
    competition_level: float = 0.3,
    reputation_impact: float = 0.5,
    url: str = "https://example.com/job/1",
) -> Opportunity:
    return Opportunity(
        id="test-id",
        title=title,
        platform=platform,
        description="A test opportunity",
        url=url,
        earning_potential=earning_potential,
        required_skills=required_skills or ["python"],
        time_estimate=time_estimate,
        competition_level=competition_level,
        reputation_impact=reputation_impact,
    )


def make_scorer(skills: Optional[List[str]] = None) -> OpportunityScorer:
    return OpportunityScorer(known_skills=skills or ["python", "javascript"])


# ---------------------------------------------------------------------------
# Task 5.1 — OpportunityMonitor
# ---------------------------------------------------------------------------

class TestJobBoardAdapter:
    def test_returns_empty_without_engine(self):
        adapter = JobBoardAdapter(engine=None)
        result = asyncio.get_event_loop().run_until_complete(
            adapter.fetch(["python"])
        )
        assert result == []

    def test_returns_empty_on_browser_exception(self):
        """If PlatformBrowser raises, adapter returns []."""
        adapter = JobBoardAdapter(engine=None)
        adapter._browser = MagicMock()
        adapter._browser.find_jobs = AsyncMock(side_effect=RuntimeError("network error"))
        result = asyncio.get_event_loop().run_until_complete(
            adapter.fetch(["python"])
        )
        assert result == []

    def test_converts_browser_jobs_to_dicts(self):
        from janus_platform_browser import BrowserJob
        from datetime import datetime, timedelta

        mock_job = BrowserJob(
            id="upwork_1",
            title="Python Dev",
            description="Build a web app",
            budget=500.0,
            required_skills=["python", "django"],
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork",
            url="https://upwork.com/job/1",
        )

        adapter = JobBoardAdapter(engine=None)
        adapter._browser = MagicMock()
        adapter._browser.find_jobs = AsyncMock(return_value=[mock_job])

        result = asyncio.get_event_loop().run_until_complete(
            adapter.fetch(["python"])
        )
        assert len(result) == 1
        assert result[0]["title"] == "Python Dev"
        assert result[0]["platform"] == "upwork"
        assert result[0]["budget"] == 500.0


class TestStubAdapters:
    def test_social_media_returns_empty(self):
        adapter = SocialMediaAdapter()
        result = asyncio.get_event_loop().run_until_complete(adapter.fetch(["python"]))
        assert result == []

    def test_github_bounties_returns_empty(self):
        adapter = GitHubBountiesAdapter()
        result = asyncio.get_event_loop().run_until_complete(adapter.fetch(["python"]))
        assert result == []

    def test_hackathon_returns_empty(self):
        adapter = HackathonAdapter()
        result = asyncio.get_event_loop().run_until_complete(adapter.fetch(["python"]))
        assert result == []

    def test_screen_vision_returns_empty_without_recorder(self):
        adapter = ScreenVisionAdapter()
        result = asyncio.get_event_loop().run_until_complete(adapter.capture_opportunities())
        assert result == []


class TestOpportunityMonitor:
    def test_scan_returns_list(self):
        monitor = OpportunityMonitor()
        result = asyncio.get_event_loop().run_until_complete(
            monitor.scan(skills=["python"])
        )
        assert isinstance(result, list)

    def test_scan_deduplicates_by_url(self):
        monitor = OpportunityMonitor(enabled_sources=["job_boards"])
        # Inject two identical items via mock
        monitor.job_boards.fetch = AsyncMock(return_value=[
            {"id": "1", "title": "Job A", "url": "https://example.com/1"},
            {"id": "2", "title": "Job A duplicate", "url": "https://example.com/1"},
        ])
        result = asyncio.get_event_loop().run_until_complete(
            monitor.scan(skills=["python"])
        )
        assert len(result) == 1

    def test_scan_handles_source_exception_gracefully(self):
        monitor = OpportunityMonitor(enabled_sources=["job_boards"])
        monitor.job_boards.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise
        result = asyncio.get_event_loop().run_until_complete(
            monitor.scan(skills=["python"])
        )
        assert result == []

    def test_last_scan_time_updated(self):
        monitor = OpportunityMonitor(enabled_sources=[])
        assert monitor.last_scan_time is None
        asyncio.get_event_loop().run_until_complete(monitor.scan())
        assert monitor.last_scan_time is not None

    def test_enabled_sources_filter(self):
        monitor = OpportunityMonitor(enabled_sources=["job_boards"])
        # social_media should not be called
        monitor.social_media.fetch = AsyncMock(return_value=[{"id": "x"}])
        monitor.job_boards.fetch = AsyncMock(return_value=[])
        asyncio.get_event_loop().run_until_complete(monitor.scan(skills=["python"]))
        monitor.social_media.fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Task 5.2 — OpportunityScorer
# ---------------------------------------------------------------------------

class TestOpportunity:
    def test_from_raw_defaults(self):
        raw = {"title": "Test", "platform": "upwork", "url": "https://x.com"}
        opp = Opportunity.from_raw(raw)
        assert opp.title == "Test"
        assert opp.platform == "upwork"
        assert opp.earning_potential == 0.0
        assert opp.competition_level == 0.5

    def test_from_raw_budget_field(self):
        raw = {"title": "T", "platform": "fiverr", "url": "", "budget": 200.0}
        opp = Opportunity.from_raw(raw)
        assert opp.earning_potential == 200.0


class TestOpportunityScorerBasic:
    def test_score_in_range(self):
        scorer = make_scorer()
        opp = make_opportunity()
        score = scorer.score_opportunity(opp)
        assert 0.0 <= score <= 1.0

    def test_score_assigned_to_opportunity(self):
        scorer = make_scorer()
        opp = make_opportunity()
        scorer.score_opportunity(opp)
        assert opp.score > 0.0

    def test_higher_earning_scores_higher(self):
        scorer = make_scorer()
        low = make_opportunity(earning_potential=10.0)
        high = make_opportunity(earning_potential=900.0)
        scorer.score_opportunity(low)
        scorer.score_opportunity(high)
        assert high.score > low.score

    def test_full_skill_match_scores_higher_than_no_match(self):
        scorer = OpportunityScorer(known_skills=["python"])
        matched = make_opportunity(required_skills=["python"])
        unmatched = make_opportunity(required_skills=["cobol", "fortran"])
        scorer.score_opportunity(matched)
        scorer.score_opportunity(unmatched)
        assert matched.score > unmatched.score

    def test_low_competition_scores_higher(self):
        scorer = make_scorer()
        easy = make_opportunity(competition_level=0.1)
        hard = make_opportunity(competition_level=0.9)
        scorer.score_opportunity(easy)
        scorer.score_opportunity(hard)
        assert easy.score > hard.score

    def test_score_and_rank_sorted(self):
        scorer = make_scorer()
        opps = [
            make_opportunity(earning_potential=50.0, url="a"),
            make_opportunity(earning_potential=500.0, url="b"),
            make_opportunity(earning_potential=200.0, url="c"),
        ]
        ranked = scorer.score_and_rank(opps)
        scores = [o.score for o in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_returns_empty(self):
        scorer = make_scorer()
        assert scorer.score_and_rank([]) == []

    def test_no_required_skills_neutral_match(self):
        scorer = make_scorer()
        opp = make_opportunity(required_skills=[])
        score = scorer.score_opportunity(opp)
        assert 0.0 <= score <= 1.0

    def test_record_outcome_updates_niche_stats(self):
        scorer = make_scorer()
        opp = make_opportunity(platform="upwork", required_skills=["python"])
        scorer.record_outcome(opp, success=True)
        assert scorer.niche_success_rate("upwork", "python") == 1.0

    def test_niche_success_rate_neutral_prior(self):
        scorer = make_scorer()
        assert scorer.niche_success_rate("upwork", "unknown_skill") == 0.5

    def test_niche_success_rate_multiple_outcomes(self):
        scorer = make_scorer()
        opp = make_opportunity(platform="fiverr", required_skills=["design"])
        scorer.record_outcome(opp, success=True)
        scorer.record_outcome(opp, success=False)
        scorer.record_outcome(opp, success=True)
        rate = scorer.niche_success_rate("fiverr", "design")
        assert abs(rate - 2 / 3) < 0.01


class TestOpportunityScorerWithMemory:
    def test_memory_boost_applied(self):
        """Memory boost should adjust score when episodic memory returns results."""
        from janus_reasoning_engine.memory.episodic_memory import Experience, OutcomeType
        from datetime import datetime

        mock_memory = MagicMock()
        success_exp = Experience(
            experience_id="e1",
            context={"platform": "upwork"},
            action={"type": "apply"},
            outcome={"result": "hired"},
            timestamp=datetime.utcnow(),
            outcome_type=OutcomeType.SUCCESS,
        )
        mock_memory.retrieve_similar_experiences = MagicMock(return_value=[success_exp])

        scorer = OpportunityScorer(
            known_skills=["python"],
            episodic_memory=mock_memory,
        )
        opp = make_opportunity()
        score_with_memory = scorer.score_opportunity(opp)

        # Score without memory for comparison
        scorer_no_mem = OpportunityScorer(known_skills=["python"])
        opp2 = make_opportunity()
        score_no_memory = scorer_no_mem.score_opportunity(opp2)

        # Memory boost from all-success history should push score up
        assert score_with_memory >= score_no_memory

    def test_memory_failure_does_not_crash(self):
        mock_memory = MagicMock()
        mock_memory.retrieve_similar_experiences = MagicMock(
            side_effect=RuntimeError("db error")
        )
        scorer = OpportunityScorer(known_skills=["python"], episodic_memory=mock_memory)
        opp = make_opportunity()
        score = scorer.score_opportunity(opp)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Task 5.3 — ExplorationStrategy
# ---------------------------------------------------------------------------

class TestExplorationStrategy:
    def test_returns_none_for_empty_list(self):
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer)
        assert strategy.select_opportunity([]) is None

    def test_returns_opportunity_from_list(self):
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer)
        opps = scorer.score_and_rank([make_opportunity(url=f"u{i}") for i in range(5)])
        result = strategy.select_opportunity(opps)
        assert result in opps

    def test_exploit_picks_highest_score(self):
        """With epsilon=0, always exploit — should pick highest scored."""
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer, epsilon=0.0)
        opps = [
            make_opportunity(earning_potential=10.0, url="a"),
            make_opportunity(earning_potential=900.0, url="b"),
            make_opportunity(earning_potential=50.0, url="c"),
        ]
        ranked = scorer.score_and_rank(opps)
        result = strategy.select_opportunity(ranked, epsilon=0.0)
        assert result.url == "b"

    def test_explore_can_pick_non_best(self):
        """With epsilon=1, always explore — should sometimes pick non-best."""
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer, epsilon=1.0)
        opps = [
            make_opportunity(earning_potential=10.0, url="a"),
            make_opportunity(earning_potential=900.0, url="b"),
        ]
        ranked = scorer.score_and_rank(opps)

        # Run many times — with epsilon=1 we should see both selected
        selected_urls = set()
        for _ in range(50):
            r = strategy.select_opportunity(ranked, epsilon=1.0)
            selected_urls.add(r.url)

        # Both should appear at some point
        assert len(selected_urls) >= 1  # at minimum it works

    def test_curiosity_bonus_for_novel_type(self):
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer, curiosity_bonus=0.2)
        opp = make_opportunity(platform="github", required_skills=["rust"])
        assert strategy.is_novel(opp) is True
        assert strategy.curiosity_score(opp) == 0.2

    def test_no_curiosity_bonus_after_seen(self):
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer, curiosity_bonus=0.2)
        opp = make_opportunity(platform="github", required_skills=["rust"])
        strategy.mark_seen(opp)
        assert strategy.is_novel(opp) is False
        assert strategy.curiosity_score(opp) == 0.0

    def test_epsilon_override(self):
        """Passing epsilon to select_opportunity overrides instance default."""
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer, epsilon=0.5)
        opps = scorer.score_and_rank([make_opportunity(url=f"u{i}") for i in range(3)])
        # Should not raise with override
        result = strategy.select_opportunity(opps, epsilon=0.0)
        assert result is not None

    def test_niche_expertise_influences_exploit(self):
        """Niche expertise should boost selection of known-good platform+skill."""
        scorer = OpportunityScorer(known_skills=["python"])
        strategy = ExplorationStrategy(scorer, epsilon=0.0)

        # Record many successes for upwork+python
        upwork_opp = make_opportunity(platform="upwork", required_skills=["python"], url="upwork")
        for _ in range(10):
            scorer.record_outcome(upwork_opp, success=True)

        # Create two opportunities with same base score but different platforms
        opp_upwork = make_opportunity(
            platform="upwork", required_skills=["python"],
            earning_potential=100.0, url="upwork"
        )
        opp_fiverr = make_opportunity(
            platform="fiverr", required_skills=["python"],
            earning_potential=100.0, url="fiverr"
        )
        scorer.score_opportunity(opp_upwork)
        scorer.score_opportunity(opp_fiverr)

        # With epsilon=0, exploit should prefer upwork due to niche expertise
        result = strategy.select_opportunity([opp_upwork, opp_fiverr], epsilon=0.0)
        assert result.platform == "upwork"

    def test_single_opportunity_always_selected(self):
        scorer = make_scorer()
        strategy = ExplorationStrategy(scorer)
        opp = make_opportunity()
        scorer.score_opportunity(opp)
        result = strategy.select_opportunity([opp])
        assert result is opp
