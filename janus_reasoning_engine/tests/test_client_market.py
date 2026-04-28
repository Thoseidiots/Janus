"""
Tests for the market module: ClientManager and MarketplaceHub.

Covers tasks 13.1, 13.2, 13.3.
"""

from __future__ import annotations

import pytest

from janus_reasoning_engine.market.client_manager import ClientLead, ClientManager
from janus_reasoning_engine.market.marketplace_hub import MarketplaceHub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return ClientManager()


@pytest.fixture
def hub():
    return MarketplaceHub()


@pytest.fixture
def sample_lead():
    return ClientLead(
        id="lead-1",
        name="Acme Corp",
        email="contact@acme.com",
        platform="upwork",
        skills_needed=["python", "ml", "api"],
        budget=500.0,
    )


# ---------------------------------------------------------------------------
# Task 13.1 — ClientManager: find_clients, qualify_lead, conversion tracking
# ---------------------------------------------------------------------------

class TestClientManagerAcquisition:
    def test_find_clients_returns_list(self, manager):
        result = manager.find_clients(skills=["python"], platforms=["upwork"])
        assert isinstance(result, list)

    def test_find_clients_stub_returns_empty(self, manager):
        # Without the optional backend, stub returns []
        result = manager.find_clients(skills=["python"], platforms=["upwork"])
        assert result == []

    def test_qualify_lead_returns_float(self, manager, sample_lead):
        score = manager.qualify_lead(sample_lead)
        assert isinstance(score, float)

    def test_qualify_lead_range(self, manager, sample_lead):
        score = manager.qualify_lead(sample_lead)
        assert 0.0 <= score <= 1.0

    def test_qualify_lead_updates_lead_score(self, manager, sample_lead):
        score = manager.qualify_lead(sample_lead)
        assert sample_lead.score == score

    def test_qualify_lead_zero_budget(self, manager):
        lead = ClientLead(
            id="l0", name="X", email="x@x.com",
            platform="fiverr", skills_needed=[], budget=0.0,
        )
        score = manager.qualify_lead(lead)
        assert score == pytest.approx(0.0)

    def test_qualify_lead_high_budget_high_skills(self, manager):
        lead = ClientLead(
            id="l1", name="BigCo", email="big@co.com",
            platform="upwork", skills_needed=["a", "b", "c", "d", "e"],
            budget=1000.0,
        )
        score = manager.qualify_lead(lead)
        assert score == pytest.approx(1.0)

    def test_track_conversion_and_rate_all_converted(self, manager):
        manager.track_conversion("lead-1", True)
        manager.track_conversion("lead-2", True)
        assert manager.get_conversion_rate() == pytest.approx(1.0)

    def test_track_conversion_and_rate_none_converted(self, manager):
        manager.track_conversion("lead-1", False)
        manager.track_conversion("lead-2", False)
        assert manager.get_conversion_rate() == pytest.approx(0.0)

    def test_track_conversion_mixed_rate(self, manager):
        manager.track_conversion("lead-1", True)
        manager.track_conversion("lead-2", False)
        assert manager.get_conversion_rate() == pytest.approx(0.5)

    def test_get_conversion_rate_empty(self, manager):
        assert manager.get_conversion_rate() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task 13.3 — ClientManager: performance incentives
# ---------------------------------------------------------------------------

class TestClientManagerPerformance:
    def test_speed_bonus_ahead_of_schedule(self, manager):
        bonus = manager.calculate_speed_bonus("job-1", completed_hours=4.0, estimated_hours=8.0)
        # (1 - 4/8) * 0.2 = 0.1
        assert bonus == pytest.approx(0.1)

    def test_speed_bonus_on_time(self, manager):
        bonus = manager.calculate_speed_bonus("job-2", completed_hours=8.0, estimated_hours=8.0)
        assert bonus == pytest.approx(0.0)

    def test_speed_bonus_over_time(self, manager):
        bonus = manager.calculate_speed_bonus("job-3", completed_hours=10.0, estimated_hours=8.0)
        assert bonus == pytest.approx(0.0)

    def test_speed_bonus_max_value(self, manager):
        # Completed in near-zero time → bonus approaches 0.2
        bonus = manager.calculate_speed_bonus("job-4", completed_hours=0.001, estimated_hours=100.0)
        assert bonus < 0.2
        assert bonus > 0.19

    def test_speed_bonus_zero_estimated_hours(self, manager):
        bonus = manager.calculate_speed_bonus("job-5", completed_hours=1.0, estimated_hours=0.0)
        assert bonus == pytest.approx(0.0)

    def test_track_quality_score_stored(self, manager):
        manager.track_quality_score("job-1", 8.5)
        metrics = manager.get_performance_metrics()
        assert metrics["avg_quality"] == pytest.approx(8.5)

    def test_track_quality_multiple_scores(self, manager):
        manager.track_quality_score("job-1", 6.0)
        manager.track_quality_score("job-1", 8.0)
        metrics = manager.get_performance_metrics()
        assert metrics["avg_quality"] == pytest.approx(7.0)

    def test_get_performance_metrics_structure(self, manager):
        metrics = manager.get_performance_metrics()
        assert "avg_quality" in metrics
        assert "avg_speed_bonus" in metrics
        assert "total_jobs" in metrics

    def test_get_performance_metrics_empty(self, manager):
        metrics = manager.get_performance_metrics()
        assert metrics["avg_quality"] == pytest.approx(0.0)
        assert metrics["avg_speed_bonus"] == pytest.approx(0.0)
        assert metrics["total_jobs"] == 0

    def test_get_performance_metrics_total_jobs(self, manager):
        manager.calculate_speed_bonus("job-a", 3.0, 5.0)
        manager.track_quality_score("job-b", 9.0)
        metrics = manager.get_performance_metrics()
        assert metrics["total_jobs"] == 2

    def test_get_performance_metrics_avg_speed_bonus(self, manager):
        manager.calculate_speed_bonus("job-1", 4.0, 8.0)   # bonus = 0.1
        manager.calculate_speed_bonus("job-2", 2.0, 8.0)   # bonus = 0.15
        metrics = manager.get_performance_metrics()
        assert metrics["avg_speed_bonus"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# Task 13.2 — MarketplaceHub
# ---------------------------------------------------------------------------

class TestMarketplaceHub:
    def test_post_job_returns_string_id(self, hub):
        job_id = hub.post_job("Build API", "REST API in Python", 500.0)
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_post_job_ids_are_unique(self, hub):
        id1 = hub.post_job("Job A", "desc", 100.0)
        id2 = hub.post_job("Job B", "desc", 200.0)
        assert id1 != id2

    def test_get_active_listings_after_post(self, hub):
        hub.post_job("ML Model", "Train a classifier", 800.0)
        listings = hub.get_active_listings()
        assert len(listings) >= 1

    def test_get_active_listings_contains_posted_job(self, hub):
        job_id = hub.post_job("Data Pipeline", "ETL pipeline", 300.0)
        listings = hub.get_active_listings()
        ids = [l["id"] for l in listings]
        assert job_id in ids

    def test_get_active_listings_empty_initially(self, hub):
        listings = hub.get_active_listings()
        assert isinstance(listings, list)

    def test_browse_jobs_returns_list(self, hub):
        result = hub.browse_jobs(skills=["python"], platforms=["upwork"])
        assert isinstance(result, list)

    def test_browse_jobs_stub_returns_empty(self, hub):
        # Without optional browser backend, returns []
        result = hub.browse_jobs(skills=["python"], platforms=["upwork"])
        assert result == []

    def test_post_job_listing_has_required_fields(self, hub):
        job_id = hub.post_job("Test Job", "Description here", 250.0)
        listings = hub.get_active_listings()
        job = next((l for l in listings if l["id"] == job_id), None)
        assert job is not None
        assert job["title"] == "Test Job"
        assert job["budget"] == 250.0
        assert job["status"] == "active"
