"""
Unit tests for IncrementalAnalysisEngine.

Tests cover:
- analyze_changed_applications with empty list returns empty dict
- analyze_changed_applications calls capability_analyzer for each app
- analyze_changed_applications updates app.last_analyzed
- analyze_changed_applications updates catalog
- recalculate_priorities with empty list returns empty list
- recalculate_priorities returns ranked list sorted by score
- generate_change_summary with no changes returns empty lists
- generate_change_summary detects new top-10 entries
- generate_change_summary detects dropped top-10 entries
- generate_change_summary detects rank changes

Requirements: 8.3, 8.4, 8.6
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, call

import pytest

from ..catalog.catalog import ApplicationCatalog
from ..core.models import (
    Application,
    Capability,
    CapabilityCategory,
    ChangeRecord,
    InterfaceType,
    Platform,
    PriorityScore,
)
from ..incremental.engine import IncrementalAnalysisEngine
from ..priority.engine import RankedCapability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(name: str = "TestApp", version: str = "1.0") -> Application:
    """Create a minimal Application for testing."""
    return Application(name=name, version=version, platform=Platform.LINUX)


def _make_capability(name: str = "cap", app_id: str = "app1") -> Capability:
    """Create a minimal Capability for testing."""
    return Capability(
        name=name,
        application_id=app_id,
        category=CapabilityCategory.FILE_PROCESSING,
        interface_type=InterfaceType.COMMAND_LINE,
        confidence_score=0.8,
    )


def _make_ranked(capability: Capability, rank: int, score: float = 0.5) -> RankedCapability:
    """Create a RankedCapability for testing."""
    ps = PriorityScore(
        usage_frequency=score,
        implementation_complexity=score,
        security_benefit=score,
        performance_impact=score,
        total_score=score,
        justification="test",
    )
    return RankedCapability(capability=capability, priority_score=ps, rank=rank)


def _make_engine() -> tuple:
    """Return (engine, mock_analyzer, mock_priority, mock_report)."""
    mock_analyzer = MagicMock()
    mock_priority = MagicMock()
    mock_report = MagicMock()
    engine = IncrementalAnalysisEngine(mock_analyzer, mock_priority, mock_report)
    return engine, mock_analyzer, mock_priority, mock_report


# ---------------------------------------------------------------------------
# analyze_changed_applications
# ---------------------------------------------------------------------------

class TestAnalyzeChangedApplications:
    def test_empty_list_returns_empty_dict(self):
        engine, mock_analyzer, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        result = engine.analyze_changed_applications([], catalog)
        assert result == {}
        mock_analyzer.analyze_application.assert_not_called()

    def test_calls_capability_analyzer_for_each_app(self):
        engine, mock_analyzer, _, _ = _make_engine()
        mock_analyzer.analyze_application.return_value = []
        catalog = ApplicationCatalog()

        apps = [_make_app("App1"), _make_app("App2"), _make_app("App3")]
        engine.analyze_changed_applications(apps, catalog)

        assert mock_analyzer.analyze_application.call_count == 3
        called_apps = [c.args[0] for c in mock_analyzer.analyze_application.call_args_list]
        assert set(a.name for a in called_apps) == {"App1", "App2", "App3"}

    def test_returns_capabilities_keyed_by_app_id(self):
        engine, mock_analyzer, _, _ = _make_engine()
        app1 = _make_app("App1")
        app2 = _make_app("App2")
        cap1 = _make_capability("cap1", app1.id)
        cap2 = _make_capability("cap2", app2.id)

        mock_analyzer.analyze_application.side_effect = lambda a: (
            [cap1] if a.id == app1.id else [cap2]
        )
        catalog = ApplicationCatalog()
        result = engine.analyze_changed_applications([app1, app2], catalog)

        assert result[app1.id] == [cap1]
        assert result[app2.id] == [cap2]

    def test_updates_app_last_analyzed(self):
        engine, mock_analyzer, _, _ = _make_engine()
        mock_analyzer.analyze_application.return_value = []
        catalog = ApplicationCatalog()

        app = _make_app("App1")
        assert app.last_analyzed is None

        before = datetime.now()
        engine.analyze_changed_applications([app], catalog)
        after = datetime.now()

        assert app.last_analyzed is not None
        assert before <= app.last_analyzed <= after

    def test_updates_catalog_with_each_app(self):
        engine, mock_analyzer, _, _ = _make_engine()
        mock_analyzer.analyze_application.return_value = []
        catalog = ApplicationCatalog()

        app1 = _make_app("App1")
        app2 = _make_app("App2")
        engine.analyze_changed_applications([app1, app2], catalog)

        assert catalog.get(app1.id) is app1
        assert catalog.get(app2.id) is app2

    def test_single_app_result_has_correct_structure(self):
        engine, mock_analyzer, _, _ = _make_engine()
        app = _make_app("SingleApp")
        caps = [_make_capability("read"), _make_capability("write")]
        mock_analyzer.analyze_application.return_value = caps
        catalog = ApplicationCatalog()

        result = engine.analyze_changed_applications([app], catalog)

        assert len(result) == 1
        assert result[app.id] == caps


# ---------------------------------------------------------------------------
# recalculate_priorities
# ---------------------------------------------------------------------------

class TestRecalculatePriorities:
    def test_empty_capabilities_returns_empty_list(self):
        engine, _, mock_priority, _ = _make_engine()
        result = engine.recalculate_priorities([], {})
        assert result == []
        mock_priority.rank_capabilities.assert_not_called()

    def test_delegates_to_priority_engine(self):
        engine, _, mock_priority, _ = _make_engine()
        cap = _make_capability("cap1")
        ranked = [_make_ranked(cap, 1, 0.9)]
        mock_priority.rank_capabilities.return_value = ranked

        result = engine.recalculate_priorities([cap], {cap.id: 10})

        mock_priority.rank_capabilities.assert_called_once()
        assert result == ranked

    def test_returns_ranked_list(self):
        engine, _, mock_priority, _ = _make_engine()
        caps = [_make_capability(f"cap{i}") for i in range(3)]
        ranked = [_make_ranked(c, i + 1, 0.9 - i * 0.1) for i, c in enumerate(caps)]
        mock_priority.rank_capabilities.return_value = ranked

        result = engine.recalculate_priorities(caps, {c.id: 10 - i for i, c in enumerate(caps)})

        assert len(result) == 3
        assert result[0].rank == 1
        assert result[1].rank == 2
        assert result[2].rank == 3

    def test_passes_correct_contexts_to_priority_engine(self):
        engine, _, mock_priority, _ = _make_engine()
        mock_priority.rank_capabilities.return_value = []

        cap1 = _make_capability("cap1")
        cap2 = _make_capability("cap2")
        usage = {cap1.id: 50, cap2.id: 100}

        engine.recalculate_priorities([cap1, cap2], usage)

        _, contexts = mock_priority.rank_capabilities.call_args.args
        assert contexts[cap1.id].usage_frequency == 50
        assert contexts[cap2.id].usage_frequency == 100
        # max_frequency should be 100
        assert contexts[cap1.id].max_frequency == 100
        assert contexts[cap2.id].max_frequency == 100

    def test_handles_empty_usage_frequencies(self):
        engine, _, mock_priority, _ = _make_engine()
        mock_priority.rank_capabilities.return_value = []
        cap = _make_capability("cap1")

        # Should not raise even with no usage data
        engine.recalculate_priorities([cap], {})

        _, contexts = mock_priority.rank_capabilities.call_args.args
        assert contexts[cap.id].usage_frequency == 0
        assert contexts[cap.id].max_frequency == 1  # clamped to at least 1

    def test_capability_not_in_usage_gets_zero_frequency(self):
        engine, _, mock_priority, _ = _make_engine()
        mock_priority.rank_capabilities.return_value = []
        cap1 = _make_capability("cap1")
        cap2 = _make_capability("cap2")

        engine.recalculate_priorities([cap1, cap2], {cap1.id: 5})

        _, contexts = mock_priority.rank_capabilities.call_args.args
        assert contexts[cap2.id].usage_frequency == 0


# ---------------------------------------------------------------------------
# generate_change_summary
# ---------------------------------------------------------------------------

class TestGenerateChangeSummary:
    def test_no_changes_returns_empty_lists(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        summary = engine.generate_change_summary(catalog, since, [], [])

        assert summary["recent_changes"] == []
        assert summary["new_top_capabilities"] == []
        assert summary["dropped_capabilities"] == []
        assert summary["rank_changes"] == []
        assert summary["total_changes"] == 0
        assert "generated_at" in summary

    def test_recent_changes_includes_records_since_cutoff(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime(2024, 1, 1, 12, 0, 0)

        old_record = ChangeRecord(
            app_id="a1", app_name="OldApp", change_type="installed",
            timestamp=since - timedelta(hours=2),
        )
        new_record = ChangeRecord(
            app_id="a2", app_name="NewApp", change_type="installed",
            timestamp=since + timedelta(hours=1),
        )
        catalog.record_change(old_record)
        catalog.record_change(new_record)

        summary = engine.generate_change_summary(catalog, since, [], [])

        assert summary["total_changes"] == 1
        assert len(summary["recent_changes"]) == 1
        assert summary["recent_changes"][0]["app_id"] == "a2"

    def test_recent_changes_have_correct_structure(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime(2024, 1, 1)
        ts = datetime(2024, 6, 1, 10, 0, 0)

        catalog.record_change(ChangeRecord(
            app_id="app1", app_name="MyApp", change_type="updated",
            timestamp=ts, previous_version="1.0", new_version="2.0",
        ))

        summary = engine.generate_change_summary(catalog, since, [], [])

        change = summary["recent_changes"][0]
        assert change["app_id"] == "app1"
        assert change["app_name"] == "MyApp"
        assert change["change_type"] == "updated"
        assert change["timestamp"] == ts.isoformat()
        assert change["previous_version"] == "1.0"
        assert change["new_version"] == "2.0"

    def test_detects_new_top10_entries(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        cap_old = _make_capability("OldCap")
        cap_new = _make_capability("NewCap")

        # Previous: OldCap is rank 1, NewCap is rank 11 (outside top 10)
        previous = [
            _make_ranked(cap_old, 1),
            _make_ranked(cap_new, 11),
        ]
        # Current: both are in top 10
        current = [
            _make_ranked(cap_old, 1),
            _make_ranked(cap_new, 5),
        ]

        summary = engine.generate_change_summary(catalog, since, previous, current)

        assert "NewCap" in summary["new_top_capabilities"]
        assert "OldCap" not in summary["new_top_capabilities"]

    def test_detects_dropped_top10_entries(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        cap_stable = _make_capability("StableCap")
        cap_dropped = _make_capability("DroppedCap")

        # Previous: both in top 10
        previous = [
            _make_ranked(cap_stable, 1),
            _make_ranked(cap_dropped, 3),
        ]
        # Current: DroppedCap fell out of top 10
        current = [
            _make_ranked(cap_stable, 1),
            _make_ranked(cap_dropped, 15),
        ]

        summary = engine.generate_change_summary(catalog, since, previous, current)

        assert "DroppedCap" in summary["dropped_capabilities"]
        assert "StableCap" not in summary["dropped_capabilities"]

    def test_detects_rank_changes(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        cap = _make_capability("MovingCap")

        previous = [_make_ranked(cap, 5)]
        current = [_make_ranked(cap, 2)]

        summary = engine.generate_change_summary(catalog, since, previous, current)

        assert len(summary["rank_changes"]) == 1
        change = summary["rank_changes"][0]
        assert change["capability_name"] == "MovingCap"
        assert change["previous_rank"] == 5
        assert change["current_rank"] == 2
        assert change["rank_delta"] == 3  # moved up by 3

    def test_no_rank_change_when_rank_unchanged(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        cap = _make_capability("StableRankCap")
        previous = [_make_ranked(cap, 3)]
        current = [_make_ranked(cap, 3)]

        summary = engine.generate_change_summary(catalog, since, previous, current)

        assert summary["rank_changes"] == []

    def test_generated_at_is_iso_string(self):
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        summary = engine.generate_change_summary(catalog, since, [], [])

        # Should parse without error
        dt = datetime.fromisoformat(summary["generated_at"])
        assert isinstance(dt, datetime)

    def test_capability_only_in_previous_not_in_rank_changes(self):
        """A capability that disappeared entirely should not appear in rank_changes."""
        engine, _, _, _ = _make_engine()
        catalog = ApplicationCatalog()
        since = datetime.now() - timedelta(hours=1)

        cap_gone = _make_capability("GoneCap")
        cap_new = _make_capability("NewCap")

        previous = [_make_ranked(cap_gone, 1)]
        current = [_make_ranked(cap_new, 1)]

        summary = engine.generate_change_summary(catalog, since, previous, current)

        # GoneCap is not in current, so no rank change entry for it
        names = [rc["capability_name"] for rc in summary["rank_changes"]]
        assert "GoneCap" not in names
