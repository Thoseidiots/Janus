"""
Property-based tests for Incremental Analysis Accuracy.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the incremental analysis components.

# Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..catalog.catalog import ApplicationCatalog
from ..core.models import (
    Application,
    Capability,
    CapabilityCategory,
    ChangeRecord,
    InterfaceType,
    Platform,
)
from ..incremental.engine import IncrementalAnalysisEngine
from ..priority.engine import AnalysisContext, PriorityEngine


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

app_strategy = st.builds(
    Application,
    name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
    version=st.from_regex(r'\d+\.\d+', fullmatch=True),
    platform=st.sampled_from(list(Platform)),
    is_accessible=st.booleans(),
)

change_record_strategy = st.builds(
    ChangeRecord,
    app_id=st.uuids().map(str),
    app_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
    change_type=st.sampled_from(["installed", "removed", "updated", "version_changed"]),
    details=st.text(max_size=50),
)

capability_strategy = st.builds(
    Capability,
    name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-",
        min_size=1,
        max_size=40,
    ),
    category=st.sampled_from(list(CapabilityCategory)),
    description=st.text(min_size=0, max_size=100),
    interface_type=st.sampled_from(list(InterfaceType)),
    parameters=st.just([]),
    supported_formats=st.just([]),
    confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)


# ---------------------------------------------------------------------------
# Property 12: Incremental Analysis Accuracy
# Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
# Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
# ---------------------------------------------------------------------------


class TestIncrementalAnalysisAccuracy:
    """
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**

    Property 12: Incremental Analysis Accuracy
    For any system changes (installations, removals, version updates, usage
    pattern changes), incremental scans SHALL detect all changes, update only
    affected applications, trigger appropriate re-analysis, and highlight
    changes in reports.
    """

    # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy

    @given(records=st.lists(change_record_strategy, min_size=1, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_change_records_stored_and_retrievable(self, records):
        """
        For any list of ChangeRecords, all records are stored and retrievable
        from catalog.

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        catalog = ApplicationCatalog()
        for record in records:
            catalog.record_change(record)
        history = catalog.get_change_history()
        assert len(history) == len(records), (
            f"Expected {len(records)} records in history, got {len(history)}"
        )

    @given(
        records=st.lists(change_record_strategy, min_size=1, max_size=10),
        cutoff_offset_hours=st.integers(min_value=-48, max_value=48),
    )
    @settings(max_examples=25, deadline=None)
    def test_recent_changes_filter_is_monotone(self, records, cutoff_offset_hours):
        """
        get_recent_changes(since=T) returns a subset of get_change_history().

        All records returned by get_recent_changes must also be in
        get_change_history.

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        catalog = ApplicationCatalog()
        for record in records:
            catalog.record_change(record)

        since = datetime.now() + timedelta(hours=cutoff_offset_hours)
        recent = catalog.get_recent_changes(since=since)
        history = catalog.get_change_history()

        # recent must be a subset of history
        history_ids = [(r.app_id, r.app_name, r.change_type) for r in history]
        for r in recent:
            assert (r.app_id, r.app_name, r.change_type) in history_ids, (
                f"Record ({r.app_id}, {r.app_name}, {r.change_type}) in recent_changes "
                f"but not in get_change_history()"
            )

        # recent must not exceed history in size
        assert len(recent) <= len(history), (
            f"get_recent_changes returned {len(recent)} records, "
            f"which exceeds get_change_history() count of {len(history)}"
        )

    @given(apps=st.lists(app_strategy, min_size=1, max_size=10))
    @settings(max_examples=25, deadline=None)
    def test_applications_added_to_catalog_are_retrievable(self, apps):
        """
        For any list of Applications, all added apps are retrievable by ID.

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        catalog = ApplicationCatalog()
        for app in apps:
            catalog.add(app)
        for app in apps:
            assert catalog.get(app.id) is not None, (
                f"Application '{app.name}' (id={app.id}) was added but not retrievable"
            )

    @given(apps=st.lists(app_strategy, min_size=1, max_size=5))
    @settings(max_examples=25, deadline=None)
    def test_incremental_analysis_updates_last_analyzed(self, apps):
        """
        After analyze_changed_applications, all apps have last_analyzed set.

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        mock_capability_analyzer = MagicMock()
        mock_capability_analyzer.analyze_application.return_value = []

        mock_priority_engine = MagicMock()
        mock_report_generator = MagicMock()

        engine = IncrementalAnalysisEngine(
            mock_capability_analyzer,
            mock_priority_engine,
            mock_report_generator,
        )
        catalog = ApplicationCatalog()

        engine.analyze_changed_applications(apps, catalog)

        for app in apps:
            assert app.last_analyzed is not None, (
                f"Application '{app.name}' (id={app.id}) has last_analyzed=None "
                f"after analyze_changed_applications"
            )

    @given(
        caps=st.lists(capability_strategy, min_size=1, max_size=10),
        freqs=st.dictionaries(
            st.uuids().map(str),
            st.integers(min_value=0, max_value=100),
            max_size=10,
        ),
    )
    @settings(max_examples=25, deadline=None)
    def test_priority_recalculation_preserves_count(self, caps, freqs):
        """
        recalculate_priorities returns same number of ranked capabilities as
        input.

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        priority_engine = PriorityEngine()
        mock_capability_analyzer = MagicMock()
        mock_report_generator = MagicMock()

        engine = IncrementalAnalysisEngine(
            mock_capability_analyzer,
            priority_engine,
            mock_report_generator,
        )

        result = engine.recalculate_priorities(caps, freqs)

        assert len(result) == len(caps), (
            f"recalculate_priorities returned {len(result)} ranked capabilities, "
            f"expected {len(caps)}"
        )

    @given(
        records=st.lists(change_record_strategy, min_size=0, max_size=10),
    )
    @settings(max_examples=25, deadline=None)
    def test_change_summary_total_matches_recent_records(self, records):
        """
        generate_change_summary total_changes equals
        len(get_recent_changes(since)).

        # Feature: janus-dependency-analyzer, Property 12: Incremental Analysis Accuracy
        """
        catalog = ApplicationCatalog()
        now = datetime.now()

        # Add all records with timestamp = now so they fall within the window
        for record in records:
            record.timestamp = now
            catalog.record_change(record)

        since = now - timedelta(hours=1)

        mock_capability_analyzer = MagicMock()
        mock_priority_engine = MagicMock()
        mock_report_generator = MagicMock()

        engine = IncrementalAnalysisEngine(
            mock_capability_analyzer,
            mock_priority_engine,
            mock_report_generator,
        )

        summary = engine.generate_change_summary(
            catalog=catalog,
            since=since,
            previous_ranked=[],
            current_ranked=[],
        )

        recent_count = len(catalog.get_recent_changes(since))
        assert summary["total_changes"] == recent_count, (
            f"summary['total_changes'] ({summary['total_changes']}) does not match "
            f"len(get_recent_changes(since)) ({recent_count})"
        )
