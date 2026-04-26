"""
Incremental Analysis Engine for the Janus Dependency Analyzer.

Coordinates re-analysis of changed applications and highlights changes
in analysis reports.

Requirements: 8.3, 8.4, 8.6
"""

from datetime import datetime
from typing import Any, Dict, List

from ..core.models import Application, Capability
from ..catalog.catalog import ApplicationCatalog
from ..priority.engine import AnalysisContext, RankedCapability


class IncrementalAnalysisEngine:
    """
    Coordinates incremental re-analysis of changed applications.

    Requirements: 8.3, 8.4, 8.6
    """

    def __init__(self, capability_analyzer, priority_engine, report_generator):
        """
        Args:
            capability_analyzer: CapabilityAnalyzerImpl instance
            priority_engine: PriorityEngine instance
            report_generator: ReportGenerator instance
        """
        self._capability_analyzer = capability_analyzer
        self._priority_engine = priority_engine
        self._report_generator = report_generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_changed_applications(
        self,
        changed_apps: List[Application],
        catalog: ApplicationCatalog,
    ) -> Dict[str, List[Capability]]:
        """
        Re-analyze capabilities for updated/new applications.

        For each app in changed_apps:
        - Run capability_analyzer.analyze_application(app)
        - Store result in returned dict keyed by app.id
        - Update app.last_analyzed = datetime.now()
        - Update catalog entry via catalog.add(app)

        Returns:
            Dict[str, List[Capability]]: app_id -> list of capabilities
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        results: Dict[str, List[Capability]] = {}
        
        if not changed_apps:
            return results
        
        # Use parallel processing with worker count based on CPU cores
        max_workers = min(multiprocessing.cpu_count() * 2, len(changed_apps))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_app = {
                executor.submit(self._capability_analyzer.analyze_application, app): app
                for app in changed_apps
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_app):
                app = future_to_app[future]
                try:
                    capabilities = future.result(timeout=60)
                    app.last_analyzed = datetime.now()
                    catalog.add(app)
                    results[app.id] = capabilities
                except Exception as exc:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to analyze {app.name}: {exc}")
                    results[app.id] = []

        return results

    def recalculate_priorities(
        self,
        all_capabilities: List[Capability],
        usage_frequencies: Dict[str, int],
    ) -> List[RankedCapability]:
        """
        Recalculate priority scores for new usage patterns.

        Args:
            all_capabilities: All capabilities to rank
            usage_frequencies: Dict mapping capability.id -> usage count

        Returns:
            List[RankedCapability]: Ranked capabilities
        """
        if not all_capabilities:
            return []

        max_frequency = max(usage_frequencies.values(), default=0) if usage_frequencies else 0
        max_frequency = max(max_frequency, 1)

        contexts: Dict[str, AnalysisContext] = {}
        for capability in all_capabilities:
            freq = usage_frequencies.get(capability.id, 0)
            contexts[capability.id] = AnalysisContext(
                usage_frequency=freq,
                max_frequency=max_frequency,
            )

        return self._priority_engine.rank_capabilities(all_capabilities, contexts)

    def generate_change_summary(
        self,
        catalog: ApplicationCatalog,
        since: datetime,
        previous_ranked: List[RankedCapability],
        current_ranked: List[RankedCapability],
    ) -> Dict[str, Any]:
        """
        Highlight changes from previous analysis runs.

        Returns a dict with:
        - recent_changes: List[Dict] — change records since `since`, each as:
            {app_id, app_name, change_type, timestamp (ISO), previous_version, new_version}
        - new_top_capabilities: List[str] — capability names that entered top-10 since previous
        - dropped_capabilities: List[str] — capability names that left top-10 since previous
        - rank_changes: List[Dict] — capabilities whose rank changed, each as:
            {capability_name, previous_rank, current_rank, rank_delta}
        - total_changes: int — total number of change records since `since`
        - generated_at: str — ISO datetime
        """
        # Gather recent change records
        recent_records = catalog.get_recent_changes(since=since)
        recent_changes = [
            {
                "app_id": r.app_id,
                "app_name": r.app_name,
                "change_type": r.change_type,
                "timestamp": r.timestamp.isoformat(),
                "previous_version": r.previous_version,
                "new_version": r.new_version,
            }
            for r in recent_records
        ]

        # Build top-10 name sets
        prev_top10_names = {
            rc.capability.name
            for rc in previous_ranked
            if rc.rank <= 10
        }
        curr_top10_names = {
            rc.capability.name
            for rc in current_ranked
            if rc.rank <= 10
        }

        new_top_capabilities = sorted(curr_top10_names - prev_top10_names)
        dropped_capabilities = sorted(prev_top10_names - curr_top10_names)

        # Build rank-change list for capabilities present in both runs
        prev_rank_by_name: Dict[str, int] = {
            rc.capability.name: rc.rank for rc in previous_ranked
        }
        curr_rank_by_name: Dict[str, int] = {
            rc.capability.name: rc.rank for rc in current_ranked
        }

        rank_changes: List[Dict[str, Any]] = []
        for name, curr_rank in curr_rank_by_name.items():
            if name in prev_rank_by_name:
                prev_rank = prev_rank_by_name[name]
                if prev_rank != curr_rank:
                    rank_changes.append(
                        {
                            "capability_name": name,
                            "previous_rank": prev_rank,
                            "current_rank": curr_rank,
                            "rank_delta": prev_rank - curr_rank,  # positive = moved up
                        }
                    )

        return {
            "recent_changes": recent_changes,
            "new_top_capabilities": new_top_capabilities,
            "dropped_capabilities": dropped_capabilities,
            "rank_changes": rank_changes,
            "total_changes": len(recent_records),
            "generated_at": datetime.now().isoformat(),
        }
