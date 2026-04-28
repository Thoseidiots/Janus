"""
Telemetry infrastructure for the Janus Reasoning Engine.

Collects and reports performance metrics, system health, and operational statistics.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TelemetrySnapshot:
    """Snapshot of telemetry data at a point in time."""
    timestamp: datetime
    metrics: Dict[str, float]
    counters: Dict[str, int]
    metadata: Dict[str, Any]


class TelemetryCollector:
    """
    Collects and manages telemetry data for the reasoning engine.
    
    Tracks metrics like:
    - Decision latency
    - Strategy success rates
    - Execution times
    - Memory usage
    - Goal completion rates
    """
    
    def __init__(self, enable_telemetry: bool = True, output_file: Optional[str] = None):
        """
        Initialize telemetry collector.
        
        Args:
            enable_telemetry: Whether to collect telemetry
            output_file: Optional file to write telemetry data
        """
        self.enabled = enable_telemetry
        self.output_file = Path(output_file) if output_file else Path("janus_telemetry.jsonl")
        
        # Metrics storage
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, float] = {}
        
        # Performance tracking
        self.decision_latencies: List[float] = []
        self.strategy_outcomes: Dict[str, List[bool]] = defaultdict(list)
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        
        # System health
        self.last_snapshot_time = datetime.utcnow()
        self.snapshots: List[TelemetrySnapshot] = []
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
        """
        if not self.enabled:
            return
        
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """
        Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        if not self.enabled:
            return
        
        self.counters[name] += amount
    
    def start_timer(self, name: str) -> None:
        """
        Start a timer for measuring duration.
        
        Args:
            name: Timer name
        """
        if not self.enabled:
            return
        
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop a timer and record the duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds, or None if timer not found
        """
        if not self.enabled:
            return None
        
        if name not in self.timers:
            return None
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        self.record_metric(f"{name}_duration", duration)
        return duration
    
    def record_decision_latency(self, latency: float) -> None:
        """
        Record decision-making latency.
        
        Args:
            latency: Latency in seconds
        """
        if not self.enabled:
            return
        
        self.decision_latencies.append(latency)
        self.record_metric("decision_latency", latency)
    
    def record_strategy_outcome(self, strategy_type: str, success: bool) -> None:
        """
        Record strategy execution outcome.
        
        Args:
            strategy_type: Type of strategy
            success: Whether strategy succeeded
        """
        if not self.enabled:
            return
        
        self.strategy_outcomes[strategy_type].append(success)
        self.increment_counter(f"strategy_{strategy_type}_{'success' if success else 'failure'}")
    
    def record_execution_time(self, task_type: str, duration: float) -> None:
        """
        Record task execution time.
        
        Args:
            task_type: Type of task
            duration: Duration in seconds
        """
        if not self.enabled:
            return
        
        self.execution_times[task_type].append(duration)
        self.record_metric(f"execution_time_{task_type}", duration)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current telemetry statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_metrics": len(self.metrics),
            "counters": dict(self.counters),
            "active_timers": len(self.timers),
        }
        
        # Decision latency stats
        if self.decision_latencies:
            stats["decision_latency"] = {
                "mean": sum(self.decision_latencies) / len(self.decision_latencies),
                "min": min(self.decision_latencies),
                "max": max(self.decision_latencies),
                "count": len(self.decision_latencies),
            }
        
        # Strategy success rates
        strategy_stats = {}
        for strategy_type, outcomes in self.strategy_outcomes.items():
            if outcomes:
                success_rate = sum(outcomes) / len(outcomes)
                strategy_stats[strategy_type] = {
                    "success_rate": success_rate,
                    "total_attempts": len(outcomes),
                }
        if strategy_stats:
            stats["strategy_success_rates"] = strategy_stats
        
        # Execution time stats
        execution_stats = {}
        for task_type, times in self.execution_times.items():
            if times:
                execution_stats[task_type] = {
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }
        if execution_stats:
            stats["execution_times"] = execution_stats
        
        return stats
    
    def create_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> TelemetrySnapshot:
        """
        Create a snapshot of current telemetry state.
        
        Args:
            metadata: Optional metadata to include
            
        Returns:
            TelemetrySnapshot
        """
        stats = self.get_statistics()
        
        snapshot = TelemetrySnapshot(
            timestamp=datetime.utcnow(),
            metrics={
                "total_metrics": stats.get("total_metrics", 0),
                "active_timers": stats.get("active_timers", 0),
            },
            counters=stats.get("counters", {}),
            metadata=metadata or {}
        )
        
        self.snapshots.append(snapshot)
        self.last_snapshot_time = snapshot.timestamp
        
        return snapshot
    
    def write_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """
        Write a snapshot to the output file.
        
        Args:
            snapshot: Snapshot to write
        """
        if not self.enabled:
            return
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        record = {
            "timestamp": snapshot.timestamp.isoformat(),
            "metrics": snapshot.metrics,
            "counters": snapshot.counters,
            "metadata": snapshot.metadata,
        }
        
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def flush(self) -> None:
        """Flush all pending telemetry data."""
        if not self.enabled:
            return
        
        snapshot = self.create_snapshot()
        self.write_snapshot(snapshot)
    
    def reset(self) -> None:
        """Reset all telemetry data."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.decision_latencies.clear()
        self.strategy_outcomes.clear()
        self.execution_times.clear()
        self.snapshots.clear()
