"""
janus_monitoring_dashboard.py
==============================
Real-time performance monitoring dashboard for Janus detection systems.

Tracks and visualizes:
- Binary decision statistics
- Loop detection patterns
- Ghost code health
- System performance metrics

Usage:
    from janus_monitoring_dashboard import JanusMonitor

    monitor = JanusMonitor()
    monitor.start()

    # In your code
    monitor.log_decision(result)
    monitor.log_loop(result)
    monitor.log_ghost_check(report)

    # View dashboard
    monitor.print_dashboard()
    monitor.export_metrics("metrics.json")
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import statistics


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionMetrics:
    """Metrics for binary decision system."""
    total_decisions: int = 0
    halt_decisions: int = 0
    loop_decisions: int = 0
    avg_confidence: float = 0.0
    avg_decision_time: float = 0.0
    confidence_history: List[float] = field(default_factory=lambda: deque(maxlen=100))
    decision_times: List[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def halt_rate(self) -> float:
        return self.halt_decisions / max(self.total_decisions, 1)

    @property
    def loop_rate(self) -> float:
        return self.loop_decisions / max(self.total_decisions, 1)


@dataclass
class LoopMetrics:
    """Metrics for loop detection system."""
    total_actions: int = 0
    detected_loops: int = 0
    prevented_loops: int = 0
    avg_similarity: float = 0.0
    most_common_patterns: List[str] = field(default_factory=list)
    pattern_history: List[str] = field(default_factory=lambda: deque(maxlen=50))

    @property
    def loop_rate(self) -> float:
        return self.detected_loops / max(self.total_actions, 1)

    @property
    def prevention_rate(self) -> float:
        return self.prevented_loops / max(self.detected_loops, 1)


@dataclass
class GhostMetrics:
    """Metrics for ghost code detection system."""
    total_checks: int = 0
    healthy_checks: int = 0
    partial_ghost_checks: int = 0
    full_ghost_checks: int = 0
    avg_confidence: float = 0.0
    ghost_components: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def health_rate(self) -> float:
        return self.healthy_checks / max(self.total_checks, 1)

    @property
    def ghost_rate(self) -> float:
        return (self.partial_ghost_checks + self.full_ghost_checks) / max(self.total_checks, 1)


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    uptime_seconds: float = 0.0
    total_operations: int = 0
    avg_operation_time: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    operation_times: List[float] = field(default_factory=lambda: deque(maxlen=1000))


# ─────────────────────────────────────────────────────────────────────────────
# Monitor Class
# ─────────────────────────────────────────────────────────────────────────────

class JanusMonitor:
    """
    Centralized monitoring system for all Janus detection components.

    Collects metrics, generates reports, and provides real-time dashboard.
    """

    def __init__(self):
        self.decision_metrics = DecisionMetrics()
        self.loop_metrics = LoopMetrics()
        self.ghost_metrics = GhostMetrics()
        self.system_metrics = SystemMetrics()

        self.start_time = time.time()
        self.last_update = self.start_time

        # Event log
        self.event_log: deque = deque(maxlen=1000)

        print("[Monitor] Janus monitoring system initialized")

    # ── Decision Logging ──────────────────────────────────────────────────────

    def log_decision(
        self,
        decision: bool,
        confidence: float,
        decision_time: float,
        reasoning: str = "",
    ):
        """Log a binary decision event."""
        self.decision_metrics.total_decisions += 1

        if decision:
            self.decision_metrics.halt_decisions += 1
        else:
            self.decision_metrics.loop_decisions += 1

        self.decision_metrics.confidence_history.append(confidence)
        self.decision_metrics.decision_times.append(decision_time)

        if self.decision_metrics.confidence_history:
            self.decision_metrics.avg_confidence = statistics.mean(
                self.decision_metrics.confidence_history
            )

        if self.decision_metrics.decision_times:
            self.decision_metrics.avg_decision_time = statistics.mean(
                self.decision_metrics.decision_times
            )

        self._log_event("decision", {
            "decision": "HALT" if decision else "LOOP",
            "confidence": confidence,
            "time_ms": decision_time * 1000,
            "reasoning": reasoning,
        })

    # ── Loop Detection Logging ────────────────────────────────────────────────

    def log_loop(
        self,
        is_loop: bool,
        similarity: float,
        pattern: Optional[str] = None,
        prevented: bool = False,
    ):
        """Log a loop detection event."""
        self.loop_metrics.total_actions += 1

        if is_loop:
            self.loop_metrics.detected_loops += 1

            if prevented:
                self.loop_metrics.prevented_loops += 1

            if pattern:
                self.loop_metrics.pattern_history.append(pattern)

        # Update average similarity (weighted moving average)
        alpha = 0.1
        self.loop_metrics.avg_similarity = (
            alpha * similarity +
            (1 - alpha) * self.loop_metrics.avg_similarity
        )

        self._log_event("loop", {
            "detected": is_loop,
            "prevented": prevented,
            "similarity": similarity,
            "pattern": pattern,
        })

    # ── Ghost Code Logging ────────────────────────────────────────────────────

    def log_ghost_check(
        self,
        component_name: str,
        confidence: float,
        status: str,
        ghost_issues: List[str],
    ):
        """Log a ghost code detection check."""
        self.ghost_metrics.total_checks += 1

        if status == "HEALTHY":
            self.ghost_metrics.healthy_checks += 1
        elif status == "PARTIAL_GHOST":
            self.ghost_metrics.partial_ghost_checks += 1
        elif status == "FULL_GHOST":
            self.ghost_metrics.full_ghost_checks += 1
            if component_name not in self.ghost_metrics.ghost_components:
                self.ghost_metrics.ghost_components.append(component_name)

        self.ghost_metrics.confidence_history.append(confidence)

        if self.ghost_metrics.confidence_history:
            self.ghost_metrics.avg_confidence = statistics.mean(
                self.ghost_metrics.confidence_history
            )

        self._log_event("ghost", {
            "component": component_name,
            "status": status,
            "confidence": confidence,
            "issues": len(ghost_issues),
        })

    # ── System Metrics ────────────────────────────────────────────────────────

    def log_operation(self, operation_time: float):
        """Log a general operation timing."""
        self.system_metrics.total_operations += 1
        self.system_metrics.operation_times.append(operation_time)

        if self.system_metrics.operation_times:
            self.system_metrics.avg_operation_time = statistics.mean(
                self.system_metrics.operation_times
            )

    def update_system_stats(self, memory_mb: float = 0.0, cpu_percent: float = 0.0):
        """Update system resource usage stats."""
        self.system_metrics.uptime_seconds = time.time() - self.start_time

        if memory_mb > self.system_metrics.peak_memory_mb:
            self.system_metrics.peak_memory_mb = memory_mb

        self.system_metrics.cpu_usage_percent = cpu_percent

    # ── Event Logging ─────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the event log."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        }
        self.event_log.append(event)

    # ── Dashboard Display ─────────────────────────────────────────────────────

    def print_dashboard(self, detailed: bool = False):
        """Print formatted monitoring dashboard."""
        current_time = time.time()
        uptime = current_time - self.start_time

        print("\n" + "=" * 80)
        print("JANUS DETECTION SYSTEMS - MONITORING DASHBOARD")
        print("=" * 80)

        # System Overview
        print(f"\n📊 SYSTEM OVERVIEW")
        print(f"  Uptime: {self._format_duration(uptime)}")
        print(f"  Total Operations: {self.system_metrics.total_operations}")
        print(f"  Avg Operation Time: {self.system_metrics.avg_operation_time * 1000:.2f}ms")
        if self.system_metrics.peak_memory_mb > 0:
            print(f"  Peak Memory: {self.system_metrics.peak_memory_mb:.1f} MB")
        if self.system_metrics.cpu_usage_percent > 0:
            print(f"  CPU Usage: {self.system_metrics.cpu_usage_percent:.1f}%")

        # Binary Decider Stats
        print(f"\n🎯 BINARY DECIDER")
        print(f"  Total Decisions: {self.decision_metrics.total_decisions}")
        print(f"  Halt Rate: {self.decision_metrics.halt_rate:.1%}")
        print(f"  Loop Rate: {self.decision_metrics.loop_rate:.1%}")
        print(f"  Avg Confidence: {self.decision_metrics.avg_confidence:.1%}")
        print(f"  Avg Decision Time: {self.decision_metrics.avg_decision_time * 1000:.2f}ms")

        if detailed and self.decision_metrics.confidence_history:
            conf_list = list(self.decision_metrics.confidence_history)
            print(f"  Confidence Range: {min(conf_list):.2f} - {max(conf_list):.2f}")
            print(f"  Confidence StdDev: {statistics.stdev(conf_list) if len(conf_list) > 1 else 0:.3f}")

        # Loop Detector Stats
        print(f"\n🔄 LOOP DETECTOR")
        print(f"  Total Actions: {self.loop_metrics.total_actions}")
        print(f"  Detected Loops: {self.loop_metrics.detected_loops}")
        print(f"  Loop Rate: {self.loop_metrics.loop_rate:.1%}")
        print(f"  Prevented Loops: {self.loop_metrics.prevented_loops}")
        print(f"  Prevention Rate: {self.loop_metrics.prevention_rate:.1%}")
        print(f"  Avg Similarity: {self.loop_metrics.avg_similarity:.2f}")

        if detailed and self.loop_metrics.pattern_history:
            patterns = list(self.loop_metrics.pattern_history)
            print(f"  Recent Patterns: {len(patterns)}")

        # Ghost Code Stats
        print(f"\n👻 GHOST CODE DETECTOR")
        print(f"  Total Checks: {self.ghost_metrics.total_checks}")
        print(f"  Healthy: {self.ghost_metrics.healthy_checks} ({self.ghost_metrics.health_rate:.1%})")
        print(f"  Partial Ghost: {self.ghost_metrics.partial_ghost_checks}")
        print(f"  Full Ghost: {self.ghost_metrics.full_ghost_checks}")
        print(f"  Ghost Rate: {self.ghost_metrics.ghost_rate:.1%}")
        print(f"  Avg Confidence: {self.ghost_metrics.avg_confidence:.1%}")

        if self.ghost_metrics.ghost_components:
            print(f"  Ghost Components: {', '.join(self.ghost_metrics.ghost_components[:5])}")

        # Recent Events
        if detailed and self.event_log:
            print(f"\n📋 RECENT EVENTS (Last 5)")
            for event in list(self.event_log)[-5:]:
                elapsed = current_time - event["timestamp"]
                print(f"  [{self._format_duration(elapsed)} ago] {event['type'].upper()}")

        print("\n" + "=" * 80)

    def print_compact_status(self):
        """Print single-line compact status."""
        ops = self.system_metrics.total_operations
        dec = self.decision_metrics.total_decisions
        loops = self.loop_metrics.detected_loops
        ghosts = self.ghost_metrics.full_ghost_checks

        status_line = (
            f"[Janus Monitor] "
            f"Ops: {ops} | "
            f"Decisions: {dec} (H:{self.decision_metrics.halt_rate:.0%}) | "
            f"Loops: {loops} | "
            f"Ghosts: {ghosts}"
        )

        print(status_line)

    # ── Export & Analysis ─────────────────────────────────────────────────────

    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file."""
        data = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "decision_metrics": {
                "total_decisions": self.decision_metrics.total_decisions,
                "halt_rate": self.decision_metrics.halt_rate,
                "loop_rate": self.decision_metrics.loop_rate,
                "avg_confidence": self.decision_metrics.avg_confidence,
                "avg_decision_time": self.decision_metrics.avg_decision_time,
            },
            "loop_metrics": {
                "total_actions": self.loop_metrics.total_actions,
                "detected_loops": self.loop_metrics.detected_loops,
                "loop_rate": self.loop_metrics.loop_rate,
                "prevented_loops": self.loop_metrics.prevented_loops,
                "avg_similarity": self.loop_metrics.avg_similarity,
            },
            "ghost_metrics": {
                "total_checks": self.ghost_metrics.total_checks,
                "health_rate": self.ghost_metrics.health_rate,
                "ghost_rate": self.ghost_metrics.ghost_rate,
                "avg_confidence": self.ghost_metrics.avg_confidence,
                "ghost_components": self.ghost_metrics.ghost_components,
            },
            "system_metrics": {
                "total_operations": self.system_metrics.total_operations,
                "avg_operation_time": self.system_metrics.avg_operation_time,
                "peak_memory_mb": self.system_metrics.peak_memory_mb,
                "cpu_usage_percent": self.system_metrics.cpu_usage_percent,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Monitor] Metrics exported to {filepath}")

    def get_health_score(self) -> float:
        """
        Calculate overall system health score (0.0 to 1.0).

        Takes into account:
        - High halt rate (good)
        - Low loop rate (good)
        - High ghost health rate (good)
        - High average confidence (good)
        """
        if self.system_metrics.total_operations == 0:
            return 1.0  # No issues if no operations

        # Component scores
        halt_score = self.decision_metrics.halt_rate
        loop_health_score = 1.0 - self.loop_metrics.loop_rate
        ghost_health_score = self.ghost_metrics.health_rate
        confidence_score = self.decision_metrics.avg_confidence

        # Weighted average
        health_score = (
            halt_score * 0.3 +
            loop_health_score * 0.3 +
            ghost_health_score * 0.3 +
            confidence_score * 0.1
        )

        return health_score

    def get_alerts(self) -> List[str]:
        """Get list of current system alerts."""
        alerts = []

        # High loop rate
        if self.loop_metrics.loop_rate > 0.3:
            alerts.append(
                f"⚠️ HIGH LOOP RATE: {self.loop_metrics.loop_rate:.1%} "
                f"({self.loop_metrics.detected_loops} loops detected)"
            )

        # High ghost rate
        if self.ghost_metrics.ghost_rate > 0.3:
            alerts.append(
                f"⚠️ HIGH GHOST RATE: {self.ghost_metrics.ghost_rate:.1%} "
                f"({self.ghost_metrics.full_ghost_checks} full ghosts)"
            )

        # Low confidence
        if self.decision_metrics.avg_confidence < 0.5:
            alerts.append(
                f"⚠️ LOW DECISION CONFIDENCE: {self.decision_metrics.avg_confidence:.1%}"
            )

        # Slow operations
        if self.system_metrics.avg_operation_time > 0.1:  # > 100ms
            alerts.append(
                f"⚠️ SLOW OPERATIONS: {self.system_metrics.avg_operation_time * 1000:.1f}ms avg"
            )

        return alerts

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def reset(self):
        """Reset all metrics."""
        self.decision_metrics = DecisionMetrics()
        self.loop_metrics = LoopMetrics()
        self.ghost_metrics = GhostMetrics()
        self.system_metrics = SystemMetrics()
        self.start_time = time.time()
        self.event_log.clear()
        print("[Monitor] All metrics reset")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_monitor: Optional[JanusMonitor] = None


def get_monitor() -> JanusMonitor:
    """Get the shared JanusMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = JanusMonitor()
    return _monitor


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("Initializing Janus Monitor Demo...")
    monitor = get_monitor()

    print("\nSimulating detection events...")

    # Simulate 50 operations
    for i in range(50):
        # Decision events
        decision = random.random() > 0.3  # 70% halt
        confidence = random.uniform(0.6, 0.95)
        decision_time = random.uniform(0.005, 0.020)
        monitor.log_decision(decision, confidence, decision_time)

        # Loop events (20% chance)
        if random.random() < 0.2:
            is_loop = True
            similarity = random.uniform(0.8, 0.95)
            prevented = random.random() > 0.5
            monitor.log_loop(is_loop, similarity, f"pattern_{i}", prevented)
        else:
            monitor.log_loop(False, random.uniform(0.4, 0.7))

        # Ghost checks (30% chance)
        if random.random() < 0.3:
            statuses = ["HEALTHY", "HEALTHY", "PARTIAL_GHOST", "FULL_GHOST"]
            status = random.choice(statuses)
            confidence = 0.9 if status == "HEALTHY" else random.uniform(0.2, 0.6)
            issues = [] if status == "HEALTHY" else ["Issue 1", "Issue 2"]
            monitor.log_ghost_check(f"Component_{i}", confidence, status, issues)

        # System operations
        monitor.log_operation(random.uniform(0.001, 0.010))

    # Update system stats
    monitor.update_system_stats(memory_mb=45.3, cpu_percent=23.5)

    # Display dashboard
    monitor.print_dashboard(detailed=True)

    # Show alerts
    print("\n🚨 ALERTS")
    alerts = monitor.get_alerts()
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("  ✓ No alerts")

    # Health score
    health_score = monitor.get_health_score()
    print(f"\n💚 OVERALL HEALTH SCORE: {health_score:.1%}")

    # Export metrics
    monitor.export_metrics("demo_metrics.json")
    print("\n✓ Demo complete")
