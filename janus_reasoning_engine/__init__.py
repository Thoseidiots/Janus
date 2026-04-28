"""
Janus Autonomous Reasoning Engine

The brain that orchestrates all Janus systems to achieve true autonomy.
Provides goal-directed reasoning, strategic planning, and adaptive decision-making.
"""

from janus_reasoning_engine.core.interfaces import (
    ReasoningEngine,
    GoalManager,
    StrategyPlanner,
    ExecutionMonitor,
)
from janus_reasoning_engine.core.config import EngineConfig
from janus_reasoning_engine.engine import JanusReasoningEngine

__version__ = "0.1.0"

__all__ = [
    "ReasoningEngine",
    "GoalManager",
    "StrategyPlanner",
    "ExecutionMonitor",
    "EngineConfig",
    "JanusReasoningEngine",
]
