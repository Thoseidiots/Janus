"""
Opportunity discovery subsystem for the Janus Reasoning Engine.
"""

from janus_reasoning_engine.discovery.opportunity_scorer import (
    Opportunity,
    OpportunityScorer,
)
from janus_reasoning_engine.discovery.opportunity_monitor import OpportunityMonitor

__all__ = ["Opportunity", "OpportunityScorer", "OpportunityMonitor"]
