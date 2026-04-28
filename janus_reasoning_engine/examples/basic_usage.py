"""
Basic usage example for the Janus Autonomous Reasoning Engine.

Demonstrates:
  - Initializing the engine
  - Setting goals
  - Running cycles
  - Checking status
  - Graceful shutdown

Requirements: REQ-9.3
"""

from __future__ import annotations

import json

from janus_reasoning_engine.engine import JanusReasoningEngine
from janus_reasoning_engine.core.config import EngineConfig


def main() -> None:
    print("=" * 60)
    print("Janus Reasoning Engine — Basic Usage Example")
    print("=" * 60)

    # ── 1. Configuration ──────────────────────────────────────────────
    print("\n[1] Creating configuration …")
    config = EngineConfig()
    config.reasoning.exploration_rate = 0.15
    config.safety.max_spending_per_action = 50.0
    config.logging.log_level = "WARNING"   # keep example output clean
    print(f"    exploration_rate : {config.reasoning.exploration_rate}")
    print(f"    max_spending     : ${config.safety.max_spending_per_action}")

    # ── 2. Initialize ─────────────────────────────────────────────────
    print("\n[2] Initializing engine …")
    engine = JanusReasoningEngine(config=config)
    engine.initialize()
    print("    Engine initialized")

    # ── 3. Set a goal ─────────────────────────────────────────────────
    print("\n[3] Setting a goal …")
    goal_id = engine.set_goal("Earn $1,000 this week via freelance work", priority=0.9)
    if goal_id:
        print(f"    Goal created: id={goal_id}")
    else:
        print("    Goal queued (goal manager unavailable in this environment)")

    # ── 4. Check status ───────────────────────────────────────────────
    print("\n[4] Engine status:")
    status = engine.get_status()
    print(f"    initialized    : {status['initialized']}")
    print(f"    cycle_count    : {status['cycle_count']}")
    print(f"    autonomous_mode: {status['config']['autonomous_mode']}")
    active_subsystems = [k for k, v in status["subsystems"].items() if v]
    print(f"    active subsystems ({len(active_subsystems)}): {', '.join(active_subsystems)}")

    # ── 5. Run a few cycles ───────────────────────────────────────────
    print("\n[5] Running 3 autonomous cycles …")
    for _ in range(3):
        outcome = engine.run_cycle()
        print(
            f"    Cycle {outcome.cycle_num}: "
            f"action={outcome.action_taken!r}  "
            f"success={outcome.success}  "
            f"earnings=${outcome.earnings:.2f}"
        )

    # ── 6. Status after cycles ────────────────────────────────────────
    print("\n[6] Status after cycles:")
    status = engine.get_status()
    print(f"    cycle_count : {status['cycle_count']}")
    print(f"    uptime      : {status['uptime_seconds']:.2f}s")

    # ── 7. Safety guardrails demo ─────────────────────────────────────
    print("\n[7] Safety guardrails demo …")
    from janus_reasoning_engine.safety.guardrails import SafetyGuardrails
    guardrails = SafetyGuardrails(budget_limit=100.0)

    budget_ok = guardrails.check_budget(50.0, 100.0)
    print(f"    check_budget($50, limit=$100) → allowed={budget_ok.allowed}")

    budget_fail = guardrails.check_budget(150.0, 100.0)
    print(f"    check_budget($150, limit=$100) → allowed={budget_fail.allowed}, reason={budget_fail.reason!r}")

    content_ok = guardrails.check_content("Write a Python script to scrape job listings")
    print(f"    check_content(safe text) → allowed={content_ok.allowed}")

    needs_perm = guardrails.requires_permission("sign up for new platform", 50.0)
    print(f"    requires_permission('sign up for new platform', $50) → {needs_perm}")

    # ── 8. Transparency logger demo ───────────────────────────────────
    print("\n[8] Transparency logger demo …")
    from janus_reasoning_engine.safety.transparency_logger import TransparencyLogger
    tl = TransparencyLogger(db_path=":memory:")

    log_id = tl.log_decision(
        action="apply for Python freelance job",
        reasoning="Best skill match and highest earning potential",
        context={"platform": "Upwork", "budget": 500},
    )
    print(f"    Decision logged: id={log_id}")

    tl.log_activity("opportunity_scan", {"sources": ["Upwork", "Fiverr"], "found": 12})
    recent = tl.get_recent_decisions(limit=5)
    print(f"    Recent decisions: {len(recent)}")

    summary = tl.get_activity_summary(hours=24)
    print(f"    Activity summary: {summary}")

    # ── 9. Advanced capabilities demo ────────────────────────────────
    print("\n[9] Advanced capabilities demo …")
    from janus_reasoning_engine.advanced.causal_horizon_bridge import CausalHorizonBridge
    from janus_reasoning_engine.advanced.advanced_concepts import AdvancedConcepts
    from janus_reasoning_engine.advanced.human_capabilities import HumanCapabilities

    bridge = CausalHorizonBridge()
    bridge.emit_signal("submitted_proposal", "client_responded", {"platform": "Upwork"})
    effects = bridge.get_propagated_effects("submitted_proposal")
    print(f"    Causal effects of 'submitted_proposal': {len(effects)}")

    ac = AdvancedConcepts()
    result = ac.bypass_computational_limit("optimise pricing strategy", 2000)
    print(f"    bypass_computational_limit result: {result!r}")

    boundaries = ac.explore_boundary("budget_limit")
    print(f"    explore_boundary('budget_limit'): {len(boundaries)} conditions")

    hc = HumanCapabilities()
    state = hc.get_personality_state()
    print(f"    Personality state: {state}")

    natural = hc.apply_natural_behavior("The task has been completed successfully.")
    print(f"    Natural behavior: {natural!r}")

    # ── 10. Shutdown ──────────────────────────────────────────────────
    print("\n[10] Shutting down engine …")
    engine.shutdown()
    print("     Engine shutdown complete")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
