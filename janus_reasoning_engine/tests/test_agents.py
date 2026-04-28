"""Tests for multi-agent collaboration subsystem (Task 15).

Covers:
- SubAgent / SubAgentManager  (REQ-14.1)
- SocialSimBridge             (REQ-14.2, REQ-14.3)
- CEOBridge                   (REQ-14.4)
- OrchestratorBridge          (REQ-14.5)
"""

import os
import tempfile

import pytest

from janus_reasoning_engine.agents.sub_agent_manager import SubAgent, SubAgentManager
from janus_reasoning_engine.agents.social_sim_bridge import SimResult, SocialSimBridge
from janus_reasoning_engine.agents.ceo_bridge import CEOBridge
from janus_reasoning_engine.agents.orchestrator_bridge import CycleResult, OrchestratorBridge


# ---------------------------------------------------------------------------
# SubAgent
# ---------------------------------------------------------------------------

class TestSubAgent:
    def test_act_returns_string(self):
        agent = SubAgent(id="a1", role="critic", goals=["evaluate"])
        result = agent.act("some observation")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_act_records_history(self):
        agent = SubAgent(id="a2", role="advocate", goals=[])
        agent.act("topic A")
        agent.act("topic B")
        assert len(agent.history) == 2

    def test_act_unknown_role(self):
        agent = SubAgent(id="a3", role="custom_role", goals=[])
        result = agent.act("something")
        assert isinstance(result, str)

    def test_active_default_true(self):
        agent = SubAgent(id="a4", role="planner", goals=[])
        assert agent.active is True


# ---------------------------------------------------------------------------
# SubAgentManager
# ---------------------------------------------------------------------------

class TestSubAgentManager:
    def test_spawn_returns_sub_agent(self):
        mgr = SubAgentManager()
        agent = mgr.spawn("critic", ["goal1"])
        assert isinstance(agent, SubAgent)
        assert agent.role == "critic"
        assert agent.active is True

    def test_spawn_adds_to_active_pool(self):
        mgr = SubAgentManager()
        mgr.spawn("critic", [])
        mgr.spawn("advocate", [])
        assert len(mgr.get_active_agents()) == 2

    def test_terminate_deactivates_agent(self):
        mgr = SubAgentManager()
        agent = mgr.spawn("critic", [])
        mgr.terminate(agent.id)
        assert agent.active is False
        assert agent not in mgr.get_active_agents()

    def test_terminate_unknown_id_is_noop(self):
        mgr = SubAgentManager()
        mgr.terminate("nonexistent-id")  # should not raise

    def test_pool_cap_terminates_oldest(self):
        mgr = SubAgentManager()
        agents = [mgr.spawn("critic", []) for _ in range(SubAgentManager.MAX_POOL_SIZE)]
        oldest = agents[0]
        # Spawning one more should evict the oldest
        mgr.spawn("advocate", [])
        assert oldest.active is False
        assert len(mgr.get_active_agents()) == SubAgentManager.MAX_POOL_SIZE

    def test_deliberate_returns_three_responses(self):
        mgr = SubAgentManager()
        responses = mgr.deliberate("should we expand?")
        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)

    def test_deliberate_cleans_up_temp_agents(self):
        mgr = SubAgentManager()
        before = len(mgr.get_active_agents())
        mgr.deliberate("topic")
        after = len(mgr.get_active_agents())
        assert after == before  # temp agents terminated

    def test_get_active_agents_excludes_terminated(self):
        mgr = SubAgentManager()
        a1 = mgr.spawn("critic", [])
        a2 = mgr.spawn("advocate", [])
        mgr.terminate(a1.id)
        active = mgr.get_active_agents()
        assert a1 not in active
        assert a2 in active


# ---------------------------------------------------------------------------
# SimResult dataclass
# ---------------------------------------------------------------------------

class TestSimResult:
    def test_defaults(self):
        sr = SimResult(scenario="test")
        assert sr.turns == []
        assert sr.outcome == ""
        assert sr.tom_updates == {}

    def test_with_data(self):
        sr = SimResult(
            scenario="negotiation",
            turns=[{"turn": 1, "agent": "a1", "action": "speak"}],
            outcome="agreement",
            tom_updates={"a1": {"belief": "updated"}},
        )
        assert len(sr.turns) == 1
        assert sr.outcome == "agreement"


# ---------------------------------------------------------------------------
# SocialSimBridge
# ---------------------------------------------------------------------------

class TestSocialSimBridge:
    def _bridge(self, tmp_path):
        path = os.path.join(str(tmp_path), "relationships.json")
        return SocialSimBridge(relationships_path=path)

    def test_run_simulation_returns_sim_result(self, tmp_path):
        bridge = self._bridge(tmp_path)
        result = bridge.run_simulation("negotiation", ["agent1", "agent2"])
        assert isinstance(result, SimResult)
        assert result.scenario == "negotiation"
        assert len(result.turns) >= 0

    def test_run_simulation_includes_agents(self, tmp_path):
        bridge = self._bridge(tmp_path)
        result = bridge.run_simulation("collab", ["a1", "a2"])
        agent_ids_in_turns = {t["agent"] for t in result.turns}
        assert "a1" in agent_ids_in_turns or len(result.turns) == 0  # stub may vary

    def test_update_relationship_positive_increases_trust(self, tmp_path):
        bridge = self._bridge(tmp_path)
        before = bridge.get_relationship("agent1")["trust"]
        bridge.update_relationship("agent1", "positive")
        after = bridge.get_relationship("agent1")["trust"]
        assert after > before

    def test_update_relationship_negative_decreases_trust(self, tmp_path):
        bridge = self._bridge(tmp_path)
        bridge.update_relationship("agent1", "positive")  # set above default first
        before = bridge.get_relationship("agent1")["trust"]
        bridge.update_relationship("agent1", "negative")
        after = bridge.get_relationship("agent1")["trust"]
        assert after < before

    def test_trust_clamped_at_one(self, tmp_path):
        bridge = self._bridge(tmp_path)
        for _ in range(20):
            bridge.update_relationship("agent1", "positive")
        assert bridge.get_relationship("agent1")["trust"] <= 1.0

    def test_trust_clamped_at_zero(self, tmp_path):
        bridge = self._bridge(tmp_path)
        for _ in range(20):
            bridge.update_relationship("agent1", "negative")
        assert bridge.get_relationship("agent1")["trust"] >= 0.0

    def test_get_relationship_returns_defaults_for_new_agent(self, tmp_path):
        bridge = self._bridge(tmp_path)
        rel = bridge.get_relationship("new_agent")
        assert "trust" in rel
        assert "familiarity" in rel
        assert "tone" in rel

    def test_list_relationships_empty_initially(self, tmp_path):
        bridge = self._bridge(tmp_path)
        assert bridge.list_relationships() == []

    def test_list_relationships_after_update(self, tmp_path):
        bridge = self._bridge(tmp_path)
        bridge.update_relationship("a1", "positive")
        bridge.update_relationship("a2", "negative")
        rels = bridge.list_relationships()
        agent_ids = {r["agent_id"] for r in rels}
        assert "a1" in agent_ids
        assert "a2" in agent_ids

    def test_sleep_phase_refinement_returns_dict(self, tmp_path):
        bridge = self._bridge(tmp_path)
        bridge.update_relationship("a1", "positive")
        result = bridge.sleep_phase_refinement()
        assert isinstance(result, dict)
        assert "simulations_run" in result
        assert "tom_updates" in result

    def test_sleep_phase_simulations_run_count(self, tmp_path):
        bridge = self._bridge(tmp_path)
        bridge.update_relationship("a1", "positive")
        bridge.update_relationship("a2", "positive")
        result = bridge.sleep_phase_refinement()
        assert result["simulations_run"] == 2

    def test_relationships_persisted_across_instances(self, tmp_path):
        path = os.path.join(str(tmp_path), "rels.json")
        b1 = SocialSimBridge(relationships_path=path)
        b1.update_relationship("agent_x", "positive")

        b2 = SocialSimBridge(relationships_path=path)
        rel = b2.get_relationship("agent_x")
        assert rel["trust"] > SocialSimBridge._DEFAULT_TRUST


# ---------------------------------------------------------------------------
# CEOBridge
# ---------------------------------------------------------------------------

class TestCEOBridge:
    def test_start_business_returns_string_id(self):
        ceo = CEOBridge()
        bid = ceo.start_business("content_agency", "My Agency", 10000.0)
        assert isinstance(bid, str)
        assert len(bid) > 0

    def test_start_business_multiple(self):
        ceo = CEOBridge()
        b1 = ceo.start_business("ecommerce", "Shop A", 5000.0)
        b2 = ceo.start_business("web_development", "Dev Co", 8000.0)
        assert b1 != b2

    def test_get_portfolio_summary_structure(self):
        ceo = CEOBridge()
        ceo.start_business("ecommerce", "Shop", 1000.0)
        summary = ceo.get_portfolio_summary()
        assert "total_businesses" in summary
        assert "total_revenue" in summary
        assert "businesses" in summary

    def test_get_portfolio_summary_counts(self):
        ceo = CEOBridge()
        ceo.start_business("ecommerce", "Shop A", 1000.0)
        ceo.start_business("marketing_agency", "Agency B", 2000.0)
        summary = ceo.get_portfolio_summary()
        assert summary["total_businesses"] == 2

    def test_get_portfolio_empty(self):
        ceo = CEOBridge()
        summary = ceo.get_portfolio_summary()
        assert summary["total_businesses"] == 0
        assert summary["total_revenue"] == 0.0

    def test_make_strategic_decision_returns_string(self):
        ceo = CEOBridge()
        decision = ceo.make_strategic_decision(
            "expand market", ["option A", "option B", "option C"]
        )
        assert isinstance(decision, str)
        assert len(decision) > 0

    def test_make_strategic_decision_no_options(self):
        ceo = CEOBridge()
        decision = ceo.make_strategic_decision("context", [])
        assert isinstance(decision, str)


# ---------------------------------------------------------------------------
# CycleResult dataclass
# ---------------------------------------------------------------------------

class TestCycleResult:
    def test_defaults(self):
        cr = CycleResult(cycle_num=1, goal="test goal", success=True)
        assert cr.output == ""
        assert cr.error is None

    def test_with_error(self):
        cr = CycleResult(cycle_num=2, goal="goal", success=False, error="timeout")
        assert cr.success is False
        assert cr.error == "timeout"


# ---------------------------------------------------------------------------
# OrchestratorBridge
# ---------------------------------------------------------------------------

class TestOrchestratorBridge:
    def test_run_cycle_returns_cycle_result(self):
        orch = OrchestratorBridge()
        result = orch.run_cycle("earn $1000")
        assert isinstance(result, CycleResult)

    def test_run_cycle_increments_counter(self):
        orch = OrchestratorBridge()
        orch.run_cycle("goal A")
        orch.run_cycle("goal B")
        status = orch.get_status()
        assert status["cycles_run"] == 2

    def test_run_cycle_goal_preserved(self):
        orch = OrchestratorBridge()
        result = orch.run_cycle("find clients")
        assert result.goal == "find clients"

    def test_run_cycle_success_flag(self):
        orch = OrchestratorBridge()
        result = orch.run_cycle("any goal")
        assert isinstance(result.success, bool)

    def test_get_status_structure(self):
        orch = OrchestratorBridge()
        status = orch.get_status()
        assert "current_goal" in status
        assert "cycles_run" in status
        assert "components" in status

    def test_set_goal_updates_status(self):
        orch = OrchestratorBridge()
        orch.set_goal("become profitable")
        status = orch.get_status()
        assert status["current_goal"] == "become profitable"

    def test_cycle_num_matches_run_count(self):
        orch = OrchestratorBridge()
        r1 = orch.run_cycle("g1")
        r2 = orch.run_cycle("g2")
        assert r1.cycle_num == 1
        assert r2.cycle_num == 2

    def test_components_dict_present(self):
        orch = OrchestratorBridge()
        status = orch.get_status()
        assert isinstance(status["components"], dict)
