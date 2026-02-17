“””
social_sim.py
────────────────────────────────────────────────────────────
Multi-agent social simulation for Janus.
• Spawn lightweight sub-instances (AgentProxy) of Janus for planning.
• Model other agents (humans or AI) with persistent belief/relationship state.
• Run internal social simulations during the SLEEP phase to refine
theory-of-mind (ToM) predictions.
• Persistent “relationships” live in relationships.json.
“””

import json
import uuid
import time
import threading
import random
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

class AgentType(Enum):
JANUS      = “janus”
HUMAN      = “human”
AI_OTHER   = “ai_other”
SUBAGENT   = “subagent”

class RelationshipTone(Enum):
NEUTRAL    = “neutral”
FRIENDLY   = “friendly”
ADVERSARIAL = “adversarial”
COLLABORATIVE = “collaborative”
DEPENDENT  = “dependent”

@dataclass
class BeliefState:
“”“Janus’s model of another agent’s internal state.”””
goals:       List[str]       = field(default_factory=list)
emotions:    Dict[str, float] = field(default_factory=dict)
knowledge:   List[str]       = field(default_factory=list)
reliability: float           = 0.5   # 0=unreliable, 1=fully reliable
updated_at:  str             = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Relationship:
agent_id:    str
agent_type:  AgentType
name:        str
tone:        RelationshipTone
trust:       float            = 0.5    # 0–1
familiarity: float            = 0.0    # 0–1
shared_goals: List[str]       = field(default_factory=list)
interaction_count: int        = 0
last_seen:   Optional[str]    = None
belief:      BeliefState      = field(default_factory=BeliefState)
memory_ids:  List[str]        = field(default_factory=list)
notes:       str              = “”

```
def to_dict(self) -> dict:
    d = asdict(self)
    d["agent_type"] = self.agent_type.value
    d["tone"]       = self.tone.value
    return d

@staticmethod
def from_dict(d: dict) -> "Relationship":
    d["agent_type"] = AgentType(d.get("agent_type", "human"))
    d["tone"]       = RelationshipTone(d.get("tone", "neutral"))
    if isinstance(d.get("belief"), dict):
        d["belief"] = BeliefState(**d["belief"])
    return Relationship(**{k: v for k, v in d.items()
                           if k in Relationship.__dataclass_fields__})
```

# ── Sub-agent proxy ────────────────────────────────────────────────────────────

@dataclass
class AgentProxy:
“”“Lightweight sub-instance of Janus used for planning simulations.”””
agent_id:    str
role:        str             # e.g. “critic”, “advocate”, “devil_advocate”
goals:       List[str]
valence:     Dict[str, float] = field(default_factory=dict)
history:     List[dict]       = field(default_factory=list)
spawned_at:  str              = field(default_factory=lambda: datetime.now().isoformat())
terminated:  bool             = False

```
def act(self, observation: str) -> str:
    """Simulate a decision given an observation (stub — wire to your LLM)."""
    role_responses = {
        "critic":         f"[CRITIC] I challenge: {observation[:60]}... Is this assumption valid?",
        "advocate":       f"[ADVOCATE] This supports: {observation[:60]}...",
        "devil_advocate": f"[DEVIL] What if the opposite is true: {observation[:60]}?",
        "planner":        f"[PLANNER] Next step for {observation[:40]}: decompose into sub-tasks.",
    }
    response = role_responses.get(self.role, f"[{self.role.upper()}] Noted: {observation[:60]}")
    self.history.append({"obs": observation[:80], "act": response, "ts": datetime.now().isoformat()})
    return response

def terminate(self):
    self.terminated = True
```

# ── Social simulation engine ───────────────────────────────────────────────────

@dataclass
class SimulationRound:
round_id: str
agents: List[str]
scenario: str
turns: List[dict] = field(default_factory=list)
outcome: Optional[str] = None
tom_update: Optional[dict] = None

class SocialSimulator:
“””
Runs internal social simulations during the SLEEP phase.
Produces theory-of-mind updates for known relationships.
“””

```
def __init__(self, relationships: Dict[str, "Relationship"]):
    self._rels = relationships

def run_negotiation_sim(self, scenario: str,
                         agent_ids: List[str],
                         n_turns: int = 4) -> SimulationRound:
    """Simulate a negotiation / collaboration between agents."""
    round_id = "sim_" + str(uuid.uuid4())[:6]
    sim = SimulationRound(round_id=round_id, agents=agent_ids, scenario=scenario)

    proxies: Dict[str, AgentProxy] = {}
    for aid in agent_ids:
        rel = self._rels.get(aid)
        proxies[aid] = AgentProxy(
            agent_id = aid,
            role     = "collaborator",
            goals    = rel.shared_goals if rel else [],
            valence  = rel.belief.emotions if rel else {},
        )

    # Add Janus itself
    janus_proxy = AgentProxy(agent_id="janus", role="advocate", goals=[scenario])
    proxies["janus"] = janus_proxy

    observation = scenario
    for turn in range(n_turns):
        for aid, proxy in proxies.items():
            if proxy.terminated:
                continue
            response = proxy.act(observation)
            sim.turns.append({"turn": turn, "agent": aid, "response": response,
                               "ts": datetime.now().isoformat()})
            observation = response   # chain responses

    # Derive outcome from last responses
    sim.outcome = f"Simulation of '{scenario}' completed in {n_turns * len(proxies)} exchanges."

    # Theory-of-mind update
    sim.tom_update = self._extract_tom_updates(sim, proxies)

    return sim

def _extract_tom_updates(self, sim: SimulationRound,
                          proxies: Dict[str, AgentProxy]) -> dict:
    updates = {}
    for aid, proxy in proxies.items():
        if aid == "janus":
            continue
        # Heuristic: agents that contributed many turns are more reliable
        turns_taken = sum(1 for t in sim.turns if t["agent"] == aid)
        reliability_delta = 0.05 * turns_taken
        updates[aid] = {
            "reliability_delta": round(reliability_delta, 3),
            "inferred_goals":    proxy.goals,
            "interaction_turns": turns_taken,
        }
    return updates
```

# ── Relationship manager ───────────────────────────────────────────────────────

class RelationshipManager:
“””
Persistent store for Janus’s relationships with other agents.
Integrates with the social simulator and SLEEP-phase ToM refinement.

```
Usage
─────
rm = RelationshipManager()
rm.meet("alice", AgentType.HUMAN, "Alice")
rm.record_interaction("alice", "Alice helped with task X", valence_delta=0.1)
sim_result = rm.sleep_phase_refinement()
"""

PERSIST_PATH = Path("relationships.json")
SUBAGENT_POOL_MAX = 8

def __init__(self):
    self._rels: Dict[str, Relationship] = {}
    self._subagents: Dict[str, AgentProxy] = {}
    self._sim = SocialSimulator(self._rels)
    self._lock = threading.Lock()
    self._load()
    print(f"[SocialSim] Loaded {len(self._rels)} relationships")

# ── Relationship CRUD ──────────────────────────────────────────────────────
def meet(self, agent_id: str, agent_type: AgentType, name: str,
         initial_tone: RelationshipTone = RelationshipTone.NEUTRAL) -> Relationship:
    with self._lock:
        if agent_id in self._rels:
            return self._rels[agent_id]
        rel = Relationship(
            agent_id   = agent_id,
            agent_type = agent_type,
            name       = name,
            tone       = initial_tone,
        )
        self._rels[agent_id] = rel
    self._persist()
    print(f"[SocialSim] New relationship: {name} ({agent_type.value})")
    return rel

def record_interaction(self, agent_id: str, summary: str,
                       valence_delta: float = 0.0):
    with self._lock:
        rel = self._rels.get(agent_id)
        if not rel:
            return
        rel.interaction_count += 1
        rel.last_seen          = datetime.now().isoformat()
        rel.familiarity        = min(1.0, rel.familiarity + 0.05)
        rel.trust              = max(0.0, min(1.0, rel.trust + valence_delta * 0.1))
        rel.belief.updated_at  = datetime.now().isoformat()
        rel.notes             += f"\n[{datetime.now():%H:%M}] {summary[:80]}"
    self._persist()

def update_belief(self, agent_id: str, goals: List[str] = None,
                  emotions: Dict[str, float] = None):
    with self._lock:
        rel = self._rels.get(agent_id)
        if not rel:
            return
        if goals:
            rel.belief.goals = goals
        if emotions:
            rel.belief.emotions.update(emotions)
        rel.belief.updated_at = datetime.now().isoformat()
    self._persist()

def get_relationship(self, agent_id: str) -> Optional[Relationship]:
    return self._rels.get(agent_id)

def list_relationships(self) -> List[dict]:
    with self._lock:
        return [r.to_dict() for r in self._rels.values()]

# ── Sub-agent management ───────────────────────────────────────────────────
def spawn_subagent(self, role: str, goals: List[str]) -> AgentProxy:
    if len(self._subagents) >= self.SUBAGENT_POOL_MAX:
        # Terminate oldest
        oldest = min(self._subagents.values(), key=lambda a: a.spawned_at)
        oldest.terminate()
        del self._subagents[oldest.agent_id]

    proxy = AgentProxy(
        agent_id = "sub_" + str(uuid.uuid4())[:6],
        role     = role,
        goals    = goals,
    )
    self._subagents[proxy.agent_id] = proxy
    print(f"[SocialSim] Spawned sub-agent: {proxy.agent_id} ({role})")
    return proxy

def terminate_subagent(self, agent_id: str):
    proxy = self._subagents.pop(agent_id, None)
    if proxy:
        proxy.terminate()

def internal_deliberation(self, topic: str) -> List[str]:
    """
    Spawn critic + advocate sub-agents to deliberate on a topic.
    Returns their outputs as a list of perspectives.
    """
    critic   = self.spawn_subagent("critic",   [topic])
    advocate = self.spawn_subagent("advocate", [topic])
    devil    = self.spawn_subagent("devil_advocate", [topic])

    perspectives = [
        critic.act(topic),
        advocate.act(topic),
        devil.act(topic),
    ]

    self.terminate_subagent(critic.agent_id)
    self.terminate_subagent(advocate.agent_id)
    self.terminate_subagent(devil.agent_id)

    return perspectives

# ── Sleep-phase refinement ─────────────────────────────────────────────────
def sleep_phase_refinement(self) -> dict:
    """
    Run social simulations for all collaborative relationships.
    Updates theory-of-mind beliefs.
    """
    report = {"simulations_run": 0, "tom_updates": {}}

    collaborative = [
        r for r in self._rels.values()
        if r.tone in (RelationshipTone.COLLABORATIVE, RelationshipTone.FRIENDLY)
        and r.shared_goals
    ]

    for rel in collaborative[:3]:   # limit per sleep cycle
        scenario = f"Advance shared goal: {rel.shared_goals[0]}" if rel.shared_goals else "General collaboration"
        sim = self._sim.run_negotiation_sim(scenario, [rel.agent_id])
        report["simulations_run"] += 1

        # Apply ToM updates
        if sim.tom_update and rel.agent_id in sim.tom_update:
            upd = sim.tom_update[rel.agent_id]
            with self._lock:
                rel.belief.reliability = min(1.0, rel.belief.reliability + upd["reliability_delta"])
                if upd.get("inferred_goals"):
                    rel.belief.goals = upd["inferred_goals"]
            report["tom_updates"][rel.agent_id] = upd

    self._persist()
    print(f"[SocialSim] Sleep refinement: {report}")
    return report

# ── Persistence ───────────────────────────────────────────────────────────
def _persist(self):
    try:
        data = {aid: r.to_dict() for aid, r in self._rels.items()}
        self.PERSIST_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[SocialSim] Persist error: {e}")

def _load(self):
    if not self.PERSIST_PATH.exists():
        return
    try:
        data = json.loads(self.PERSIST_PATH.read_text())
        for aid, rdata in data.items():
            self._rels[aid] = Relationship.from_dict(rdata)
    except Exception as e:
        print(f"[SocialSim] Load error: {e}")
```