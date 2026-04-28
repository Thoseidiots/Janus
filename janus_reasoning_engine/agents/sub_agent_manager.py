"""Sub-agent management system for internal deliberation and role-based reasoning.

Requirements: REQ-14.1
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List


# Role-specific response templates
_ROLE_RESPONSES = {
    "critic": "As a critic, I challenge this: {observation}. Key weaknesses to consider.",
    "advocate": "As an advocate, I support this: {observation}. Key strengths to highlight.",
    "devil_advocate": "As devil's advocate, what if the opposite is true about: {observation}?",
    "planner": "As a planner, here is a structured approach to: {observation}.",
}

_MAX_POOL_SIZE = 8


@dataclass
class SubAgent:
    """Lightweight sub-agent instance for planning and deliberation."""

    id: str
    role: str
    goals: List[str]
    history: List[str] = field(default_factory=list)
    active: bool = True

    def act(self, observation: str) -> str:
        """Return a role-appropriate response to the given observation."""
        template = _ROLE_RESPONSES.get(
            self.role,
            "As {role}, my perspective on: {observation}".format(
                role=self.role, observation="{observation}"
            ),
        )
        response = template.format(observation=observation)
        self.history.append(f"obs={observation!r} -> {response!r}")
        return response


class SubAgentManager:
    """Manages a pool of sub-agents for internal deliberation.

    Pool is capped at MAX_POOL_SIZE (8). When full, the oldest active agent
    is terminated before spawning a new one.
    """

    MAX_POOL_SIZE = _MAX_POOL_SIZE

    def __init__(self) -> None:
        self._agents: List[SubAgent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn(self, role: str, goals: List[str]) -> SubAgent:
        """Spawn a new sub-agent with the given role and goals.

        If the pool is at capacity, the oldest active agent is terminated first.
        """
        active = self.get_active_agents()
        if len(active) >= self.MAX_POOL_SIZE:
            # Terminate the oldest active agent
            self.terminate(active[0].id)

        agent = SubAgent(id=str(uuid.uuid4()), role=role, goals=list(goals))
        self._agents.append(agent)
        return agent

    def deliberate(self, topic: str) -> List[str]:
        """Spawn critic, advocate, and devil_advocate; each acts on topic.

        Returns their responses and then terminates the temporary agents.
        """
        roles = ["critic", "advocate", "devil_advocate"]
        temp_agents: List[SubAgent] = []
        for role in roles:
            agent = self.spawn(role=role, goals=[f"deliberate on: {topic}"])
            temp_agents.append(agent)

        responses = [agent.act(topic) for agent in temp_agents]

        for agent in temp_agents:
            self.terminate(agent.id)

        return responses

    def terminate(self, agent_id: str) -> None:
        """Mark the agent with the given id as inactive."""
        for agent in self._agents:
            if agent.id == agent_id:
                agent.active = False
                return

    def get_active_agents(self) -> List[SubAgent]:
        """Return all currently active agents in spawn order."""
        return [a for a in self._agents if a.active]
