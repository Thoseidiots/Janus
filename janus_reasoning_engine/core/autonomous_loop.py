"""
autonomous_loop.py
==================
Main autonomous loop for the Janus Reasoning Engine.

Wires together all subsystems into a single run_cycle() method that
executes one full iteration of the autonomous loop:

  1. Assess state  (wallet balance, active goals, working memory context)
  2. Decide action (continue active work OR discover new opportunities)
  3. Execute action
  4. Learn and adapt
  5. Manage resources

All subsystems are optional — pass None (or omit) and the loop degrades
gracefully.

Requirements: REQ-9.1, REQ-1.1, REQ-6.3, REQ-2.1, REQ-2.2, REQ-4.1,
              REQ-4.3, REQ-5.4, REQ-3.2, REQ-1.2
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.core.autonomous_loop")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LoopState:
    """Snapshot of the autonomous loop's current state."""
    cycle_count: int = 0
    current_goal: Optional[str] = None          # goal description or ID
    active_work: Optional[str] = None           # opportunity ID currently being worked
    balance: float = 0.0                        # last known wallet balance
    last_reflection_at: Optional[datetime] = None


@dataclass
class CycleOutcome:
    """Result of a single autonomous loop cycle."""
    cycle_num: int
    action_taken: str
    success: bool
    earnings: float = 0.0
    notes: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a single opportunity."""
    opportunity_id: str
    success: bool
    earnings: float = 0.0
    duration_seconds: float = 0.0
    notes: str = ""


# ── AutonomousLoop ────────────────────────────────────────────────────────────

class AutonomousLoop:
    """
    Orchestrates all Janus subsystems in a continuous autonomous loop.

    Each call to :meth:`run_cycle` executes one full iteration:
    assess → decide → execute → learn → manage resources.

    All subsystem arguments are optional.  When a subsystem is None the
    loop skips that capability gracefully rather than raising an error.

    Args:
        opportunity_monitor:   OpportunityMonitor for multi-source scanning.
        opportunity_scorer:    OpportunityScorer for ranking opportunities.
        exploration_strategy:  ExplorationStrategy for exploit/explore balance.
        multi_step_planner:    MultiStepPlanner for creating execution plans.
        execution_monitor:     ExecutionMonitor for tracking plan execution.
        metacognition:         Metacognition for self-reflection.
        progress_tracker:      ProgressTracker for goal progress.
        skill_inventory:       SkillInventory for skill management.
        episodic_memory:       EpisodicMemory for experience storage.
        working_memory:        WorkingMemory for active context.
        financial_manager:     FinancialManager for wallet/budget queries.
        low_balance_threshold: Balance below which earning is prioritised.
    """

    LOW_BALANCE_DEFAULT = 50.0          # USD
    REFLECT_EVERY_N_CYCLES = 5         # run metacognition every N cycles

    def __init__(
        self,
        opportunity_monitor=None,
        opportunity_scorer=None,
        exploration_strategy=None,
        multi_step_planner=None,
        execution_monitor=None,
        metacognition=None,
        progress_tracker=None,
        skill_inventory=None,
        episodic_memory=None,
        working_memory=None,
        financial_manager=None,
        low_balance_threshold: float = LOW_BALANCE_DEFAULT,
    ) -> None:
        self.opportunity_monitor = opportunity_monitor
        self.opportunity_scorer = opportunity_scorer
        self.exploration_strategy = exploration_strategy
        self.multi_step_planner = multi_step_planner
        self.execution_monitor = execution_monitor
        self.metacognition = metacognition
        self.progress_tracker = progress_tracker
        self.skill_inventory = skill_inventory
        self.episodic_memory = episodic_memory
        self.working_memory = working_memory
        self.financial_manager = financial_manager
        self.low_balance_threshold = low_balance_threshold

        self.state = LoopState()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_cycle(self) -> CycleOutcome:
        """
        Execute one full autonomous loop iteration.

        Steps:
          1. Assess state
          2. Decide next action
          3. Execute action
          4. Learn and adapt
          5. Manage resources

        Returns:
            CycleOutcome describing what happened this cycle.
        """
        self.state.cycle_count += 1
        cycle_num = self.state.cycle_count
        logger.info("=== Autonomous loop cycle %d ===", cycle_num)

        # 1. Assess state
        self._assess_state()

        # 2. Decide and execute
        action_taken = "idle"
        success = True
        earnings = 0.0
        notes = ""

        try:
            if self.state.active_work:
                # Continue existing work — for now we treat it as a no-op
                # (real continuation would resume an ExecutionSession)
                action_taken = f"continue_work:{self.state.active_work}"
                notes = "Continuing active work from previous cycle"
                logger.info("Continuing active work: %s", self.state.active_work)
            else:
                # Discover and execute a new opportunity
                opportunity = self._discover_and_select()
                if opportunity is not None:
                    self.state.active_work = opportunity.id
                    exec_result = self._execute_opportunity(opportunity)
                    action_taken = f"execute_opportunity:{opportunity.id}"
                    success = exec_result.success
                    earnings = exec_result.earnings
                    notes = exec_result.notes
                    if exec_result.success:
                        self.state.active_work = None  # work complete
                    # 4. Learn from execution
                    self._update_knowledge(exec_result)
                else:
                    action_taken = "no_opportunity_found"
                    notes = "No suitable opportunity discovered this cycle"
                    logger.info("No opportunity found — idling this cycle")

        except Exception as exc:
            success = False
            notes = f"Cycle error: {exc}"
            logger.exception("Error during cycle %d: %s", cycle_num, exc)

        outcome = CycleOutcome(
            cycle_num=cycle_num,
            action_taken=action_taken,
            success=success,
            earnings=earnings,
            notes=notes,
        )

        # 4. Reflect and learn (periodic)
        self._reflect_and_learn(outcome)

        # 5. Manage resources
        self._manage_resources()

        logger.info(
            "Cycle %d complete — action=%s success=%s earnings=%.2f",
            cycle_num, action_taken, success, earnings,
        )
        return outcome

    # ── Step 1: Assess state ──────────────────────────────────────────────────

    def _assess_state(self) -> None:
        """
        Assess current state: wallet balance, active goals, working memory.

        Updates self.state in-place.  All subsystem calls are wrapped in
        try/except so a missing subsystem never crashes the loop.
        """
        # Check wallet balance
        if self.financial_manager is not None:
            try:
                budget = self.financial_manager.get_budget_status()
                self.state.balance = float(
                    getattr(budget, "current_balance", 0.0)
                    or getattr(budget, "available", 0.0)
                    or 0.0
                )
                logger.debug("Balance: %.2f", self.state.balance)
            except Exception as exc:
                logger.warning("Could not read balance: %s", exc)

        # Recall active goals
        if self.progress_tracker is not None:
            try:
                stats = self.progress_tracker.get_statistics()
                logger.debug("Progress tracker stats: %s", stats)
            except Exception as exc:
                logger.warning("Could not read progress stats: %s", exc)

        # Recall recent context from working memory
        if self.working_memory is not None:
            try:
                # Attempt to get recent context items
                ctx = getattr(self.working_memory, "get_recent_context", None)
                if ctx is not None:
                    items = ctx(limit=5)
                    logger.debug("Working memory context items: %d", len(items))
            except Exception as exc:
                logger.warning("Could not read working memory: %s", exc)

    # ── Step 2: Discover and select ───────────────────────────────────────────

    def _discover_and_select(self):
        """
        Discover opportunities and select the best one.

        Uses OpportunityMonitor → OpportunityScorer → ExplorationStrategy.
        Returns None if no subsystems are available or no opportunities found.

        Returns:
            Opportunity or None.
        """
        if self.opportunity_monitor is None:
            logger.debug("No OpportunityMonitor — skipping discovery")
            return None

        # Determine skills to search for
        skills: List[str] = []
        if self.skill_inventory is not None:
            try:
                skill_dict = self.skill_inventory.as_dict()
                # Use top skills (confidence >= 0.5) as search terms
                skills = [
                    name for name, conf in skill_dict.items() if conf >= 0.5
                ][:10]
            except Exception as exc:
                logger.warning("Could not read skill inventory: %s", exc)

        # Run async scan synchronously
        raw_opportunities: List[Dict[str, Any]] = []
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                raw_opportunities = loop.run_until_complete(
                    self.opportunity_monitor.scan(skills=skills)
                )
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("Opportunity scan failed: %s", exc)
            return None

        if not raw_opportunities:
            return None

        # Convert raw dicts to Opportunity objects
        try:
            from janus_reasoning_engine.discovery.opportunity_scorer import Opportunity
            opportunities = [Opportunity.from_raw(r) for r in raw_opportunities]
        except Exception as exc:
            logger.warning("Could not parse opportunities: %s", exc)
            return None

        # Score opportunities
        if self.opportunity_scorer is not None:
            try:
                opportunities = self.opportunity_scorer.score_and_rank(opportunities)
            except Exception as exc:
                logger.warning("Scoring failed: %s", exc)

        if not opportunities:
            return None

        # Select via exploration strategy or fall back to top-scored
        if self.exploration_strategy is not None:
            try:
                selected = self.exploration_strategy.select_opportunity(opportunities)
                return selected
            except Exception as exc:
                logger.warning("Exploration strategy failed: %s", exc)

        return opportunities[0]

    # ── Step 3: Execute opportunity ───────────────────────────────────────────

    def _execute_opportunity(self, opportunity) -> ExecutionResult:
        """
        Create a plan for *opportunity* and execute it via ExecutionMonitor.

        Args:
            opportunity: Opportunity to execute.

        Returns:
            ExecutionResult with success/earnings/notes.
        """
        start_time = time.monotonic()
        opp_id = getattr(opportunity, "id", str(uuid.uuid4()))

        # Build a plan
        plan = None
        if self.multi_step_planner is not None:
            try:
                goal_desc = (
                    f"Complete opportunity: {getattr(opportunity, 'title', opp_id)} "
                    f"on {getattr(opportunity, 'platform', 'unknown')}"
                )
                context = {
                    "earning_potential": getattr(opportunity, "earning_potential", 0),
                    "required_skills": getattr(opportunity, "required_skills", []),
                    "url": getattr(opportunity, "url", ""),
                }
                plan = self.multi_step_planner.create_plan(goal_desc, context)
                logger.info("Created plan %s for opportunity %s", plan.id, opp_id)
            except Exception as exc:
                logger.warning("Plan creation failed: %s", exc)

        # Execute the plan
        success = False
        earnings = 0.0
        notes = ""

        if plan is not None and self.execution_monitor is not None:
            try:
                session = self.execution_monitor.start_plan(plan)
                # Advance through all steps
                max_steps = len(plan.steps) + 1
                for _ in range(max_steps):
                    result = self.execution_monitor.advance(session)
                    if not session.is_active:
                        break
                    if result.metadata.get("plan_complete"):
                        break

                status = self.execution_monitor.get_status(session)
                success = status.get("plan_status") == "completed"
                notes = f"Plan {plan.id}: {status.get('plan_status', 'unknown')}"
                logger.info("Execution finished: success=%s", success)
            except Exception as exc:
                notes = f"Execution error: {exc}"
                logger.warning("Execution failed: %s", exc)
        elif plan is None and self.execution_monitor is None:
            # No subsystems — simulate a minimal execution
            notes = "No planner/monitor available — simulated execution"
            success = True
            logger.debug("Simulated execution for opportunity %s", opp_id)
        else:
            notes = "Partial subsystems — skipping execution"
            logger.debug("Skipping execution (missing planner or monitor)")

        duration = time.monotonic() - start_time
        return ExecutionResult(
            opportunity_id=opp_id,
            success=success,
            earnings=earnings,
            duration_seconds=duration,
            notes=notes,
        )

    # ── Step 4a: Reflect and learn ────────────────────────────────────────────

    def _reflect_and_learn(self, cycle_outcome: CycleOutcome) -> None:
        """
        Periodic metacognitive reflection using Metacognition + ProgressTracker.

        Runs every REFLECT_EVERY_N_CYCLES cycles.

        Args:
            cycle_outcome: Outcome of the current cycle.
        """
        should_reflect = (
            self.state.cycle_count % self.REFLECT_EVERY_N_CYCLES == 0
        )
        if not should_reflect:
            return

        self.state.last_reflection_at = datetime.utcnow()
        logger.info("Running reflection at cycle %d", self.state.cycle_count)

        # Metacognition reflection
        if self.metacognition is not None:
            try:
                task_result = {
                    "task_id": f"cycle_{cycle_outcome.cycle_num}",
                    "success": cycle_outcome.success,
                    "notes": cycle_outcome.notes,
                    "duration_minutes": 0,
                }
                reflection = self.metacognition.reflect(task_result)
                logger.info(
                    "Reflection: success=%s delta=%+.2f patterns=%s",
                    reflection.success,
                    reflection.confidence_delta,
                    reflection.patterns_identified,
                )
            except Exception as exc:
                logger.warning("Metacognition reflection failed: %s", exc)

        # Progress tracker reflection
        if self.progress_tracker is not None and self.state.current_goal:
            try:
                reflection_data = self.progress_tracker.reflect(
                    self.state.current_goal
                )
                logger.debug("Progress reflection: %s", reflection_data)
            except Exception as exc:
                logger.warning("Progress tracker reflection failed: %s", exc)

    # ── Step 4b: Update knowledge ─────────────────────────────────────────────

    def _update_knowledge(self, execution_result: ExecutionResult) -> None:
        """
        Update skill inventory and episodic memory after an execution.

        Args:
            execution_result: Result of the completed execution.
        """
        # Update skill inventory
        if self.skill_inventory is not None:
            try:
                delta = 0.05 if execution_result.success else -0.02
                # We don't have the opportunity object here, so update a generic skill
                self.skill_inventory.update_confidence("autonomous_execution", delta)
                logger.debug(
                    "Updated skill 'autonomous_execution' by %+.2f", delta
                )
            except Exception as exc:
                logger.warning("Skill inventory update failed: %s", exc)

        # Store experience in episodic memory
        if self.episodic_memory is not None:
            try:
                from janus_reasoning_engine.memory.episodic_memory import OutcomeType
                outcome_type = (
                    OutcomeType.SUCCESS if execution_result.success else OutcomeType.FAILURE
                )
                self.episodic_memory.store_experience(
                    context={"opportunity_id": execution_result.opportunity_id},
                    action={"type": "execute_opportunity"},
                    outcome={
                        "success": execution_result.success,
                        "earnings": execution_result.earnings,
                        "notes": execution_result.notes,
                    },
                    outcome_type=outcome_type,
                    earnings=execution_result.earnings,
                    time_spent=execution_result.duration_seconds / 3600,
                )
                logger.debug(
                    "Stored experience for opportunity %s (outcome=%s)",
                    execution_result.opportunity_id,
                    outcome_type.value,
                )
            except Exception as exc:
                logger.warning("Episodic memory update failed: %s", exc)

    # ── Step 5: Manage resources ──────────────────────────────────────────────

    def _manage_resources(self) -> None:
        """
        Check balance threshold and prioritise earning if balance is low.

        When balance < low_balance_threshold, clears active_work so the next
        cycle will focus on discovering earning opportunities.
        """
        if self.state.balance < self.low_balance_threshold:
            logger.warning(
                "Balance %.2f below threshold %.2f — prioritising earning",
                self.state.balance,
                self.low_balance_threshold,
            )
            # Clear any non-earning active work to force opportunity discovery
            if self.state.active_work is not None:
                logger.info(
                    "Clearing active_work '%s' to prioritise earning",
                    self.state.active_work,
                )
                self.state.active_work = None

            # Update working memory with urgency signal
            if self.working_memory is not None:
                try:
                    add_ctx = getattr(self.working_memory, "add_context", None)
                    if add_ctx is not None:
                        add_ctx(
                            key="resource_urgency",
                            value={
                                "balance": self.state.balance,
                                "threshold": self.low_balance_threshold,
                                "action": "prioritise_earning",
                            },
                        )
                except Exception as exc:
                    logger.warning("Could not update working memory urgency: %s", exc)
        else:
            logger.debug(
                "Balance %.2f is healthy (threshold=%.2f)",
                self.state.balance,
                self.low_balance_threshold,
            )
