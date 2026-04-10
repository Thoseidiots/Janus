"""
janus_system_orchestrator.py
==============================
Master coordinator for the Janus autonomous system.
Connects all subsystems into a single running loop.

Architecture:
    AvusInference          — the brain
    ScreenInterpreter      — vision
    SkillExecutor          — hands (OS control)
    GameGenerationPipeline — creative output
    CEOAgent               — strategy and goals
    AgentLoop              — observe/plan/act cycle

Usage:
    from janus_system_orchestrator import JanusOrchestrator

    janus = JanusOrchestrator()
    janus.start()

    # Or run a single cycle
    janus.load()
    result = janus.cycle()

    # Or give it a specific goal
    janus.load()
    janus.set_goal("Generate a AAA game environment asset pack")
    janus.run(cycles=10)
"""

import os
import sys
import json
import time
import asyncio
import logging
import platform
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
log = logging.getLogger("janus.orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    candidates = [
        Path(__file__).parent,
        Path("/kaggle/working/Janus"),
        Path("/teamspace/studios/this_studio/Janus"),
        Path(os.path.expanduser("~/Janus")),
        Path("C:/Users") / os.environ.get("USERNAME", "") / "Janus",
    ]
    for p in candidates:
        if (p / "model.py").exists():
            return p
    return Path.cwd()


REPO_ROOT = _find_repo_root()

def _add_path(p):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

_add_path(REPO_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# CycleResult — output of one orchestrator cycle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleResult:
    cycle_num:      int
    timestamp:      str           = field(default_factory=lambda: datetime.now().isoformat())
    phase:          str           = "idle"
    goal:           Optional[str] = None
    screen_desc:    Optional[str] = None
    avus_output:    Optional[str] = None
    action_taken:   Optional[Dict]= None
    asset_generated:bool          = False
    asset_name:     Optional[str] = None
    success:        bool          = False
    error:          Optional[str] = None
    duration_ms:    float         = 0.0
    notes:          List[str]     = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cycle":          self.cycle_num,
            "timestamp":      self.timestamp,
            "phase":          self.phase,
            "goal":           self.goal,
            "screen_desc":    self.screen_desc,
            "avus_output":    self.avus_output,
            "action_taken":   self.action_taken,
            "asset_generated":self.asset_generated,
            "asset_name":     self.asset_name,
            "success":        self.success,
            "error":          self.error,
            "duration_ms":    self.duration_ms,
            "notes":          self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# JanusOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class JanusOrchestrator:
    """
    Master coordinator for all Janus subsystems.

    Runs the core loop:
        OBSERVE  → read screen / system state
        PLAN     → CEO decides what to do
        ACT      → Avus generates output or action
        EXECUTE  → SkillExecutor / GamePipeline acts
        REVIEW   → record result, update CEO
        IMPROVE  → adjust strategy

    Degrades gracefully — each subsystem is optional.
    If a component fails to load the others continue.
    """

    def __init__(self, name: str = "Janus"):
        self.name         = name
        self.cycle_count  = 0
        self.history:     List[CycleResult] = []
        self.current_goal: Optional[str]    = None
        self._running     = False

        # Subsystems (loaded lazily)
        self.avus          = None
        self.screen        = None
        self.skill_exec    = None
        self.game_pipeline = None
        self.ceo           = None
        self.janus_os      = None

        # Subsystem availability flags
        self._has_avus     = False
        self._has_screen   = False
        self._has_skills   = False
        self._has_game     = False
        self._has_ceo      = False
        self._has_os       = False

        # State persistence
        self._state_file   = REPO_ROOT / "janus_orchestrator_state.json"

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, weights_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Load all subsystems. Returns dict of component availability.
        """
        log.info(f"Loading {self.name} orchestrator...")

        self._load_avus(weights_path)
        self._load_screen_interpreter()
        self._load_skill_executor()
        self._load_game_pipeline(weights_path)
        self._load_ceo()
        self._load_janus_os()

        status = self._status()
        log.info(f"Load complete: {status}")
        return status

    def _load_avus(self, weights_path):
        try:
            from avus_inference import AvusInference
            self.avus      = AvusInference()
            self._has_avus = self.avus.load(weights_path)
            log.info(f"Avus: {'OK' if self._has_avus else 'FAILED'}")
        except Exception as e:
            log.warning(f"Avus load failed: {e}")
            self._has_avus = False

    def _load_screen_interpreter(self):
        try:
            from screen_interpreter import ScreenInterpreter
            self.screen      = ScreenInterpreter()
            self._has_screen = True
            log.info(f"ScreenInterpreter: OK  caps={self.screen.capabilities}")
        except Exception as e:
            log.warning(f"ScreenInterpreter load failed: {e}")
            self._has_screen = False

    def _load_skill_executor(self):
        try:
            from skill_executor import SkillExecutor

            # SkillExecutor needs a video_learner — use a stub if not available
            try:
                from video_learner import VideoLearner
                vl = VideoLearner()
            except Exception:
                vl = _StubVideoLearner()

            self.skill_exec  = SkillExecutor(vl)
            self._has_skills = True
            log.info("SkillExecutor: OK")
        except Exception as e:
            log.warning(f"SkillExecutor load failed: {e}")
            self._has_skills = False

    def _load_game_pipeline(self, weights_path):
        try:
            from game_generation_pipeline import GameGenerationPipeline
            self.game_pipeline = GameGenerationPipeline()
            self._has_game     = self.game_pipeline.load(weights_path)
            log.info(f"GamePipeline: {'OK' if self._has_game else 'partial'}")
        except Exception as e:
            log.warning(f"GamePipeline load failed: {e}")
            self._has_game = False

    def _load_ceo(self):
        try:
            from ceo_agent import CEOAgent, GoalPriority
            self.ceo      = CEOAgent(self.name)
            self._has_ceo = True
            log.info("CEOAgent: OK")

            # Set default goals
            self.ceo.set_goal(
                "Generate AAA Game Assets",
                "Use Avus and aaa_stack to generate high quality game assets",
                GoalPriority.HIGH,
                target_value=100,
                metric="assets_generated",
                deadline_days=30
            )
            self.ceo.set_goal(
                "Autonomous Task Completion",
                "Complete desktop tasks autonomously using screen vision",
                GoalPriority.MEDIUM,
                target_value=50,
                metric="tasks_completed",
                deadline_days=30
            )
        except Exception as e:
            log.warning(f"CEOAgent load failed: {e}")
            self._has_ceo = False

    def _load_janus_os(self):
        if platform.system() != "Windows":
            log.info("JanusOS: skipped (not Windows)")
            return
        try:
            from os_human_interface import JanusOS
            self.janus_os  = JanusOS()
            self._has_os   = True
            log.info("JanusOS: OK")
        except Exception as e:
            log.warning(f"JanusOS load failed: {e}")
            self._has_os = False

    # ── goal management ───────────────────────────────────────────────────────

    def set_goal(self, goal: str):
        """Set the current task goal for the next cycle."""
        self.current_goal = goal
        log.info(f"Goal set: {goal}")

    # ── main cycle ────────────────────────────────────────────────────────────

    def cycle(self) -> CycleResult:
        """
        Run one full OBSERVE → PLAN → ACT → EXECUTE → REVIEW cycle.
        Returns a CycleResult with full details.
        """
        self.cycle_count += 1
        t0     = time.time()
        result = CycleResult(cycle_num=self.cycle_count)
        result.goal = self.current_goal

        log.info(f"\n{'═'*60}")
        log.info(f"CYCLE {self.cycle_count}  goal={self.current_goal or 'auto'}")
        log.info(f"{'═'*60}")

        try:
            # ── PHASE 1: OBSERVE ──────────────────────────────────
            result.phase       = "observe"
            result.screen_desc = self._observe()
            result.notes.append(f"Screen: {result.screen_desc[:80]}...")

            # ── PHASE 2: PLAN ─────────────────────────────────────
            result.phase = "plan"
            plan         = self._plan(result.screen_desc)
            result.notes.append(f"Plan: {plan}")

            # ── PHASE 3: ACT ──────────────────────────────────────
            result.phase       = "act"
            avus_out, act_type = self._act(plan, result.screen_desc)
            result.avus_output = avus_out
            result.notes.append(f"Act type: {act_type}  output: {str(avus_out)[:80]}")

            # ── PHASE 4: EXECUTE ──────────────────────────────────
            result.phase = "execute"
            exec_result  = self._execute(avus_out, act_type)
            result.notes.append(f"Execute: {exec_result}")

            if act_type == "action" and isinstance(avus_out, dict):
                result.action_taken = avus_out
            elif act_type == "asset" and exec_result:
                result.asset_generated = True
                result.asset_name      = exec_result.get("name")

            # ── PHASE 5: REVIEW ───────────────────────────────────
            result.phase   = "review"
            result.success = True
            self._review(result)

        except Exception as e:
            result.error   = str(e)
            result.success = False
            log.error(f"Cycle {self.cycle_count} failed: {e}")
            traceback.print_exc()

        result.duration_ms = (time.time() - t0) * 1000
        result.phase       = "complete"

        self.history.append(result)
        self._save_state()

        log.info(f"Cycle {self.cycle_count} "
                 f"{'OK' if result.success else 'FAILED'} "
                 f"in {result.duration_ms:.0f}ms")
        return result

    def run(self, cycles: int = 10,
            delay_seconds: float = 2.0) -> List[CycleResult]:
        """
        Run multiple cycles with a delay between them.
        Returns list of all CycleResults.
        """
        self._running = True
        results       = []

        log.info(f"Starting {self.name} — {cycles} cycles, "
                 f"{delay_seconds}s delay between cycles")

        for i in range(cycles):
            if not self._running:
                log.info("Stopped by request.")
                break

            result = self.cycle()
            results.append(result)

            if i < cycles - 1:
                time.sleep(delay_seconds)

        self._running = False
        self._print_summary(results)
        return results

    def stop(self):
        """Signal the run() loop to stop after the current cycle."""
        self._running = False

    # ── phase implementations ─────────────────────────────────────────────────

    def _observe(self) -> str:
        """OBSERVE: Read screen state or use goal as context."""
        # Try real screen capture first
        if self._has_os and self._has_screen:
            try:
                raw, w, h = self.janus_os.capture_screen()
                return self.screen.interpret(raw, w, h, self.current_goal)
            except Exception as e:
                log.warning(f"Screen capture failed: {e}")

        # Fall back to mock description based on goal
        if self._has_screen and self.current_goal:
            return self.screen.mock_description(
                "Janus Workspace",
                [f"Working toward goal: {self.current_goal}"],
                goal=self.current_goal
            )

        return f"System state nominal. Current goal: {self.current_goal or 'none set'}."

    def _plan(self, observation: str) -> str:
        """PLAN: CEO decides what to do based on observation."""
        if self._has_ceo:
            try:
                recs = self.ceo.recommend_action(observation)
                if recs["recommendations"]:
                    top = recs["recommendations"][0]
                    return top.get("action", "continue current task")
            except Exception as e:
                log.warning(f"CEO plan failed: {e}")

        # Simple fallback planning
        if self.current_goal:
            if "game" in self.current_goal.lower() or "asset" in self.current_goal.lower():
                return f"generate_asset: {self.current_goal}"
            elif "click" in self.current_goal.lower() or "screen" in self.current_goal.lower():
                return f"screen_action: {self.current_goal}"
            else:
                return f"process: {self.current_goal}"

        return "generate_asset: a procedural stone pillar"

    def _act(self, plan: str, observation: str):
        """
        ACT: Avus generates output based on plan.
        Returns (output, act_type) where act_type is 'action', 'asset', or 'text'.
        """
        if not self._has_avus:
            return self._fallback_act(plan)

        plan_lower = plan.lower()

        # Screen action
        if any(k in plan_lower for k in ["screen_action", "click", "type",
                                          "scroll", "press"]):
            action = self.avus.generate_action(observation)
            if action:
                return action, "action"

        # Game asset generation
        if any(k in plan_lower for k in ["generate_asset", "game", "asset",
                                          "3d", "terrain", "character"]):
            prompt = self.current_goal or plan.replace("generate_asset:", "").strip()
            params = self.avus.generate_3d_params(prompt)
            return params or {}, "asset"

        # General text
        output = self.avus.generate(plan, max_new_tokens=128)
        return output, "text"

    def _fallback_act(self, plan: str):
        """Act without Avus — use structured defaults."""
        if "asset" in plan.lower() or "game" in plan.lower():
            return {
                "object_name": "default_asset",
                "geometry":    {"primitive_type": "sphere",
                                "geometry_params": {"radius": 1.0}},
                "material":    {"material_type": "stone",
                                "material_params": {"resolution": [512, 512]}}
            }, "asset"
        return {"type": "wait", "duration": 1.0}, "action"

    def _execute(self, output: Any, act_type: str) -> Optional[Dict]:
        """EXECUTE: Carry out the action or generate the asset."""

        if act_type == "action":
            return self._execute_action(output)

        elif act_type == "asset":
            return self._execute_asset(output)

        elif act_type == "text":
            log.info(f"Text output: {str(output)[:200]}")
            return {"type": "text", "output": str(output)[:200]}

        return None

    def _execute_action(self, action: Any) -> Optional[Dict]:
        """Execute a desktop action via SkillExecutor."""
        if not isinstance(action, dict):
            return None

        if self._has_skills:
            try:
                # Build a minimal skill with this single action
                skill_name = f"dynamic_{action.get('type', 'action')}"
                dry_run    = not self._has_os  # only real execution on Windows

                # Direct dispatch via skill_executor
                result = self.skill_exec._execute_action(action)
                log.info(f"Action executed: {action.get('type')} → {result}")
                return result
            except Exception as e:
                log.warning(f"Action execution failed: {e}")

        log.info(f"[DRY] Action: {action}")
        return {"type": action.get("type", "unknown"), "status": "dry_run"}

    def _execute_asset(self, params: Any) -> Optional[Dict]:
        """Generate a game asset via GameGenerationPipeline."""
        if not self._has_game:
            log.info(f"[DRY] Would generate asset with params: "
                     f"{str(params)[:100]}")
            return {"type": "asset", "status": "dry_run"}

        try:
            prompt = self.current_goal or "A procedural game asset"
            asset  = self.game_pipeline.generate(prompt)
            log.info(f"Asset generated: {asset.summary()}")

            if self._has_ceo:
                # Update CEO goal progress
                for goal in self.ceo.get_active_goals():
                    if goal.metric == "assets_generated":
                        self.ceo.update_goal_progress(
                            goal.id, goal.current_value + 1)

            return asset.to_dict()
        except Exception as e:
            log.error(f"Asset generation failed: {e}")
            return None

    def _review(self, result: CycleResult):
        """REVIEW: Update CEO with results, log performance."""
        if not self._has_ceo:
            return

        try:
            if result.success:
                self.ceo.record_transaction(
                    "revenue", 0,
                    f"Cycle {result.cycle_count} completed: {result.phase}")

            # Update task completion goal
            for goal in self.ceo.get_active_goals():
                if goal.metric == "tasks_completed" and result.success:
                    self.ceo.update_goal_progress(
                        goal.id, goal.current_value + 1)
        except Exception as e:
            log.warning(f"Review update failed: {e}")

    # ── state persistence ─────────────────────────────────────────────────────

    def _save_state(self):
        """Save orchestrator state to disk."""
        try:
            state = {
                "name":         self.name,
                "cycle_count":  self.cycle_count,
                "current_goal": self.current_goal,
                "timestamp":    datetime.now().isoformat(),
                "status":       self._status(),
                "last_cycle":   self.history[-1].to_dict() if self.history else None,
            }
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"State save failed: {e}")

    def load_state(self) -> bool:
        """Restore orchestrator state from disk."""
        if not self._state_file.exists():
            return False
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            self.cycle_count  = state.get("cycle_count", 0)
            self.current_goal = state.get("current_goal")
            log.info(f"State restored: cycle={self.cycle_count} "
                     f"goal={self.current_goal}")
            return True
        except Exception:
            return False

    # ── reporting ─────────────────────────────────────────────────────────────

    def _status(self) -> Dict[str, bool]:
        return {
            "avus":     self._has_avus,
            "screen":   self._has_screen,
            "skills":   self._has_skills,
            "game":     self._has_game,
            "ceo":      self._has_ceo,
            "os":       self._has_os,
        }

    def status(self) -> str:
        s = self._status()
        lines = [f"Janus Orchestrator — {self.name}",
                 f"Cycles run: {self.cycle_count}",
                 f"Current goal: {self.current_goal or 'none'}",
                 "Components:"]
        for k, v in s.items():
            lines.append(f"  {k:10s} {'✅' if v else '❌'}")
        return "\n".join(lines)

    def _print_summary(self, results: List[CycleResult]):
        total   = len(results)
        success = sum(1 for r in results if r.success)
        assets  = sum(1 for r in results if r.asset_generated)
        avg_ms  = sum(r.duration_ms for r in results) / max(total, 1)
        print(f"\n{'═'*60}")
        print(f"JANUS RUN SUMMARY")
        print(f"{'═'*60}")
        print(f"Cycles:        {total}")
        print(f"Successful:    {success}/{total}")
        print(f"Assets made:   {assets}")
        print(f"Avg cycle time:{avg_ms:.0f}ms")
        if self._has_ceo:
            report = self.ceo.get_performance_report()
            print(f"CEO goals done:{report['goals']['completed']}")
        print(f"{'═'*60}\n")

    def __repr__(self):
        return (f"JanusOrchestrator(name={self.name}, "
                f"cycles={self.cycle_count}, "
                f"components={self._status()})")


# ─────────────────────────────────────────────────────────────────────────────
# Stub video learner (used when real VideoLearner not available)
# ─────────────────────────────────────────────────────────────────────────────

class _StubVideoLearner:
    def list_skills(self):  return []
    def get_skill(self, name): return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Starting Janus Orchestrator...\n")

    janus = JanusOrchestrator("Janus")
    janus.load()

    print("\n" + janus.status() + "\n")

    # Run a demo with game generation goal
    janus.set_goal("Generate a AAA quality stone fortress wall asset")
    results = janus.run(cycles=3, delay_seconds=1.0)

    # Show last result
    if results:
        print("\nLast cycle result:")
        print(json.dumps(results[-1].to_dict(), indent=2))


if __name__ == "__main__":
    main()
