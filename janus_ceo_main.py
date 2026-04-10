"""
Janus CEO Agent - Full Autonomous Integration
Combines vision, learning, automation, and strategic decision-making
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("janus_ceo")

from ceo_agent import CEOAgent, GoalPriority
from skill_executor import SkillExecutor
from video_learner import VideoLearner
from local_vision import LocalVisionAnalyzer


class JanusCEOAutonomousAgent:
    """
    Complete CEO-level autonomous agent.
    Combines strategic planning, learning, and automation.
    """
    
    def __init__(self, name: str = "Janus CEO"):
        self.name = name
        self.ceo = CEOAgent(name)
        self.skill_executor = None
        self.video_learner = VideoLearner()
        self.vision_analyzer = LocalVisionAnalyzer()
        
        self.execution_log = []
        self.learning_history = []
        
        logger.info(f"Janus CEO Agent initialized: {name}")
    
    async def initialize(self):
        """Initialize all components."""
        self.skill_executor = SkillExecutor(self.video_learner)
        logger.info("All components initialized")
    
    async def autonomous_ceo_cycle(self, num_cycles: int = 1):
        """
        Main CEO autonomous cycle:
        OBSERVE → PLAN → DECIDE → EXECUTE → REVIEW → IMPROVE
        """
        
        for cycle_num in range(1, num_cycles + 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"AUTONOMOUS CEO CYCLE {cycle_num}/{num_cycles}")
            logger.info(f"{'='*70}")
            
            # PHASE 1: OBSERVE - Get current state
            await self._phase_observe()
            
            # PHASE 2: PLAN - Create or update strategic plans
            await self._phase_plan()
            
            # PHASE 3: DECIDE - Make strategic decisions
            await self._phase_decide()
            
            # PHASE 4: EXECUTE - Run automation tasks
            await self._phase_execute()
            
            # PHASE 5: REVIEW - Analyze results
            await self._phase_review()
            
            # PHASE 6: IMPROVE - Learn and adapt
            await self._phase_improve()
            
            await asyncio.sleep(2)
    
    async def _phase_observe(self):
        """Observe: Analyze current state and opportunities."""
        logger.info("\n[PHASE 1] OBSERVE - Current State Analysis")
        
        # Get financial snapshot
        financial = self.ceo.get_financial_summary()
        logger.info(f"  Financial: ${financial['net_worth']} net worth")
        
        # Count available skills
        available_skills = self.video_learner.list_skills()
        logger.info(f"  Available skills: {len(available_skills)}")
        if available_skills:
            logger.info(f"    Skills: {', '.join(available_skills)}")
        
        # Check active goals
        active_goals = self.ceo.get_active_goals()
        logger.info(f"  Active goals: {len(active_goals)}")
        for goal in active_goals[:3]:
            progress = (goal.current_value / goal.target_value * 100) if goal.target_value > 0 else 0
            logger.info(f"    • {goal.name}: {progress:.0f}% (target: {goal.target_value} {goal.metric})")
    
    async def _phase_plan(self):
        """Plan: Create or update strategic plans."""
        logger.info("\n[PHASE 2] PLAN - Strategic Planning")
        
        active_goals = self.ceo.get_active_goals()
        
        if not active_goals:
            # Auto-create default goals
            logger.info("  Creating default goals...")
            self.ceo.set_goal(
                "Generate Monthly Revenue",
                "Automate high-value financial tasks",
                GoalPriority.HIGH,
                target_value=1000,
                metric="revenue",
                deadline_days=30
            )
            self.ceo.set_goal(
                "Build Savings",
                "Reduce expenses and accumulate reserves",
                GoalPriority.MEDIUM,
                target_value=500,
                metric="savings",
                deadline_days=60
            )
        
        # Create plans for goals without plans
        for goal in self.ceo.get_active_goals():
            if goal.id not in self.ceo.plans:
                logger.info(f"  Creating plan for: {goal.name}")
                plan = self.ceo.create_plan(goal)
                logger.info(f"    Plan has {len(plan.steps)} steps")
    
    async def _phase_decide(self):
        """Decide: Make strategic decisions."""
        logger.info("\n[PHASE 3] DECIDE - Strategic Decision Making")
        
        # Get AI recommendations
        recommendations = self.ceo.recommend_action("What should we do next?")
        
        for rec in recommendations["recommendations"]:
            logger.info(f"  [{rec['priority']}] {rec['action']}")
            logger.info(f"    Reason: {rec['reason']}")
            if "steps" in rec:
                for step in rec["steps"]:
                    logger.info(f"      → {step}")
    
    async def _phase_execute(self):
        """Execute: Run automation tasks."""
        logger.info("\n[PHASE 4] EXECUTE - Task Automation")
        
        available_skills = self.video_learner.list_skills()
        
        if available_skills:
            # Execute top priority skill
            skill_to_execute = available_skills[0]
            logger.info(f"  Executing skill: {skill_to_execute}")
            
            try:
                result = self.skill_executor.execute_skill(skill_to_execute, dry_run=True)
                
                if result.get("success"):
                    # Record successful execution
                    self.ceo.record_transaction(
                        "revenue",
                        100,  # Default value per execution
                        f"Executed skill: {skill_to_execute}"
                    )
                    
                    # Update corresponding goal if revenue-related
                    for goal in self.ceo.get_active_goals():
                        if goal.metric == "revenue":
                            self.ceo.update_goal_progress(goal.id, goal.current_value + 100)
                    
                    logger.info(f"    ✓ Skill executed successfully")
                    self.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "skill": skill_to_execute,
                        "status": "success"
                    })
                else:
                    logger.error(f"    ✗ Skill execution failed: {result.get('error')}")
            except Exception as e:
                logger.error(f"    ✗ Execution error: {e}")
        else:
            logger.info("  No skills available yet - watch videos to learn skills")
    
    async def _phase_review(self):
        """Review: Analyze execution results."""
        logger.info("\n[PHASE 5] REVIEW - Results Analysis")
        
        # Review recent executions
        recent_executions = self.execution_log[-5:]
        if recent_executions:
            successful = len([e for e in recent_executions if e["status"] == "success"])
            logger.info(f"  Recent executions: {successful}/{len(recent_executions)} successful")
        
        # Review financial performance
        financial = self.ceo.get_financial_summary()
        logger.info(f"  Financial performance:")
        logger.info(f"    Revenue: ${financial['total_revenue']}")
        logger.info(f"    Expenses: ${financial['total_expenses']}")
        logger.info(f"    Profit: ${financial['net_profit']}")
        logger.info(f"    Net worth: ${financial['net_worth']}")
    
    async def _phase_improve(self):
        """Improve: Learn from experience and adapt."""
        logger.info("\n[PHASE 6] IMPROVE - Learning & Adaptation")
        
        # Check for new videos to learn from
        video_dir = Path("videos")
        if video_dir.exists():
            video_files = list(video_dir.glob("*.mp4"))
            existing_skills = self.video_learner.list_skills()
            
            # Learn from new videos
            for video_file in video_files[:1]:  # One per cycle
                skill_name = video_file.stem.lower()
                if skill_name not in existing_skills:
                    logger.info(f"  Learning from new video: {video_file.name}")
                    try:
                        skill = self.video_learner.create_skill_from_video(
                            str(video_file),
                            skill_name,
                            self.vision_analyzer
                        )
                        logger.info(f"    ✓ Learned skill with {len(skill['actions'])} actions")
                        self.learning_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "skill": skill_name,
                            "actions": len(skill["actions"]),
                            "confidence": skill["confidence"]
                        })
                    except Exception as e:
                        logger.error(f"    ✗ Learning failed: {e}")
        
        # Adjust strategy based on performance
        goals = self.ceo.get_active_goals()
        for goal in goals:
            if goal.current_value >= goal.target_value:
                logger.info(f"  ✓ Goal achieved: {goal.name}")
            elif goal.deadline < datetime.now():
                logger.warning(f"  ⚠ Goal overdue: {goal.name}")
    
    async def generate_executive_summary(self) -> Dict:
        """Generate executive summary of agent performance."""
        
        report = self.ceo.get_performance_report()
        
        summary = {
            "report_date": datetime.now().isoformat(),
            "agent_name": self.name,
            "ceo_status": {
                "active_goals": report["goals"]["active"],
                "completed_goals": report["goals"]["completed"],
                "total_decisions": report["decisions"]["total_made"],
                "positive_decisions": report["decisions"]["positive_impact"]
            },
            "financial_status": {
                "net_worth": report["financial"]["net_worth"],
                "cash": report["financial"]["cash"],
                "revenue": report["financial"]["total_revenue"],
                "expenses": report["financial"]["total_expenses"],
                "profit": report["financial"]["net_profit"]
            },
            "operations": {
                "skills_learned": len(self.video_learner.list_skills()),
                "executions_completed": len(self.execution_log),
                "execution_success_rate": len([e for e in self.execution_log if e["status"] == "success"]) / max(len(self.execution_log), 1)
            },
            "recommendations": self.ceo.recommend_action("Summary")["recommendations"]
        }
        
        return summary
    
    async def start(self, num_cycles: int = 1):
        """Start the CEO agent."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING {self.name} - CEO-LEVEL AUTONOMOUS AGENT")
        logger.info(f"{'='*70}")
        
        try:
            await self.initialize()
            await self.autonomous_ceo_cycle(num_cycles)
            
            # Generate summary
            summary = await self.generate_executive_summary()
            
            logger.info(f"\n{'='*70}")
            logger.info("EXECUTIVE SUMMARY")
            logger.info(f"{'='*70}")
            logger.info(json.dumps(summary, indent=2))
            
            # Save state
            self.ceo.save_state("ceo_state.json")
            
            # Save learning history
            with open("learning_history.json", "w") as f:
                json.dump(self.learning_history, f, indent=2)
            
            logger.info("\n✓ CEO agent cycle completed successfully")
            
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)


async def main():
    """Entry point."""
    agent = JanusCEOAutonomousAgent("Janus CEO")
    await agent.start(num_cycles=1)


if __name__ == "__main__":
    asyncio.run(main())
