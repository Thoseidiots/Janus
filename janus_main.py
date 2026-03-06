"""
Enhanced Janus Agent with Video Learning
Main autonomous loop that learns from videos and executes skills
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("janus")

try:
    from video_learner import VideoLearner, SimpleVisionAnalyzer, SkillExecutor
    HAS_VIDEO_LEARNING = True
except ImportError:
    logger.warning("video_learner module not available")
    HAS_VIDEO_LEARNING = False


class JanusAutonomousAgent:
    """
    Autonomous AI agent that learns from videos and executes skills.
    Can be run headless or with GUI.
    """
    
    def __init__(self, mode: str = "headless"):
        self.mode = mode
        self.running = False
        self.identity = self.load_identity()
        self.task_log = []
        self.learning_stats = {}
        
        # Initialize video learning if available
        if HAS_VIDEO_LEARNING:
            self.video_learner = VideoLearner()
            self.vision_analyzer = SimpleVisionAnalyzer()
            self.skill_executor = SkillExecutor(self.video_learner)
            logger.info("Video learning module initialized")
        else:
            logger.warning("Video learning disabled (optional dependencies missing)")
        
        logger.info(f"Janus {self.identity.get('version', '0.1.0')} initialized (mode: {mode})")
    
    def load_identity(self) -> dict:
        """Load persistent identity."""
        identity_file = Path("identity_object.json")
        if identity_file.exists():
            try:
                with open(identity_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "name": "Janus",
            "version": "0.2.0",
            "mode": "autonomous_learner",
            "created": datetime.now().isoformat()
        }
    
    def log_task(self, task_name: str, status: str, details: Dict = None):
        """Log a task execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task_name,
            "status": status,
            "details": details or {}
        }
        self.task_log.append(entry)
        logger.info(f"[{status.upper()}] {task_name}")
        if details:
            logger.debug(f"  Details: {details}")
    
    async def watch_and_learn(self, video_directory: str = "videos") -> Dict:
        """
        Watch tutorial videos and learn skills autonomously.
        """
        if not HAS_VIDEO_LEARNING:
            logger.error("Video learning not available")
            return {"success": False, "error": "Video learning disabled"}
        
        logger.info(f"Starting learning phase - watching videos from {video_directory}")
        self.log_task("watch_and_learn", "started", {"source": video_directory})
        
        video_dir = Path(video_directory)
        if not video_dir.exists():
            video_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created video directory: {video_directory}")
            logger.info("Place tutorial videos here and re-run")
            return {
                "success": False,
                "message": f"Video directory created: {video_directory}",
                "next_step": "Add tutorial videos and restart agent"
            }
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
        
        if not videos:
            logger.info(f"No videos found in {video_directory}")
            return {
                "success": False,
                "message": "No tutorial videos found",
                "video_directory": str(video_directory),
                "next_step": "Add video files to the videos/ directory"
            }
        
        logger.info(f"Found {len(videos)} video(s) to learn from")
        
        learned_skills = []
        for video_path in videos[:5]:  # Limit to 5 videos per run
            try:
                logger.info(f"\nLearning from: {video_path.name}")
                skill_name = video_path.stem.replace(" ", "_").lower()
                
                skill = self.video_learner.create_skill_from_video(
                    str(video_path),
                    skill_name,
                    self.vision_analyzer,
                    description=f"Learned from {video_path.name}"
                )
                
                learned_skills.append(skill_name)
                self.log_task("learn_skill", "completed", {
                    "skill": skill_name,
                    "actions": len(skill["actions"]),
                    "confidence": skill["confidence"]
                })
                
                await asyncio.sleep(1)  # Small delay between videos
            
            except Exception as e:
                logger.error(f"Error learning from {video_path.name}: {e}")
                self.log_task("learn_skill", "failed", {"video": str(video_path), "error": str(e)})
        
        self.learning_stats = {
            "total_videos_processed": len(videos[:5]),
            "skills_learned": len(learned_skills),
            "learned_skills": learned_skills
        }
        
        self.log_task("watch_and_learn", "completed", self.learning_stats)
        return {
            "success": True,
            "skills_learned": learned_skills,
            "total": len(videos)
        }
    
    async def execute_learned_skill(self, skill_name: str, dry_run: bool = True) -> Dict:
        """Execute a skill the agent learned from video."""
        
        if not HAS_VIDEO_LEARNING:
            logger.error("Video learning not available")
            return {"success": False, "error": "Video learning disabled"}
        
        logger.info(f"Executing skill: {skill_name}")
        self.log_task("execute_skill", "started", {"skill": skill_name, "dry_run": dry_run})
        
        try:
            result = self.skill_executor.execute_skill(skill_name, dry_run=dry_run)
            self.log_task("execute_skill", "completed", result)
            return result
        except Exception as e:
            logger.error(f"Error executing skill: {e}")
            self.log_task("execute_skill", "failed", {"skill": skill_name, "error": str(e)})
            return {"success": False, "error": str(e)}
    
    async def autonomous_cycle(self, num_cycles: int = 3):
        """
        Main autonomous cycle:
        1. Watch and learn from videos
        2. Build skill library
        3. Execute skills
        4. Improve and iterate
        """
        logger.info(f"Starting {num_cycles} autonomous cycles")
        
        for cycle in range(1, num_cycles + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"AUTONOMOUS CYCLE {cycle}/{num_cycles}")
            logger.info(f"{'='*60}")
            
            # Phase 1: Observe & Learn
            logger.info("\n[PHASE 1] OBSERVE & LEARN")
            learn_result = await self.watch_and_learn("videos")
            
            if learn_result.get("skills_learned"):
                # Phase 2: Execute
                logger.info("\n[PHASE 2] EXECUTE & TEST")
                for skill in learn_result["skills_learned"]:
                    await self.execute_learned_skill(skill, dry_run=True)
                    await asyncio.sleep(0.5)
            
            # Phase 3: Reflect & Improve
            logger.info("\n[PHASE 3] REFLECT & IMPROVE")
            self.log_task("reflect", "in_progress", {
                "cycle": cycle,
                "skills_count": len(self.video_learner.list_skills())
            })
            
            # Phase 4: Plan Next Actions
            logger.info("\n[PHASE 4] PLAN NEXT ACTIONS")
            next_actions = self.plan_next_actions()
            self.log_task("plan", "completed", next_actions)
            
            await asyncio.sleep(2)
        
        logger.info("\n" + "="*60)
        logger.info("AUTONOMOUS CYCLES COMPLETED")
        logger.info("="*60)
    
    def plan_next_actions(self) -> Dict:
        """Decide what to do next based on learned skills."""
        skills = self.video_learner.list_skills()
        
        if not skills:
            return {
                "next_step": "Continue learning",
                "reason": "No skills in library yet",
                "action": "Wait for more videos"
            }
        
        return {
            "next_step": "Execute skills",
            "available_skills": skills,
            "reason": f"Learned {len(skills)} skills, ready to execute",
            "action": "Start task execution phase"
        }
    
    async def start(self):
        """Start the autonomous agent."""
        logger.info(f"Starting Janus Autonomous Agent: {self.identity.get('name')}")
        logger.info(f"Mode: {self.mode}")
        
        self.running = True
        
        try:
            # Run autonomous learning cycles
            await self.autonomous_cycle(num_cycles=1)
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("SUMMARY")
            logger.info("="*60)
            logger.info(self.video_learner.get_skill_summary() if HAS_VIDEO_LEARNING else "Video learning disabled")
            logger.info(f"Total tasks logged: {len(self.task_log)}")
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown and save state."""
        self.running = False
        logger.info("Shutting down Janus Agent")
        
        # Save task log
        log_file = Path("janus_task_log.json")
        with open(log_file, 'w') as f:
            json.dump(self.task_log, f, indent=2)
        logger.info(f"Task log saved to {log_file}")
        
        # Save skill library
        if HAS_VIDEO_LEARNING:
            logger.info(f"Skill library saved with {len(self.video_learner.list_skills())} skills")


async def main():
    """Entry point."""
    agent = JanusAutonomousAgent(mode="learning")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
