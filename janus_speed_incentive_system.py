"""
Janus Speed Incentive System - 15 Minute Challenge

Implements a time-based incentive system that rewards the AI for completing
tasks quickly and efficiently. The faster the completion, the higher the bonus.

15-MINUTE INCENTIVE STRUCTURE:
- < 5 minutes: 50% bonus
- < 10 minutes: 25% bonus  
- < 15 minutes: 10% bonus
- > 15 minutes: No bonus

This encourages efficiency without sacrificing quality.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeedTier(Enum):
    """Speed incentive tiers"""
    LIGHTNING = "lightning"    # < 5 minutes
    FAST = "fast"             # < 10 minutes
    STANDARD = "standard"     # < 15 minutes
    SLOW = "slow"             # > 15 minutes

@dataclass
class SpeedIncentive:
    """Speed incentive configuration"""
    tier: SpeedTier
    time_limit_minutes: int
    bonus_percentage: float
    bonus_multiplier: float
    quality_requirement: float  # Minimum quality score
    reward_description: str

@dataclass
class TaskPerformance:
    """Task performance metrics"""
    task_id: str
    service_type: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    quality_score: float
    base_value: float
    speed_bonus: float
    total_earned: float
    speed_tier: SpeedTier
    efficiency_rating: float

class JanusSpeedIncentiveSystem:
    """15-minute speed incentive system"""
    
    def __init__(self):
        self.incentives = self._initialize_incentives()
        self.task_history = []
        self.speed_stats = {
            "total_tasks": 0,
            "lightning_tasks": 0,
            "fast_tasks": 0,
            "standard_tasks": 0,
            "slow_tasks": 0,
            "average_completion_time": 0.0,
            "total_bonus_earned": 0.0,
            "efficiency_improvement": 0.0
        }
        
        # AI motivation prompts
        self.motivation_prompts = {
            "lightning": "LIGHTNING SPEED CHALLENGE! Complete this in under 5 minutes for a 50% bonus! Be quick but maintain quality!",
            "fast": "SPEED BONUS! Complete in under 10 minutes for a 25% bonus! Focus on efficiency!",
            "standard": "EFFICIENCY REWARD! Complete in under 15 minutes for a 10% bonus! Stay focused!",
            "slow": "STANDARD COMPLETION. Take your time to ensure quality. No time pressure."
        }
        
        logger.info("Janus Speed Incentive System initialized")
    
    def _initialize_incentives(self) -> Dict[SpeedTier, SpeedIncentive]:
        """Initialize speed incentive tiers"""
        return {
            SpeedTier.LIGHTNING: SpeedIncentive(
                tier=SpeedTier.LIGHTNING,
                time_limit_minutes=5,
                bonus_percentage=0.50,  # 50% bonus
                bonus_multiplier=1.5,
                quality_requirement=0.85,  # 85% quality minimum
                reward_description="LIGHTNING SPEED - 50% bonus!"
            ),
            SpeedTier.FAST: SpeedIncentive(
                tier=SpeedTier.FAST,
                time_limit_minutes=10,
                bonus_percentage=0.25,  # 25% bonus
                bonus_multiplier=1.25,
                quality_requirement=0.80,  # 80% quality minimum
                reward_description="FAST COMPLETION - 25% bonus!"
            ),
            SpeedTier.STANDARD: SpeedIncentive(
                tier=SpeedTier.STANDARD,
                time_limit_minutes=15,
                bonus_percentage=0.10,  # 10% bonus
                bonus_multiplier=1.10,
                quality_requirement=0.75,  # 75% quality minimum
                reward_description="EFFICIENT WORK - 10% bonus!"
            ),
            SpeedTier.SLOW: SpeedIncentive(
                tier=SpeedTier.SLOW,
                time_limit_minutes=999,  # No limit
                bonus_percentage=0.00,  # No bonus
                bonus_multiplier=1.0,
                quality_requirement=0.70,  # 70% quality minimum
                reward_description="STANDARD COMPLETION - No time bonus"
            )
        }
    
    def start_speed_challenge(self, task_id: str, service_type: str, base_value: float) -> Dict:
        """Start a speed challenge for a task"""
        start_time = datetime.now()
        
        # Determine target tier based on task complexity
        target_tier = self._determine_target_tier(service_type, base_value)
        incentive = self.incentives[target_tier]
        
        challenge = {
            "task_id": task_id,
            "service_type": service_type,
            "base_value": base_value,
            "start_time": start_time,
            "target_tier": target_tier,
            "incentive": incentive,
            "motivation_prompt": self.motivation_prompts[target_tier.value],
            "time_limit": start_time + timedelta(minutes=incentive.time_limit_minutes),
            "potential_bonus": base_value * incentive.bonus_percentage,
            "potential_total": base_value * incentive.bonus_multiplier
        }
        
        logger.info(f"Speed challenge started for {task_id}")
        logger.info(f"Target: {target_tier.value.upper()} - {incentive.reward_description}")
        logger.info(f"Potential bonus: ${challenge['potential_bonus']:.2f}")
        logger.info(f"Potential total: ${challenge['potential_total']:.2f}")
        
        return challenge
    
    def _determine_target_tier(self, service_type: str, base_value: float) -> SpeedTier:
        """Determine appropriate speed tier based on task"""
        # Simple tasks can aim for lightning speed
        simple_tasks = ["content_writing", "data_analysis", "basic_research"]
        if service_type in simple_tasks and base_value < 200:
            return SpeedTier.LIGHTNING
        
        # Medium tasks can aim for fast completion
        medium_tasks = ["code_development", "business_analysis", "market_research"]
        if service_type in medium_tasks and base_value < 500:
            return SpeedTier.FAST
        
        # Complex tasks get standard time
        return SpeedTier.STANDARD
    
    def generate_speed_aware_prompt(self, challenge: Dict, original_prompt: str) -> str:
        """Generate a prompt that includes speed motivation"""
        motivation = challenge["motivation_prompt"]
        time_remaining = challenge["time_limit"] - datetime.now()
        
        if time_remaining.total_seconds() > 0:
            minutes_left = int(time_remaining.total_seconds() / 60)
            seconds_left = int(time_remaining.total_seconds() % 60)
            
            speed_prompt = f"""
{motivation}

TIME REMAINING: {minutes_left}:{seconds_left:02d}

ORIGINAL TASK:
{original_prompt}

REMEMBER: Quality matters! Complete quickly but ensure the work meets the quality standard.
"""
        else:
            speed_prompt = f"""
Time limit exceeded! Focus on quality completion.

ORIGINAL TASK:
{original_prompt}
"""
        
        return speed_prompt
    
    def complete_speed_challenge(self, challenge: Dict, work_content: str, quality_score: float) -> TaskPerformance:
        """Complete a speed challenge and calculate rewards"""
        end_time = datetime.now()
        duration = (end_time - challenge["start_time"]).total_seconds()
        
        # Determine actual speed tier achieved
        speed_tier = self._calculate_speed_tier(duration)
        incentive = self.incentives[speed_tier]
        
        # Calculate speed bonus
        speed_bonus = 0.0
        if speed_tier != SpeedTier.SLOW and quality_score >= incentive.quality_requirement:
            speed_bonus = challenge["base_value"] * incentive.bonus_percentage
        
        # Calculate total earned
        total_earned = challenge["base_value"] + speed_bonus
        
        # Calculate efficiency rating
        efficiency_rating = self._calculate_efficiency_rating(duration, quality_score, speed_tier)
        
        performance = TaskPerformance(
            task_id=challenge["task_id"],
            service_type=challenge["service_type"],
            start_time=challenge["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            quality_score=quality_score,
            base_value=challenge["base_value"],
            speed_bonus=speed_bonus,
            total_earned=total_earned,
            speed_tier=speed_tier,
            efficiency_rating=efficiency_rating
        )
        
        # Update statistics
        self._update_speed_stats(performance)
        
        # Log results
        self._log_performance_results(performance, challenge)
        
        return performance
    
    def _calculate_speed_tier(self, duration_seconds: float) -> SpeedTier:
        """Calculate speed tier based on completion time"""
        minutes = duration_seconds / 60
        
        if minutes < 5:
            return SpeedTier.LIGHTNING
        elif minutes < 10:
            return SpeedTier.FAST
        elif minutes < 15:
            return SpeedTier.STANDARD
        else:
            return SpeedTier.SLOW
    
    def _calculate_efficiency_rating(self, duration: float, quality: float, tier: SpeedTier) -> float:
        """Calculate overall efficiency rating"""
        # Base efficiency from speed
        if tier == SpeedTier.LIGHTNING:
            speed_efficiency = 1.0
        elif tier == SpeedTier.FAST:
            speed_efficiency = 0.85
        elif tier == SpeedTier.STANDARD:
            speed_efficiency = 0.70
        else:
            speed_efficiency = 0.50
        
        # Quality factor
        quality_factor = quality / 100.0
        
        # Combined efficiency
        efficiency = (speed_efficiency * 0.6) + (quality_factor * 0.4)
        
        return min(1.0, efficiency)
    
    def _update_speed_stats(self, performance: TaskPerformance):
        """Update speed statistics"""
        self.speed_stats["total_tasks"] += 1
        
        # Update tier counts
        if performance.speed_tier == SpeedTier.LIGHTNING:
            self.speed_stats["lightning_tasks"] += 1
        elif performance.speed_tier == SpeedTier.FAST:
            self.speed_stats["fast_tasks"] += 1
        elif performance.speed_tier == SpeedTier.STANDARD:
            self.speed_stats["standard_tasks"] += 1
        else:
            self.speed_stats["slow_tasks"] += 1
        
        # Update average completion time
        if performance.duration_seconds:
            total_time = self.speed_stats["average_completion_time"] * (self.speed_stats["total_tasks"] - 1)
            total_time += performance.duration_seconds
            self.speed_stats["average_completion_time"] = total_time / self.speed_stats["total_tasks"]
        
        # Update total bonus earned
        self.speed_stats["total_bonus_earned"] += performance.speed_bonus
        
        # Calculate efficiency improvement
        if self.speed_stats["total_tasks"] > 1:
            fast_completion_rate = (self.speed_stats["lightning_tasks"] + self.speed_stats["fast_tasks"]) / self.speed_stats["total_tasks"]
            self.speed_stats["efficiency_improvement"] = fast_completion_rate * 100
    
    def _log_performance_results(self, performance: TaskPerformance, challenge: Dict):
        """Log performance results"""
        duration_minutes = performance.duration_seconds / 60 if performance.duration_seconds else 0
        
        logger.info(f"Speed Challenge Completed: {performance.task_id}")
        logger.info(f"Duration: {duration_minutes:.2f} minutes")
        logger.info(f"Speed Tier: {performance.speed_tier.value.upper()}")
        logger.info(f"Quality Score: {performance.quality_score:.1f}%")
        logger.info(f"Base Value: ${performance.base_value:.2f}")
        logger.info(f"Speed Bonus: ${performance.speed_bonus:.2f}")
        logger.info(f"Total Earned: ${performance.total_earned:.2f}")
        logger.info(f"Efficiency Rating: {performance.efficiency_rating:.2f}")
        
        if performance.speed_bonus > 0:
            logger.info(f"SUCCESS: Speed bonus achieved! {challenge['incentive'].reward_description}")
        else:
            logger.info(f"STANDARD: No speed bonus earned")
    
    def get_speed_dashboard(self) -> Dict:
        """Get comprehensive speed dashboard"""
        total_tasks = self.speed_stats["total_tasks"]
        
        if total_tasks == 0:
            return {"message": "No tasks completed yet"}
        
        # Calculate percentages
        lightning_pct = (self.speed_stats["lightning_tasks"] / total_tasks) * 100
        fast_pct = (self.speed_stats["fast_tasks"] / total_tasks) * 100
        standard_pct = (self.speed_stats["standard_tasks"] / total_tasks) * 100
        slow_pct = (self.speed_stats["slow_tasks"] / total_tasks) * 100
        
        # Calculate earnings
        avg_bonus_per_task = self.speed_stats["total_bonus_earned"] / total_tasks if total_tasks > 0 else 0
        
        return {
            "total_tasks_completed": total_tasks,
            "speed_distribution": {
                "lightning": {
                    "count": self.speed_stats["lightning_tasks"],
                    "percentage": lightning_pct,
                    "description": "< 5 minutes (50% bonus)"
                },
                "fast": {
                    "count": self.speed_stats["fast_tasks"],
                    "percentage": fast_pct,
                    "description": "< 10 minutes (25% bonus)"
                },
                "standard": {
                    "count": self.speed_stats["standard_tasks"],
                    "percentage": standard_pct,
                    "description": "< 15 minutes (10% bonus)"
                },
                "slow": {
                    "count": self.speed_stats["slow_tasks"],
                    "percentage": slow_pct,
                    "description": "> 15 minutes (no bonus)"
                }
            },
            "performance_metrics": {
                "average_completion_time_minutes": self.speed_stats["average_completion_time"] / 60,
                "total_bonus_earned": self.speed_stats["total_bonus_earned"],
                "average_bonus_per_task": avg_bonus_per_task,
                "efficiency_improvement_percentage": self.speed_stats["efficiency_improvement"]
            },
            "incentive_impact": {
                "tasks_with_bonus": self.speed_stats["lightning_tasks"] + self.speed_stats["fast_tasks"] + self.speed_stats["standard_tasks"],
                "bonus_achievement_rate": ((self.speed_stats["lightning_tasks"] + self.speed_stats["fast_tasks"] + self.speed_stats["standard_tasks"]) / total_tasks) * 100,
                "total_extra_revenue": self.speed_stats["total_bonus_earned"]
            }
        }
    
    def get_speed_recommendations(self) -> List[str]:
        """Get recommendations for improving speed"""
        recommendations = []
        
        if self.speed_stats["total_tasks"] == 0:
            return ["Start completing tasks to get speed recommendations"]
        
        # Analyze performance patterns
        slow_rate = self.speed_stats["slow_tasks"] / self.speed_stats["total_tasks"]
        
        if slow_rate > 0.5:
            recommendations.append("Focus on completing tasks faster - aim for under 15 minutes")
            recommendations.append("Break down complex tasks into smaller, manageable chunks")
        
        if self.speed_stats["lightning_tasks"] < self.speed_stats["total_tasks"] * 0.1:
            recommendations.append("Try to complete simple tasks in under 5 minutes for maximum bonus")
        
        avg_time = self.speed_stats["average_completion_time"] / 60
        if avg_time > 12:
            recommendations.append("Your average completion time is high - focus on efficiency")
        
        if self.speed_stats["total_bonus_earned"] > 0:
            recommendations.append("Great job earning speed bonuses! Keep up the fast work")
        
        # General tips
        recommendations.extend([
            "Start with the most important parts of the task first",
            "Don't overthink simple requests",
            "Use templates for common task types",
            "Practice quick decision-making",
            "Focus on delivering value quickly rather than perfection"
        ])
        
        return recommendations
    
    def simulate_speed_challenge(self, service_type: str, base_value: float, difficulty: str = "medium") -> Dict:
        """Simulate a speed challenge for testing"""
        task_id = f"sim_{int(time.time())}"
        
        # Start challenge
        challenge = self.start_speed_challenge(task_id, service_type, base_value)
        
        # Simulate completion time based on difficulty
        if difficulty == "easy":
            duration = random.uniform(2, 8)  # 2-8 minutes
        elif difficulty == "medium":
            duration = random.uniform(5, 18)  # 5-18 minutes
        else:  # hard
            duration = random.uniform(10, 25)  # 10-25 minutes
        
        # Simulate quality score
        if duration < 5:
            quality = random.uniform(75, 90)  # Lower quality for very fast work
        elif duration < 15:
            quality = random.uniform(80, 95)  # Good quality for reasonable speed
        else:
            quality = random.uniform(85, 98)  # Higher quality for slow work
        
        # Simulate work content
        work_content = f"Simulated work for {service_type} task"
        
        # Complete challenge
        performance = self.complete_speed_challenge(challenge, work_content, quality)
        
        return {
            "simulation": True,
            "difficulty": difficulty,
            "simulated_duration_minutes": duration / 60,
            "simulated_quality": quality,
            "performance": performance,
            "challenge": challenge
        }
    
    def export_speed_data(self, filename: str = "speed_incentive_data.json"):
        """Export speed incentive data"""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "incentives": {
                    tier.value: {
                        "time_limit_minutes": incentive.time_limit_minutes,
                        "bonus_percentage": incentive.bonus_percentage,
                        "bonus_multiplier": incentive.bonus_multiplier,
                        "quality_requirement": incentive.quality_requirement,
                        "reward_description": incentive.reward_description
                    }
                    for tier, incentive in self.incentives.items()
                },
                "speed_statistics": self.speed_stats,
                "task_history": [
                    {
                        "task_id": task.task_id,
                        "service_type": task.service_type,
                        "duration_seconds": task.duration_seconds,
                        "quality_score": task.quality_score,
                        "speed_tier": task.speed_tier.value,
                        "base_value": task.base_value,
                        "speed_bonus": task.speed_bonus,
                        "total_earned": task.total_earned,
                        "efficiency_rating": task.efficiency_rating,
                        "completion_time": task.end_time.isoformat() if task.end_time else None
                    }
                    for task in self.task_history
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Speed incentive data exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export speed data: {e}")
            return False

def main():
    """Main function for speed incentive system demo"""
    print("Janus Speed Incentive System")
    print("=" * 40)
    print("15-Minute Challenge - Complete tasks faster for bonuses!")
    print()
    
    # Initialize system
    speed_system = JanusSpeedIncentiveSystem()
    
    # Simulate some challenges
    print("SIMULATING SPEED CHALLENGES:")
    print("-" * 35)
    
    simulations = [
        ("content_writing", 100.0, "easy"),
        ("code_development", 250.0, "medium"),
        ("data_analysis", 300.0, "hard"),
        ("business_analysis", 150.0, "easy"),
        ("market_research", 200.0, "medium")
    ]
    
    for service_type, base_value, difficulty in simulations:
        result = speed_system.simulate_speed_challenge(service_type, base_value, difficulty)
        
        perf = result["performance"]
        print(f"Task: {service_type} (${base_value:.2f})")
        print(f"  Duration: {perf.duration_seconds / 60:.1f} minutes")
        print(f"  Speed Tier: {perf.speed_tier.value}")
        print(f"  Quality: {perf.quality_score:.1f}%")
        print(f"  Bonus: ${perf.speed_bonus:.2f}")
        print(f"  Total: ${perf.total_earned:.2f}")
        print()
    
    # Show dashboard
    dashboard = speed_system.get_speed_dashboard()
    print("SPEED DASHBOARD:")
    print("-" * 20)
    print(f"Total Tasks: {dashboard['total_tasks_completed']}")
    print(f"Average Time: {dashboard['performance_metrics']['average_completion_time_minutes']:.1f} minutes")
    print(f"Total Bonus: ${dashboard['performance_metrics']['total_bonus_earned']:.2f}")
    print(f"Bonus Achievement Rate: {dashboard['incentive_impact']['bonus_achievement_rate']:.1f}%")
    print()
    
    # Export data
    speed_system.export_speed_data()
    
    print("Speed incentive system demo complete!")

if __name__ == "__main__":
    import random
    main()
