"""
Autonomous Task Selection Engine
Janus decides what to work on without human input
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("task_selection")


class OpportunityScannerEnum(Enum):
    """Types of opportunities to scan for."""
    FREELANCE_WORK = "freelance"  # Fiverr, Upwork style
    SERVICE_AUTOMATION = "service"  # Offer services
    CODE_GENERATION = "code_gen"  # Generate and sell code
    DATA_ANALYSIS = "data"  # Analyze data for clients
    CONTENT_CREATION = "content"  # Write articles, docs
    SKILL_IMPROVEMENT = "learning"  # Learn new skills


class OpportunityPriority(Enum):
    """How valuable is this opportunity."""
    CRITICAL = 1  # High revenue, low effort
    HIGH = 2      # Good revenue or low effort
    MEDIUM = 3    # Moderate opportunity
    LOW = 4       # Small or hard opportunity


class Opportunity:
    """Represents a potential task/revenue source."""
    
    def __init__(self, title: str, description: str, opportunity_type: OpportunityScannerEnum,
                 estimated_revenue: float, estimated_effort_hours: float,
                 priority: OpportunityPriority, deadline: Optional[datetime] = None):
        self.id = f"opp_{datetime.now().timestamp()}"
        self.title = title
        self.description = description
        self.type = opportunity_type
        self.estimated_revenue = estimated_revenue
        self.estimated_effort = estimated_effort_hours
        self.priority = priority
        self.deadline = deadline or datetime.now() + timedelta(days=30)
        self.discovered_at = datetime.now()
        self.status = "pending"  # pending, in_progress, completed, rejected
        self.roi = estimated_revenue / max(estimated_effort_hours, 0.1)  # ROI calculation
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "revenue": self.estimated_revenue,
            "effort_hours": self.estimated_effort,
            "priority": self.priority.name,
            "roi": self.roi,
            "deadline": self.deadline.isoformat(),
            "status": self.status
        }


class AutonomousTaskSelector:
    """
    Janus autonomously selects and prioritizes tasks.
    No human input needed for task selection.
    """
    
    def __init__(self):
        self.opportunities: Dict[str, Opportunity] = {}
        self.current_task: Optional[Opportunity] = None
        self.completed_tasks: List[Opportunity] = []
        self.rejected_tasks: List[Opportunity] = []
        self.decision_log = []
        
        logger.info("Autonomous Task Selector initialized")
    
    def scan_opportunities(self) -> List[Opportunity]:
        """
        Scan for available opportunities.
        In production, would scan:
        - Freelance platforms (Upwork, Fiverr, Freelancer)
        - Social media for requests
        - Email for client inquiries
        - Forum posts needing solutions
        """
        
        logger.info("Scanning for opportunities...")
        
        opportunities = []
        
        # Example opportunities (in production, these come from APIs/scrapers)
        opportunities.append(Opportunity(
            title="Build code generator for client",
            description="Client needs automated code generation tool for their startup",
            opportunity_type=OpportunityScannerEnum.CODE_GENERATION,
            estimated_revenue=500,
            estimated_effort_hours=8,
            priority=OpportunityPriority.HIGH
        ))
        
        opportunities.append(Opportunity(
            title="Automate data analysis reports",
            description="Generate daily reports from CSV files for small business",
            opportunity_type=OpportunityScannerEnum.DATA_ANALYSIS,
            estimated_revenue=200,
            estimated_effort_hours=4,
            priority=OpportunityPriority.MEDIUM
        ))
        
        opportunities.append(Opportunity(
            title="Write technical documentation",
            description="Create API docs for new startup",
            opportunity_type=OpportunityScannerEnum.CONTENT_CREATION,
            estimated_revenue=300,
            estimated_effort_hours=6,
            priority=OpportunityPriority.MEDIUM
        ))
        
        opportunities.append(Opportunity(
            title="Learn Kubernetes for future work",
            description="Improve skills in container orchestration",
            opportunity_type=OpportunityScannerEnum.SKILL_IMPROVEMENT,
            estimated_revenue=0,  # No immediate revenue
            estimated_effort_hours=10,
            priority=OpportunityPriority.LOW
        ))
        
        opportunities.append(Opportunity(
            title="Scrape and analyze competitor data",
            description="Competitor analysis tool for marketing agency",
            opportunity_type=OpportunityScannerEnum.SERVICE_AUTOMATION,
            estimated_revenue=800,
            estimated_effort_hours=12,
            priority=OpportunityPriority.HIGH
        ))
        
        # Store opportunities
        for opp in opportunities:
            self.opportunities[opp.id] = opp
            logger.info(f"Found opportunity: {opp.title} (ROI: {opp.roi:.2f})")
        
        return opportunities
    
    def rank_opportunities(self) -> List[Opportunity]:
        """
        Rank opportunities by various metrics.
        Considers: ROI, priority, deadline, effort, revenue.
        """
        
        # Get pending opportunities
        pending = [o for o in self.opportunities.values() if o.status == "pending"]
        
        # Calculate scores
        scores = []
        for opp in pending:
            # Scoring algorithm
            roi_score = opp.roi / 100  # Higher ROI = higher score
            priority_score = (5 - opp.priority.value) * 20  # Critical=80, Low=20
            revenue_score = opp.estimated_revenue / 100  # Higher revenue = higher
            effort_penalty = opp.estimated_effort / 10 * -5  # Lower effort = higher (less penalty)
            
            total_score = roi_score + priority_score + revenue_score + effort_penalty
            
            scores.append((opp, total_score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Opportunities ranked:")
        for opp, score in scores[:5]:
            logger.info(f"  {opp.title}: {score:.2f}")
        
        return [opp for opp, _ in scores]
    
    def select_next_task(self) -> Optional[Opportunity]:
        """
        Autonomously select the next best task to work on.
        """
        
        logger.info("Selecting next task...")
        
        ranked = self.rank_opportunities()
        
        if not ranked:
            logger.warning("No tasks available")
            return None
        
        # Pick the top-ranked task
        selected = ranked[0]
        selected.status = "in_progress"
        self.current_task = selected
        
        logger.info(f"Selected task: {selected.title}")
        logger.info(f"  Revenue: ${selected.estimated_revenue}")
        logger.info(f"  Effort: {selected.estimated_effort}h")
        logger.info(f"  ROI: {selected.roi:.2f}")
        
        self.decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "decision": "task_selected",
            "task": selected.title,
            "reason": f"Best ROI ({selected.roi:.2f}), Priority: {selected.priority.name}"
        })
        
        return selected
    
    def evaluate_task(self, task_id: str, success: bool, 
                     actual_revenue: Optional[float] = None,
                     actual_effort: Optional[float] = None,
                     notes: str = "") -> Dict:
        """
        After completing a task, evaluate results.
        Learn for future task selection.
        """
        
        if task_id not in self.opportunities:
            return {"error": "Task not found"}
        
        task = self.opportunities[task_id]
        
        if success:
            task.status = "completed"
            self.completed_tasks.append(task)
            
            # Update estimates for learning
            if actual_revenue is not None:
                revenue_accuracy = actual_revenue / max(task.estimated_revenue, 1)
            else:
                revenue_accuracy = 1.0
            
            if actual_effort is not None:
                effort_accuracy = actual_effort / max(task.estimated_effort, 0.1)
            else:
                effort_accuracy = 1.0
            
            logger.info(f"Task completed: {task.title}")
            logger.info(f"  Revenue accuracy: {revenue_accuracy:.2f}x")
            logger.info(f"  Effort accuracy: {effort_accuracy:.2f}x")
            
            self.decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "decision": "task_completed",
                "task": task.title,
                "revenue": actual_revenue or task.estimated_revenue,
                "effort": actual_effort or task.estimated_effort,
                "notes": notes
            })
            
            return {
                "status": "completed",
                "revenue": actual_revenue or task.estimated_revenue,
                "effort": actual_effort or task.estimated_effort
            }
        else:
            task.status = "rejected"
            self.rejected_tasks.append(task)
            
            logger.warning(f"Task rejected: {task.title}")
            logger.warning(f"  Reason: {notes}")
            
            self.decision_log.append({
                "timestamp": datetime.now().isoformat(),
                "decision": "task_rejected",
                "task": task.title,
                "reason": notes
            })
            
            return {"status": "rejected", "reason": notes}
    
    def get_work_summary(self) -> Dict:
        """Get summary of work activity."""
        
        total_revenue = sum(
            t.estimated_revenue for t in self.completed_tasks
        )
        total_effort = sum(
            t.estimated_effort for t in self.completed_tasks
        )
        
        return {
            "completed_tasks": len(self.completed_tasks),
            "rejected_tasks": len(self.rejected_tasks),
            "pending_tasks": len([o for o in self.opportunities.values() if o.status == "pending"]),
            "total_revenue_generated": total_revenue,
            "total_effort_hours": total_effort,
            "hourly_rate": total_revenue / max(total_effort, 0.1),
            "recent_decisions": self.decision_log[-10:]
        }


class AutonomousGoalGenerator:
    """
    Generate goals for Janus based on opportunities and performance.
    """
    
    def __init__(self, task_selector: AutonomousTaskSelector):
        self.selector = task_selector
        self.goals = []
    
    def generate_goals(self, timeframe_days: int = 30) -> List[Dict]:
        """
        Generate autonomous goals.
        Not based on human input, but on opportunity analysis.
        """
        
        logger.info(f"Generating goals for next {timeframe_days} days...")
        
        goals = []
        
        # Goal 1: Revenue generation
        opportunities = self.selector.scan_opportunities()
        total_potential = sum(o.estimated_revenue for o in opportunities)
        
        goals.append({
            "name": "Generate revenue",
            "target": total_potential * 0.8,  # 80% of potential
            "timeframe_days": timeframe_days,
            "metric": "revenue",
            "strategy": "Complete high-ROI tasks first"
        })
        
        # Goal 2: Skill development
        learning_opps = [o for o in opportunities if o.type == OpportunityScannerEnum.SKILL_IMPROVEMENT]
        if learning_opps:
            goals.append({
                "name": "Improve capabilities",
                "target": 10,  # 10 hours of learning
                "timeframe_days": timeframe_days,
                "metric": "hours",
                "strategy": "Dedicate time to high-impact skills"
            })
        
        # Goal 3: Task completion
        goals.append({
            "name": "Complete available work",
            "target": len([o for o in opportunities if o.status == "pending"]),
            "timeframe_days": timeframe_days,
            "metric": "tasks",
            "strategy": "Work through ranked opportunity list"
        })
        
        self.goals = goals
        
        logger.info(f"Generated {len(goals)} autonomous goals")
        for goal in goals:
            logger.info(f"  {goal['name']}: {goal['target']} {goal['metric']}")
        
        return goals


if __name__ == "__main__":
    print("Autonomous Task Selection Engine")
    print("=" * 60)
    
    selector = AutonomousTaskSelector()
    
    # Scan for opportunities
    print("\n[Step 1] Scanning for opportunities...")
    opps = selector.scan_opportunities()
    print(f"Found {len(opps)} opportunities")
    
    # Rank them
    print("\n[Step 2] Ranking opportunities...")
    ranked = selector.rank_opportunities()
    
    # Select next task
    print("\n[Step 3] Selecting next task...")
    task = selector.select_next_task()
    
    if task:
        print(f"\nSelected: {task.title}")
        print(f"Revenue: ${task.estimated_revenue}")
        print(f"Effort: {task.estimated_effort}h")
    
    # Generate autonomous goals
    print("\n[Step 4] Generating autonomous goals...")
    goal_gen = AutonomousGoalGenerator(selector)
    goals = goal_gen.generate_goals()
    
    for goal in goals:
        print(f"Goal: {goal['name']} - {goal['target']} {goal['metric']}")
    
    # Summary
    print("\n[Summary]")
    summary = selector.get_work_summary()
    print(json.dumps(summary, indent=2))
