"""
CEO-Level Autonomy Module
Strategic decision-making, planning, goal management, and adaptation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("janus_ceo")


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class GoalStatus(Enum):
    """Goal execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Goal:
    """Represents a high-level business goal."""
    
    def __init__(self, name: str, description: str, priority: GoalPriority,
                 target_value: float, metric: str, deadline: datetime = None):
        self.id = f"goal_{datetime.now().timestamp()}"
        self.name = name
        self.description = description
        self.priority = priority
        self.target_value = target_value
        self.metric = metric  # e.g., "revenue", "savings", "portfolio_value"
        self.deadline = deadline or datetime.now() + timedelta(days=30)
        self.status = GoalStatus.PENDING
        self.current_value = 0
        self.created_at = datetime.now()
        self.progress = []  # Track progress over time
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "metric": self.metric,
            "status": self.status.value,
            "deadline": self.deadline.isoformat(),
            "completion_percentage": (self.current_value / self.target_value * 100) if self.target_value > 0 else 0,
            "progress": self.progress
        }


class Decision:
    """Represents a strategic decision."""
    
    def __init__(self, context: str, options: List[Dict], recommendation: str, reasoning: str):
        self.id = f"decision_{datetime.now().timestamp()}"
        self.context = context
        self.options = options  # [{"action": "...", "risk": 0-1, "upside": 0-1}, ...]
        self.recommendation = recommendation
        self.reasoning = reasoning
        self.made_at = datetime.now()
        self.outcome = None
        self.impact = 0.0  # -1 to 1
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "context": self.context,
            "options": self.options,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "made_at": self.made_at.isoformat(),
            "outcome": self.outcome,
            "impact": self.impact
        }


class StrategicPlan:
    """Multi-step strategic plan to achieve goals."""
    
    def __init__(self, goal: Goal, timeframe_days: int = 30):
        self.id = f"plan_{datetime.now().timestamp()}"
        self.goal = goal
        self.timeframe_days = timeframe_days
        self.steps = []
        self.created_at = datetime.now()
        self.status = "draft"
        self.kpis = {}
    
    def add_step(self, step_name: str, tasks: List[str], duration_days: int, 
                 dependencies: List[str] = None, risk_level: str = "medium"):
        """Add a step to the plan."""
        self.steps.append({
            "name": step_name,
            "tasks": tasks,
            "duration_days": duration_days,
            "dependencies": dependencies or [],
            "risk_level": risk_level,
            "status": "pending",
            "start_date": None,
            "completion_date": None
        })
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "goal": self.goal.name,
            "timeframe_days": self.timeframe_days,
            "steps": self.steps,
            "status": self.status,
            "kpis": self.kpis
        }


class CEOAgent:
    """
    CEO-level autonomous agent.
    Makes strategic decisions, manages goals, plans multi-step initiatives.
    """
    
    def __init__(self, name: str = "Janus CEO"):
        self.name = name
        self.goals: Dict[str, Goal] = {}
        self.decisions: Dict[str, Decision] = {}
        self.plans: Dict[str, StrategicPlan] = {}
        self.financial_state = {
            "cash": 1000,  # Starting capital
            "investments": {},
            "expenses": 0,
            "revenue": 0,
            "net_worth": 1000
        }
        self.performance_history = []
        self.strategy_log = []
        
        logger.info(f"CEO Agent initialized: {name}")
    
    # ==================== GOAL MANAGEMENT ====================
    
    def set_goal(self, name: str, description: str, priority: GoalPriority,
                target_value: float, metric: str, deadline_days: int = 30) -> Goal:
        """Set a high-level business goal."""
        
        goal = Goal(
            name=name,
            description=description,
            priority=priority,
            target_value=target_value,
            metric=metric,
            deadline=datetime.now() + timedelta(days=deadline_days)
        )
        
        self.goals[goal.id] = goal
        self.strategy_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "set_goal",
            "goal": goal.name,
            "target": f"{target_value} {metric}"
        })
        
        logger.info(f"Goal set: {name} (target: {target_value} {metric})")
        return goal
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active (non-completed) goals."""
        return [g for g in self.goals.values() 
                if g.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]]
    
    def prioritize_goals(self) -> List[Goal]:
        """Return goals sorted by priority."""
        return sorted(self.get_active_goals(), key=lambda g: g.priority.value)
    
    def update_goal_progress(self, goal_id: str, current_value: float):
        """Update goal progress."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.current_value = current_value
            goal.progress.append({
                "timestamp": datetime.now().isoformat(),
                "value": current_value,
                "percentage": (current_value / goal.target_value * 100) if goal.target_value > 0 else 0
            })
            
            # Check if goal completed
            if current_value >= goal.target_value:
                goal.status = GoalStatus.COMPLETED
                logger.info(f"Goal completed: {goal.name}")
    
    # ==================== DECISION MAKING ====================
    
    def evaluate_options(self, context: str, options: List[Dict]) -> Dict:
        """
        Evaluate decision options.
        Each option should have: "action", "upside" (0-1), "risk" (0-1), "timeframe"
        """
        
        scored_options = []
        
        for i, option in enumerate(options):
            upside = option.get("upside", 0.5)
            risk = option.get("risk", 0.3)
            
            # Risk-adjusted return: upside * (1 - risk)
            score = upside * (1 - risk)
            
            scored_options.append({
                "index": i,
                "action": option.get("action"),
                "score": score,
                "upside": upside,
                "risk": risk,
                "timeframe": option.get("timeframe", "unknown")
            })
        
        # Rank by score
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "context": context,
            "options_evaluated": len(options),
            "ranked_options": scored_options,
            "recommendation": scored_options[0]["action"] if scored_options else None,
            "confidence": scored_options[0]["score"] if scored_options else 0
        }
    
    def make_decision(self, context: str, options: List[Dict], 
                     reasoning: str = "") -> Decision:
        """Make a strategic decision."""
        
        evaluation = self.evaluate_options(context, options)
        recommendation = evaluation["recommendation"]
        
        decision = Decision(
            context=context,
            options=options,
            recommendation=recommendation,
            reasoning=reasoning or f"Selected best risk-adjusted option (score: {evaluation['confidence']:.2f})"
        )
        
        self.decisions[decision.id] = decision
        
        self.strategy_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "decision_made",
            "context": context,
            "recommendation": recommendation
        })
        
        logger.info(f"Decision made: {recommendation} (confidence: {evaluation['confidence']:.2f})")
        return decision
    
    # ==================== PLANNING ====================
    
    def create_plan(self, goal: Goal, timeframe_days: int = 30) -> StrategicPlan:
        """Create strategic plan to achieve goal."""
        
        plan = StrategicPlan(goal, timeframe_days)
        
        # Auto-generate plan based on goal type
        if goal.metric == "revenue":
            self._generate_revenue_plan(plan)
        elif goal.metric == "savings":
            self._generate_savings_plan(plan)
        elif goal.metric == "portfolio_value":
            self._generate_investment_plan(plan)
        else:
            self._generate_generic_plan(plan)
        
        plan.status = "active"
        self.plans[plan.id] = plan
        
        logger.info(f"Plan created for goal: {goal.name} ({len(plan.steps)} steps)")
        return plan
    
    def _generate_revenue_plan(self, plan: StrategicPlan):
        """Generate revenue growth plan."""
        plan.add_step(
            "Market Research",
            ["Analyze market opportunities", "Identify customer segments", "Research competitors"],
            duration_days=5,
            risk_level="low"
        )
        plan.add_step(
            "Product/Service Development",
            ["Design offering", "Create MVP", "Test with users"],
            duration_days=10,
            dependencies=["Market Research"],
            risk_level="medium"
        )
        plan.add_step(
            "Launch & Marketing",
            ["Set up sales channels", "Marketing campaign", "Customer outreach"],
            duration_days=10,
            dependencies=["Product/Service Development"],
            risk_level="medium"
        )
        plan.add_step(
            "Scale & Optimize",
            ["Track metrics", "Optimize conversion", "Expand channels"],
            duration_days=5,
            dependencies=["Launch & Marketing"],
            risk_level="high"
        )
    
    def _generate_savings_plan(self, plan: StrategicPlan):
        """Generate savings/cost reduction plan."""
        plan.add_step(
            "Expense Audit",
            ["Track all expenses", "Categorize spending", "Identify waste"],
            duration_days=7,
            risk_level="low"
        )
        plan.add_step(
            "Optimization",
            ["Negotiate contracts", "Automate processes", "Cut unnecessary costs"],
            duration_days=14,
            dependencies=["Expense Audit"],
            risk_level="medium"
        )
        plan.add_step(
            "Monitor & Adjust",
            ["Track savings", "Maintain discipline", "Look for new opportunities"],
            duration_days=9,
            dependencies=["Optimization"],
            risk_level="low"
        )
    
    def _generate_investment_plan(self, plan: StrategicPlan):
        """Generate investment portfolio plan."""
        plan.add_step(
            "Risk Assessment",
            ["Determine risk tolerance", "Set allocation targets", "Research options"],
            duration_days=5,
            risk_level="low"
        )
        plan.add_step(
            "Portfolio Construction",
            ["Diversify across assets", "Dollar-cost average", "Set rebalancing rules"],
            duration_days=10,
            dependencies=["Risk Assessment"],
            risk_level="medium"
        )
        plan.add_step(
            "Active Management",
            ["Monitor performance", "Rebalance as needed", "Adjust for life changes"],
            duration_days=15,
            dependencies=["Portfolio Construction"],
            risk_level="medium"
        )
    
    def _generate_generic_plan(self, plan: StrategicPlan):
        """Generic plan template."""
        plan.add_step(
            "Planning Phase",
            ["Define requirements", "Research options", "Set milestones"],
            duration_days=7,
            risk_level="low"
        )
        plan.add_step(
            "Execution Phase",
            ["Execute tasks", "Monitor progress", "Handle blockers"],
            duration_days=20,
            dependencies=["Planning Phase"],
            risk_level="medium"
        )
        plan.add_step(
            "Review & Optimize",
            ["Measure results", "Gather feedback", "Plan improvements"],
            duration_days=3,
            dependencies=["Execution Phase"],
            risk_level="low"
        )
    
    # ==================== FINANCIAL MANAGEMENT ====================
    
    def get_financial_summary(self) -> Dict:
        """Get current financial state."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cash": self.financial_state["cash"],
            "investments_value": sum(self.financial_state["investments"].values()),
            "total_assets": self.financial_state["cash"] + sum(self.financial_state["investments"].values()),
            "total_revenue": self.financial_state["revenue"],
            "total_expenses": self.financial_state["expenses"],
            "net_profit": self.financial_state["revenue"] - self.financial_state["expenses"],
            "net_worth": self.financial_state["cash"] + sum(self.financial_state["investments"].values())
        }
    
    def record_transaction(self, transaction_type: str, amount: float, description: str):
        """Record financial transaction."""
        
        if transaction_type == "revenue":
            self.financial_state["revenue"] += amount
            self.financial_state["cash"] += amount
        elif transaction_type == "expense":
            self.financial_state["expenses"] += amount
            self.financial_state["cash"] -= amount
        elif transaction_type == "investment":
            self.financial_state["cash"] -= amount
            self.financial_state["investments"][description] = amount
        
        self.strategy_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "transaction",
            "type": transaction_type,
            "amount": amount,
            "description": description
        })
        
        logger.info(f"Transaction: {transaction_type} ${amount} - {description}")
    
    def recommend_action(self, situation: str) -> Dict:
        """
        AI-powered recommendation based on current state.
        No ML model needed - uses heuristics and goal alignment.
        """
        
        financial_summary = self.get_financial_summary()
        active_goals = self.prioritize_goals()
        
        recommendations = []
        
        # Financial health checks
        if financial_summary["cash"] < 100:
            recommendations.append({
                "priority": "CRITICAL",
                "action": "Increase cash reserves",
                "reason": "Low cash balance",
                "steps": ["Execute revenue-generating skill", "Cut non-essential expenses"]
            })
        
        if financial_summary["net_profit"] < 0:
            recommendations.append({
                "priority": "HIGH",
                "action": "Reduce expenses or increase revenue",
                "reason": "Operating at a loss",
                "steps": ["Audit expenses", "Launch revenue initiative"]
            })
        
        # Goal-aligned recommendations
        for goal in active_goals:
            if goal.status == GoalStatus.PENDING:
                recommendations.append({
                    "priority": goal.priority.name,
                    "action": f"Start working toward: {goal.name}",
                    "reason": f"High priority goal with {(goal.current_value / goal.target_value * 100):.0f}% progress",
                    "target": f"{goal.target_value} {goal.metric}",
                    "deadline": goal.deadline.isoformat()
                })
        
        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return {
            "situation": situation,
            "recommendations": recommendations[:3],  # Top 3
            "financial_health": financial_summary,
            "active_goals": len(active_goals)
        }
    
    # ==================== REPORTING ====================
    
    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        
        return {
            "report_date": datetime.now().isoformat(),
            "goals": {
                "active": len(self.get_active_goals()),
                "completed": len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED]),
                "failed": len([g for g in self.goals.values() if g.status == GoalStatus.FAILED]),
                "details": [g.to_dict() for g in self.prioritize_goals()]
            },
            "decisions": {
                "total_made": len(self.decisions),
                "positive_impact": len([d for d in self.decisions.values() if d.impact > 0]),
                "negative_impact": len([d for d in self.decisions.values() if d.impact < 0])
            },
            "financial": self.get_financial_summary(),
            "recent_actions": self.strategy_log[-10:]
        }
    
    def save_state(self, filename: str = "ceo_state.json"):
        """Save CEO state to file."""
        state = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "goals": {gid: g.to_dict() for gid, g in self.goals.items()},
            "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
            "financial": self.financial_state,
            "strategy_log": self.strategy_log
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"CEO state saved to {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("JANUS CEO AGENT - Strategic Autonomy Module")
    print("=" * 60)
    
    # Initialize CEO
    ceo = CEOAgent("Janus")
    
    # Example: Set business goals
    print("\n[1] Setting business goals...")
    goal1 = ceo.set_goal(
        "Increase Monthly Revenue",
        "Grow revenue by automating high-value tasks",
        GoalPriority.HIGH,
        target_value=5000,
        metric="revenue",
        deadline_days=90
    )
    
    goal2 = ceo.set_goal(
        "Build Investment Portfolio",
        "Grow wealth through smart investments",
        GoalPriority.HIGH,
        target_value=10000,
        metric="portfolio_value",
        deadline_days=180
    )
    
    # Example: Make strategic decisions
    print("\n[2] Making strategic decisions...")
    decision = ceo.make_decision(
        context="How to achieve revenue goal?",
        options=[
            {
                "action": "Automate financial analysis for clients",
                "upside": 0.8,
                "risk": 0.3,
                "timeframe": "30 days"
            },
            {
                "action": "Develop new product",
                "upside": 0.9,
                "risk": 0.7,
                "timeframe": "90 days"
            },
            {
                "action": "Expand existing services",
                "upside": 0.6,
                "risk": 0.2,
                "timeframe": "14 days"
            }
        ]
    )
    
    # Example: Create strategic plans
    print("\n[3] Creating strategic plans...")
    plan1 = ceo.create_plan(goal1, timeframe_days=90)
    
    # Example: Record transactions
    print("\n[4] Recording financial transactions...")
    ceo.record_transaction("revenue", 500, "Completed financial analysis task")
    ceo.record_transaction("expense", 50, "Operating costs")
    ceo.record_transaction("investment", 200, "Stock portfolio")
    
    # Example: Get recommendations
    print("\n[5] Getting AI recommendations...")
    recommendations = ceo.recommend_action("Should I scale or optimize?")
    
    # Example: Generate report
    print("\n[6] Performance report...")
    report = ceo.get_performance_report()
    
    # Output
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    
    # Save state
    ceo.save_state()
    print("\n✓ CEO state saved to ceo_state.json")
