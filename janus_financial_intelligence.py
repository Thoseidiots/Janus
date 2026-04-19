"""
janus_financial_intelligence.py
================================
Financial intelligence and budget management for Janus

Features:
- Financial reporting and analytics
- Budget allocation and tracking
- Goal tracking (e.g., saving for robot body)
- Autonomous financial decision-making
- Cash flow prediction
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialReport:
    """Financial report for a period"""
    period_start: datetime
    period_end: datetime
    total_income: Decimal
    total_expenses: Decimal
    net_profit: Decimal
    income_by_source: Dict[str, Decimal]
    expenses_by_category: Dict[str, Decimal]
    balance_change: Decimal
    savings_rate: float
    top_income_sources: List[Tuple[str, Decimal]]
    top_expense_categories: List[Tuple[str, Decimal]]


@dataclass
class FinancialGoal:
    """Financial goal (e.g., save for robot body)"""
    goal_id: str
    name: str
    target_amount: Decimal
    current_amount: Decimal
    deadline: Optional[datetime]
    priority: int
    category: str
    status: str = "active"
    
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.target_amount == 0:
            return 0.0
        return float((self.current_amount / self.target_amount) * 100)
    
    def days_remaining(self) -> Optional[int]:
        """Calculate days remaining until deadline"""
        if not self.deadline:
            return None
        delta = self.deadline - datetime.now()
        return max(0, delta.days)
    
    def required_daily_savings(self) -> Decimal:
        """Calculate required daily savings to reach goal"""
        if not self.deadline:
            return Decimal("0")
        
        days = self.days_remaining()
        if not days or days == 0:
            return Decimal("0")
        
        remaining = self.target_amount - self.current_amount
        return remaining / Decimal(str(days))


@dataclass
class BudgetAllocation:
    """Budget allocation strategy"""
    emergency_fund: Decimal = Decimal("0.20")  # 20%
    compute_resources: Decimal = Decimal("0.30")  # 30%
    api_costs: Decimal = Decimal("0.20")  # 20%
    savings_goals: Decimal = Decimal("0.20")  # 20%
    reinvestment: Decimal = Decimal("0.10")  # 10%


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialIntelligence:
    """Financial intelligence and analytics"""
    
    def __init__(self, wallet):
        """Initialize financial intelligence"""
        self.wallet = wallet
        self.goals: Dict[str, FinancialGoal] = {}
        self.allocation_strategy = BudgetAllocation()
        self._load_goals()
        logger.info("Financial intelligence initialized")
    
    def _load_goals(self):
        """Load financial goals from database"""
        conn = self.wallet.core.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT goal_id, name, target_amount, current_amount, deadline, 
                   priority, category, status
            FROM financial_goals
            WHERE status = 'active'
        ''')
        
        for row in cursor.fetchall():
            goal_id, name, target, current, deadline_str, priority, category, status = row
            
            deadline = None
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str)
                except:
                    pass
            
            self.goals[goal_id] = FinancialGoal(
                goal_id=goal_id,
                name=name,
                target_amount=Decimal(target),
                current_amount=Decimal(current),
                deadline=deadline,
                priority=priority,
                category=category,
                status=status
            )
        
        conn.close()
        logger.info(f"Loaded {len(self.goals)} financial goals")
    
    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> FinancialReport:
        """Generate financial report for period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Get transactions for period
        transactions = self.wallet.transactions.get_transactions(
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        # Calculate totals
        total_income = Decimal("0")
        total_expenses = Decimal("0")
        income_by_source: Dict[str, Decimal] = {}
        expenses_by_category: Dict[str, Decimal] = {}
        
        for tx in transactions:
            if tx.type.value == "income":
                total_income += tx.amount
                source = tx.source or "unknown"
                income_by_source[source] = income_by_source.get(source, Decimal("0")) + tx.amount
            elif tx.type.value == "expense":
                total_expenses += tx.amount
                category = tx.category.value
                expenses_by_category[category] = expenses_by_category.get(category, Decimal("0")) + tx.amount
        
        net_profit = total_income - total_expenses
        savings_rate = float((net_profit / total_income * 100)) if total_income > 0 else 0.0
        
        # Top sources and categories
        top_income = sorted(income_by_source.items(), key=lambda x: x[1], reverse=True)[:5]
        top_expenses = sorted(expenses_by_category.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return FinancialReport(
            period_start=start_date,
            period_end=end_date,
            total_income=total_income,
            total_expenses=total_expenses,
            net_profit=net_profit,
            income_by_source=income_by_source,
            expenses_by_category=expenses_by_category,
            balance_change=net_profit,
            savings_rate=savings_rate,
            top_income_sources=top_income,
            top_expense_categories=top_expenses
        )
    
    def create_goal(
        self,
        name: str,
        target_amount: Decimal,
        deadline: Optional[datetime] = None,
        priority: int = 0,
        category: str = "general"
    ) -> FinancialGoal:
        """Create a financial goal"""
        import uuid
        goal_id = f"goal_{uuid.uuid4().hex[:16]}"
        
        goal = FinancialGoal(
            goal_id=goal_id,
            name=name,
            target_amount=target_amount,
            current_amount=Decimal("0"),
            deadline=deadline,
            priority=priority,
            category=category
        )
        
        # Save to database
        conn = self.wallet.core.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO financial_goals 
            (goal_id, name, target_amount, current_amount, deadline, priority, category, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            goal_id, name, str(target_amount), "0",
            deadline.isoformat() if deadline else None,
            priority, category, "active"
        ))
        
        conn.commit()
        conn.close()
        
        self.goals[goal_id] = goal
        logger.info(f"Created financial goal: {name} (${target_amount})")
        return goal
    
    def update_goal_progress(self, goal_id: str, amount: Decimal):
        """Update progress toward a goal"""
        if goal_id not in self.goals:
            logger.warning(f"Goal not found: {goal_id}")
            return
        
        goal = self.goals[goal_id]
        goal.current_amount += amount
        
        # Update database
        conn = self.wallet.core.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE financial_goals 
            SET current_amount = ?, updated_at = CURRENT_TIMESTAMP
            WHERE goal_id = ?
        ''', (str(goal.current_amount), goal_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated goal '{goal.name}': ${goal.current_amount} / ${goal.target_amount}")
    
    def allocate_income(self, amount: Decimal) -> Dict[str, Decimal]:
        """Allocate income according to strategy"""
        allocation = {
            "emergency_fund": amount * self.allocation_strategy.emergency_fund,
            "compute_resources": amount * self.allocation_strategy.compute_resources,
            "api_costs": amount * self.allocation_strategy.api_costs,
            "savings_goals": amount * self.allocation_strategy.savings_goals,
            "reinvestment": amount * self.allocation_strategy.reinvestment
        }
        
        logger.info(f"Allocated ${amount}: {allocation}")
        return allocation
    
    def should_spend(self, amount: Decimal, category: str) -> bool:
        """Decide if Janus should spend money"""
        # Check current balance
        balance = self.wallet.get_balance("USD")
        
        # Don't spend if it would leave less than $100
        if balance - amount < Decimal("100"):
            logger.warning(f"Spending ${amount} would leave balance too low")
            return False
        
        # Check if expense is within reasonable limits
        daily_limit = Decimal("100")
        if amount > daily_limit:
            logger.warning(f"Expense ${amount} exceeds daily limit ${daily_limit}")
            return False
        
        return True
    
    def predict_cash_flow(self, days: int = 30) -> Dict[str, Decimal]:
        """Predict cash flow for next N days"""
        # Get historical data
        start_date = datetime.now() - timedelta(days=30)
        report = self.generate_report(start_date, datetime.now())
        
        # Calculate daily averages
        days_in_period = (datetime.now() - start_date).days
        avg_daily_income = report.total_income / Decimal(str(days_in_period))
        avg_daily_expenses = report.total_expenses / Decimal(str(days_in_period))
        
        # Predict future
        predicted_income = avg_daily_income * Decimal(str(days))
        predicted_expenses = avg_daily_expenses * Decimal(str(days))
        predicted_balance = self.wallet.get_balance("USD") + predicted_income - predicted_expenses
        
        return {
            "predicted_income": predicted_income,
            "predicted_expenses": predicted_expenses,
            "predicted_balance": predicted_balance,
            "avg_daily_income": avg_daily_income,
            "avg_daily_expenses": avg_daily_expenses
        }
    
    def print_report(self, days: int = 30):
        """Print financial report"""
        start_date = datetime.now() - timedelta(days=days)
        report = self.generate_report(start_date, datetime.now())
        
        print("\n" + "="*60)
        print(f"FINANCIAL REPORT ({days} days)")
        print("="*60)
        
        print(f"\nINCOME: ${report.total_income:,.2f}")
        print(f"EXPENSES: ${report.total_expenses:,.2f}")
        print(f"NET PROFIT: ${report.net_profit:,.2f}")
        print(f"SAVINGS RATE: {report.savings_rate:.1f}%")
        
        if report.top_income_sources:
            print(f"\nTOP INCOME SOURCES:")
            for source, amount in report.top_income_sources:
                print(f"  {source}: ${amount:,.2f}")
        
        if report.top_expense_categories:
            print(f"\nTOP EXPENSES:")
            for category, amount in report.top_expense_categories:
                print(f"  {category}: ${amount:,.2f}")
        
        # Show goals
        if self.goals:
            print(f"\nFINANCIAL GOALS:")
            for goal in sorted(self.goals.values(), key=lambda g: g.priority, reverse=True):
                progress = goal.progress_percentage()
                print(f"  {goal.name}: ${goal.current_amount:,.2f} / ${goal.target_amount:,.2f} ({progress:.1f}%)")
                if goal.deadline:
                    days_left = goal.days_remaining()
                    daily_needed = goal.required_daily_savings()
                    print(f"    {days_left} days left, need ${daily_needed:.2f}/day")
        
        # Cash flow prediction
        prediction = self.predict_cash_flow(30)
        print(f"\n30-DAY PREDICTION:")
        print(f"  Expected income: ${prediction['predicted_income']:,.2f}")
        print(f"  Expected expenses: ${prediction['predicted_expenses']:,.2f}")
        print(f"  Predicted balance: ${prediction['predicted_balance']:,.2f}")
        
        print("\n" + "="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo financial intelligence"""
    print("Financial Intelligence Demo")
    print("This requires an initialized wallet")


if __name__ == "__main__":
    demo()
