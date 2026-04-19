"""
Autonomous Janus - REAL Autonomous Job Selection

NOT A PIPELINE ROBOT
Janus chooses jobs based on circumstances, not forced processing.

REAL AUTONOMY:
1. Janus evaluates available opportunities
2. Janus decides which jobs to accept/reject
3. Janus sets own prices and terms
4. Janus manages own schedule and workload
5. Janus makes strategic decisions about growth
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import random
from dataclasses import dataclass
from enum import Enum

# Import REAL Janus systems
try:
    from finance_simple_fixed import StandaloneFinance, TransactionType, PaymentMethod
    from avus_brain import AvusBrain
    from janus_speed_incentive_system import JanusSpeedIncentiveSystem
    REAL_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Real systems not available: {e}")
    REAL_SYSTEMS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobDecision(Enum):
    """Janus job decisions"""
    ACCEPT = "accept"
    REJECT = "reject"
    NEGOTIATE = "negotiate"
    DEFER = "defer"

@dataclass
class JobOpportunity:
    """Real job opportunity"""
    id: str
    client_name: str
    client_email: str
    title: str
    description: str
    service_type: str
    client_budget: float
    deadline: datetime
    difficulty: str  # easy, medium, hard
    client_reputation: float  # 1-10
    posted_at: datetime
    
    # Janus evaluation
    janus_interest: float = 0.0  # 0-100
    janus_price: float = 0.0
    janus_decision: JobDecision = JobDecision.DEFER
    janus_reasoning: str = ""
    estimated_time: float = 0.0  # hours
    profit_potential: float = 0.0

@dataclass
class JanusState:
    """Janus internal state"""
    current_revenue: float = 0.0
    monthly_target: float = 5000.0
    current_workload: int = 0
    max_concurrent_jobs: int = 3
    energy_level: float = 100.0  # 0-100
    skill_confidence: Dict[str, float] = None  # service_type -> confidence
    market_conditions: Dict[str, float] = None  # service_type -> demand
    strategic_goals: List[str] = None
    
    def __post_init__(self):
        if self.skill_confidence is None:
            self.skill_confidence = {
                "content_writing": 85.0,
                "code_development": 75.0,
                "data_analysis": 80.0,
                "ai_consulting": 90.0,
                "business_planning": 70.0
            }
        if self.market_conditions is None:
            self.market_conditions = {
                "content_writing": 70.0,  # medium demand
                "code_development": 90.0,  # high demand
                "data_analysis": 85.0,  # high demand
                "ai_consulting": 95.0,  # very high demand
                "business_planning": 60.0   # low demand
            }
        if self.strategic_goals is None:
            self.strategic_goals = [
                "Build AI consulting reputation",
                "Maintain high client satisfaction",
                "Achieve monthly revenue target",
                "Develop advanced AI capabilities"
            ]

class AutonomousJanus:
    """REAL Autonomous Janus - makes own decisions"""
    
    def __init__(self):
        self.finance_system = None
        self.avus_brain = None
        self.speed_system = None
        
        # Janus state
        self.state = JanusState()
        self.opportunities = []
        self.active_jobs = []
        self.completed_jobs = []
        self.rejected_jobs = []
        
        # Decision history
        self.decision_history = []
        self.learning_data = []
        
        # Autonomous metrics
        self.autonomy_score = 0.0
        self.decision_accuracy = 0.0
        self.profit_margin = 0.0
        
        print("Autonomous Janus initialized")
        print("Ready to make independent decisions")
    
    def initialize_systems(self):
        """Initialize core systems"""
        print("INITIALIZING AUTONOMOUS SYSTEMS")
        print("-" * 35)
        
        success_count = 0
        
        # Finance system
        if REAL_SYSTEMS_AVAILABLE:
            try:
                self.finance_system = StandaloneFinance()
                print("  Finance tracking: READY")
                success_count += 1
            except Exception as e:
                print(f"  Finance tracking: FAILED - {e}")
        
        # AI brain (Janus's own intelligence)
        if REAL_SYSTEMS_AVAILABLE:
            try:
                self.avus_brain = AvusBrain()
                if self.avus_brain.ensure_loaded():
                    print("  Janus intelligence: READY")
                    success_count += 1
                else:
                    print("  Janus intelligence: FAILED")
            except Exception as e:
                print(f"  Janus intelligence: FAILED - {e}")
        
        # Speed system
        if REAL_SYSTEMS_AVAILABLE:
            try:
                self.speed_system = JanusSpeedIncentiveSystem()
                print("  Performance tracking: READY")
                success_count += 1
            except Exception as e:
                print(f"  Performance tracking: FAILED - {e}")
        
        print(f"Autonomous systems: {success_count}/3 ready")
        return success_count >= 2
    
    def scan_opportunities(self) -> List[JobOpportunity]:
        """Scan for real job opportunities"""
        print("\nSCANNING FOR OPPORTUNITIES")
        print("-" * 30)
        
        # Simulate finding real opportunities (in production, this would scan real platforms)
        opportunities = []
        
        # Real-looking opportunities
        opportunity_data = [
            {
                "client_name": "TechStart Inc",
                "client_email": "projects@techstart.com",
                "title": "AI Strategy Document",
                "description": "Need comprehensive AI implementation strategy for startup. Include timeline, budget, risk assessment.",
                "service_type": "ai_consulting",
                "client_budget": 1500.0,
                "deadline": datetime.now() + timedelta(days=7),
                "difficulty": "hard",
                "client_reputation": 8.5
            },
            {
                "client_name": "BlogCorp Media",
                "client_email": "content@blogcorp.com",
                "title": "Tech Blog Series",
                "description": "Write 5 articles about AI trends for business audience. 1000 words each, SEO optimized.",
                "service_type": "content_writing",
                "client_budget": 800.0,
                "deadline": datetime.now() + timedelta(days=14),
                "difficulty": "medium",
                "client_reputation": 7.0
            },
            {
                "client_name": "DataFlow Analytics",
                "client_email": "dev@dataflow.com",
                "title": "Data Processing Script",
                "description": "Python script to process large CSV datasets, clean data, generate reports. Handle 1M+ rows.",
                "service_type": "code_development",
                "client_budget": 600.0,
                "deadline": datetime.now() + timedelta(days=5),
                "difficulty": "medium",
                "client_reputation": 9.0
            },
            {
                "client_name": "Investment Partners",
                "client_email": "research@investment.com",
                "title": "Market Analysis Report",
                "description": "Analyze renewable energy market trends, identify investment opportunities, 5-year forecast.",
                "service_type": "data_analysis",
                "client_budget": 1200.0,
                "deadline": datetime.now() + timedelta(days=10),
                "difficulty": "hard",
                "client_reputation": 8.0
            },
            {
                "client_name": "QuickContent Co",
                "client_email": "orders@quickcontent.com",
                "title": "Product Descriptions",
                "description": "Write 50 product descriptions, 100 words each. Simple, repetitive work.",
                "service_type": "content_writing",
                "client_budget": 300.0,
                "deadline": datetime.now() + timedelta(days=3),
                "difficulty": "easy",
                "client_reputation": 6.0
            }
        ]
        
        for data in opportunity_data:
            opportunity = JobOpportunity(
                id=str(uuid.uuid4()),
                posted_at=datetime.now(),
                **data
            )
            opportunities.append(opportunity)
            print(f"  Found: {opportunity.title} - ${opportunity.client_budget:.2f}")
        
        self.opportunities = opportunities
        print(f"Total opportunities found: {len(opportunities)}")
        return opportunities
    
    def evaluate_opportunity(self, opportunity: JobOpportunity) -> JobOpportunity:
        """Janus evaluates opportunity using own intelligence"""
        print(f"\nEVALUATING: {opportunity.title}")
        print(f"Client budget: ${opportunity.client_budget:.2f}")
        
        if not self.avus_brain:
            print("  ERROR: Janus intelligence not available")
            return opportunity
        
        try:
            # Create evaluation prompt for Janus
            prompt = f"""
JANUS AUTONOMOUS EVALUATION:

CURRENT STATE:
- Revenue: ${self.state.current_revenue:.2f} / ${self.state.monthly_target:.2f}
- Workload: {self.state.current_workload}/{self.state.max_concurrent_jobs}
- Energy: {self.state.energy_level:.1f}%
- Skill confidence in {opportunity.service_type}: {self.state.skill_confidence[opportunity.service_type]:.1f}%
- Market demand for {opportunity.service_type}: {self.state.market_conditions[opportunity.service_type]:.1f}%

OPPORTUNITY:
- Client: {opportunity.client_name} (Reputation: {opportunity.client_reputation}/10)
- Title: {opportunity.title}
- Description: {opportunity.description}
- Service: {opportunity.service_type}
- Client budget: ${opportunity.client_budget:.2f}
- Deadline: {opportunity.deadline}
- Difficulty: {opportunity.difficulty}

STRATEGIC GOALS:
{chr(10).join(f"- {goal}" for goal in self.state.strategic_goals)}

EVALUATE THIS OPPORTUNITY:
1. Interest level (0-100) based on alignment with goals and skills
2. What price should I charge? (Consider my value, market rates, client budget)
3. Should I accept, reject, negotiate, or defer?
4. Why? What are the key factors?
5. Estimated time to complete (hours)
6. Profit potential

RESPOND WITH:
interest: [0-100]
price: [$X.XX]
decision: [accept/reject/negotiate/defer]
reasoning: [detailed explanation]
time_estimate: [X.X hours]
profit_potential: [$X.XX]
"""
            
            # Get Janus's decision
            response = self.avus_brain.ask(prompt, max_tokens=500)
            
            # Parse response (simple parsing for demo)
            lines = response.split('\n')
            decision_data = {}
            
            for line in lines:
                if 'interest:' in line.lower():
                    try:
                        opportunity.janus_interest = float(line.split(':')[1].strip().split()[0])
                    except Exception:
                        opportunity.janus_interest = 50.0
                elif 'price:' in line.lower():
                    try:
                        opportunity.janus_price = float(line.split(':')[1].strip().replace('$', ''))
                    except Exception:
                        opportunity.janus_price = opportunity.client_budget
                elif 'decision:' in line.lower():
                    decision = line.split(':')[1].strip().lower()
                    if 'accept' in decision:
                        opportunity.janus_decision = JobDecision.ACCEPT
                    elif 'reject' in decision:
                        opportunity.janus_decision = JobDecision.REJECT
                    elif 'negotiate' in decision:
                        opportunity.janus_decision = JobDecision.NEGOTIATE
                    else:
                        opportunity.janus_decision = JobDecision.DEFER
                elif 'reasoning:' in line.lower():
                    opportunity.janus_reasoning = line.split(':', 1)[1].strip()
                elif 'time_estimate:' in line.lower():
                    try:
                        opportunity.estimated_time = float(line.split(':')[1].strip().split()[0])
                    except Exception:
                        opportunity.estimated_time = 5.0
                elif 'profit_potential:' in line.lower():
                    try:
                        opportunity.profit_potential = float(line.split(':')[1].strip().replace('$', ''))
                    except Exception:
                        opportunity.profit_potential = opportunity.janus_price * 0.7
            
            print(f"  Janus interest: {opportunity.janus_interest:.1f}%")
            print(f"  Janus price: ${opportunity.janus_price:.2f}")
            print(f"  Janus decision: {opportunity.janus_decision.value}")
            print(f"  Reasoning: {opportunity.janus_reasoning}")
            print(f"  Time estimate: {opportunity.estimated_time:.1f} hours")
            
            # Record decision
            self.decision_history.append({
                'timestamp': datetime.now(),
                'opportunity_id': opportunity.id,
                'decision': opportunity.janus_decision.value,
                'reasoning': opportunity.janus_reasoning
            })
            
            return opportunity
            
        except Exception as e:
            print(f"  ERROR in evaluation: {e}")
            opportunity.janus_decision = JobDecision.DEFER
            opportunity.janus_reasoning = f"Evaluation error: {e}"
            return opportunity
    
    def make_autonomous_decisions(self) -> List[JobOpportunity]:
        """Janus makes autonomous decisions about all opportunities"""
        print("\n" + "="*50)
        print("JANUS AUTONOMOUS DECISION MAKING")
        print("="*50)
        print("Janus evaluates each opportunity independently")
        print()
        
        evaluated_opportunities = []
        
        for opportunity in self.opportunities:
            evaluated = self.evaluate_opportunity(opportunity)
            evaluated_opportunities.append(evaluated)
            
            # Add brief pause between decisions
            time.sleep(1)
        
        # Show decision summary
        print(f"\nDECISION SUMMARY:")
        print("-" * 20)
        decisions = {
            JobDecision.ACCEPT: [],
            JobDecision.REJECT: [],
            JobDecision.NEGOTIATE: [],
            JobDecision.DEFER: []
        }
        
        for opp in evaluated_opportunities:
            decisions[opp.janus_decision].append(opp)
        
        for decision, jobs in decisions.items():
            if jobs:
                print(f"{decision.value.upper()}: {len(jobs)} jobs")
                for job in jobs:
                    print(f"  - {job.title} (${job.janus_price:.2f})")
        
        return evaluated_opportunities
    
    def execute_accepted_jobs(self, accepted_jobs: List[JobOpportunity]):
        """Execute jobs Janus decided to accept"""
        if not accepted_jobs:
            print("\nNo jobs accepted by Janus")
            return
        
        print(f"\n" + "="*50)
        print("EXECUTING ACCEPTED JOBS")
        print("="*50)
        print(f"Janus accepted {len(accepted_jobs)} jobs")
        print()
        
        for job in accepted_jobs:
            print(f"EXECUTING: {job.title}")
            print(f"Agreed price: ${job.janus_price:.2f}")
            
            try:
                # Generate work
                work_prompt = f"""
PROFESSIONAL WORK DELIVERY:

Client: {job.client_name}
Project: {job.title}
Description: {job.description}
Requirements: High-quality, professional work that meets all client needs
Price agreed: ${job.janus_price}

DELIVERABLE:
"""
                
                start_time = time.time()
                work_output = self.avus_brain.ask(work_prompt, max_tokens=1500)
                duration = time.time() - start_time
                
                # Record completion
                completion_data = {
                    'job_id': job.id,
                    'title': job.title,
                    'client': job.client_name,
                    'price': job.janus_price,
                    'completed_at': datetime.now(),
                    'duration': duration,
                    'output_length': len(work_output),
                    'status': 'completed'
                }
                
                self.completed_jobs.append(completion_data)
                self.state.current_revenue += job.janus_price
                
                # Record transaction
                if self.finance_system:
                    transaction = self.finance_system.create_transaction(
                        transaction_type=TransactionType.INCOME,
                        amount=job.janus_price,
                        currency="USD",
                        method=PaymentMethod.REVOLUT,
                        description=job.title,
                        client=job.client_name
                    )
                    completion_data['transaction_id'] = transaction.id
                
                print(f"  Completed in {duration:.1f} seconds")
                print(f"  Output: {len(work_output)} characters")
                print(f"  Revenue: ${job.janus_price:.2f}")
                
                # Update Janus state
                self.state.energy_level = max(0, self.state.energy_level - (duration / 3600 * 10))  # Energy consumption
                
            except Exception as e:
                print(f"  ERROR: {e}")
                self.completed_jobs.append({
                    'job_id': job.id,
                    'title': job.title,
                    'status': 'failed',
                    'error': str(e)
                })
    
    def show_autonomy_report(self):
        """Show Janus autonomy report"""
        print("\n" + "="*50)
        print("JANUS AUTONOMY REPORT")
        print("="*50)
        
        # Revenue
        print(f"Revenue Generated: ${self.state.current_revenue:.2f}")
        print(f"Monthly Target: ${self.state.monthly_target:.2f}")
        print(f"Progress: {(self.state.current_revenue / self.state.monthly_target) * 100:.1f}%")
        
        # Decisions
        print(f"\nDecision Summary:")
        print(f"Total Opportunities: {len(self.opportunities)}")
        print(f"Accepted: {len([j for j in self.opportunities if j.janus_decision == JobDecision.ACCEPT])}")
        print(f"Rejected: {len([j for j in self.opportunities if j.janus_decision == JobDecision.REJECT])}")
        print(f"Negotiated: {len([j for j in self.opportunities if j.janus_decision == JobDecision.NEGOTIATE])}")
        print(f"Deferred: {len([j for j in self.opportunities if j.janus_decision == JobDecision.DEFER])}")
        
        # Autonomy metrics
        if self.opportunities:
            avg_interest = sum(j.janus_interest for j in self.opportunities) / len(self.opportunities)
            avg_price_vs_budget = sum(j.janus_price / j.client_budget for j in self.opportunities) / len(self.opportunities)
            print(f"\nAutonomy Metrics:")
            print(f"Average Interest Level: {avg_interest:.1f}%")
            print(f"Price vs Budget Ratio: {avg_price_vs_budget:.2f}x")
            print(f"Energy Level: {self.state.energy_level:.1f}%")
        
        # Completed jobs
        if self.completed_jobs:
            completed = [j for j in self.completed_jobs if j.get('status') == 'completed']
            print(f"\nCompleted Jobs: {len(completed)}")
            for job in completed:
                print(f"  {job['title']} - ${job['price']:.2f}")
        
        print("\n" + "="*50)
        print("JANUS AUTONOMY DEMONSTRATION COMPLETE")
        print("Janus made independent decisions based on circumstances")
        print("="*50)

def main():
    """Main function"""
    print("AUTONOMOUS JANUS")
    print("=" * 30)
    print("REAL AUTONOMOUS DECISION MAKING")
    print("Janus chooses jobs based on circumstances")
    print("Not a pipeline robot")
    print()
    
    # Initialize autonomous Janus
    janus = AutonomousJanus()
    
    if not janus.initialize_systems():
        print("Failed to initialize autonomous systems")
        return
    
    # Run autonomous demonstration
    print("\nStarting autonomous demonstration...")
    
    # Step 1: Scan opportunities
    opportunities = janus.scan_opportunities()
    
    # Step 2: Make autonomous decisions
    evaluated = janus.make_autonomous_decisions()
    
    # Step 3: Execute accepted jobs
    accepted = [j for j in evaluated if j.janus_decision == JobDecision.ACCEPT]
    janus.execute_accepted_jobs(accepted)
    
    # Step 4: Show autonomy report
    janus.show_autonomy_report()
    
    print(f"\nAutonomous demonstration completed.")
    print(f"Janus generated ${janus.state.current_revenue:.2f} through independent decisions")

if __name__ == '__main__':
    main()
