"""
Janus Payout Analysis - Complete Payout Type Analysis

Comprehensive analysis of all payout types and scenarios for production system.
This ensures we handle every possible payment situation correctly.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PayoutCategory(Enum):
    """Main payout categories"""
    PLATFORM_AUTOMATIC = "platform_automatic"  # Platform handles everything
    CLIENT_MANUAL = "client_manual"           # Client pays manually
    HYBRID = "hybrid"                        # Mix of automatic and manual
    ESCROW = "escrow"                        # Funds held in escrow
    MILESTONE = "milestone"                  # Paid in milestones

class PayoutStatus(Enum):
    """Payout status tracking"""
    PENDING = "pending"           # Awaiting payment
    PROCESSING = "processing"     # Payment being processed
    COMPLETED = "completed"       # Payment received
    FAILED = "failed"           # Payment failed
    DISPUTED = "disputed"       # Payment disputed
    REFUNDED = "refunded"       # Payment refunded

@dataclass
class PayoutScenario:
    """Complete payout scenario analysis"""
    category: PayoutCategory
    platform: str
    payment_method: str
    processing_time: str
    fees: float
    verification_required: bool
    retry_possible: bool
    escrow_held: bool
    milestones: bool
    auto_release: bool
    client_action_required: bool
    failure_rate: float
    minimum_amount: float
    maximum_amount: float

class JanusPayoutAnalysis:
    """Comprehensive payout analysis system"""
    
    def __init__(self):
        self.payout_scenarios = self._analyze_all_payout_types()
        self.payout_history = []
        self.failure_patterns = {}
        self.optimization_suggestions = []
        
        logger.info("Janus Payout Analysis initialized")
    
    def _analyze_all_payout_types(self) -> Dict[str, PayoutScenario]:
        """Analyze all possible payout scenarios"""
        scenarios = {}
        
        # Upwork Payout Scenarios
        scenarios["upwork_fixed"] = PayoutScenario(
            category=PayoutCategory.ESCROW,
            platform="Upwork",
            payment_method="Platform Escrow",
            processing_time="5-7 days after approval",
            fees=0.20,  # 20% platform fee
            verification_required=True,
            retry_possible=False,
            escrow_held=True,
            milestones=False,
            auto_release=True,
            client_action_required=True,  # Client must approve
            failure_rate=0.05,
            minimum_amount=5.00,
            maximum_amount=10000.00
        )
        
        scenarios["upwork_hourly"] = PayoutScenario(
            category=PayoutCategory.PLATFORM_AUTOMATIC,
            platform="Upwork",
            payment_method="Weekly Automatic",
            processing_time="Weekly (Wednesday)",
            fees=0.20,
            verification_required=False,
            retry_possible=True,
            escrow_held=False,
            milestones=False,
            auto_release=True,
            client_action_required=False,
            failure_rate=0.02,
            minimum_amount=1.00,
            maximum_amount=5000.00
        )
        
        scenarios["upwork_milestone"] = PayoutScenario(
            category=PayoutCategory.MILESTONE,
            platform="Upwork",
            payment_method="Milestone Escrow",
            processing_time="After each milestone approval",
            fees=0.20,
            verification_required=True,
            retry_possible=False,
            escrow_held=True,
            milestones=True,
            auto_release=True,
            client_action_required=True,
            failure_rate=0.08,
            minimum_amount=25.00,
            maximum_amount=15000.00
        )
        
        # Fiverr Payout Scenarios
        scenarios["fiverr_gig"] = PayoutScenario(
            category=PayoutCategory.PLATFORM_AUTOMATIC,
            platform="Fiverr",
            payment_method="Order Completion",
            processing_time="14 days after delivery",
            fees=0.20,  # 20% platform fee
            verification_required=False,
            retry_possible=True,
            escrow_held=True,
            milestones=False,
            auto_release=True,
            client_action_required=True,  # Client must accept delivery
            failure_rate=0.03,
            minimum_amount=5.00,
            maximum_amount=1000.00
        )
        
        scenarios["fiverr_custom"] = PayoutScenario(
            category=PayoutCategory.ESCROW,
            platform="Fiverr",
            payment_method="Custom Order Escrow",
            processing_time="14 days after approval",
            fees=0.20,
            verification_required=True,
            retry_possible=False,
            escrow_held=True,
            milestones=False,
            auto_release=True,
            client_action_required=True,
            failure_rate=0.05,
            minimum_amount=100.00,
            maximum_amount=10000.00
        )
        
        # Freelancer Payout Scenarios
        scenarios["freelancer_fixed"] = PayoutScenario(
            category=PayoutCategory.ESCROW,
            platform="Freelancer",
            payment_method="Milestone Escrow",
            processing_time="After milestone approval",
            fees=0.10,  # 10% platform fee
            verification_required=True,
            retry_possible=False,
            escrow_held=True,
            milestones=True,
            auto_release=True,
            client_action_required=True,
            failure_rate=0.07,
            minimum_amount=30.00,
            maximum_amount=5000.00
        )
        
        scenarios["freelancer_hourly"] = PayoutScenario(
            category=PayoutCategory.PLATFORM_AUTOMATIC,
            platform="Freelancer",
            payment_method="Weekly Automatic",
            processing_time="Weekly",
            fees=0.10,
            verification_required=False,
            retry_possible=True,
            escrow_held=False,
            milestones=False,
            auto_release=True,
            client_action_required=False,
            failure_rate=0.04,
            minimum_amount=1.00,
            maximum_amount=2000.00
        )
        
        # Direct Payment Scenarios
        scenarios["revolut_direct"] = PayoutScenario(
            category=PayoutCategory.CLIENT_MANUAL,
            platform="Direct",
            payment_method="Revolut.me Link",
            processing_time="Instant",
            fees=0.00,
            verification_required=False,
            retry_possible=True,
            escrow_held=False,
            milestones=False,
            auto_release=True,
            client_action_required=True,  # Client must click link
            failure_rate=0.15,  # Higher failure rate for manual
            minimum_amount=1.00,
            maximum_amount=10000.00
        )
        
        scenarios["paypal_direct"] = PayoutScenario(
            category=PayoutCategory.CLIENT_MANUAL,
            platform="Direct",
            payment_method="PayPal Link",
            processing_time="Instant",
            fees=0.029 + 0.30,  # PayPal fees
            verification_required=False,
            retry_possible=True,
            escrow_held=False,
            milestones=False,
            auto_release=True,
            client_action_required=True,
            failure_rate=0.12,
            minimum_amount=1.00,
            maximum_amount=10000.00
        )
        
        scenarios["bank_transfer"] = PayoutScenario(
            category=PayoutCategory.CLIENT_MANUAL,
            platform="Direct",
            payment_method="Bank Transfer",
            processing_time="1-3 business days",
            fees=0.00,
            verification_required=True,
            retry_possible=True,
            escrow_held=False,
            milestones=False,
            auto_release=True,
            client_action_required=True,
            failure_rate=0.08,
            minimum_amount=50.00,
            maximum_amount=50000.00
        )
        
        # Hybrid Scenarios
        scenarios["hybrid_partial"] = PayoutScenario(
            category=PayoutCategory.HYBRID,
            platform="Multiple",
            payment_method="Partial Platform + Partial Direct",
            processing_time="Varies",
            fees=0.15,
            verification_required=True,
            retry_possible=True,
            escrow_held=True,
            milestones=True,
            auto_release=False,
            client_action_required=True,
            failure_rate=0.10,
            minimum_amount=100.00,
            maximum_amount=20000.00
        )
        
        return scenarios
    
    def analyze_payout_for_opportunity(self, opportunity: Dict) -> PayoutScenario:
        """Analyze the best payout scenario for an opportunity"""
        platform = opportunity.get("platform", "").lower()
        budget = opportunity.get("budget", 0)
        service_type = opportunity.get("service", "")
        
        # Select appropriate scenario based on platform and job type
        if "upwork" in platform:
            if "hourly" in service_type.lower():
                return self.payout_scenarios["upwork_hourly"]
            elif "milestone" in opportunity.get("description", "").lower():
                return self.payout_scenarios["upwork_milestone"]
            else:
                return self.payout_scenarios["upwork_fixed"]
        
        elif "fiverr" in platform:
            if "custom" in opportunity.get("description", "").lower():
                return self.payout_scenarios["fiverr_custom"]
            else:
                return self.payout_scenarios["fiverr_gig"]
        
        elif "freelancer" in platform:
            if "hourly" in service_type.lower():
                return self.payout_scenarios["freelancer_hourly"]
            else:
                return self.payout_scenarios["freelancer_fixed"]
        
        else:
            # Default to direct payment for unknown platforms
            if budget >= 100:
                return self.payout_scenarios["revolut_direct"]
            else:
                return self.payout_scenarios["paypal_direct"]
    
    def calculate_expected_payout(self, opportunity: Dict) -> Dict:
        """Calculate expected payout with all factors"""
        scenario = self.analyze_payout_for_opportunity(opportunity)
        gross_amount = opportunity.get("budget", 0)
        
        # Calculate fees
        if scenario.fees < 1.0:  # Percentage fee
            fee_amount = gross_amount * scenario.fees
        else:  # Fixed fee
            fee_amount = scenario.fees
        
        net_amount = gross_amount - fee_amount
        
        # Calculate expected timeline
        processing_days = self._parse_processing_time(scenario.processing_time)
        
        # Calculate success probability
        success_probability = 1.0 - scenario.failure_rate
        
        # Expected value (considering failure rate)
        expected_value = net_amount * success_probability
        
        return {
            "scenario": scenario,
            "gross_amount": gross_amount,
            "fee_amount": fee_amount,
            "net_amount": net_amount,
            "processing_days": processing_days,
            "success_probability": success_probability,
            "expected_value": expected_value,
            "client_action_required": scenario.client_action_required,
            "verification_required": scenario.verification_required,
            "retry_possible": scenario.retry_possible
        }
    
    def _parse_processing_time(self, processing_time: str) -> int:
        """Parse processing time string to days"""
        if "instant" in processing_time.lower():
            return 0
        elif "daily" in processing_time.lower():
            return 1
        elif "weekly" in processing_time.lower():
            return 7
        elif "14 days" in processing_time:
            return 14
        elif "5-7 days" in processing_time:
            return 6
        elif "1-3 business days" in processing_time:
            return 3
        else:
            return 7  # Default to 1 week
    
    def generate_payout_recommendations(self, opportunities: List[Dict]) -> List[Dict]:
        """Generate payout recommendations for opportunities"""
        recommendations = []
        
        for opportunity in opportunities:
            payout_analysis = self.calculate_expected_payout(opportunity)
            
            recommendation = {
                "opportunity_id": opportunity.get("id", "unknown"),
                "platform": opportunity.get("platform", "unknown"),
                "budget": opportunity.get("budget", 0),
                "recommended_scenario": payout_analysis["scenario"].category.value,
                "payment_method": payout_analysis["scenario"].payment_method,
                "expected_net": payout_analysis["net_amount"],
                "expected_value": payout_analysis["expected_value"],
                "processing_days": payout_analysis["processing_days"],
                "success_rate": payout_analysis["success_probability"],
                "client_action_needed": payout_analysis["client_action_required"],
                "risk_level": self._calculate_risk_level(payout_analysis),
                "priority_score": self._calculate_priority_score(payout_analysis)
            }
            
            recommendations.append(recommendation)
        
        # Sort by priority score (highest first)
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return recommendations
    
    def _calculate_risk_level(self, payout_analysis: Dict) -> str:
        """Calculate risk level for payout"""
        success_prob = payout_analysis["success_probability"]
        client_action = payout_analysis["client_action_required"]
        verification = payout_analysis["verification_required"]
        
        risk_factors = 0
        
        if success_prob < 0.8:
            risk_factors += 1
        if client_action:
            risk_factors += 1
        if verification:
            risk_factors += 1
        
        if risk_factors == 0:
            return "LOW"
        elif risk_factors == 1:
            return "MEDIUM"
        elif risk_factors == 2:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _calculate_priority_score(self, payout_analysis: Dict) -> float:
        """Calculate priority score for opportunity"""
        expected_value = payout_analysis["expected_value"]
        success_prob = payout_analysis["success_probability"]
        processing_days = payout_analysis["processing_days"]
        client_action = payout_analysis["client_action_required"]
        
        # Base score from expected value
        score = expected_value / 100  # Normalize to 0-10 scale
        
        # Bonus for high success rate
        score += success_prob * 2
        
        # Penalty for long processing time
        score -= (processing_days / 30)  # Penalty up to 1 point
        
        # Penalty for client action required
        if client_action:
            score -= 0.5
        
        return max(0, score)  # Ensure non-negative
    
    def track_payout_performance(self, opportunity: Dict, actual_result: Dict):
        """Track actual payout performance"""
        scenario = self.analyze_payout_for_opportunity(opportunity)
        
        performance_record = {
            "timestamp": datetime.now(),
            "opportunity_id": opportunity.get("id"),
            "platform": opportunity.get("platform"),
            "scenario_category": scenario.category.value,
            "expected_amount": opportunity.get("budget", 0),
            "actual_amount": actual_result.get("amount", 0),
            "expected_processing_days": self._parse_processing_time(scenario.processing_time),
            "actual_processing_days": actual_result.get("processing_days", 0),
            "success": actual_result.get("success", False),
            "failure_reason": actual_result.get("failure_reason", ""),
            "client_paid": actual_result.get("client_paid", False)
        }
        
        self.payout_history.append(performance_record)
        
        # Update failure patterns
        if not actual_result.get("success", False):
            failure_type = actual_result.get("failure_reason", "unknown")
            if failure_type not in self.failure_patterns:
                self.failure_patterns[failure_type] = 0
            self.failure_patterns[failure_type] += 1
    
    def generate_failure_analysis(self) -> Dict:
        """Generate analysis of payout failures"""
        if not self.payout_history:
            return {"message": "No payout history available"}
        
        total_payouts = len(self.payout_history)
        successful_payouts = len([p for p in self.payout_history if p["success"]])
        failed_payouts = total_payouts - successful_payouts
        
        # Analyze by scenario
        scenario_performance = {}
        for record in self.payout_history:
            scenario = record["scenario_category"]
            if scenario not in scenario_performance:
                scenario_performance[scenario] = {"total": 0, "success": 0}
            scenario_performance[scenario]["total"] += 1
            if record["success"]:
                scenario_performance[scenario]["success"] += 1
        
        # Calculate success rates
        for scenario in scenario_performance:
            total = scenario_performance[scenario]["total"]
            success = scenario_performance[scenario]["success"]
            scenario_performance[scenario]["success_rate"] = success / total if total > 0 else 0
        
        return {
            "total_payouts": total_payouts,
            "successful_payouts": successful_payouts,
            "failed_payouts": failed_payouts,
            "overall_success_rate": successful_payouts / total_payouts if total_payouts > 0 else 0,
            "scenario_performance": scenario_performance,
            "failure_patterns": self.failure_patterns,
            "recommendations": self._generate_failure_recommendations()
        }
    
    def _generate_failure_recommendations(self) -> List[str]:
        """Generate recommendations based on failure patterns"""
        recommendations = []
        
        for failure_type, count in self.failure_patterns.items():
            if failure_type == "client_non_payment":
                recommendations.append("Focus on platform escrow payments to reduce client non-payment risk")
            elif failure_type == "verification_failed":
                recommendations.append("Improve verification process and documentation")
            elif failure_type == "processing_delay":
                recommendations.append("Prioritize platforms with faster processing times")
            elif failure_type == "dispute":
                recommendations.append("Improve work quality and communication to reduce disputes")
        
        return recommendations
    
    def get_payout_optimization_suggestions(self) -> List[str]:
        """Get suggestions for payout optimization"""
        suggestions = [
            "Prioritize Upwork hourly projects for consistent weekly payments",
            "Use Revolut direct payments for small, quick projects",
            "Avoid milestone projects unless client has good reputation",
            "Set minimum project amounts to justify processing fees",
            "Focus on platforms with automatic release when possible",
            "Maintain multiple payment methods for flexibility",
            "Track payment processing times to identify bottlenecks",
            "Build client relationships to reduce payment failures"
        ]
        
        return suggestions
    
    def export_payout_analysis(self, filename: str = "payout_analysis.json"):
        """Export complete payout analysis"""
        analysis_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "payout_scenarios": {
                name: {
                    "category": scenario.category.value,
                    "platform": scenario.platform,
                    "payment_method": scenario.payment_method,
                    "processing_time": scenario.processing_time,
                    "fees": scenario.fees,
                    "verification_required": scenario.verification_required,
                    "retry_possible": scenario.retry_possible,
                    "escrow_held": scenario.escrow_held,
                    "milestones": scenario.milestones,
                    "auto_release": scenario.auto_release,
                    "client_action_required": scenario.client_action_required,
                    "failure_rate": scenario.failure_rate,
                    "minimum_amount": scenario.minimum_amount,
                    "maximum_amount": scenario.maximum_amount
                }
                for name, scenario in self.payout_scenarios.items()
            },
            "payout_history": [
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "opportunity_id": record["opportunity_id"],
                    "platform": record["platform"],
                    "scenario_category": record["scenario_category"],
                    "success": record["success"],
                    "failure_reason": record["failure_reason"]
                }
                for record in self.payout_history
            ],
            "failure_patterns": self.failure_patterns,
            "optimization_suggestions": self.get_payout_optimization_suggestions()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"Payout analysis exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export payout analysis: {e}")
            return False

def main():
    """Main function for payout analysis"""
    print("Janus Payout Analysis")
    print("=" * 40)
    print("Comprehensive payout type analysis")
    print()
    
    analyzer = JanusPayoutAnalysis()
    
    # Analyze sample opportunities
    sample_opportunities = [
        {
            "id": "sample1",
            "platform": "upwork",
            "service": "content_writing",
            "budget": 500.0,
            "description": "Fixed price project"
        },
        {
            "id": "sample2", 
            "platform": "fiverr",
            "service": "code_development",
            "budget": 150.0,
            "description": "Gig order"
        },
        {
            "id": "sample3",
            "platform": "direct",
            "service": "data_analysis",
            "budget": 300.0,
            "description": "Direct client project"
        }
    ]
    
    # Generate recommendations
    recommendations = analyzer.generate_payout_recommendations(sample_opportunities)
    
    print("PAYOUT RECOMMENDATIONS:")
    print("-" * 30)
    for rec in recommendations:
        print(f"Platform: {rec['platform']}")
        print(f"Budget: ${rec['budget']:.2f}")
        print(f"Payment Method: {rec['payment_method']}")
        print(f"Expected Net: ${rec['expected_net']:.2f}")
        print(f"Success Rate: {rec['success_rate']:.1%}")
        print(f"Risk Level: {rec['risk_level']}")
        print(f"Priority Score: {rec['priority_score']:.2f}")
        print()
    
    # Export analysis
    analyzer.export_payout_analysis()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
