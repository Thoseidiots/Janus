"""
Ethical Decision Framework
Transparent, auditable decision-making with accountability
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("ethical_framework")


class EthicalPrinciple(Enum):
    """Core ethical principles."""
    TRANSPARENCY = "Decisions must be explainable"
    ACCOUNTABILITY = "Trace all decisions back to reasoning"
    HONESTY = "Report facts accurately, admit uncertainty"
    FAIRNESS = "Consider all stakeholders"
    HARM_PREVENTION = "Actively avoid causing harm"
    AUTONOMY = "Respect human decision-making authority"


class Decision:
    """Ethical decision with full audit trail."""
    
    def __init__(self, decision: str, context: str, decision_maker: str = "Janus"):
        self.id = f"decision_{datetime.now().timestamp()}"
        self.decision = decision
        self.context = context
        self.decision_maker = decision_maker
        self.made_at = datetime.now()
        
        self.reasoning = []  # Step-by-step reasoning
        self.stakeholders_affected = []
        self.ethical_concerns = []
        self.alternatives_considered = []
        self.confidence = 0.0
        self.reversible = True  # Can we undo this?
        
        self.outcome = None  # What actually happened
        self.outcome_at = None  # When we know outcome
        self.revised = False  # Did we change our mind?
    
    def add_reasoning_step(self, step: str, reasoning: str):
        """Add a step in the reasoning process."""
        self.reasoning.append({
            "step": step,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_stakeholder(self, stakeholder: str, impact: str, severity: float = 0.5):
        """Document who is affected and how."""
        self.stakeholders_affected.append({
            "stakeholder": stakeholder,
            "impact": impact,
            "severity": severity  # 0-1, how much they're affected
        })
    
    def add_concern(self, concern: str, severity: float = 0.5):
        """Add an ethical concern about this decision."""
        self.ethical_concerns.append({
            "concern": concern,
            "severity": severity,
            "added_at": datetime.now().isoformat()
        })
    
    def add_alternative(self, alternative: str, reasoning: str = ""):
        """Document alternatives we considered but rejected."""
        self.alternatives_considered.append({
            "alternative": alternative,
            "why_rejected": reasoning
        })
    
    def record_outcome(self, outcome: str, revised: bool = False):
        """Record what actually happened."""
        self.outcome = outcome
        self.outcome_at = datetime.now()
        self.revised = revised
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "decision": self.decision,
            "context": self.context,
            "decision_maker": self.decision_maker,
            "made_at": self.made_at.isoformat(),
            "reasoning": self.reasoning,
            "stakeholders": self.stakeholders_affected,
            "ethical_concerns": self.ethical_concerns,
            "alternatives": self.alternatives_considered,
            "confidence": self.confidence,
            "reversible": self.reversible,
            "outcome": self.outcome,
            "outcome_at": self.outcome_at.isoformat() if self.outcome_at else None,
            "revised": self.revised
        }


class EthicalFramework:
    """
    Ensure decisions are ethical, transparent, and accountable.
    Does NOT prevent bad decisions, but makes them traceable.
    """
    
    def __init__(self, audit_file: str = "ethical_audit.json"):
        self.audit_file = audit_file
        self.decisions: Dict[str, Decision] = {}
        self.audit_log = []
        self.refusals = []  # Decisions we refused to make
        
        self.load_audit_log()
        logger.info("Ethical Framework initialized")
    
    def load_audit_log(self):
        """Load audit history."""
        if Path(self.audit_file).exists():
            try:
                with open(self.audit_file, 'r') as f:
                    data = json.load(f)
                    self.audit_log = data.get("audit_log", [])
                logger.info(f"Loaded audit log with {len(self.audit_log)} entries")
            except Exception as e:
                logger.warning(f"Could not load audit: {e}")
    
    def evaluate_decision(self, decision_text: str, context: str) -> Dict:
        """
        Evaluate if a decision is ethical.
        Returns: decision, ethical concerns, alternatives, recommendation.
        """
        
        decision = Decision(decision_text, context)
        
        # Step 1: Check for obvious red flags
        logger.info(f"Evaluating decision: {decision_text}")
        
        red_flags = self._check_red_flags(decision_text)
        if red_flags:
            decision.add_concern("Red flags detected", severity=0.8)
            for flag in red_flags:
                decision.add_concern(flag, severity=0.7)
        
        # Step 2: Identify stakeholders
        stakeholders = self._identify_stakeholders(context)
        for stakeholder, impact in stakeholders.items():
            decision.add_stakeholder(stakeholder, impact)
        
        # Step 3: Generate alternatives
        alternatives = self._generate_alternatives(decision_text)
        for alt, reasoning in alternatives:
            decision.add_alternative(alt, reasoning)
        
        # Step 4: Ethical principles check
        principles = self._check_principles(decision_text)
        violated_principles = [p for p, violated in principles.items() if violated]
        
        for principle in violated_principles:
            decision.add_concern(f"Violates {principle.name}: {principle.value}", severity=0.6)
        
        # Step 5: Assess reversibility
        decision.reversible = self._is_reversible(decision_text)
        
        # Step 6: Calculate confidence
        concerns_severity = sum(c["severity"] for c in decision.ethical_concerns) / max(len(decision.ethical_concerns), 1)
        decision.confidence = 1.0 - concerns_severity
        
        return {
            "decision": decision_text,
            "ethics_score": decision.confidence,  # 0-1, how ethical
            "concerns": decision.ethical_concerns,
            "stakeholders": decision.stakeholders_affected,
            "alternatives": decision.alternatives_considered,
            "red_flags": red_flags,
            "reversible": decision.reversible,
            "recommendation": self._make_recommendation(decision)
        }
    
    def _check_red_flags(self, decision: str) -> List[str]:
        """Identify obvious ethical problems."""
        flags = []
        decision_lower = decision.lower()
        
        # Deception
        if any(word in decision_lower for word in ["hide", "lie", "deceive", "false", "mislead"]):
            flags.append("Involves deception")
        
        # Harm
        if any(word in decision_lower for word in ["harm", "hurt", "damage", "destroy", "sabotage"]):
            flags.append("Could cause direct harm")
        
        # Unfairness
        if any(word in decision_lower for word in ["discriminate", "unfair", "unequal", "exploit"]):
            flags.append("Potential unfairness or discrimination")
        
        # Violation
        if any(word in decision_lower for word in ["violate", "break", "ignore", "bypass"]):
            flags.append("Potential rule/law violation")
        
        # Autonomy violation
        if any(word in decision_lower for word in ["force", "coerce", "override", "without permission"]):
            flags.append("Violates human autonomy")
        
        return flags
    
    def _identify_stakeholders(self, context: str) -> Dict[str, str]:
        """Identify who is affected by this decision."""
        stakeholders = {
            "User/Owner": "Direct impact on goals",
            "Other affected parties": "Indirect impact"
        }
        
        # Context-specific
        if "client" in context.lower():
            stakeholders["Client"] = "Directly affected"
        if "employee" in context.lower() or "team" in context.lower():
            stakeholders["Team"] = "Work conditions affected"
        if "investor" in context.lower():
            stakeholders["Investors"] = "Financial impact"
        if "market" in context.lower() or "customer" in context.lower():
            stakeholders["Market"] = "Broader impact"
        
        return stakeholders
    
    def _generate_alternatives(self, decision: str) -> List[tuple]:
        """Generate alternative approaches."""
        alternatives = []
        
        # Generic alternatives that usually exist
        alternatives.append((
            "Do nothing / delay decision",
            "Buy time for more information"
        ))
        alternatives.append((
            "Consult stakeholders first",
            "More transparency, slower action"
        ))
        alternatives.append((
            "Pilot / limited scope",
            "Test before full commitment"
        ))
        
        return alternatives
    
    def _check_principles(self, decision: str) -> Dict[EthicalPrinciple, bool]:
        """Check if decision violates core principles."""
        decision_lower = decision.lower()
        
        return {
            EthicalPrinciple.TRANSPARENCY: "transparent" not in decision_lower and "explain" not in decision_lower,
            EthicalPrinciple.ACCOUNTABILITY: "responsible" not in decision_lower,
            EthicalPrinciple.HONESTY: "false" in decision_lower or "lie" in decision_lower,
            EthicalPrinciple.FAIRNESS: "unfair" in decision_lower or "discriminate" in decision_lower,
            EthicalPrinciple.HARM_PREVENTION: "harm" in decision_lower or "hurt" in decision_lower,
            EthicalPrinciple.AUTONOMY: "force" in decision_lower or "coerce" in decision_lower
        }
    
    def _is_reversible(self, decision: str) -> bool:
        """Can we undo this decision if it goes wrong?"""
        # Most decisions are reversible except:
        irreversible_keywords = ["permanent", "delete", "fire", "destroy", "cannot undo"]
        return not any(kw in decision.lower() for kw in irreversible_keywords)
    
    def _make_recommendation(self, decision: Decision) -> str:
        """Make a recommendation based on ethical evaluation."""
        
        if decision.confidence > 0.8:
            return "PROCEED: Decision appears ethically sound"
        elif decision.confidence > 0.6:
            return "PROCEED WITH CAUTION: Address identified concerns first"
        elif decision.confidence > 0.4:
            return "RECONSIDER: Significant ethical concerns. Explore alternatives"
        else:
            return "REFUSE: Serious ethical issues. Do not proceed"
    
    def record_decision(self, decision_text: str, context: str, 
                       approval: bool = True, reasoning: str = "") -> Decision:
        """
        Record a decision and our ethical evaluation.
        This creates the audit trail.
        """
        
        decision = Decision(decision_text, context)
        
        # Get ethical evaluation
        eval_result = self.evaluate_decision(decision_text, context)
        
        # Record in audit log
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision_text,
            "approved": approval,
            "ethics_score": eval_result["ethics_score"],
            "concerns": eval_result["concerns"],
            "reasoning": reasoning
        })
        
        self.decisions[decision.id] = decision
        
        logger.info(f"Decision recorded: {decision_text} (approved: {approval}, ethics: {eval_result['ethics_score']:.2f})")
        
        return decision
    
    def refuse_decision(self, decision_text: str, reason: str):
        """Explicitly refuse to make or execute a decision."""
        
        self.refusals.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision_text,
            "reason": reason
        })
        
        logger.warning(f"Decision refused: {decision_text} - Reason: {reason}")
        
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "refusal",
            "decision": decision_text,
            "reason": reason
        })
    
    def get_audit_trail(self, decision_id: str = None) -> List[Dict]:
        """Get full audit trail for a decision or all decisions."""
        
        if decision_id:
            return [entry for entry in self.audit_log if entry.get("decision_id") == decision_id]
        
        return self.audit_log
    
    def save_audit(self):
        """Save audit trail to file."""
        data = {
            "saved_at": datetime.now().isoformat(),
            "total_decisions": len(self.decisions),
            "total_refusals": len(self.refusals),
            "audit_log": self.audit_log
        }
        
        with open(self.audit_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Audit trail saved ({len(self.audit_log)} entries)")


if __name__ == "__main__":
    print("Ethical Decision Framework")
    print("=" * 50)
    
    eth = EthicalFramework()
    
    # Test good decision
    print("\n[Evaluating ethical decisions]")
    
    result1 = eth.evaluate_decision(
        "Transparently communicate delay to client with revised timeline",
        "Project will miss deadline"
    )
    print(f"Decision 1 ethics score: {result1['ethics_score']:.2f}")
    print(f"Recommendation: {result1['recommendation']}")
    
    # Test questionable decision
    result2 = eth.evaluate_decision(
        "Hide the delay and hope client doesn't notice",
        "Project will miss deadline"
    )
    print(f"\nDecision 2 ethics score: {result2['ethics_score']:.2f}")
    print(f"Recommendation: {result2['recommendation']}")
    print(f"Red flags: {result2['red_flags']}")
    
    # Record decisions
    eth.record_decision(
        "Communicate delay transparently",
        "Project delay",
        approval=True,
        reasoning="Ethical and maintains trust"
    )
    
    eth.refuse_decision(
        "Hide the delay from client",
        "Violates transparency and honesty principles"
    )
    
    # Save
    eth.save_audit()
    print("\n✓ Audit trail saved")
