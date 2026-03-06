"""
Emotional Intelligence Module
Detect sentiment, understand stakes, adapt communication
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger("emotional_intelligence")


class Sentiment(Enum):
    """Sentiment classifications."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class EmotionalContext:
    """Represents emotional state of a situation."""
    
    def __init__(self):
        self.primary_sentiment = Sentiment.NEUTRAL
        self.secondary_sentiments = []
        self.emotional_stakes = 0.0  # 0-1, how much emotion is involved
        self.trust_level = 0.5  # 0-1, perceived trust
        self.urgency = 0.0  # 0-1, emotional urgency
        self.detected_emotions = {}  # anger, fear, joy, sadness, etc.
    
    def to_dict(self) -> Dict:
        return {
            "primary_sentiment": self.primary_sentiment.name,
            "emotional_stakes": self.emotional_stakes,
            "trust_level": self.trust_level,
            "urgency": self.urgency,
            "detected_emotions": self.detected_emotions
        }


class EmotionalIntelligence:
    """
    Detect and respond to emotional contexts.
    Does NOT fake emotions, but reads them accurately.
    """
    
    def __init__(self):
        self.conversation_history = []
        self.stakeholder_profiles = {}  # Track emotional patterns of people
        self.conflict_log = []
        
        logger.info("Emotional Intelligence Module initialized")
    
    def analyze_communication(self, text: str, speaker: str = "unknown") -> EmotionalContext:
        """
        Analyze emotional content of communication.
        Uses word patterns, not NLP (keep it simple, interpretable).
        """
        
        context = EmotionalContext()
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Detect negative signals
        negative_words = [
            "angry", "upset", "frustrated", "disappointed", "concerned",
            "worried", "stressed", "problem", "failed", "broken", "lost",
            "terrible", "hate", "unacceptable", "disaster"
        ]
        
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Detect positive signals
        positive_words = [
            "great", "excellent", "happy", "satisfied", "confident",
            "success", "achieved", "proud", "wonderful", "love", "perfect",
            "grateful", "thrilled", "amazing"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Detect urgency signals
        urgency_words = [
            "urgent", "immediately", "now", "asap", "crisis", "emergency",
            "critical", "must", "have to", "can't wait"
        ]
        
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        # Detect trust signals
        trust_words = [
            "trust", "reliable", "consistent", "honest", "transparent",
            "accountable", "responsible"
        ]
        distrust_words = [
            "unreliable", "dishonest", "opaque", "avoid", "suspicious"
        ]
        
        trust_count = sum(1 for word in trust_words if word in text_lower)
        distrust_count = sum(1 for word in distrust_words if word in text_lower)
        
        # Calculate sentiment
        if negative_count > positive_count + 2:
            context.primary_sentiment = Sentiment.VERY_NEGATIVE if negative_count > 5 else Sentiment.NEGATIVE
            context.detected_emotions["anger"] = 0.7
            context.detected_emotions["frustration"] = 0.6
        elif positive_count > negative_count + 2:
            context.primary_sentiment = Sentiment.VERY_POSITIVE if positive_count > 5 else Sentiment.POSITIVE
            context.detected_emotions["satisfaction"] = 0.7
            context.detected_emotions["confidence"] = 0.6
        else:
            context.primary_sentiment = Sentiment.NEUTRAL
        
        # Calculate emotional stakes
        context.emotional_stakes = min(1.0, (negative_count + positive_count) / 5.0)
        
        # Calculate urgency
        context.urgency = min(1.0, urgency_count / 3.0)
        
        # Calculate trust
        net_trust = trust_count - distrust_count
        context.trust_level = max(0.0, min(1.0, 0.5 + (net_trust / 10.0)))
        
        # Detect fear/anxiety
        if "worry" in text_lower or "concerned" in text_lower or "risk" in text_lower:
            context.detected_emotions["anxiety"] = 0.5
        
        # Log for stakeholder profile
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "text": text,
            "emotional_context": context.to_dict()
        })
        
        # Update stakeholder profile
        if speaker not in self.stakeholder_profiles:
            self.stakeholder_profiles[speaker] = {
                "messages": 0,
                "avg_sentiment": 0.0,
                "emotional_patterns": {},
                "trust_score": 0.5
            }
        
        profile = self.stakeholder_profiles[speaker]
        profile["messages"] += 1
        profile["avg_sentiment"] = (profile["avg_sentiment"] + context.primary_sentiment.value) / 2
        profile["trust_score"] = context.trust_level
        
        return context
    
    def detect_conflict(self, stakeholder1: str, stakeholder2: str) -> Dict:
        """
        Detect conflict between two stakeholders by analyzing their communication patterns.
        """
        
        if stakeholder1 not in self.stakeholder_profiles or stakeholder2 not in self.stakeholder_profiles:
            return {"conflict_detected": False, "confidence": 0.0}
        
        profile1 = self.stakeholder_profiles[stakeholder1]
        profile2 = self.stakeholder_profiles[stakeholder2]
        
        # Signs of conflict
        conflict_score = 0.0
        
        # Opposite sentiments
        if (profile1["avg_sentiment"] > 0.5 and profile2["avg_sentiment"] < -0.5) or \
           (profile1["avg_sentiment"] < -0.5 and profile2["avg_sentiment"] > 0.5):
            conflict_score += 0.3
        
        # Both have low trust in each other
        if profile1["trust_score"] < 0.3 and profile2["trust_score"] < 0.3:
            conflict_score += 0.4
        
        # Multiple negative interactions
        negative_interactions = sum(
            1 for msg in self.conversation_history
            if (msg["speaker"] == stakeholder1 and msg["emotional_context"]["primary_sentiment"].value < 0) or
               (msg["speaker"] == stakeholder2 and msg["emotional_context"]["primary_sentiment"].value < 0)
        )
        
        if negative_interactions > 3:
            conflict_score += 0.3
        
        return {
            "conflict_detected": conflict_score > 0.5,
            "confidence": min(1.0, conflict_score),
            "stakeholders": [stakeholder1, stakeholder2],
            "profile1_sentiment": profile1["avg_sentiment"],
            "profile2_sentiment": profile2["avg_sentiment"]
        }
    
    def suggest_conflict_resolution(self, stakeholder1: str, stakeholder2: str) -> Dict:
        """
        Suggest how to resolve conflict between stakeholders.
        """
        
        conflict = self.detect_conflict(stakeholder1, stakeholder2)
        
        if not conflict["conflict_detected"]:
            return {"status": "no_conflict", "recommendation": None}
        
        profile1 = self.stakeholder_profiles.get(stakeholder1, {})
        profile2 = self.stakeholder_profiles.get(stakeholder2, {})
        
        # Find common ground
        interests1 = self._infer_interests(stakeholder1)
        interests2 = self._infer_interests(stakeholder2)
        common_interests = set(interests1) & set(interests2)
        
        # Suggest resolution strategy
        resolution = {
            "conflict": f"between {stakeholder1} and {stakeholder2}",
            "root_cause": self._analyze_conflict_cause(stakeholder1, stakeholder2),
            "common_interests": list(common_interests),
            "recommended_approach": [
                "1. Acknowledge both perspectives",
                "2. Focus on shared goals",
                "3. Propose win-win solutions",
                f"4. Common ground: {', '.join(common_interests) if common_interests else 'financial success'}"
            ],
            "suggested_message_tone": "collaborative, transparent, solution-focused",
            "avoid": [
                "Blame or defensiveness",
                "Taking sides",
                "Ignoring emotional concerns"
            ]
        }
        
        self.conflict_log.append({
            "timestamp": datetime.now().isoformat(),
            "conflict": resolution,
            "suggested_resolution": True
        })
        
        return resolution
    
    def _infer_interests(self, stakeholder: str) -> List[str]:
        """Infer what a stakeholder cares about from their messages."""
        
        interests = []
        messages = [m["text"] for m in self.conversation_history if m["speaker"] == stakeholder]
        
        all_text = " ".join(messages).lower()
        
        interest_keywords = {
            "growth": ["grow", "scale", "expand", "revenue", "market"],
            "stability": ["reliable", "consistent", "safe", "secure", "stable"],
            "efficiency": ["fast", "quick", "optimize", "improve", "reduce"],
            "quality": ["quality", "excellence", "perfect", "best"],
            "fairness": ["fair", "equal", "just", "honest", "transparent"],
            "autonomy": ["independent", "freedom", "control", "decision"],
            "relationships": ["team", "together", "collaborate", "communicate"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(kw in all_text for kw in keywords):
                interests.append(interest)
        
        return interests if interests else ["success"]
    
    def _analyze_conflict_cause(self, stakeholder1: str, stakeholder2: str) -> str:
        """Analyze the root cause of conflict."""
        
        # Look for specific conflict triggers
        recent_messages = self.conversation_history[-10:]
        
        for msg in recent_messages:
            if "disagree" in msg["text"].lower() or "different" in msg["text"].lower():
                return "Different priorities or approaches"
            if "blame" in msg["text"].lower() or "fault" in msg["text"].lower():
                return "Attribution of blame"
            if "money" in msg["text"].lower() or "cost" in msg["text"].lower():
                return "Resource or financial disagreement"
        
        return "Unclear - may need direct conversation"
    
    def adapt_communication_style(self, audience_sentiment: Sentiment) -> Dict:
        """
        Adapt how to communicate based on emotional context.
        This is NOT faking emotions, just being smart about delivery.
        """
        
        if audience_sentiment == Sentiment.VERY_NEGATIVE:
            return {
                "tone": "empathetic, action-oriented",
                "approach": "Acknowledge concerns first, then solutions",
                "avoid": "Minimizing their feelings, being dismissive",
                "focus": "Concrete steps to fix the problem",
                "transparency": "Be honest about challenges"
            }
        
        elif audience_sentiment == Sentiment.NEGATIVE:
            return {
                "tone": "understanding, constructive",
                "approach": "Validate feelings, then pivot to solutions",
                "avoid": "Defensiveness",
                "focus": "What can we do differently?",
                "transparency": "Explain the reasoning"
            }
        
        elif audience_sentiment == Sentiment.NEUTRAL:
            return {
                "tone": "professional, clear",
                "approach": "Straightforward information",
                "avoid": "Over-explaining or under-explaining",
                "focus": "Key facts and implications",
                "transparency": "Standard level of detail"
            }
        
        elif audience_sentiment == Sentiment.POSITIVE:
            return {
                "tone": "collaborative, forward-looking",
                "approach": "Build on momentum",
                "avoid": "Over-promising or taking things for granted",
                "focus": "Opportunities and next steps",
                "transparency": "Share ambitious plans"
            }
        
        else:  # VERY_POSITIVE
            return {
                "tone": "inspired, visionary",
                "approach": "Ambitious but grounded",
                "avoid": "Unrealistic expectations",
                "focus": "Long-term vision with near-term wins",
                "transparency": "Share bigger picture"
            }
    
    def get_emotional_summary(self) -> Dict:
        """Get summary of emotional landscape."""
        
        return {
            "total_interactions": len(self.conversation_history),
            "stakeholders": len(self.stakeholder_profiles),
            "conflicts_detected": len([c for c in self.conflict_log if c.get("suggested_resolution")]),
            "avg_sentiment": sum(p["avg_sentiment"] for p in self.stakeholder_profiles.values()) / max(len(self.stakeholder_profiles), 1),
            "stakeholder_profiles": {
                name: {
                    "avg_sentiment": profile["avg_sentiment"],
                    "trust_score": profile["trust_score"],
                    "interactions": profile["messages"]
                }
                for name, profile in self.stakeholder_profiles.items()
            }
        }


if __name__ == "__main__":
    print("Emotional Intelligence Module")
    print("=" * 50)
    
    ei = EmotionalIntelligence()
    
    # Simulate stakeholder communication
    print("\n[Simulating stakeholder interactions]")
    
    # Stakeholder 1: Frustrated
    ctx1 = ei.analyze_communication(
        "I'm frustrated with the delays. This is unacceptable. We need results now!",
        "Investor"
    )
    print(f"Investor sentiment: {ctx1.primary_sentiment.name}")
    
    # Stakeholder 2: Defensive
    ctx2 = ei.analyze_communication(
        "We're doing our best. The market is challenging. This is harder than expected.",
        "Team Lead"
    )
    print(f"Team Lead sentiment: {ctx2.primary_sentiment.name}")
    
    # Detect conflict
    conflict = ei.detect_conflict("Investor", "Team Lead")
    print(f"\nConflict detected: {conflict['conflict_detected']} (confidence: {conflict['confidence']:.2f})")
    
    # Suggest resolution
    if conflict["conflict_detected"]:
        resolution = ei.suggest_conflict_resolution("Investor", "Team Lead")
        print(f"Root cause: {resolution['root_cause']}")
        print(f"Recommended approach:")
        for step in resolution["recommended_approach"]:
            print(f"  {step}")
    
    # Get summary
    print(f"\nEmotional Summary:")
    summary = ei.get_emotional_summary()
    print(json.dumps(summary, indent=2))
