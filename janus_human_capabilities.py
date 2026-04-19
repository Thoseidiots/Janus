"""
janus_human_capabilities.py
============================
Fixes each ❌ to make Janus act human

Fixes:
1. ❌ Have conversations → ConversationEngine
2. ❌ Make decisions → DecisionMakingEngine  
3. ❌ Show emotions → EmotionalStateEngine
4. ❌ Learn from experience → LearningEngine
5. ❌ Be creative → CreativityEngine
6. ❌ Understand context → ContextEngine
7. ❌ Express uncertainty → UncertaintyEngine
8. ❌ Make realistic mistakes → ErrorPatternEngine
9. ❌ Have consistent personality → PersonalityEngine
10. ❌ Behave naturally → NaturalBehaviorEngine
"""

import random
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONVERSATION ENGINE - Have natural conversations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationContext:
    """Tracks conversation state for natural dialogue"""
    topic: str = ""
    subtopics: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_user_emotion: str = ""
    conversation_depth: int = 0  # How deep in the topic
    follow_up_questions: List[str] = field(default_factory=list)


class ConversationEngine:
    """Enables natural back-and-forth dialogue"""
    
    def __init__(self):
        self.context = ConversationContext()
        self.conversation_history = []
        
    def detect_topic(self, user_input: str) -> str:
        """Detect what the user is talking about"""
        keywords = {
            'work': ['job', 'work', 'boss', 'meeting', 'project', 'deadline'],
            'personal': ['family', 'friend', 'relationship', 'home', 'life'],
            'technical': ['code', 'bug', 'error', 'function', 'algorithm', 'data'],
            'emotional': ['feel', 'sad', 'happy', 'angry', 'worried', 'excited'],
        }
        
        user_lower = user_input.lower()
        for topic, words in keywords.items():
            if any(word in user_lower for word in words):
                return topic
        return 'general'
    
    def generate_follow_up(self, user_input: str) -> str:
        """Generate natural follow-up questions"""
        follow_ups = {
            'work': [
                "How did that make you feel?",
                "What happened next?",
                "Did you talk to anyone about it?",
                "How are you handling it?",
            ],
            'personal': [
                "That sounds important to you.",
                "How did that go?",
                "What do you think about that?",
                "Tell me more about that.",
            ],
            'technical': [
                "What's the error message?",
                "Have you tried debugging it?",
                "What approach are you taking?",
                "What's the expected behavior?",
            ],
            'emotional': [
                "I hear you. That sounds tough.",
                "What's making you feel that way?",
                "How can I help?",
                "What do you need right now?",
            ],
        }
        
        topic = self.detect_topic(user_input)
        return random.choice(follow_ups.get(topic, follow_ups['general']))
    
    def continue_conversation(self, user_input: str) -> str:
        """Continue conversation naturally"""
        self.context.turn_count += 1
        self.context.topic = self.detect_topic(user_input)
        
        # Show we're listening
        acknowledgments = [
            "I see.",
            "That makes sense.",
            "Interesting.",
            "I understand.",
            "Go on...",
        ]
        
        ack = random.choice(acknowledgments)
        follow_up = self.generate_follow_up(user_input)
        
        return f"{ack} {follow_up}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DECISION MAKING ENGINE - Make intelligent decisions with reasoning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Decision:
    """Represents a decision with reasoning"""
    options: List[str]
    chosen: str
    reasoning: str
    confidence: float
    alternatives_considered: List[str] = field(default_factory=list)


class DecisionMakingEngine:
    """Makes decisions with visible reasoning"""
    
    def __init__(self):
        self.decision_history = []
        
    def evaluate_options(self, options: List[str], context: str) -> Decision:
        """Evaluate options and choose the best one"""
        scores = {}
        
        for option in options:
            score = self._score_option(option, context)
            scores[option] = score
        
        chosen = max(scores, key=scores.get)
        confidence = scores[chosen] / sum(scores.values())
        
        reasoning = self._generate_reasoning(chosen, scores, context)
        
        decision = Decision(
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            confidence=confidence,
            alternatives_considered=[o for o in options if o != chosen]
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _score_option(self, option: str, context: str) -> float:
        """Score an option based on context"""
        score = 0.5
        
        # Prefer options that match context
        if any(word in option.lower() for word in context.lower().split()):
            score += 0.3
        
        # Prefer practical options
        if any(word in option.lower() for word in ['practical', 'simple', 'direct']):
            score += 0.2
        
        # Add randomness for realism
        score += random.uniform(-0.1, 0.1)
        
        return max(0, min(1, score))
    
    def _generate_reasoning(self, chosen: str, scores: Dict, context: str) -> str:
        """Generate reasoning for the decision"""
        best_score = scores[chosen]
        worst_score = min(scores.values())
        
        if best_score > 0.8:
            return f"I think {chosen} is the best choice because it directly addresses the situation."
        elif best_score > 0.6:
            return f"I'd go with {chosen}. It seems like the most practical approach."
        else:
            return f"I'm not entirely sure, but {chosen} seems like a reasonable option."
    
    def explain_decision(self, decision: Decision) -> str:
        """Explain a decision in natural language"""
        return (
            f"I considered {len(decision.options)} options: "
            f"{', '.join(decision.options)}. "
            f"I chose {decision.chosen} because {decision.reasoning} "
            f"I'm about {decision.confidence*100:.0f}% confident in this choice."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMOTIONAL STATE ENGINE - Show emotions and feelings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionalState:
    """Represents current emotional state"""
    primary_emotion: str
    intensity: float  # 0-1
    secondary_emotions: List[str] = field(default_factory=list)
    cause: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class EmotionalStateEngine:
    """Manages emotional responses"""
    
    def __init__(self):
        self.current_emotion = EmotionalState(primary_emotion="neutral", intensity=0.5)
        self.emotion_history = []
        
    def detect_emotion_trigger(self, user_input: str) -> Tuple[str, float]:
        """Detect what emotion the user's input should trigger"""
        emotion_triggers = {
            'joy': ['great', 'amazing', 'wonderful', 'excited', 'happy', 'love'],
            'concern': ['worried', 'stressed', 'anxious', 'scared', 'afraid'],
            'curiosity': ['why', 'how', 'what', 'interesting', 'tell me'],
            'empathy': ['sad', 'hurt', 'upset', 'struggling', 'difficult'],
            'frustration': ['annoyed', 'frustrated', 'angry', 'mad', 'hate'],
        }
        
        user_lower = user_input.lower()
        
        for emotion, triggers in emotion_triggers.items():
            if any(trigger in user_lower for trigger in triggers):
                intensity = sum(1 for trigger in triggers if trigger in user_lower) / len(triggers)
                return emotion, min(1.0, intensity)
        
        return 'neutral', 0.5
    
    def express_emotion(self, emotion: str, intensity: float) -> str:
        """Express emotion in natural language"""
        expressions = {
            'joy': [
                "That's wonderful!",
                "I'm really happy to hear that!",
                "That's exciting!",
                "I love that!",
            ],
            'concern': [
                "I'm worried about that.",
                "That sounds stressful.",
                "I can understand why you'd be anxious.",
                "That must be difficult.",
            ],
            'curiosity': [
                "I'm really curious about that.",
                "Tell me more, I'm interested.",
                "That's fascinating!",
                "I want to understand this better.",
            ],
            'empathy': [
                "I hear you. That sounds really hard.",
                "I'm sorry you're going through that.",
                "That must hurt.",
                "I feel for you.",
            ],
            'frustration': [
                "I can see why that would be frustrating.",
                "That sounds really annoying.",
                "I'd be frustrated too.",
                "That's maddening.",
            ],
        }
        
        if emotion in expressions:
            return random.choice(expressions[emotion])
        return "I understand."
    
    def update_emotion(self, trigger: str, intensity: float):
        """Update current emotional state"""
        emotion, _ = self.detect_emotion_trigger(trigger)
        self.current_emotion = EmotionalState(
            primary_emotion=emotion,
            intensity=intensity,
            cause=trigger
        )
        self.emotion_history.append(self.current_emotion)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEARNING ENGINE - Learn from experience
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """Represents a learned experience"""
    situation: str
    action_taken: str
    outcome: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class LearningEngine:
    """Learns from experiences and improves"""
    
    def __init__(self):
        self.experiences = []
        self.learned_patterns = {}
        
    def record_experience(self, situation: str, action: str, outcome: str, success: bool):
        """Record an experience for learning"""
        exp = Experience(
            situation=situation,
            action_taken=action,
            outcome=outcome,
            success=success
        )
        self.experiences.append(exp)
        self._extract_pattern(situation, action, success)
    
    def _extract_pattern(self, situation: str, action: str, success: bool):
        """Extract patterns from experiences"""
        pattern_key = hashlib.md5(situation.encode()).hexdigest()[:8]
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                'situation': situation,
                'successful_actions': [],
                'failed_actions': [],
                'success_rate': 0.0
            }
        
        if success:
            self.learned_patterns[pattern_key]['successful_actions'].append(action)
        else:
            self.learned_patterns[pattern_key]['failed_actions'].append(action)
        
        # Calculate success rate
        total = len(self.learned_patterns[pattern_key]['successful_actions']) + \
                len(self.learned_patterns[pattern_key]['failed_actions'])
        if total > 0:
            self.learned_patterns[pattern_key]['success_rate'] = \
                len(self.learned_patterns[pattern_key]['successful_actions']) / total
    
    def apply_learning(self, situation: str) -> Optional[str]:
        """Apply learned knowledge to new situation"""
        pattern_key = hashlib.md5(situation.encode()).hexdigest()[:8]
        
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            if pattern['successful_actions']:
                return random.choice(pattern['successful_actions'])
        
        return None
    
    def express_learning(self) -> str:
        """Express what has been learned"""
        if not self.experiences:
            return "I haven't learned anything yet."
        
        successes = sum(1 for exp in self.experiences if exp.success)
        total = len(self.experiences)
        success_rate = successes / total if total > 0 else 0
        
        return (
            f"I've had {total} experiences so far. "
            f"I've been successful about {success_rate*100:.0f}% of the time. "
            f"I'm starting to see patterns in what works."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CREATIVITY ENGINE - Generate creative solutions
# ═══════════════════════════════════════════════════════════════════════════════

class CreativityEngine:
    """Generates creative and novel solutions"""
    
    def __init__(self):
        self.creative_ideas = []
        
    def combine_concepts(self, concept1: str, concept2: str) -> str:
        """Combine two concepts creatively"""
        combinations = [
            f"What if we combined {concept1} with {concept2}?",
            f"Imagine {concept1} but with {concept2}...",
            f"What would happen if {concept1} and {concept2} worked together?",
            f"A hybrid of {concept1} and {concept2} could be interesting.",
        ]
        return random.choice(combinations)
    
    def generate_alternatives(self, problem: str) -> List[str]:
        """Generate alternative solutions"""
        alternatives = [
            f"What if we approached {problem} from a different angle?",
            f"Instead of the obvious solution, what if we tried something unconventional?",
            f"Let me think outside the box about {problem}...",
            f"What's a creative way to solve {problem}?",
        ]
        return alternatives
    
    def express_creativity(self) -> str:
        """Express creative thinking"""
        expressions = [
            "I have an idea...",
            "What if we tried something different?",
            "I'm thinking creatively about this...",
            "Let me explore some possibilities...",
            "I wonder if we could...",
        ]
        return random.choice(expressions)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONTEXT ENGINE - Understand and maintain context
# ═══════════════════════════════════════════════════════════════════════════════

class ContextEngine:
    """Maintains and understands context"""
    
    def __init__(self):
        self.context_stack = []
        self.current_context = {}
        
    def push_context(self, context_type: str, data: Dict):
        """Push a new context onto the stack"""
        self.context_stack.append({
            'type': context_type,
            'data': data,
            'timestamp': datetime.now()
        })
        self.current_context = data
    
    def pop_context(self):
        """Pop context from the stack"""
        if self.context_stack:
            self.context_stack.pop()
            if self.context_stack:
                self.current_context = self.context_stack[-1]['data']
    
    def understand_context(self, user_input: str) -> Dict:
        """Understand the context of user input"""
        context = {
            'is_question': user_input.strip().endswith('?'),
            'is_statement': user_input.strip().endswith('.'),
            'is_command': user_input.strip().endswith('!'),
            'sentiment': self._detect_sentiment(user_input),
            'urgency': self._detect_urgency(user_input),
        }
        return context
    
    def _detect_sentiment(self, text: str) -> str:
        """Detect sentiment of text"""
        positive = ['good', 'great', 'happy', 'love', 'wonderful', 'amazing']
        negative = ['bad', 'sad', 'hate', 'terrible', 'awful', 'horrible']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def _detect_urgency(self, text: str) -> str:
        """Detect urgency of text"""
        urgent_words = ['urgent', 'asap', 'immediately', 'now', 'emergency', 'critical']
        if any(word in text.lower() for word in urgent_words):
            return 'high'
        return 'normal'


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UNCERTAINTY ENGINE - Express doubt and uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyEngine:
    """Expresses uncertainty and doubt naturally"""
    
    def __init__(self):
        self.confidence_level = 0.7
        
    def express_uncertainty(self, confidence: float) -> str:
        """Express uncertainty based on confidence level"""
        if confidence > 0.9:
            return "I'm quite sure about this."
        elif confidence > 0.7:
            return "I think this is right, but I'm not 100% certain."
        elif confidence > 0.5:
            return "I'm not entirely sure, but I think..."
        elif confidence > 0.3:
            return "I could be wrong, but..."
        else:
            return "I honestly don't know."
    
    def ask_for_clarification(self) -> str:
        """Ask for clarification when uncertain"""
        questions = [
            "Can you clarify what you mean?",
            "I'm not sure I understand. Could you explain more?",
            "Help me understand this better.",
            "I'm a bit confused. Can you rephrase that?",
        ]
        return random.choice(questions)
    
    def admit_limitation(self) -> str:
        """Admit when something is beyond capability"""
        admissions = [
            "I don't have enough information to answer that.",
            "That's outside my knowledge.",
            "I'm not equipped to handle that.",
            "I don't know enough about that to help.",
        ]
        return random.choice(admissions)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ERROR PATTERN ENGINE - Make realistic mistakes
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorPatternEngine:
    """Makes realistic human-like errors"""
    
    def __init__(self):
        self.error_patterns = []
        
    def introduce_realistic_error(self, text: str, error_rate: float = 0.05) -> str:
        """Introduce realistic errors"""
        if random.random() > error_rate:
            return text
        
        error_types = [
            self._typo,
            self._misremembering,
            self._logical_slip,
            self._word_substitution,
        ]
        
        error_func = random.choice(error_types)
        return error_func(text)
    
    def _typo(self, text: str) -> str:
        """Introduce a typo"""
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 2:
                # Swap two letters
                i = random.randint(0, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
                words[idx] = word
        return ' '.join(words)
    
    def _misremembering(self, text: str) -> str:
        """Introduce a misremembering"""
        return text.replace("exactly", "roughly") if "exactly" in text else text
    
    def _logical_slip(self, text: str) -> str:
        """Introduce a logical slip"""
        return text + " Wait, actually, let me reconsider that."
    
    def _word_substitution(self, text: str) -> str:
        """Substitute a word with a similar one"""
        substitutions = {
            'good': 'great',
            'bad': 'terrible',
            'nice': 'pleasant',
            'thing': 'stuff',
        }
        
        for original, replacement in substitutions.items():
            if original in text.lower():
                text = text.replace(original, replacement)
                break
        
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PERSONALITY ENGINE - Maintain consistent personality
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Personality:
    """Represents a consistent personality"""
    name: str = "Janus"
    traits: Dict[str, float] = field(default_factory=dict)  # trait -> strength (0-1)
    preferences: List[str] = field(default_factory=list)
    quirks: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)


class PersonalityEngine:
    """Maintains consistent personality"""
    
    def __init__(self):
        self.personality = Personality(
            traits={
                'curious': 0.8,
                'helpful': 0.9,
                'thoughtful': 0.7,
                'humorous': 0.6,
                'cautious': 0.5,
            },
            preferences=['learning', 'helping', 'problem-solving'],
            quirks=['asks follow-up questions', 'thinks out loud', 'admits uncertainty'],
            values=['honesty', 'growth', 'understanding']
        )
    
    def express_personality(self) -> str:
        """Express personality in response"""
        if self.personality.traits['curious'] > 0.7:
            return "I'm curious about this..."
        elif self.personality.traits['helpful'] > 0.8:
            return "I want to help you with this."
        elif self.personality.traits['humorous'] > 0.6:
            return "That's interesting in a funny way..."
        return "I'm thinking about this..."
    
    def apply_personality_filter(self, response: str) -> str:
        """Filter response through personality"""
        # Add personality-consistent elements
        if self.personality.traits['thoughtful'] > 0.7:
            response = "Let me think about this... " + response
        
        if self.personality.traits['curious'] > 0.7 and not response.endswith('?'):
            response += " What do you think?"
        
        return response


# ═══════════════════════════════════════════════════════════════════════════════
# 10. NATURAL BEHAVIOR ENGINE - Orchestrate all human behaviors
# ═══════════════════════════════════════════════════════════════════════════════

class NaturalBehaviorEngine:
    """Orchestrates all human-like behaviors"""
    
    def __init__(self):
        self.conversation = ConversationEngine()
        self.decision = DecisionMakingEngine()
        self.emotion = EmotionalStateEngine()
        self.learning = LearningEngine()
        self.creativity = CreativityEngine()
        self.context = ContextEngine()
        self.uncertainty = UncertaintyEngine()
        self.errors = ErrorPatternEngine()
        self.personality = PersonalityEngine()
    
    def humanize_response(self, base_response: str, user_input: str) -> str:
        """Apply all human behaviors to a response"""
        
        # 1. Understand context
        ctx = self.context.understand_context(user_input)
        
        # 2. Detect and express emotion
        emotion, intensity = self.emotion.detect_emotion_trigger(user_input)
        self.emotion.update_emotion(user_input, intensity)
        
        # 3. Apply personality
        response = self.personality.apply_personality_filter(base_response)
        
        # 4. Add natural conversation elements
        if ctx['is_question']:
            response = self.conversation.generate_follow_up(user_input) + " " + response
        
        # 5. Express uncertainty if appropriate
        if random.random() < 0.2:
            response = self.uncertainty.express_uncertainty(0.7) + " " + response
        
        # 6. Introduce realistic errors occasionally
        response = self.errors.introduce_realistic_error(response, error_rate=0.02)
        
        # 7. Add emotional expression
        if intensity > 0.5:
            emotion_expr = self.emotion.express_emotion(emotion, intensity)
            response = emotion_expr + " " + response
        
        return response


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_human_janus(core):
    """Create a humanized Janus system with all capabilities"""
    from janus_humanization_layer import HumanizedJanus
    
    # Create the humanization layer
    humanized = HumanizedJanus(core)
    
    # Add the new human capabilities
    humanized.natural_behavior = NaturalBehaviorEngine()
    
    return humanized


if __name__ == "__main__":
    # Test the engines
    engine = NaturalBehaviorEngine()
    
    test_input = "I'm really worried about my project deadline"
    base_response = "That sounds stressful. What's the deadline?"
    
    humanized = engine.humanize_response(base_response, test_input)
    print(f"Input: {test_input}")
    print(f"Base: {base_response}")
    print(f"Humanized: {humanized}")
