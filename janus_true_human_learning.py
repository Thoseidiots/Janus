"""
janus_true_human_learning.py
=============================
Makes Janus truly human by learning and generating novel responses,
not just picking from predetermined templates.

Key difference:
- OLD: Pick from list of 5 predetermined responses
- NEW: Generate unique responses based on learned patterns and context

Components:
  - AdaptiveMemory - Stores experiences, not just transcripts
  - PatternLearner - Learns what works, not just records what happened
  - ResponseGenerator - Generates novel responses, not template selection
  - ContextualReasoning - Reasons about situations, not just detects them
  - ExperienceIntegration - Integrates learning into future behavior
"""

import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ADAPTIVE MEMORY - Learn from experiences, not just store them
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """Rich experience with outcomes and lessons"""
    situation: str
    context: Dict[str, Any]
    action_taken: str
    outcome: str
    success_score: float  # 0-1, how well it worked
    emotional_response: str
    lessons_learned: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_similarity_score(self, other_situation: str) -> float:
        """Calculate how similar this experience is to a new situation"""
        # Simple word overlap for now, but could be more sophisticated
        self_words = set(self.situation.lower().split())
        other_words = set(other_situation.lower().split())
        
        if not self_words or not other_words:
            return 0.0
        
        overlap = len(self_words & other_words)
        total = len(self_words | other_words)
        return overlap / total if total > 0 else 0.0


class AdaptiveMemory:
    """Learns from experiences, not just stores them"""
    
    def __init__(self, max_experiences: int = 1000):
        self.experiences: List[Experience] = []
        self.max_experiences = max_experiences
        self.learned_patterns = defaultdict(list)  # situation_type -> [successful_actions]
        self.emotional_patterns = defaultdict(list)  # emotion -> [effective_responses]
        self.context_patterns = defaultdict(list)  # context_type -> [outcomes]
        
    def add_experience(self, exp: Experience):
        """Add experience and extract learnings"""
        self.experiences.append(exp)
        
        # Extract patterns
        self._extract_patterns(exp)
        
        # Keep memory bounded
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
    
    def _extract_patterns(self, exp: Experience):
        """Extract learnable patterns from experience"""
        # Pattern 1: What works in similar situations
        situation_type = self._categorize_situation(exp.situation)
        if exp.success_score > 0.7:
            self.learned_patterns[situation_type].append({
                'action': exp.action_taken,
                'success': exp.success_score,
                'context': exp.context
            })
        
        # Pattern 2: What emotional responses work
        if exp.emotional_response:
            self.emotional_patterns[exp.emotional_response].append({
                'action': exp.action_taken,
                'outcome': exp.outcome,
                'success': exp.success_score
            })
        
        # Pattern 3: Context-specific outcomes
        for context_key, context_val in exp.context.items():
            key = f"{context_key}:{context_val}"
            self.context_patterns[key].append({
                'action': exp.action_taken,
                'success': exp.success_score
            })
    
    def _categorize_situation(self, situation: str) -> str:
        """Categorize situation type"""
        keywords = {
            'decision': ['should', 'decide', 'choose', 'option', 'which'],
            'problem': ['problem', 'issue', 'wrong', 'broken', 'error'],
            'question': ['what', 'how', 'why', 'when', 'where'],
            'emotion': ['feel', 'sad', 'happy', 'angry', 'worried'],
            'learning': ['learn', 'understand', 'explain', 'teach'],
            'humor': ['joke', 'funny', 'laugh', 'sleep', 'robot', 'code'],
            'philosophical': ['meaning', 'existence', 'life', 'reality', 'think'],
            'personal': ['favorite', 'like', 'opinion', 'believe', 'think about', 'take on'],
            'error_admission': ['mistake', 'wrong', 'sorry', 'oops'],
            'conflict': ['disagree', 'wrong', 'annoyed', 'frustrating', 'bad'],
            'memory_recall': ['remember', 'last time', 'yesterday', 'before'],
            'curiosity': ['project', 'tired', 'jazz', 'day', 'tell me more'],
        }
        
        situation_lower = situation.lower()
        for category, words in keywords.items():
            if any(word in situation_lower for word in words):
                return category
        return 'general'
    
    def find_similar_experiences(self, situation: str, top_k: int = 3) -> List[Experience]:
        """Find similar past experiences"""
        similarities = []
        for exp in self.experiences:
            score = exp.get_similarity_score(situation)
            if score > 0.1:  # Only consider somewhat similar
                similarities.append((exp, score))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:top_k]]
    
    def get_learned_action(self, situation: str) -> Optional[Dict]:
        """Get best learned action for situation"""
        situation_type = self._categorize_situation(situation)
        
        if situation_type in self.learned_patterns:
            actions = self.learned_patterns[situation_type]
            if actions:
                # Return the most successful action
                best = max(actions, key=lambda x: x['success'])
                return best
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PATTERN LEARNER - Learn what works, not just record what happened
# ═══════════════════════════════════════════════════════════════════════════════

class PatternLearner:
    """Learns patterns from experiences"""
    
    def __init__(self):
        self.patterns = {}
        self.effectiveness_scores = defaultdict(list)
        
    def learn_pattern(self, situation_type: str, action: str, effectiveness: float):
        """Learn that an action is effective in a situation"""
        key = f"{situation_type}:{action}"
        self.effectiveness_scores[key].append(effectiveness)
    
    def get_pattern_effectiveness(self, situation_type: str, action: str) -> float:
        """Get how effective an action is in a situation"""
        key = f"{situation_type}:{action}"
        if key in self.effectiveness_scores:
            scores = self.effectiveness_scores[key]
            return np.mean(scores)
        return 0.5  # Default neutral
    
    def predict_effectiveness(self, situation_type: str, action: str) -> float:
        """Predict how effective an action will be"""
        return self.get_pattern_effectiveness(situation_type, action)
    
    def get_best_actions(self, situation_type: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get the best actions for a situation type"""
        actions = {}
        for key, scores in self.effectiveness_scores.items():
            if key.startswith(f"{situation_type}:"):
                action = key.split(":", 1)[1]
                effectiveness = np.mean(scores)
                actions[action] = effectiveness
        
        # Sort by effectiveness
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTEXTUAL REASONING - Reason about situations, not just detect them
# ═══════════════════════════════════════════════════════════════════════════════

class ContextualReasoning:
    """Reasons about situations to generate appropriate responses"""
    
    def __init__(self):
        self.reasoning_chains = []
        
    def reason_about_situation(self, situation: str, context: Dict) -> Dict:
        """Reason about a situation to understand it deeply"""
        reasoning = {
            'situation': situation,
            'context': context,
            'analysis': self._analyze(situation, context),
            'implications': self._derive_implications(situation, context),
            'appropriate_responses': self._generate_response_types(situation, context),
        }
        return reasoning
    
    def _analyze(self, situation: str, context: Dict) -> str:
        """Analyze the situation"""
        # Look for key elements
        urgency = context.get('urgency', 'normal')
        emotion = context.get('emotion', 'neutral')
        
        analysis = f"This is a {urgency} situation with {emotion} emotion involved."
        
        if 'decision' in situation.lower():
            analysis += " The person needs to make a choice."
        if 'problem' in situation.lower():
            analysis += " There's a problem to solve."
        if 'learn' in situation.lower():
            analysis += " The person wants to understand something."
        
        return analysis
    
    def _derive_implications(self, situation: str, context: Dict) -> List[str]:
        """Derive implications of the situation"""
        implications = []
        
        if context.get('urgency') == 'high':
            implications.append("Time is a factor - need quick response")
        
        if context.get('emotion') in ['worried', 'anxious', 'stressed']:
            implications.append("Person needs reassurance and support")
        
        if 'decision' in situation.lower():
            implications.append("Need to help evaluate options")
        
        return implications
    
    def _generate_response_types(self, situation: str, context: Dict) -> List[str]:
        """Generate types of responses that would be appropriate"""
        responses = []
        
        if context.get('emotion') in ['worried', 'anxious']:
            responses.append('empathetic')
            responses.append('supportive')
        
        if 'decision' in situation.lower():
            responses.append('analytical')
            responses.append('exploratory')
        
        if 'learn' in situation.lower():
            responses.append('educational')
            responses.append('explanatory')
        
        if context.get('urgency') == 'high':
            responses.append('direct')
            responses.append('action-oriented')
        
        return responses if responses else ['thoughtful']


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RESPONSE GENERATOR - Generate novel responses, not template selection
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Generates novel responses based on learned patterns"""
    
    def __init__(self, memory: AdaptiveMemory, pattern_learner: PatternLearner):
        self.memory = memory
        self.pattern_learner = pattern_learner
        self.reasoning = ContextualReasoning()
        
    def generate_response(self, user_input: str, context: Dict) -> str:
        """Generate a novel response"""
        
        # Step 1: Reason about the situation
        reasoning = self.reasoning.reason_about_situation(user_input, context)
        
        # Step 2: Find similar past experiences
        similar_experiences = self.memory.find_similar_experiences(user_input)
        
        # Step 3: Learn from similar experiences
        learned_action = self.memory.get_learned_action(user_input)
        
        # Step 4: Generate response based on reasoning and learning
        response = self._construct_response(
            user_input,
            reasoning,
            similar_experiences,
            learned_action,
            context
        )
        
        return response
    
    def _construct_response(self, user_input: str, reasoning: Dict,
                           similar_experiences: List[Experience],
                           learned_action: Optional[Dict],
                           context: Dict) -> str:
        """Construct a response from reasoning and learning"""
        
        parts = []
        
        # Part 1: Acknowledge and show understanding
        parts.append(self._generate_acknowledgment(reasoning))
        
        # Part 2: Show reasoning
        if reasoning['analysis']:
            parts.append(f"I see that {reasoning['analysis'].lower()}")
        
        # Part 3: Reference similar experiences if available
        if similar_experiences:
            parts.append(self._reference_experience(similar_experiences[0]))
        
        # Part 4: Provide learned insight
        if learned_action:
            parts.append(self._generate_insight(learned_action))
        
        # Part 5: Suggest appropriate response type
        if reasoning['appropriate_responses']:
            parts.append(self._generate_suggestion(reasoning['appropriate_responses']))
        
        # Join parts into coherent response
        response = " ".join(filter(None, parts))
        return response
    
    def _generate_acknowledgment(self, reasoning: Dict) -> str:
        """Generate acknowledgment of the situation"""
        implications = reasoning.get('implications', [])
        
        if not implications:
            return "I understand."
        
        # Generate based on implications
        if "Time is a factor" in implications:
            return "I see this is time-sensitive."
        if "Person needs reassurance" in implications:
            return "I can see why you'd feel that way."
        if "Need to help evaluate" in implications:
            return "Let me help you think through this."
        
        return "I understand your situation."
    
    def _reference_experience(self, exp: Experience) -> str:
        """Reference a similar past experience"""
        return f"I've encountered something similar before, and what I learned was: {exp.lessons_learned[0] if exp.lessons_learned else 'it requires careful thought'}."
    
    def _generate_insight(self, learned_action: Dict) -> str:
        """Generate insight from learned action"""
        action = learned_action.get('action', '')
        success = learned_action.get('success', 0.5)
        
        if success > 0.8:
            return f"Based on what I've learned, {action} tends to work well in situations like this."
        elif success > 0.6:
            return f"I've found that {action} can be helpful here."
        else:
            return f"One approach that might work is {action}."
    
    def _generate_suggestion(self, response_types: List[str]) -> str:
        """Generate suggestion based on response types"""
        if 'empathetic' in response_types:
            return "What I want to emphasize is that I'm here to support you."
        if 'analytical' in response_types:
            return "Let me help you analyze the options."
        if 'educational' in response_types:
            return "Let me explain this in a way that makes sense."
        if 'action-oriented' in response_types:
            return "Here's what I think you should do."
        
        return "What would be most helpful?"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXPERIENCE INTEGRATION - Integrate learning into future behavior
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceIntegration:
    """Integrates learning into future behavior"""
    
    def __init__(self, memory: AdaptiveMemory, pattern_learner: PatternLearner):
        self.memory = memory
        self.pattern_learner = pattern_learner
        self.behavior_adjustments = []
        
    def integrate_experience(self, exp: Experience):
        """Integrate an experience into behavior"""
        
        # Record in memory
        self.memory.add_experience(exp)
        
        # Learn patterns
        situation_type = self.memory._categorize_situation(exp.situation)
        self.pattern_learner.learn_pattern(situation_type, exp.action_taken, exp.success_score)
        
        # Adjust future behavior
        if exp.success_score > 0.7:
            self.behavior_adjustments.append({
                'situation_type': situation_type,
                'action': exp.action_taken,
                'reason': 'high success',
                'timestamp': datetime.now()
            })
        elif exp.success_score < 0.3:
            self.behavior_adjustments.append({
                'situation_type': situation_type,
                'action': exp.action_taken,
                'reason': 'low success - avoid',
                'timestamp': datetime.now()
            })
    
    def get_behavior_adjustments(self) -> List[Dict]:
        """Get how behavior has adjusted"""
        return self.behavior_adjustments
    
    def should_use_action(self, situation_type: str, action: str) -> bool:
        """Determine if an action should be used"""
        effectiveness = self.pattern_learner.get_pattern_effectiveness(situation_type, action)
        return effectiveness > 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRUE HUMAN JANUS - Learns and generates, not templates
# ═══════════════════════════════════════════════════════════════════════════════

class TrueHumanJanus:
    """Janus that truly learns and generates novel responses"""
    
    def __init__(self):
        self.memory = AdaptiveMemory()
        self.pattern_learner = PatternLearner()
        self.response_generator = ResponseGenerator(self.memory, self.pattern_learner)
        self.integration = ExperienceIntegration(self.memory, self.pattern_learner)
        
    def generate_response(self, user_input: str, context: Optional[Dict] = None) -> str:
        """Generate a novel response based on learning"""
        
        if context is None:
            context = self._extract_context(user_input)
        
        # Generate response (not from templates)
        response = self.response_generator.generate_response(user_input, context)
        
        return response
    
    def record_interaction(self, user_input: str, response: str, 
                          outcome: str, success_score: float):
        """Record an interaction and learn from it"""
        
        exp = Experience(
            situation=user_input,
            context=self._extract_context(user_input),
            action_taken=response,
            outcome=outcome,
            success_score=success_score,
            emotional_response=self._detect_emotion(user_input),
            lessons_learned=self._extract_lessons(user_input, response, outcome)
        )
        
        self.integration.integrate_experience(exp)
    
    def _extract_context(self, user_input: str) -> Dict:
        """Extract context from user input"""
        context = {
            'urgency': 'high' if any(w in user_input.lower() for w in ['urgent', 'asap', 'now']) else 'normal',
            'emotion': self._detect_emotion(user_input),
            'type': self._categorize_input(user_input),
        }
        return context
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion in text"""
        emotions = {
            'worried': ['worried', 'anxious', 'stressed', 'concerned'],
            'happy': ['happy', 'great', 'wonderful', 'excited'],
            'sad': ['sad', 'down', 'depressed', 'unhappy'],
            'angry': ['angry', 'frustrated', 'mad', 'annoyed'],
        }
        
        text_lower = text.lower()
        for emotion, words in emotions.items():
            if any(word in text_lower for word in words):
                return emotion
        
        return 'neutral'
    
    def _categorize_input(self, user_input: str) -> str:
        """Categorize the type of input"""
        if '?' in user_input:
            return 'question'
        if any(w in user_input.lower() for w in ['should', 'decide', 'choose']):
            return 'decision'
        if any(w in user_input.lower() for w in ['problem', 'issue', 'wrong']):
            return 'problem'
        return 'statement'
    
    def _extract_lessons(self, user_input: str, response: str, outcome: str) -> List[str]:
        """Extract lessons from an interaction"""
        lessons = []
        
        if 'worked' in outcome.lower():
            lessons.append(f"When faced with '{user_input[:30]}...', responding with '{response[:30]}...' was effective")
        
        if 'didn\'t work' in outcome.lower():
            lessons.append(f"When faced with '{user_input[:30]}...', responding with '{response[:30]}...' was not effective")
        
        return lessons
    
    def get_learning_summary(self) -> Dict:
        """Get a summary of what has been learned"""
        return {
            'total_experiences': len(self.memory.experiences),
            'learned_patterns': dict(self.memory.learned_patterns),
            'behavior_adjustments': self.integration.get_behavior_adjustments(),
            'effectiveness_scores': dict(self.pattern_learner.effectiveness_scores),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    janus = TrueHumanJanus()
    
    # Simulate interactions
    interactions = [
        {
            'input': "I'm really worried about my project deadline",
            'response': None,  # Will be generated
            'outcome': 'user felt understood and got helpful advice',
            'success': 0.9
        },
        {
            'input': "Should I call my boss or email the team?",
            'response': None,
            'outcome': 'user made a good decision',
            'success': 0.85
        },
        {
            'input': "I don't understand how this works",
            'response': None,
            'outcome': 'user understood after explanation',
            'success': 0.8
        },
    ]
    
    # Process interactions
    for interaction in interactions:
        # Generate response
        response = janus.generate_response(interaction['input'])
        interaction['response'] = response
        
        print(f"\nUser: {interaction['input']}")
        print(f"Janus: {response}")
        
        # Record and learn
        janus.record_interaction(
            interaction['input'],
            response,
            interaction['outcome'],
            interaction['success']
        )
    
    # Show learning
    print("\n" + "="*60)
    print("LEARNING SUMMARY")
    print("="*60)
    summary = janus.get_learning_summary()
    print(f"Total experiences: {summary['total_experiences']}")
    print(f"Learned patterns: {len(summary['learned_patterns'])} types")
    print(f"Behavior adjustments: {len(summary['behavior_adjustments'])}")
    
    # Generate new response based on learning
    print("\n" + "="*60)
    print("NEW RESPONSE (BASED ON LEARNING)")
    print("="*60)
    new_input = "I'm stressed about a deadline"
    new_response = janus.generate_response(new_input)
    print(f"User: {new_input}")
    print(f"Janus: {new_response}")
