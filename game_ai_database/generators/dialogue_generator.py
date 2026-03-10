"""
Dialogue Generator
Generates conversations, NPC interactions, and spoken content
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from schemas.data_schemas import DialogueNode
from utils.procedural_algorithms import ProceduralNameGenerator, ProceduralDescriptionGenerator


class DialogueGenerator:
    """Enterprise-grade dialogue generation with branching conversations and emotional depth"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize procedural generators
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)

        # Dialogue archetypes and styles
        self.dialogue_styles = {
            'formal': ['courteous', 'diplomatic', 'ceremonial', 'official', 'regal'],
            'casual': ['friendly', 'conversational', 'relaxed', 'informal', 'colloquial'],
            'aggressive': ['confrontational', 'hostile', 'intimidating', 'demanding', 'belligerent'],
            'mysterious': ['enigmatic', 'cryptic', 'allusive', 'oblique', 'inscrutable'],
            'scholarly': ['learned', 'pedantic', 'erudite', 'intellectual', 'academic'],
            'folksy': ['rustic', 'homespun', 'provincial', 'earthy', 'vernacular'],
            'military': ['disciplined', 'commanding', 'tactical', 'strategic', 'authoritative'],
            'merchant': ['bargaining', 'persuasive', 'calculating', 'entrepreneurial', 'commercial']
        }

        self.emotion_types = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
            'trust', 'anticipation', 'love', 'guilt', 'shame', 'pride',
            'excitement', 'anxiety', 'contentment', 'frustration'
        ]

        self.speech_patterns = [
            'direct', 'circumlocutory', 'rhetorical', 'interrogative', 'declarative',
            'imperative', 'exclamatory', 'hesitant', 'confident', 'sarcastic'
        ]

        self.conversation_topics = [
            'personal_history', 'current_events', 'future_plans', 'philosophical_questions',
            'practical_advice', 'warnings', 'offers', 'requests', 'gossip', 'rumors',
            'quests', 'trade', 'politics', 'magic', 'technology', 'nature', 'society'
        ]

    def generate_dialogue_node(self,
                               speaker_id: str,
                               context: str = 'general',
                               emotion: str = None,
                               complexity: int = 3) -> DialogueNode:
        """Generate a complete dialogue node with responses"""

        node = DialogueNode()
        node.id = str(uuid.uuid4())
        node.speaker_id = speaker_id

        # Generate dialogue content
        node.text = self._generate_dialogue_text(context, complexity)

        # Generate emotion and delivery
        if not emotion:
            emotion = self.rng.choice(self.emotion_types)
        node.emotion = emotion
        node.delivery_style = self._generate_delivery_style(emotion)

        # Generate player responses
        node.responses = self._generate_responses(context, complexity)

        # Generate metadata
        node.metadata = self._generate_dialogue_metadata(complexity, context)

        return node

    def generate_conversation_tree(self,
                                  speaker_id: str,
                                  depth: int = 3,
                                  breadth: int = 2,
                                  context: str = 'general') -> Dict[str, Any]:
        """Generate a complete conversation tree"""

        tree = {
            'id': str(uuid.uuid4()),
            'speaker_id': speaker_id,
            'root_node': None,
            'all_nodes': {},
            'conversation_flow': []
        }

        # Generate root node
        root_node = self.generate_dialogue_node(speaker_id, context, complexity=3)
        tree['root_node'] = root_node.id
        tree['all_nodes'][root_node.id] = root_node

        # Generate conversation branches
        self._generate_conversation_branches(
            tree, root_node, depth, breadth, context, []
        )

        return tree

    def _generate_dialogue_text(self, context: str, complexity: int) -> str:
        """Generate dialogue text based on context"""

        dialogue_templates = {
            'general': [
                "I have something important to discuss with you.",
                "The world is changing, and we must adapt.",
                "What brings you to this place?",
                "I've been expecting someone like you.",
                "There are matters we should address."
            ],
            'quest': [
                "I have a task that requires your particular skills.",
                "There's a problem that needs solving.",
                "I seek someone who can help with a delicate matter.",
                "A great opportunity awaits those brave enough to seize it.",
                "The fate of many rests on what happens next."
            ],
            'trade': [
                "I have goods that might interest you.",
                "Let's discuss a mutually beneficial arrangement.",
                "I can offer you something quite valuable.",
                "Your needs and my supplies seem perfectly matched.",
                "Business opportunities abound for the discerning."
            ],
            'combat': [
                "You dare challenge me?",
                "This ends here and now!",
                "Your arrogance will be your undoing.",
                "Fight or flee, the choice is yours.",
                "Blood will be spilled before this is over."
            ],
            'romantic': [
                "I've never met anyone quite like you.",
                "My heart beats faster in your presence.",
                "There are feelings I cannot ignore.",
                "Our paths were meant to cross.",
                "What we have is something special."
            ],
            'mysterious': [
                "There are secrets that must remain hidden.",
                "Some truths are too dangerous to speak.",
                "What you seek lies beyond the veil.",
                "Ancient powers stir in the shadows.",
                "The answers you want come at a price."
            ]
        }

        templates = dialogue_templates.get(context, dialogue_templates['general'])

        # Select base template
        base_text = self.rng.choice(templates)

        # Add complexity-based elaboration
        if complexity > 3:
            elaborations = [
                " Let me explain further.",
                " This requires careful consideration.",
                " The details are quite intricate.",
                " There are layers to this situation.",
                " Multiple factors are at play here."
            ]
            if self.rng.random() > 0.5:
                base_text += self.rng.choice(elaborations)

        # Add emotional context
        if complexity > 5:
            emotional_additions = [
                " I must confess I'm quite concerned.",
                " This fills me with great hope.",
                " My instincts tell me this is right.",
                " I fear what might come to pass.",
                " This brings me great joy to discuss."
            ]
            if self.rng.random() > 0.6:
                base_text += self.rng.choice(emotional_additions)

        return base_text

    def _generate_delivery_style(self, emotion: str) -> Dict[str, Any]:
        """Generate delivery style based on emotion"""

        style_map = {
            'joy': {
                'tone': 'warm',
                'pace': 'animated',
                'volume': 'louder',
                'gestures': 'expressive'
            },
            'anger': {
                'tone': 'sharp',
                'pace': 'rapid',
                'volume': 'loud',
                'gestures': 'aggressive'
            },
            'sadness': {
                'tone': 'soft',
                'pace': 'slow',
                'volume': 'quiet',
                'gestures': 'subdued'
            },
            'fear': {
                'tone': 'trembling',
                'pace': 'erratic',
                'volume': 'variable',
                'gestures': 'nervous'
            },
            'surprise': {
                'tone': 'sharp',
                'pace': 'sudden',
                'volume': 'loud',
                'gestures': 'startled'
            }
        }

        base_style = style_map.get(emotion, {
            'tone': 'neutral',
            'pace': 'steady',
            'volume': 'normal',
            'gestures': 'moderate'
        })

        # Add variations
        variations = {
            'tone': ['confident', 'hesitant', 'sarcastic', 'sincere'],
            'pace': ['deliberate', 'hurried', 'measured', 'halting'],
            'volume': ['whispering', 'booming', 'muttering', 'projecting'],
            'gestures': ['minimal', 'sweeping', 'pointing', 'folding']
        }

        for aspect in base_style:
            if self.rng.random() > 0.7:
                base_style[aspect] = self.rng.choice(variations[aspect])

        return base_style

    def _generate_responses(self, context: str, complexity: int) -> List[Dict[str, Any]]:
        """Generate player response options"""

        responses = []
        num_responses = min(4, 2 + complexity // 2)

        response_templates = {
            'general': [
                "Tell me more about this.",
                "I'm not sure I understand.",
                "What do you want me to do?",
                "That sounds interesting.",
                "I need time to think about this."
            ],
            'quest': [
                "I'm interested. What's involved?",
                "What's the reward for this task?",
                "Why me specifically?",
                "What are the risks involved?",
                "I need more information first."
            ],
            'trade': [
                "What do you have to offer?",
                "How much does this cost?",
                "Can we negotiate the price?",
                "What are the terms of this deal?",
                "I need to see the merchandise first."
            ],
            'combat': [
                "Let's settle this!",
                "I don't want to fight.",
                "Stand down or face the consequences.",
                "This doesn't have to end in violence.",
                "Your move."
            ]
        }

        templates = response_templates.get(context, response_templates['general'])

        for i in range(num_responses):
            response = {
                'id': str(uuid.uuid4()),
                'text': self.rng.choice(templates),
                'effect': self._generate_response_effect(),
                'conditions': self._generate_response_conditions(complexity)
            }

            # Remove used template to avoid duplicates
            if response['text'] in templates:
                templates.remove(response['text'])

            responses.append(response)

        return responses

    def _generate_response_effect(self) -> str:
        """Generate the effect of a response"""

        effects = [
            'continue_conversation',
            'end_conversation',
            'increase_relationship',
            'decrease_relationship',
            'unlock_information',
            'trigger_quest',
            'initiate_combat',
            'offer_deal',
            'request_action',
            'reveal_secret'
        ]

        return self.rng.choice(effects)

    def _generate_response_conditions(self, complexity: int) -> List[str]:
        """Generate conditions for response availability"""

        conditions = []

        if complexity > 2:
            possible_conditions = [
                'high_relationship',
                'low_relationship',
                'has_skill_diplomacy',
                'has_item_special',
                'completed_quest',
                'faction_member',
                'high_reputation',
                'specific_knowledge'
            ]

            num_conditions = min(complexity // 2, len(possible_conditions))
            conditions = self.rng.sample(possible_conditions, num_conditions)

        return conditions

    def _generate_dialogue_metadata(self, complexity: int, context: str) -> Dict[str, Any]:
        """Generate dialogue metadata"""

        return {
            'complexity': complexity,
            'context': context,
            'emotional_intensity': self.rng.randint(1, 10),
            'persuasion_potential': self.rng.randint(1, 10),
            'information_value': self.rng.randint(1, 10),
            'relationship_impact': self.rng.choice([-2, -1, 0, 1, 2]),
            'time_pressure': self.rng.random() > 0.8,
            'moral_choice': self.rng.random() > 0.9,
            'branching_potential': self.rng.randint(1, 5)
        }

    def _generate_conversation_branches(self,
                                       tree: Dict,
                                       current_node: DialogueNode,
                                       depth: int,
                                       breadth: int,
                                       context: str,
                                       visited_topics: List[str]):
        """Recursively generate conversation branches"""

        if depth <= 0:
            return

        # Generate responses for current node
        for response in current_node.responses:
            if len(tree['all_nodes']) >= 50:  # Prevent infinite growth
                break

            # Create follow-up node
            followup_context = self._evolve_context(context, response['effect'])
            followup_node = self.generate_dialogue_node(
                current_node.speaker_id,
                followup_context,
                complexity=max(1, depth)
            )

            # Avoid topic repetition
            if followup_node.metadata['context'] not in visited_topics:
                tree['all_nodes'][followup_node.id] = followup_node
                tree['conversation_flow'].append({
                    'from_node': current_node.id,
                    'response_id': response['id'],
                    'to_node': followup_node.id
                })

                # Recurse with reduced depth
                new_visited = visited_topics + [followup_context]
                self._generate_conversation_branches(
                    tree, followup_node, depth - 1, breadth, followup_context, new_visited
                )

    def _evolve_context(self, current_context: str, response_effect: str) -> str:
        """Evolve conversation context based on response"""

        context_transitions = {
            'continue_conversation': {
                'general': ['personal', 'quest', 'trade'],
                'quest': ['details', 'rewards', 'risks'],
                'trade': ['negotiation', 'inspection', 'terms'],
                'combat': ['intimidation', 'diplomacy', 'retreat']
            },
            'increase_relationship': ['personal', 'trust', 'friendship'],
            'decrease_relationship': ['tension', 'hostility', 'suspicion'],
            'unlock_information': ['revelation', 'secret', 'knowledge'],
            'trigger_quest': ['commitment', 'obligation', 'adventure'],
            'offer_deal': ['negotiation', 'bargaining', 'agreement']
        }

        transitions = context_transitions.get(response_effect, [current_context])
        return self.rng.choice(transitions)

    def generate_batch(self, count: int, speaker_ids: List[str] = None) -> List[DialogueNode]:
        """Generate multiple dialogue nodes"""

        if not speaker_ids:
            speaker_ids = [f"speaker_{i}" for i in range(count)]

        dialogues = []

        for i in range(count):
            speaker_id = speaker_ids[i % len(speaker_ids)] if speaker_ids else f"speaker_{i}"
            context = self.rng.choice(list(self.conversation_topics))
            complexity = self.rng.randint(1, 5)

            dialogue = self.generate_dialogue_node(speaker_id, context, complexity=complexity)
            dialogues.append(dialogue)

        return dialogues

    def generate_character_dialogue_set(self,
                                       character_id: str,
                                       personality_traits: List[str],
                                       num_dialogues: int = 10) -> Dict[str, Any]:
        """Generate a complete set of dialogues for a character"""

        dialogue_set = {
            'character_id': character_id,
            'personality_traits': personality_traits,
            'dialogues': [],
            'conversation_themes': set(),
            'emotional_range': set()
        }

        contexts = self.conversation_topics[:num_dialogues]

        for context in contexts:
            dialogue = self.generate_dialogue_node(character_id, context)
            dialogue_set['dialogues'].append(dialogue)
            dialogue_set['conversation_themes'].add(context)
            dialogue_set['emotional_range'].add(dialogue.emotion)

        return dialogue_set