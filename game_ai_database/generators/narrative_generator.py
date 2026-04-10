"""
Narrative Generator
Generates complete narrative structures and story elements
"""
import random
from typing import List, Dict, Any, Optional
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralDescriptionGenerator,
    ProceduralGraphGenerator,
    ProceduralNumberGenerator
)
from schemas.data_schemas import NarrativeTheme

class NarrativeGenerator:
    """Generates complete narrative structures and story elements"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.graph_gen = ProceduralGraphGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)

        # Original thematic building blocks
        self.core_conflicts = [
            "the struggle between preservation and progress",
            "the cost of power and its corrupting influence",
            "identity fragmentation in a world of masks",
            "the weight of inherited purpose versus chosen destiny",
            "survival of the collective versus individual freedom",
            "the boundaries between creation and destruction",
            "memory as both prison and salvation",
            "the search for meaning in apparent chaos",
            "reconciliation of opposing fundamental forces",
            "transcendence through understanding limitation"
        ]

        self.philosophical_questions = [
            "What defines the boundary between self and other?",
            "Can true change occur without destruction?",
            "Is purpose found or created?",
            "Where does responsibility end and fate begin?",
            "What is preserved when everything transforms?",
            "Can understanding exist without experience?",
            "Is unity achieved through similarity or acceptance of difference?",
            "What remains when all illusions fall away?",
            "Does power corrupt or reveal?",
            "Can the past be redeemed or only released?"
        ]

        self.symbolic_element_pools = {
            'natural': ['the dying tree', 'the endless ocean', 'the frozen peak',
                       'the burning sun', 'the hidden spring', 'the storm that cleanses'],
            'constructed': ['the broken bridge', 'the sealed door', 'the empty throne',
                           'the shattered mirror', 'the rusted key', 'the unfinished monument'],
            'abstract': ['the echo that responds', 'the shadow without source',
                        'the flame that does not burn', 'the silence between words',
                        'the weight of unspoken truth', 'the pattern recognized']
        }

        self.narrative_tones = [
            'melancholic hope', 'grim determination', 'tragic beauty',
            'dark wonder', 'bittersweet resolution', 'quiet defiance',
            'profound uncertainty', 'cautious optimism', 'noble sacrifice',
            'transformative struggle'
        ]

        self.resolution_types = [
            'pyrrhic_victory', 'bittersweet_success', 'unexpected_synthesis',
            'personal_transformation', 'cyclical_continuation', 'sacrificial_resolution',
            'understanding_without_solution', 'partial_reconciliation',
            'transcendent_acceptance', 'defiant_hope'
        ]

        # Plot structure templates
        self.beat_types = [
            'inciting_incident', 'first_threshold', 'rising_action',
            'point_of_no_return', 'dark_night', 'revelation',
            'climactic_confrontation', 'resolution', 'denouement'
        ]

        self.emotional_shifts = [
            'wonder_to_unease', 'confidence_to_doubt', 'isolation_to_connection',
            'ignorance_to_awareness', 'hope_to_despair', 'despair_to_determination',
            'certainty_to_questioning', 'fear_to_acceptance', 'anger_to_understanding'
        ]

    def generate_theme(self) -> Dict[str, Any]:
        """Generate a complete narrative theme"""

        theme_id = str(uuid.uuid4())

        # Select and combine elements
        core_conflict = self.rng.choice(self.core_conflicts)
        philosophical_question = self.rng.choice(self.philosophical_questions)

        # Build symbolic vocabulary
        num_symbol_categories = self.rng.randint(2, 4)
        symbol_categories = self.rng.sample(list(self.symbolic_element_pools.keys()),
                                            num_symbol_categories)

        symbolic_elements = []
        for category in symbol_categories:
            symbols = self.rng.sample(self.symbolic_element_pools[category],
                                      self.rng.randint(1, 3))
            symbolic_elements.extend(symbols)

        tone = self.rng.choice(self.narrative_tones)

        # Select compatible resolution types
        resolution_types = self.rng.sample(self.resolution_types, self.rng.randint(2, 4))

        # Generate thematic name
        theme_name = self._generate_theme_name(core_conflict, tone)

        return {
            'id': theme_id,
            'name': theme_name,
            'core_conflict': core_conflict,
            'philosophical_question': philosophical_question,
            'symbolic_elements': symbolic_elements,
            'tone': tone,
            'resolution_types': resolution_types,
            'thematic_keywords': self._extract_keywords(core_conflict, philosophical_question)
        }

    def _generate_theme_name(self, conflict: str, tone: str) -> str:
        """Generate evocative theme name"""

        name_patterns = [
            "The {noun} of {abstract}",
            "{adjective} {noun}",
            "Beyond the {noun}",
            "The {abstract} Within",
            "{noun} and {noun2}",
            "Where {noun} {verb}"
        ]

        nouns = ['Veil', 'Edge', 'Threshold', 'Echo', 'Shadow', 'Dawn',
                'Remnant', 'Burden', 'Path', 'Flame', 'Silence', 'Storm']

        abstracts = ['Memory', 'Truth', 'Change', 'Loss', 'Hope', 'Darkness',
                    'Purpose', 'Freedom', 'Sacrifice', 'Understanding']

        adjectives = ['Shattered', 'Eternal', 'Hidden', 'Final', 'Forgotten',
                     'Burning', 'Silent', 'Broken', 'Rising', 'Fading']

        verbs = ['Falls', 'Rises', 'Breaks', 'Fades', 'Burns', 'Waits']

        pattern = self.rng.choice(name_patterns)

        name = pattern.format(
            noun=self.rng.choice(nouns),
            noun2=self.rng.choice(nouns),
            abstract=self.rng.choice(abstracts),
            adjective=self.rng.choice(adjectives),
            verb=self.rng.choice(verbs)
        )

        return name

    def _extract_keywords(self, conflict: str, question: str) -> List[str]:
        """Extract thematic keywords from narrative elements"""

        # Simple keyword extraction without NLP libraries
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by',
                     'from', 'or', 'and', 'but', 'if', 'when', 'where', 'what',
                     'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'between', 'versus', 'through', 'without', 'all', 'its'}

        text = f"{conflict} {question}".lower()

        # Remove punctuation
        for char in '.,?!;:\'"()[]{}':
            text = text.replace(char, ' ')

        words = text.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        return list(set(keywords))[:10]

    def generate_plot_structure(self,
                                theme: Dict[str, Any],
                                complexity: int = 5) -> Dict[str, Any]:
        """Generate a complete plot structure based on theme"""

        plot_id = str(uuid.uuid4())

        # Determine number of acts and beats
        num_acts = 3 if complexity < 7 else (5 if complexity < 9 else 7)

        acts = []
        tension_baseline = 0.2

        for act_num in range(num_acts):
            act_beats = []

            # Determine beats per act
            if act_num == 0:
                beat_count = 2 + complexity // 3
                tension_range = (0.1, 0.4)
            elif act_num == num_acts - 1:
                beat_count = 2 + complexity // 2
                tension_range = (0.6, 1.0)
            else:
                beat_count = 3 + complexity // 2
                tension_range = (0.3, 0.8)

            for beat_num in range(beat_count):
                beat = self._generate_plot_beat(
                    act_num, beat_num, beat_count,
                    theme, tension_range
                )
                act_beats.append(beat)

            acts.append({
                'act_number': act_num + 1,
                'act_name': self._generate_act_name(act_num, num_acts),
                'beats': act_beats,
                'thematic_focus': self.rng.choice(theme['symbolic_elements'])
            })

        # Generate branching possibilities
        branch_graph = self.graph_gen.generate_branching_structure(
            depth=complexity,
            min_branches=1,
            max_branches=2 + complexity // 3,
            convergence_chance=0.2
        )

        return {
            'id': plot_id,
            'theme_id': theme['id'],
            'acts': acts,
            'total_beats': sum(len(act['beats']) for act in acts),
            'branch_structure': branch_graph,
            'narrative_hooks': self._generate_narrative_hooks(theme, complexity),
            'foreshadowing_elements': self._generate_foreshadowing(theme),
            'climax_type': self.rng.choice(['confrontation', 'revelation',
                                            'transformation', 'choice', 'sacrifice'])
        }

    def _generate_plot_beat(self,
                            act_num: int,
                            beat_num: int,
                            total_beats: int,
                            theme: Dict[str, Any],
                            tension_range: tuple) -> Dict[str, Any]:
        """Generate individual plot beat"""

        beat_id = str(uuid.uuid4())

        # Determine beat type based on position
        progression = beat_num / max(1, total_beats - 1)

        if act_num == 0:
            if beat_num == 0:
                beat_type = 'setup'
            elif beat_num == total_beats - 1:
                beat_type = 'first_threshold'
            else:
                beat_type = 'establishing'
        elif act_num == 1 or (act_num > 0 and act_num < 2):
            if progression < 0.5:
                beat_type = 'rising_action'
            else:
                beat_type = 'complication'
        else:
            if beat_num == 0:
                beat_type = 'dark_moment'
            elif beat_num == total_beats - 1:
                beat_type = 'resolution'
            else:
                beat_type = 'climactic'

        # Calculate tension
        tension = tension_range[0] + (tension_range[1] - tension_range[0]) * progression
        tension += self.rng.uniform(-0.1, 0.1)
        tension = max(0.0, min(1.0, tension))

        # Generate beat description
        description = self._generate_beat_description(beat_type, theme)

        return {
            'id': beat_id,
            'beat_type': beat_type,
            'description': description,
            'emotional_shift': self.rng.choice(self.emotional_shifts),
            'tension_level': round(tension, 2),
            'required_elements': self._get_required_elements(beat_type),
            'optional_branches': self.rng.sample(
                ['alternate_path', 'hidden_consequence', 'character_death',
                 'ally_betrayal', 'unexpected_ally', 'revelation'],
                self.rng.randint(0, 2)
            )
        }

    def _generate_beat_description(self, beat_type: str, theme: Dict[str, Any]) -> str:
        """Generate description for a plot beat"""

        beat_templates = {
            'setup': [
                "The world is introduced through the lens of {symbol}.",
                "Establish the status quo, with {symbol} hinting at deeper truths.",
                "The protagonist exists in a state of {tone}, unaware of coming change."
            ],
            'establishing': [
                "Deeper layers of the world emerge, connected to {conflict}.",
                "Characters and their motivations become clearer through {symbol}.",
                "The nature of {conflict} begins to manifest in subtle ways."
            ],
            'first_threshold': [
                "An irreversible action commits the protagonist to face {conflict}.",
                "The {symbol} transforms from symbol to lived reality.",
                "The old world falls away, replaced by the journey toward {question}."
            ],
            'rising_action': [
                "Challenges mount as {conflict} intensifies.",
                "Allies and enemies reveal their true natures against {symbol}.",
                "The protagonist gains power but begins to question {question}."
            ],
            'complication': [
                "What seemed clear becomes murky in light of {symbol}.",
                "The path forward fractures into impossible choices.",
                "The true nature of {conflict} reveals itself."
            ],
            'dark_moment': [
                "All seems lost as {conflict} appears insurmountable.",
                "The protagonist must confront {question} without easy answers.",
                "The {symbol} appears broken beyond repair."
            ],
            'climactic': [
                "The final confrontation with {conflict} begins.",
                "Everything built leads to this moment of {tone}.",
                "The {symbol} reaches its ultimate significance."
            ],
            'resolution': [
                "The aftermath reveals the true cost and meaning of the journey.",
                "The answer to {question} is found, though not as expected.",
                "A new equilibrium emerges from {conflict}."
            ]
        }

        templates = beat_templates.get(beat_type, beat_templates['rising_action'])
        template = self.rng.choice(templates)

        return template.format(
            symbol=self.rng.choice(theme['symbolic_elements']),
            conflict=theme['core_conflict'].split()[3:6],  # Extract key phrase
            question=theme['philosophical_question'][:50],
            tone=theme['tone']
        )

    def _generate_act_name(self, act_num: int, total_acts: int) -> str:
        """Generate evocative act names"""

        if total_acts == 3:
            names = [
                ['The World Before', 'Departure', 'Awakening'],
                ['The Descent', 'Trials', 'Transformation'],
                ['The Return', 'Resolution', 'New Dawn']
            ]
        elif total_acts == 5:
            names = [
                ['Prelude', 'First Steps', 'The Familiar World'],
                ['Crossing Over', 'Into the Unknown', 'First Trials'],
                ['The Deepening', 'Complications', 'Shifting Ground'],
                ['The Crisis', 'Dark Night', 'Transformation'],
                ['Resolution', 'Return', 'New Equilibrium']
            ]
        else:
            names = [[f"Movement {i+1}"] for i in range(total_acts)]

        return self.rng.choice(names[act_num])

    def _get_required_elements(self, beat_type: str) -> List[str]:
        """Get required narrative elements for beat type"""

        requirements = {
            'setup': ['world_establishment', 'protagonist_introduction', 'status_quo'],
            'establishing': ['character_development', 'world_building', 'theme_introduction'],
            'first_threshold': ['call_to_action', 'point_of_no_return', 'stakes_established'],
            'rising_action': ['challenge', 'growth', 'complication'],
            'complication': ['twist', 'moral_complexity', 'relationship_shift'],
            'dark_moment': ['loss', 'doubt', 'lowest_point'],
            'climactic': ['confrontation', 'climax', 'transformation'],
            'resolution': ['consequence', 'meaning', 'new_status']
        }

        return requirements.get(beat_type, ['narrative_progression'])

    def _generate_narrative_hooks(self, theme: Dict[str, Any], complexity: int) -> List[Dict]:
        """Generate narrative hooks and mysteries"""

        hook_types = ['mystery', 'prophecy', 'hidden_truth', 'character_secret',
                     'world_secret', 'ancient_event', 'looming_threat']

        num_hooks = 2 + complexity // 2
        hooks = []

        for _ in range(num_hooks):
            hook = {
                'id': str(uuid.uuid4()),
                'type': self.rng.choice(hook_types),
                'description': f"A {self.rng.choice(hook_types)} connected to {self.rng.choice(theme['symbolic_elements'])}",
                'reveal_timing': self.rng.choice(['early', 'mid', 'late', 'climax', 'epilogue']),
                'impact_level': self.rng.choice(['minor', 'moderate', 'major', 'critical'])
            }
            hooks.append(hook)

        return hooks

    def _generate_foreshadowing(self, theme: Dict[str, Any]) -> List[Dict]:
        """Generate foreshadowing elements"""

        elements = []

        for symbol in theme['symbolic_elements'][:3]:
            element = {
                'symbol': symbol,
                'early_appearance': f"A subtle hint of {symbol} in unexpected context",
                'middle_appearance': f"The {symbol} appears more prominently, its meaning unclear",
                'revelation': f"The true significance of {symbol} becomes clear"
            }
            elements.append(element)

        return elements

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple complete narrative packages"""

        narratives = []

        for i in range(count):
            theme = self.generate_theme()
            complexity = self.rng.randint(3, 10)
            plot = self.generate_plot_structure(theme, complexity)

            narrative_package = {
                'id': str(uuid.uuid4()),
                'theme': theme,
                'plot_structure': plot,
                'complexity_rating': complexity,
                'estimated_playtime_hours': complexity * 2 + self.rng.randint(5, 20),
                'generation_seed': self.seed + i
            }

            narratives.append(narrative_package)

        return narratives