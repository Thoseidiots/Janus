"""
Mechanics Generator
Generates game mechanics, rules, and interactive systems
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from schemas.data_schemas import GameMechanic
from utils.procedural_algorithms import ProceduralNameGenerator, ProceduralDescriptionGenerator, ProceduralNumberGenerator


class MechanicsGenerator:
    """Enterprise-grade game mechanics generation with complex rule systems"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize procedural generators
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)

        # Mechanic categories and their properties
        self.mechanic_categories = {
            'combat': {
                'subcategories': ['damage', 'defense', 'special_attacks', 'combos', 'status_effects'],
                'parameters': ['damage_multiplier', 'defense_rating', 'attack_speed', 'critical_chance'],
                'complexity_range': (1, 10)
            },
            'progression': {
                'subcategories': ['leveling', 'skill_trees', 'mastery', 'prestige', 'specialization'],
                'parameters': ['experience_rate', 'skill_points', 'mastery_threshold', 'prestige_bonus'],
                'complexity_range': (1, 8)
            },
            'resource': {
                'subcategories': ['currency', 'materials', 'energy', 'health', 'stamina'],
                'parameters': ['regeneration_rate', 'capacity', 'efficiency', 'depletion_rate'],
                'complexity_range': (1, 7)
            },
            'social': {
                'subcategories': ['reputation', 'relationships', 'alliances', 'rivalries', 'influence'],
                'parameters': ['trust_level', 'influence_radius', 'relationship_strength', 'diplomacy_power'],
                'complexity_range': (1, 6)
            },
            'exploration': {
                'subcategories': ['discovery', 'mapping', 'navigation', 'survival', 'archaeology'],
                'parameters': ['exploration_speed', 'discovery_chance', 'mapping_accuracy', 'survival_rate'],
                'complexity_range': (1, 8)
            },
            'crafting': {
                'subcategories': ['gathering', 'refining', 'assembly', 'enchanting', 'upgrading'],
                'parameters': ['crafting_speed', 'quality_multiplier', 'success_rate', 'material_efficiency'],
                'complexity_range': (1, 9)
            },
            'inventory': {
                'subcategories': ['storage', 'organization', 'equipment', 'consumables', 'artifacts'],
                'parameters': ['capacity', 'weight_limit', 'organization_bonus', 'durability'],
                'complexity_range': (1, 5)
            },
            'equipment': {
                'subcategories': ['weapons', 'armor', 'accessories', 'tools', 'mounts'],
                'parameters': ['durability', 'enhancement_level', 'compatibility', 'upgrade_potential'],
                'complexity_range': (1, 8)
            },
            'skills': {
                'subcategories': ['active', 'passive', 'combat', 'utility', 'crafting'],
                'parameters': ['cooldown', 'effect_strength', 'range', 'duration'],
                'complexity_range': (1, 10)
            },
            'achievements': {
                'subcategories': ['combat', 'exploration', 'social', 'crafting', 'collection'],
                'parameters': ['progress_rate', 'reward_scale', 'visibility', 'rarity'],
                'complexity_range': (1, 6)
            }
        }

        self.interaction_types = [
            'amplifies', 'diminishes', 'triggers', 'prevents', 'modifies',
            'combines_with', 'conflicts_with', 'enhances', 'weakens', 'neutralizes'
        ]

        self.balance_factors = [
            'power_level', 'resource_cost', 'time_investment', 'risk_reward_ratio',
            'accessibility', 'replayability', 'skill_requirement', 'luck_factor'
        ]

    def generate_mechanic(self,
                         category: str = None,
                         subcategory: str = None,
                         complexity: int = 5) -> GameMechanic:
        """Generate a complete game mechanic"""

        mechanic = GameMechanic()
        mechanic.id = str(uuid.uuid4())

        # Select category and subcategory
        if not category:
            category = self.rng.choice(list(self.mechanic_categories.keys()))
        mechanic.category = category

        cat_info = self.mechanic_categories[category]
        if not subcategory:
            subcategory = self.rng.choice(cat_info['subcategories'])
        mechanic.subcategory = subcategory

        # Generate basic properties
        mechanic.name = self._generate_mechanic_name(category, subcategory)
        mechanic.description = self._generate_mechanic_description(category, subcategory, complexity)
        mechanic.complexity_rating = complexity

        # Generate rules
        mechanic.rules = self._generate_rules(category, complexity)

        # Generate parameters
        mechanic.parameters = self._generate_parameters(cat_info['parameters'], complexity)

        # Generate interactions
        mechanic.interactions = self._generate_interactions(complexity)

        # Generate progression hooks
        mechanic.progression_hooks = self._generate_progression_hooks(category)

        # Generate balance considerations
        mechanic.balance_considerations = self._generate_balance_considerations()

        # Generate player feedback
        mechanic.player_feedback = self._generate_player_feedback(category)

        # Generate edge cases
        mechanic.edge_cases = self._generate_edge_cases(complexity)

        return mechanic

    def _generate_mechanic_name(self, category: str, subcategory: str) -> str:
        """Generate mechanic name"""

        name_templates = {
            'combat': [
                "{adjective} {subcategory}",
                "{subcategory} Mastery",
                "Advanced {subcategory}",
                "{subcategory} System"
            ],
            'progression': [
                "{subcategory} Progression",
                "Path of {subcategory}",
                "{subcategory} Development",
                "Mastery in {subcategory}"
            ],
            'resource': [
                "{subcategory} Management",
                "{subcategory} System",
                "Resource {subcategory}",
                "{subcategory} Control"
            ],
            'social': [
                "{subcategory} Dynamics",
                "Social {subcategory}",
                "{subcategory} Network",
                "Relationship {subcategory}"
            ],
            'exploration': [
                "{subcategory} Techniques",
                "Art of {subcategory}",
                "{subcategory} Mastery",
                "Exploration {subcategory}"
            ],
            'crafting': [
                "{subcategory} Craft",
                "Master {subcategory}",
                "{subcategory} Expertise",
                "Crafting {subcategory}"
            ],
            'inventory': [
                "{subcategory} Organization",
                "Inventory {subcategory}",
                "{subcategory} Management",
                "Storage {subcategory}"
            ],
            'equipment': [
                "{subcategory} Enhancement",
                "Equipment {subcategory}",
                "{subcategory} System",
                "Gear {subcategory}"
            ],
            'skills': [
                "{subcategory} Abilities",
                "Skill {subcategory}",
                "{subcategory} Powers",
                "Mastery {subcategory}"
            ],
            'achievements': [
                "{subcategory} Recognition",
                "Achievement {subcategory}",
                "{subcategory} Honors",
                "Triumph in {subcategory}"
            ]
        }

        templates = name_templates.get(category, ["{subcategory} System"])
        template = self.rng.choice(templates)

        return template.replace("{subcategory}", subcategory.replace('_', ' ').title())

    def _generate_mechanic_description(self, category: str, subcategory: str, complexity: int) -> str:
        """Generate mechanic description"""

        return self.desc_gen.generate_description(
            theme=f"mechanic_{category}_{subcategory}",
            complexity=complexity,
            length='medium'
        )

    def _generate_rules(self, category: str, complexity: int) -> List[str]:
        """Generate mechanic rules"""

        rule_templates = {
            'combat': [
                "Damage scales with weapon level and character strength",
                "Critical hits occur at {percent}% base chance",
                "Status effects last {duration} rounds",
                "Combo multipliers cap at {multiplier}x damage",
                "Defense reduces incoming damage by {percent}%",
                "Special attacks consume {resource} points",
                "Attack speed determines action priority",
                "Elemental weaknesses increase damage by {percent}%"
            ],
            'progression': [
                "Experience requirements increase by {rate} per level",
                "Skill points awarded every {interval} levels",
                "Mastery unlocks at level {threshold}",
                "Prestige resets progress for {percent}% bonus",
                "Specialization reduces costs by {percent}%",
                "Level caps increase with game progression",
                "Milestone rewards scale with achievement difficulty",
                "Training time reduces with skill investment"
            ],
            'resource': [
                "Resources regenerate at {rate} per minute",
                "Capacity increases with container upgrades",
                "Efficiency bonuses reduce consumption by {percent}%",
                "Depletion occurs during active use only",
                "Resource sharing possible with party members",
                "Storage limits prevent overflow",
                "Conversion rates vary by resource type",
                "Emergency reserves activate below {threshold}%"
            ],
            'social': [
                "Reputation changes by ±{amount} per interaction",
                "Relationship strength affects dialogue options",
                "Alliance bonuses provide {percent}% advantages",
                "Rivalry penalties increase encounter difficulty",
                "Influence radius expands with reputation",
                "Diplomacy success based on charisma and history",
                "Faction quests unlock at reputation {tier}",
                "Betrayal consequences scale with relationship depth"
            ],
            'exploration': [
                "Discovery chance increases with perception skill",
                "Mapping accuracy improves with cartography tools",
                "Navigation speed affected by terrain familiarity",
                "Survival rate depends on preparation level",
                "Archaeology finds scale with excavation skill",
                "Hidden locations revealed through special means",
                "Exploration rewards diminish with area knowledge",
                "Danger level affects encounter frequency"
            ],
            'crafting': [
                "Success rate improves with crafting skill level",
                "Quality determined by material purity and technique",
                "Time requirements scale with item complexity",
                "Material efficiency increases by {rate} per mastery level",
                "Experimentation has {chance}% chance of {outcome}",
                "Time reduction caps at {cap}% with {skill} skill"
            ]
        }

        templates = rule_templates.get(category, rule_templates['combat'])
        num_rules = 3 + complexity // 2

        rules = []
        for template in self.rng.sample(templates, min(num_rules, len(templates))):
            # Fill in template values
            rule = template.format(
                stat=self.rng.choice(['strength', 'agility', 'intelligence', 'willpower']),
                condition=self.rng.choice(['hit succeeds', 'target is vulnerable', 'timing is perfect']),
                multiplier=round(self.rng.uniform(1.5, 3.0), 1),
                formula=self.rng.choice(['flat reduction', 'percentage reduction', 'scaling formula']),
                trigger=self.rng.choice(['attack roll', 'damage threshold', 'critical hit']),
                percent=self.rng.randint(5, 25),
                base=self.rng.randint(100, 1000),
                exponent=round(self.rng.uniform(1.5, 2.5), 2),
                interval=self.rng.randint(2, 5),
                curve=self.rng.choice(['linear', 'exponential', 'logarithmic', 'stepped']),
                threshold=self.rng.randint(75, 100),
                amount=self.rng.randint(5, 20),
                resource=self.rng.choice(['health', 'energy', 'mana', 'stamina']),
                scaling=round(self.rng.uniform(0.5, 5.0), 1),
                cap=self.rng.randint(30, 70),
                ratio=f"{self.rng.randint(1, 5)}:{self.rng.randint(1, 5)}",
                rate=round(self.rng.uniform(0.1, 2.0), 2),
                affected_systems=self.rng.choice(['shops', 'quests', 'dialogue', 'access']),
                tier=self.rng.choice(['friendly', 'trusted', 'allied', 'honored']),
                value=self.rng.randint(500, 2000),
                limiting_factor=self.rng.choice(['character level', 'total reputation', 'quest completion']),
                perception=self.rng.choice(['perception', 'awareness', 'investigation']),
                factors=self.rng.choice(['terrain', 'equipment', 'abilities', 'weather']),
                difficulty=self.rng.choice(['area level', 'region tier', 'exploration depth']),
                skill=self.rng.choice(['crafting', 'smithing', 'alchemy', 'enchanting']),
                tiers='Common/Uncommon/Rare/Epic/Legendary',
                values='0/50/100/200/500',
                chance=self.rng.randint(5, 30),
                outcome=self.rng.choice(['discovery', 'failure', 'mutation', 'enhancement'])
            )
            rules.append(rule)

        return rules

    def _generate_parameters(self, param_types: List[str], complexity: int) -> Dict[str, Any]:
        """Generate mechanic parameters"""

        parameters = {}

        for param in param_types:
            param_config = {
                'base_value': round(self.rng.uniform(1.0, 100.0), 2),
                'min_value': round(self.rng.uniform(0.0, 10.0), 2),
                'max_value': round(self.rng.uniform(100.0, 1000.0), 2),
                'scaling_type': self.rng.choice(['linear', 'exponential', 'logarithmic', 'stepped']),
                'affected_by': self.rng.sample([
                    'character_level', 'skill_level', 'equipment', 'buffs',
                    'environment', 'time', 'resources'
                ], self.rng.randint(1, 3))
            }

            if complexity > 5:
                param_config['breakpoints'] = self.num_gen.generate_curve(
                    param_config['min_value'],
                    param_config['max_value'],
                    complexity,
                    param_config['scaling_type'].replace('stepped', 'linear')
                )

            parameters[param] = param_config

        return parameters

    def _generate_interactions(self, complexity: int) -> List[Dict]:
        """Generate mechanic interactions with other systems"""

        interactions = []
        num_interactions = 2 + complexity // 2

        systems = ['combat', 'progression', 'resource', 'social', 'exploration', 'crafting',
                  'inventory', 'equipment', 'skills', 'achievements', 'world_state']

        for _ in range(num_interactions):
            interaction = {
                'target_system': self.rng.choice(systems),
                'interaction_type': self.rng.choice(self.interaction_types),
                'conditions': self.rng.sample([
                    'always_active', 'when_in_combat', 'when_threshold_met',
                    'when_resource_low', 'when_skill_active', 'when_status_applied'
                ], self.rng.randint(1, 2)),
                'effect_strength': self.rng.choice(['minor', 'moderate', 'major', 'scaling']),
                'bidirectional': self.rng.random() > 0.5
            }
            interactions.append(interaction)

        return interactions

    def _generate_progression_hooks(self, category: str) -> List[str]:
        """Generate hooks for progression systems"""

        hooks = {
            'combat': [
                'damage_dealt_total', 'critical_hits_landed', 'enemies_defeated',
                'combos_completed', 'perfect_blocks', 'status_effects_applied'
            ],
            'progression': [
                'levels_gained', 'skills_unlocked', 'milestones_reached',
                'achievements_completed', 'mastery_points_earned'
            ],
            'resource': [
                'resources_gathered', 'resources_spent', 'efficiency_achieved',
                'conversions_made', 'thresholds_triggered'
            ],
            'social': [
                'reputation_gained', 'relationships_formed', 'factions_allied',
                'influence_exerted', 'negotiations_completed'
            ],
            'exploration': [
                'areas_discovered', 'secrets_found', 'distance_traveled',
                'maps_completed', 'rare_encounters'
            ],
            'crafting': [
                'items_created', 'quality_achieved', 'recipes_discovered',
                'experiments_succeeded', 'master_works_completed'
            ]
        }

        return hooks.get(category, ['general_progress'])

    def _generate_balance_considerations(self) -> Dict[str, float]:
        """Generate balance consideration weights"""

        considerations = {}

        for factor in self.balance_factors:
            considerations[factor] = round(self.rng.uniform(0.3, 1.0), 2)

        return considerations

    def _generate_player_feedback(self, category: str) -> Dict[str, str]:
        """Generate player feedback specifications"""

        feedback = {
            'visual': self.rng.choice([
                'particle_effects', 'screen_flash', 'ui_highlight',
                'color_change', 'animation_trigger', 'icon_display'
            ]),
            'audio': self.rng.choice([
                'sound_effect', 'music_change', 'voice_line',
                'ambient_shift', 'notification_sound'
            ]),
            'haptic': self.rng.choice([
                'controller_rumble', 'impact_feedback', 'sustained_vibration',
                'pulse_pattern', 'none'
            ]),
            'ui': self.rng.choice([
                'number_popup', 'progress_bar', 'notification',
                'tooltip', 'status_icon', 'meter_fill'
            ]),
            'timing': self.rng.choice([
                'immediate', 'delayed', 'sustained', 'triggered'
            ])
        }

        return feedback

    def _generate_edge_cases(self, complexity: int) -> List[Dict]:
        """Generate edge case handling"""

        edge_cases = []
        num_cases = 1 + complexity // 3

        case_types = [
            'overflow', 'underflow', 'simultaneous_triggers', 'conflicting_effects',
            'resource_exhaustion', 'state_corruption', 'timing_conflict',
            'invalid_target', 'cancelled_action', 'interrupted_process'
        ]

        for _ in range(num_cases):
            case = {
                'type': self.rng.choice(case_types),
                'handling': self.rng.choice([
                    'clamp_to_bounds', 'fail_safely', 'queue_for_later',
                    'prioritize_by_rules', 'cancel_operation', 'partial_execution'
                ]),
                'notification': self.rng.choice([
                    'silent', 'warning', 'error_message', 'visual_cue'
                ])
            }
            edge_cases.append(case)

        return edge_cases

    def generate_system(self,
                        category: str,
                        num_mechanics: int = 5,
                        complexity: int = 5) -> Dict[str, Any]:
        """Generate a complete game system with multiple interacting mechanics"""

        cat_info = self.mechanic_categories.get(category, self.mechanic_categories['combat'])

        system = {
            'id': str(uuid.uuid4()),
            'name': f"{category.title()} System",
            'category': category,
            'mechanics': [],
            'system_rules': [],
            'global_parameters': {},
            'interaction_matrix': []
        }

        # Generate mechanics for each subcategory
        for subcat in cat_info['subcategories'][:num_mechanics]:
            mechanic = self.generate_mechanic(category, subcat, complexity)
            system['mechanics'].append(mechanic)

        # Generate system-level rules
        system['system_rules'] = [
            f"All {category} mechanics share a common resource pool",
            f"Effects from multiple mechanics stack {self.rng.choice(['additively', 'multiplicatively', 'to a cap'])}",
            f"Global cooldown of {self.rng.uniform(0.5, 2.0):.1f} seconds between major actions",
            f"System state resets upon {self.rng.choice(['rest', 'area change', 'combat end', 'time passage'])}"
        ]

        # Generate global parameters
        for param in cat_info['parameters']:
            system['global_parameters'][f"global_{param}"] = {
                'value': round(self.rng.uniform(1.0, 10.0), 2),
                'affects_all_mechanics': True
            }

        # Generate interaction matrix between mechanics
        mechanics_ids = [m['id'] for m in system['mechanics']]
        for i, m1 in enumerate(mechanics_ids):
            for m2 in mechanics_ids[i+1:]:
                if self.rng.random() > 0.5:
                    system['interaction_matrix'].append({
                        'mechanic_a': m1,
                        'mechanic_b': m2,
                        'interaction_type': self.rng.choice(self.interaction_types),
                        'strength': round(self.rng.uniform(0.5, 2.0), 2)
                    })

        return system

    def generate_batch(self, count: int) -> List[GameMechanic]:
        """Generate multiple mechanics"""

        mechanics = []

        for _ in range(count):
            category = self.rng.choice(list(self.mechanic_categories.keys()))
            complexity = self.rng.randint(3, 10)

            mechanic = self.generate_mechanic(category=category, complexity=complexity)
            mechanics.append(mechanic)

        return mechanics