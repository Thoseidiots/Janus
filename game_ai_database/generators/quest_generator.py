"""
Quest Generator
Generates complex quests with branching narratives, moral choices, and dynamic rewards
"""
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid

from schemas.data_schemas import Quest
from utils.procedural_algorithms import ProceduralNameGenerator, ProceduralDescriptionGenerator, ProceduralNumberGenerator


class QuestGenerator:
    """Enterprise-grade quest generation with complex branching and moral systems"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize procedural generators
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)

        # Quest archetypes and themes
        self.quest_themes = {
            'heroic': ['save_village', 'defeat_monster', 'find_artifact', 'protect_caravan'],
            'mysterious': ['investigate_ruins', 'decode_ancient_text', 'find_hidden_temple', 'uncover_conspiracy'],
            'personal': ['avenge_family', 'find_lost_loved_one', 'prove_worth', 'gain_redemption'],
            'political': ['assassinate_leader', 'steal_documents', 'infiltrate_fortress', 'negotiate_treaty'],
            'economic': ['clear_trade_route', 'establish_business', 'collect_debt', 'mine_resources'],
            'exploration': ['map_territory', 'discover_new_species', 'chart_underwater_ruins', 'find_lost_city']
        }

        self.objective_types = [
            'kill', 'collect', 'deliver', 'escort', 'investigate', 'craft',
            'negotiate', 'stealth', 'puzzle', 'survive', 'time_limit', 'moral_choice'
        ]

        self.reward_types = [
            'gold', 'experience', 'item', 'reputation', 'title', 'land',
            'alliance', 'knowledge', 'artifact', 'blessing', 'curse'
        ]

        self.branch_triggers = [
            'time_pressure', 'moral_choice', 'skill_check', 'random_event',
            'player_decision', 'external_factor', 'relationship_status'
        ]

        self.reward_categories = {
            'tangible': ['currency', 'equipment', 'consumables', 'materials', 'property'],
            'progression': ['experience', 'skill_points', 'ability_unlock', 'reputation'],
            'narrative': ['information', 'alliance', 'access', 'resolution', 'secret_revealed'],
            'permanent': ['stat_increase', 'new_companion', 'world_change', 'title']
        }

    def generate_quest(self,
                       quest_type: str = 'main',
                       complexity: int = 5,
                       theme: str = None) -> Quest:
        """Generate a complete quest with all components"""

        quest = Quest()
        quest.id = str(uuid.uuid4())
        quest.quest_type = quest_type
        quest.complexity_rating = complexity

        # Select theme
        if not theme:
            theme = self.rng.choice(list(self.quest_themes.keys()))
        quest.theme = theme

        # Generate basic info
        quest.title = self._generate_title(theme)
        quest.description = self._generate_description(theme, complexity)
        quest.quest_category = self._select_category(theme)

        # Generate objectives
        quest.objectives = self._generate_objectives(complexity, theme)

        # Generate branches
        quest.branches = self._generate_branches(complexity)

        # Generate rewards
        quest.rewards = self._generate_rewards(quest.quest_category, complexity)

        # Generate metadata
        quest.metadata = self._generate_metadata(complexity, theme)

        return quest

    def _generate_title(self, theme: str) -> str:
        """Generate quest title"""

        title_templates = {
            'heroic': [
                "The {adjective} {noun}",
                "{action} of the {group}",
                "A {hero_type}'s {task}",
                "The {crisis} at {location}"
            ],
            'mysterious': [
                "Whispers of the {ancient}",
                "The {secret} of {location}",
                "Shadows of the {forgotten}",
                "Echoes from the {past}"
            ],
            'personal': [
                "A {emotion} {quest}",
                "The {family_member}'s {fate}",
                "Lost in {oblivion}",
                "The {redemption} of {name}"
            ],
            'political': [
                "The {political} {conspiracy}",
                "Betrayal in the {court}",
                "The {throne} of {kingdom}",
                "Power and {deception}"
            ],
            'economic': [
                "The {trade} {crisis}",
                "Gold and {greed}",
                "The {merchant}'s {burden}",
                "Fortunes of the {realm}"
            ],
            'exploration': [
                "Beyond the {horizon}",
                "The {unknown} {lands}",
                "Secrets of the {wilderness}",
                "The {explorer}'s {dream}"
            ]
        }

        templates = title_templates.get(theme, title_templates['heroic'])
        template = self.rng.choice(templates)

        # Fill template variables
        variables = {
            'adjective': self.rng.choice(['Lost', 'Ancient', 'Forgotten', 'Sacred', 'Cursed', 'Blessed']),
            'noun': self.rng.choice(['Sword', 'Crown', 'Temple', 'Dragon', 'Prophecy', 'Legacy']),
            'action': self.rng.choice(['Rise', 'Fall', 'Return', 'Awakening', 'Conquest', 'Defense']),
            'group': self.rng.choice(['Guardians', 'Elders', 'Warriors', 'Mages', 'Thieves', 'Priests']),
            'hero_type': self.rng.choice(['Warrior', 'Mage', 'Rogue', 'Paladin', 'Ranger', 'Bard']),
            'task': self.rng.choice(['Quest', 'Journey', 'Trial', 'Test', 'Challenge', 'Adventure']),
            'crisis': self.rng.choice(['Siege', 'Plague', 'Invasion', 'Betrayal', 'Eclipse', 'Cataclysm']),
            'location': self.name_gen.generate_name('location'),
            'ancient': self.rng.choice(['Ancients', 'Elders', 'Gods', 'Titans', 'Dragons', 'Spirits']),
            'secret': self.rng.choice(['Secret', 'Mystery', 'Enigma', 'Riddle', 'Puzzle', 'Conundrum']),
            'forgotten': self.rng.choice(['Forgotten', 'Lost', 'Abandoned', 'Vanished', 'Extinct', 'Erased']),
            'past': self.rng.choice(['Past', 'History', 'Ancestry', 'Heritage', 'Legacy', 'Origins']),
            'emotion': self.rng.choice(['Heartfelt', 'Desperate', 'Passionate', 'Tormented', 'Hopeful', 'Grieving']),
            'quest': self.rng.choice(['Quest', 'Journey', 'Search', 'Pursuit', 'Mission', 'Venture']),
            'family_member': self.rng.choice(['Father', 'Mother', 'Brother', 'Sister', 'Child', 'Love']),
            'fate': self.rng.choice(['Fate', 'Destiny', 'Doom', 'Fortune', 'End', 'Legacy']),
            'oblivion': self.rng.choice(['Oblivion', 'Darkness', 'Void', 'Shadows', 'Abyss', 'Nothingness']),
            'redemption': self.rng.choice(['Redemption', 'Salvation', 'Forgiveness', 'Atonement', 'Absolution', 'Grace']),
            'name': self.name_gen.generate_name('character'),
            'political': self.rng.choice(['Political', 'Royal', 'Imperial', 'Tribal', 'Noble', 'Courtly']),
            'conspiracy': self.rng.choice(['Conspiracy', 'Plot', 'Scheme', 'Intrigue', 'Machination', 'Cabal']),
            'court': self.rng.choice(['Court', 'Palace', 'Castle', 'Throne', 'Council', 'Senate']),
            'throne': self.rng.choice(['Throne', 'Crown', 'Scepter', 'Regalia', 'Power', 'Authority']),
            'kingdom': self.name_gen.generate_name('kingdom'),
            'deception': self.rng.choice(['Deception', 'Betrayal', 'Treachery', 'Duplicity', 'Fraud', 'Deceit']),
            'trade': self.rng.choice(['Trade', 'Commerce', 'Merchant', 'Caravan', 'Market', 'Exchange']),
            'greed': self.rng.choice(['Greed', 'Ambition', 'Avarice', 'Covetousness', 'Rapacity', 'Insatiability']),
            'merchant': self.rng.choice(['Merchant', 'Trader', 'Vendor', 'Dealer', 'Broker', 'Speculator']),
            'burden': self.rng.choice(['Burden', 'Load', 'Weight', 'Yoke', 'Cross', 'Trial']),
            'realm': self.name_gen.generate_name('realm'),
            'horizon': self.rng.choice(['Horizon', 'Edge', 'Boundary', 'Limit', 'Frontier', 'Verge']),
            'unknown': self.rng.choice(['Unknown', 'Uncharted', 'Mysterious', 'Enigmatic', 'Arcane', 'Esoteric']),
            'lands': self.rng.choice(['Lands', 'Territories', 'Realms', 'Domains', 'Regions', 'Provinces']),
            'wilderness': self.rng.choice(['Wilderness', 'Wilds', 'Outlands', 'Frontier', 'Badlands', 'Wasteland']),
            'explorer': self.rng.choice(['Explorer', 'Adventurer', 'Pioneer', 'Pathfinder', 'Voyager', 'Trailblazer']),
            'dream': self.rng.choice(['Dream', 'Vision', 'Aspiration', 'Goal', 'Ambition', 'Desire'])
        }

        title = template
        for var, options in variables.items():
            if f"{{{var}}}" in title:
                title = title.replace(f"{{{var}}}", str(options))

        return title

    def _generate_description(self, theme: str, complexity: int) -> str:
        """Generate quest description"""

        return self.desc_gen.generate_description(
            theme=f"quest_{theme}",
            complexity=complexity,
            length='medium'
        )

    def _select_category(self, theme: str) -> str:
        """Select quest category based on theme"""

        category_map = {
            'heroic': 'main',
            'mysterious': 'side',
            'personal': 'personal',
            'political': 'faction',
            'economic': 'side',
            'exploration': 'exploration'
        }

        return category_map.get(theme, 'side')

    def _generate_objectives(self, complexity: int, theme: str) -> List[Dict]:
        """Generate quest objectives"""

        objectives = []
        num_objectives = 2 + complexity // 2

        for i in range(num_objectives):
            obj = QuestObjective()
            obj.id = str(uuid.uuid4())
            obj.objective_type = self.rng.choice(self.objective_types)
            obj.description = self._generate_objective_description(obj.objective_type, theme)
            obj.is_optional = self.rng.random() > 0.7
            obj.is_hidden = self.rng.random() > 0.8

            # Generate requirements
            obj.requirements = self._generate_requirements(complexity)

            # Generate consequences
            obj.success_consequences = self._generate_consequences('success', complexity)
            obj.failure_consequences = self._generate_consequences('failure', complexity)

            objectives.append(obj)

        return objectives

    def _generate_objective_description(self, obj_type: str, theme: str) -> str:
        """Generate objective description"""

        templates = {
            'kill': [
                "Defeat {count} {enemy_type} {location}",
                "Slay the {enemy_title} known as {name}",
                "Eliminate the threat of {enemy_type} in {area}"
            ],
            'collect': [
                "Gather {count} {item_type} from {location}",
                "Find and retrieve {count} {item_type}",
                "Collect {resource} for the {beneficiary}"
            ],
            'deliver': [
                "Deliver {item} to {recipient} in {location}",
                "Transport {cargo} safely to {destination}",
                "Bring {message} to {contact} before {time}"
            ],
            'escort': [
                "Protect {person} during their journey to {destination}",
                "Guide {group} safely through {dangerous_area}",
                "Escort {vip} to the {event} in {location}"
            ],
            'investigate': [
                "Investigate the {mystery} at {location}",
                "Uncover the truth about {event} in {area}",
                "Examine the {clue} and report findings"
            ],
            'craft': [
                "Create {item} using {materials}",
                "Forge {weapon} with {special_ingredient}",
                "Build {structure} in {location}"
            ],
            'negotiate': [
                "Negotiate {deal} with {faction}",
                "Bargain for {item} from {merchant}",
                "Convince {person} to {action}"
            ],
            'stealth': [
                "Infiltrate {location} without being detected",
                "Steal {item} from {guardian}",
                "Spy on {target} and gather intelligence"
            ],
            'puzzle': [
                "Solve the {riddle} in {location}",
                "Decipher the {code} hidden in {artifact}",
                "Unlock the {mystery} of {ancient_structure}"
            ],
            'survive': [
                "Survive {duration} in {hostile_environment}",
                "Endure the {trial} of {challenge}",
                "Withstand {threat} for {time_period}"
            ],
            'time_limit': [
                "Complete the task before {deadline}",
                "Race against time to {action} in {location}",
                "Prevent {catastrophe} within {timeframe}"
            ],
            'moral_choice': [
                "Choose between {option_a} and {option_b}",
                "Decide the fate of {person} in {situation}",
                "Make a choice that will affect {consequences}"
            ]
        }

        template_list = templates.get(obj_type, ["Complete the objective"])
        template = self.rng.choice(template_list)

        # Fill template variables
        variables = {
            'count': self.rng.randint(1, 10),
            'enemy_type': self.rng.choice(['bandits', 'monsters', 'undead', 'demons', 'beasts', 'cultists']),
            'location': self.name_gen.generate_name('location'),
            'enemy_title': self.rng.choice(['Terror', 'Scourge', 'Menace', 'Horror', 'Nightmare', 'Doom']),
            'name': self.name_gen.generate_name('character'),
            'area': self.name_gen.generate_name('region'),
            'item_type': self.rng.choice(['artifacts', 'relics', 'ingredients', 'documents', 'gems', 'scrolls']),
            'resource': self.rng.choice(['minerals', 'herbs', 'wood', 'stone', 'metal', 'crystals']),
            'beneficiary': self.rng.choice(['village', 'temple', 'guild', 'kingdom', 'merchant', 'scholar']),
            'item': self.rng.choice(['package', 'letter', 'artifact', 'weapon', 'document', 'treasure']),
            'recipient': self.name_gen.generate_name('character'),
            'cargo': self.rng.choice(['supplies', 'treasure', 'prisoner', 'message', 'artifact', 'goods']),
            'destination': self.name_gen.generate_name('location'),
            'message': self.rng.choice(['warning', 'invitation', 'threat', 'proposal', 'secret', 'plea']),
            'contact': self.name_gen.generate_name('character'),
            'time': self.rng.choice(['dawn', 'noon', 'dusk', 'midnight', 'next moon', 'festival']),
            'person': self.name_gen.generate_name('character'),
            'group': self.rng.choice(['caravan', 'pilgrims', 'refugees', 'merchants', 'soldiers', 'scholars']),
            'dangerous_area': self.name_gen.generate_name('dangerous_location'),
            'vip': self.rng.choice(['noble', 'priest', 'merchant', 'general', 'scholar', 'heir']),
            'event': self.rng.choice(['ceremony', 'meeting', 'ritual', 'festival', 'trial', 'coronation']),
            'mystery': self.rng.choice(['disappearance', 'murder', 'curse', 'haunting', 'theft', 'phenomenon']),
            'event': self.rng.choice(['incident', 'occurrence', 'happening', 'affair', 'matter', 'case']),
            'clue': self.rng.choice(['symbol', 'message', 'artifact', 'witness', 'scene', 'pattern']),
            'materials': self.rng.choice(['rare metals', 'enchanted wood', 'dragon scales', 'unicorn horn', 'mithril', 'adamant']),
            'weapon': self.rng.choice(['sword', 'bow', 'staff', 'dagger', 'hammer', 'shield']),
            'special_ingredient': self.rng.choice(['dragon heart', 'phoenix feather', 'unicorn blood', 'demon essence', 'angel tear', 'god shard']),
            'structure': self.rng.choice(['bridge', 'tower', 'shrine', 'fortress', 'portal', 'obelisk']),
            'deal': self.rng.choice(['trade agreement', 'alliance', 'truce', 'surrender', 'marriage', 'partnership']),
            'faction': self.name_gen.generate_name('faction'),
            'merchant': self.name_gen.generate_name('character'),
            'action': self.rng.choice(['join', 'leave', 'fight', 'surrender', 'reveal secret', 'change allegiance']),
            'guardian': self.rng.choice(['dragon', 'golem', 'guardian spirit', 'elite guard', 'ancient warden', 'cursed knight']),
            'target': self.name_gen.generate_name('character'),
            'riddle': self.rng.choice(['ancient riddle', 'magical puzzle', 'prophecy', 'enigma', 'mystery', 'conundrum']),
            'code': self.rng.choice(['ancient runes', 'magical symbols', 'secret language', 'cipher', 'glyphs', 'hieroglyphs']),
            'artifact': self.rng.choice(['statue', 'tome', 'crystal', 'amulet', 'crown', 'scepter']),
            'ancient_structure': self.name_gen.generate_name('ancient_structure'),
            'duration': self.rng.choice(['one hour', 'one day', 'one week', 'one month', 'until dawn', 'until full moon']),
            'hostile_environment': self.name_gen.generate_name('dangerous_location'),
            'trial': self.rng.choice(['fire', 'ice', 'poison', 'darkness', 'light', 'chaos']),
            'challenge': self.rng.choice(['endurance', 'courage', 'wisdom', 'strength', 'faith', 'sacrifice']),
            'threat': self.rng.choice(['storm', 'earthquake', 'flood', 'plague', 'invasion', 'cataclysm']),
            'time_period': self.rng.choice(['one hour', 'one day', 'one week', 'one month', 'one year', 'eternity']),
            'deadline': self.rng.choice(['sunset', 'midnight', 'next moon', 'festival', 'eclipse', 'solstice']),
            'catastrophe': self.rng.choice(['flood', 'earthquake', 'plague', 'war', 'famine', 'apocalypse']),
            'timeframe': self.rng.choice(['one hour', 'one day', 'one week', 'one month', 'one year', 'decade']),
            'option_a': self.rng.choice(['save the innocent', 'destroy the evil', 'seek mercy', 'show compassion', 'fight honorably', 'protect the weak']),
            'option_b': self.rng.choice(['sacrifice for power', 'embrace darkness', 'seek vengeance', 'show ruthlessness', 'fight dirty', 'abandon the weak']),
            'situation': self.rng.choice(['moral dilemma', 'life or death choice', 'sacrifice decision', 'loyalty test', 'honor vs duty', 'love vs power']),
            'consequences': self.rng.choice(['the realm', 'your companions', 'future generations', 'the balance of power', 'innocent lives', 'ancient prophecies'])
        }

        description = template
        for var, options in variables.items():
            if f"{{{var}}}" in description:
                description = description.replace(f"{{{var}}}", str(options))

        return description

    def _generate_requirements(self, complexity: int) -> Dict[str, Any]:
        """Generate objective requirements"""

        requirements = {
            'level': self.rng.randint(max(1, complexity - 2), complexity + 2),
            'skills': []
        }

        # Add skill requirements
        skill_types = ['combat', 'stealth', 'magic', 'social', 'crafting', 'exploration']
        num_skills = min(complexity // 2, len(skill_types))

        for skill in self.rng.sample(skill_types, num_skills):
            requirements['skills'].append({
                'skill': skill,
                'level': self.rng.randint(1, 10)
            })

        # Add item requirements occasionally
        if self.rng.random() > 0.6:
            requirements['items'] = self.rng.sample([
                'special key', 'magic amulet', 'enchanted weapon', 'ancient map',
                'sacred relic', 'royal seal', 'guild badge', 'mystical artifact'
            ], self.rng.randint(1, 3))

        return requirements

    def _generate_consequences(self, outcome: str, complexity: int) -> List[Dict]:
        """Generate objective consequences"""

        consequences = []

        if outcome == 'success':
            positive_effects = [
                'gain_reputation', 'unlock_quest', 'learn_skill', 'receive_item',
                'form_alliance', 'discover_location', 'gain_follower', 'increase_faction_standing'
            ]
            num_effects = 1 + complexity // 3

            for _ in range(num_effects):
                effect = {
                    'type': self.rng.choice(positive_effects),
                    'magnitude': self.rng.randint(1, complexity),
                    'description': f"Positive consequence from {outcome}"
                }
                consequences.append(effect)

        else:  # failure
            negative_effects = [
                'lose_reputation', 'close_quest_path', 'lose_item', 'damage_relationship',
                'create_enemy', 'trigger_event', 'reduce_faction_standing', 'personal_cost'
            ]
            num_effects = 1 + complexity // 4

            for _ in range(num_effects):
                effect = {
                    'type': self.rng.choice(negative_effects),
                    'magnitude': self.rng.randint(1, complexity),
                    'description': f"Negative consequence from {outcome}"
                }
                consequences.append(effect)

        return consequences

    def _generate_branches(self, complexity: int) -> List[Dict]:
        """Generate quest branches"""

        branches = []
        num_branches = 1 + complexity // 3

        for i in range(num_branches):
            branch = QuestBranch()
            branch.id = str(uuid.uuid4())
            branch.trigger_condition = self.rng.choice(self.branch_triggers)
            branch.branch_description = self._generate_branch_description(branch.trigger_condition)

            # Generate branch paths
            branch.paths = self._generate_branch_paths(complexity)

            # Generate branch consequences
            branch.consequences = self._generate_branch_consequences(complexity)

            branches.append(branch)

        return branches

    def _generate_branch_description(self, trigger: str) -> str:
        """Generate branch description"""

        descriptions = {
            'time_pressure': "Time is running out - make a decision quickly!",
            'moral_choice': "A difficult moral decision must be made.",
            'skill_check': "Your skills will determine the path forward.",
            'random_event': "An unexpected event changes everything.",
            'player_decision': "The choice is yours to make.",
            'external_factor': "External circumstances force a change in plans.",
            'relationship_status': "Your relationships affect the available options."
        }

        return descriptions.get(trigger, "A branching point in the quest.")

    def _generate_branch_paths(self, complexity: int) -> List[Dict]:
        """Generate branch paths"""

        paths = []
        num_paths = 2 + self.rng.randint(0, complexity // 2)

        for i in range(num_paths):
            path = {
                'id': str(uuid.uuid4()),
                'description': f"Path {i+1}: {self._generate_path_description()}",
                'requirements': self._generate_path_requirements(),
                'difficulty': self.rng.randint(1, 10),
                'risk_level': self.rng.choice(['low', 'medium', 'high', 'extreme'])
            }
            paths.append(path)

        return paths

    def _generate_path_description(self) -> str:
        """Generate path description"""

        templates = [
            "Take the direct approach and confront the problem head-on.",
            "Use stealth and cunning to achieve your goal.",
            "Seek help from allies and build a coalition.",
            "Research and prepare thoroughly before acting.",
            "Make a sacrifice to achieve a greater good.",
            "Embrace the darkness to defeat a greater evil.",
            "Follow the path of honor and righteousness.",
            "Bend the rules to achieve justice."
        ]

        return self.rng.choice(templates)

    def _generate_path_requirements(self) -> Dict[str, Any]:
        """Generate path requirements"""

        requirements = {}

        if self.rng.random() > 0.5:
            requirements['alignment'] = self.rng.choice(['good', 'neutral', 'evil'])

        if self.rng.random() > 0.6:
            requirements['reputation'] = self.rng.choice(['hero', 'villain', 'unknown'])

        if self.rng.random() > 0.7:
            requirements['companion'] = self.rng.choice(['warrior', 'mage', 'rogue', 'priest'])

        return requirements

    def _generate_branch_consequences(self, complexity: int) -> Dict[str, Any]:
        """Generate branch consequences"""

        return {
            'success_bonus': self.rng.randint(1, complexity * 10),
            'failure_penalty': self.rng.randint(1, complexity * 5),
            'relationship_changes': self._generate_relationship_changes(),
            'world_state_changes': self._generate_world_changes(complexity)
        }

    def _generate_relationship_changes(self) -> List[Dict]:
        """Generate relationship changes"""

        changes = []
        num_changes = self.rng.randint(1, 3)

        for _ in range(num_changes):
            change = {
                'faction': self.name_gen.generate_name('faction'),
                'change_type': self.rng.choice(['increase', 'decrease', 'alliance', 'rivalry']),
                'magnitude': self.rng.randint(1, 5)
            }
            changes.append(change)

        return changes

    def _generate_world_changes(self, complexity: int) -> List[Dict]:
        """Generate world state changes"""

        changes = []
        num_changes = self.rng.randint(0, complexity // 2)

        change_types = [
            'location_access', 'resource_availability', 'faction_status',
            'environmental_change', 'population_change', 'economic_shift'
        ]

        for _ in range(num_changes):
            change = {
                'type': self.rng.choice(change_types),
                'location': self.name_gen.generate_name('location'),
                'effect': self.rng.choice(['positive', 'negative', 'neutral']),
                'duration': self.rng.choice(['temporary', 'permanent', 'conditional'])
            }
            changes.append(change)

        return changes

    def _generate_rewards(self, category: str, complexity: int) -> Dict[str, Any]:
        """Generate quest rewards"""
        
        rewards = {}
        
        # Base rewards
        rewards['experience'] = 100 * complexity + self.rng.randint(50, 200)
        rewards['currency'] = 50 * complexity + self.rng.randint(20, 100)
        
        # Category-specific rewards
        if category == 'main':
            rewards['ability_unlock'] = self.rng.choice([
                'new_skill', 'power_upgrade', 'unique_ability', 'companion_ability'
            ])
            rewards['world_change'] = self.rng.choice([
                'region_unlocked', 'faction_attitude_shift', 'npc_fate_changed',
                'resource_availability_changed'
            ])
        
        if category == 'faction':
            rewards['reputation'] = {
                'faction': self.name_gen.generate_name('ancient', 2, 3),
                'amount': 50 + complexity * 10
            }
        
        # Random additional rewards
        if self.rng.random() > 0.5:
            reward_category = self.rng.choice(list(self.reward_categories.keys()))
            reward_type = self.rng.choice(self.reward_categories[reward_category])
            rewards[reward_type] = True
        
        # Item rewards
        if self.rng.random() > 0.3:
            rewards['items'] = self.num_gen.generate_loot_table(
                self.rng.randint(1, 3)
            )
        
        return rewards

    def _generate_reward_description(self, reward_type: str, theme: str) -> str:
        """Generate reward description"""

        descriptions = {
            'gold': [
                "{amount} gold coins",
                "A purse containing {amount} gold pieces",
                "Payment of {amount} gold for services rendered"
            ],
            'experience': [
                "{amount} experience points",
                "Knowledge and wisdom worth {amount} experience",
                "Lessons learned granting {amount} experience"
            ],
            'item': [
                "A {quality} {item_type}",
                "An enchanted {item_type} of {element}",
                "A legendary {item_type} lost to time"
            ],
            'reputation': [
                "Increased standing with {faction}",
                "Hero status among the {group}",
                "Legendary reputation in {region}"
            ],
            'title': [
                "The title of {title}",
                "Noble rank as {title}",
                "Honorary position as {title}"
            ],
            'land': [
                "Ownership of {location}",
                "A plot of land in {region}",
                "Territory rights in {area}"
            ],
            'alliance': [
                "Alliance with {faction}",
                "Friendship of the {group}",
                "Pact with {organization}"
            ],
            'knowledge': [
                "Ancient secrets revealed",
                "Forbidden knowledge unlocked",
                "Mystical wisdom imparted"
            ],
            'artifact': [
                "A powerful {artifact_type}",
                "An ancient {artifact_type} of legend",
                "A {artifact_type} of unimaginable power"
            ],
            'blessing': [
                "Divine blessing from {deity}",
                "Magical boon granted",
                "Cursed no more - blessed instead"
            ],
            'curse': [
                "A terrible curse upon you",
                "Dark magic now courses through your veins",
                "The gods have turned against you"
            ]
        }

        template_list = descriptions.get(reward_type, ["A valuable reward"])
        template = self.rng.choice(template_list)

        # Fill template variables
        variables = {
            'amount': self.rng.randint(100, 10000),
            'quality': self.rng.choice(['masterwork', 'enchanted', 'legendary', 'cursed', 'blessed', 'ordinary']),
            'item_type': self.rng.choice(['sword', 'shield', 'armor', 'amulet', 'ring', 'cloak', 'boots', 'gloves']),
            'element': self.rng.choice(['fire', 'ice', 'lightning', 'shadow', 'light', 'nature', 'spirit', 'void']),
            'faction': self.name_gen.generate_name('faction'),
            'group': self.rng.choice(['peasants', 'nobles', 'merchants', 'warriors', 'mages', 'priests']),
            'region': self.name_gen.generate_name('region'),
            'title': self.rng.choice(['Hero', 'Champion', 'Guardian', 'Savior', 'Legend', 'Master']),
            'location': self.name_gen.generate_name('location'),
            'area': self.name_gen.generate_name('region'),
            'organization': self.name_gen.generate_name('organization'),
            'artifact_type': self.rng.choice(['relic', 'tome', 'crystal', 'orb', 'statue', 'weapon', 'armor']),
            'deity': self.name_gen.generate_name('deity')
        }

        description = template
        for var, options in variables.items():
            if f"{{{var}}}" in description:
                description = description.replace(f"{{{var}}}", str(options))

        return description

    def _calculate_reward_value(self, reward_type: str, complexity: int) -> int:
        """Calculate reward value"""

        base_values = {
            'gold': 100,
            'experience': 50,
            'item': 500,
            'reputation': 200,
            'title': 1000,
            'land': 2000,
            'alliance': 800,
            'knowledge': 300,
            'artifact': 1500,
            'blessing': 600,
            'curse': -500
        }

        base = base_values.get(reward_type, 100)
        multiplier = 1 + (complexity - 1) * 0.5

        return int(base * multiplier * self.rng.uniform(0.8, 1.2))

    def _generate_reward_conditions(self) -> List[str]:
        """Generate reward conditions"""

        conditions = []

        possible_conditions = [
            "Complete all objectives",
            "Maintain good reputation",
            "Finish within time limit",
            "No civilian casualties",
            "Help was not required",
            "Perfect execution",
            "Moral choice made correctly"
        ]

        num_conditions = self.rng.randint(0, 3)
        conditions = self.rng.sample(possible_conditions, num_conditions)

        return conditions

    def _generate_metadata(self, complexity: int, theme: str) -> Dict[str, Any]:
        """Generate quest metadata"""

        return {
            'estimated_duration': self._estimate_duration(complexity),
            'recommended_level': self.rng.randint(max(1, complexity - 2), complexity + 3),
            'difficulty_rating': self.rng.choice(['easy', 'medium', 'hard', 'expert', 'master']),
            'moral_complexity': self.rng.randint(1, 10),
            'replayability': self.rng.randint(1, 10),
            'branching_factor': len(self._generate_branches(complexity)),
            'hidden_elements': self.rng.randint(0, complexity),
            'faction_influence': self._generate_faction_influence(),
            'world_impact': self.rng.choice(['minimal', 'moderate', 'significant', 'world_shaking'])
        }

    def _estimate_duration(self, complexity: int) -> str:
        """Estimate quest duration"""

        durations = {
            1: "30 minutes",
            2: "1 hour",
            3: "2 hours",
            4: "4 hours",
            5: "8 hours",
            6: "1 day",
            7: "2 days",
            8: "1 week",
            9: "2 weeks",
            10: "1 month"
        }

        return durations.get(min(complexity, 10), "Several days")

    def _generate_faction_influence(self) -> List[Dict]:
        """Generate faction influence changes"""

        factions = []
        num_factions = self.rng.randint(1, 4)

        for _ in range(num_factions):
            faction = {
                'name': self.name_gen.generate_name('faction'),
                'influence_change': self.rng.randint(-5, 5),
                'relationship_type': self.rng.choice(['ally', 'enemy', 'neutral', 'rival', 'patron'])
            }
            factions.append(faction)

        return factions

    def generate_batch(self, count: int) -> List[Quest]:
        """Generate multiple quests"""

        quests = []

        for _ in range(count):
            complexity = self.rng.randint(3, 10)
            theme = self.rng.choice(list(self.quest_themes.keys()))

            quest = self.generate_quest(complexity=complexity, theme=theme)
            quests.append(quest)

        return quests