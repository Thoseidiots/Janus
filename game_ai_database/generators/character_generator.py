"""
Character Generator
Generates detailed character profiles with personalities, backgrounds, and development arcs
"""
import random
import math
from typing import List, Dict, Any, Tuple, Optional
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralDescriptionGenerator,
    ProceduralGraphGenerator,
    ProceduralNumberGenerator
)
from schemas.data_schemas import Character, CharacterRole

class CharacterGenerator:
    """Generates complete character profiles with rich backgrounds and personalities"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.graph_gen = ProceduralGraphGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)
        
        # Character generation parameters
        self.personality_dimensions = {
            'openness': {'range': (0, 100), 'traits': ['curious', 'conventional', 'imaginative', 'practical']},
            'conscientiousness': {'range': (0, 100), 'traits': ['organized', 'careless', 'disciplined', 'spontaneous']},
            'extraversion': {'range': (0, 100), 'traits': ['outgoing', 'reserved', 'sociable', 'solitary']},
            'agreeableness': {'range': (0, 100), 'traits': ['compassionate', 'competitive', 'cooperative', 'selfish']},
            'neuroticism': {'range': (0, 100), 'traits': ['calm', 'anxious', 'confident', 'insecure']}
        }
        
        self.skill_categories = {
            'combat': ['melee', 'ranged', 'unarmed', 'defense', 'tactics'],
            'social': ['persuasion', 'deception', 'intimidation', 'empathy', 'leadership'],
            'technical': ['crafting', 'engineering', 'alchemy', 'magic', 'technology'],
            'exploration': ['survival', 'navigation', 'tracking', 'stealth', 'perception'],
            'knowledge': ['history', 'arcana', 'nature', 'religion', 'lore']
        }
        
        self.motivation_types = [
            'revenge', 'redemption', 'discovery', 'power', 'wealth', 'love',
            'justice', 'freedom', 'knowledge', 'survival', 'honor', 'legacy'
        ]
        
        self.flaw_types = [
            'greed', 'pride', 'wrath', 'envy', 'lust', 'gluttony', 'sloth',
            'paranoia', 'recklessness', 'dishonesty', 'cruelty', 'cowardice'
        ]
        
        self.arc_types = [
            'redemption', 'corruption', 'enlightenment', 'revenge', 'discovery',
            'sacrifice', 'ascension', 'fall', 'reconciliation', 'transformation'
        ]
    
    def generate_character(self, 
                          character_class: str = None,
                          background_theme: str = None,
                          power_level: str = 'standard') -> Dict[str, Any]:
        """Generate a complete character profile"""
        
        char_id = str(uuid.uuid4())
        
        # Generate basic identity
        identity = self._generate_identity(character_class, background_theme)
        
        # Generate personality profile
        personality = self._generate_personality_profile()
        
        # Generate background and history
        background = self._generate_background(identity, background_theme)
        
        # Generate skills and abilities
        skills = self._generate_skills(power_level, identity, background)
        
        # Generate motivations and goals
        motivations = self._generate_motivations(identity, background)
        
        # Generate character arc
        character_arc = self._generate_character_arc(identity, background, motivations)
        
        # Generate relationships
        relationships = self._generate_relationships(identity, background)
        
        # Generate appearance and mannerisms
        appearance = self._generate_appearance(identity)
        
        # Calculate derived stats
        derived_stats = self._calculate_derived_stats(personality, skills, background)
        
        character = Character(
            name=identity['name'],
            title=identity.get('title', ''),
            role=CharacterRole.NEUTRAL,
            backstory=background['detailed_history'],
            personality_traits=[trait['name'] for trait in personality['dominant_traits']],
            motivations=motivations['primary_motivations'],
            fears=motivations.get('flaws', []),
            skills=skills['skill_list'],
            relationships=relationships,
            dialogue_style={},
            visual_descriptors={'appearance': appearance['description']},
            arc_progression=[],
            combat_style='',
            faction_affiliations=[],
            secrets=[]
        )
        
        char_dict = character.to_dict()
        
        # Add additional generated content
        char_dict.update({
            'id': char_id,
            'identity': identity,
            'personality_profile': personality,
            'skills_detailed': skills,
            'motivations_detailed': motivations,
            'character_arc': character_arc,
            'relationships_detailed': relationships,
            'appearance_detailed': appearance,
            'derived_stats': derived_stats,
            'power_level': power_level,
            'creation_seed': self.seed
        })
        
        return char_dict
    
    def _generate_identity(self, 
                          character_class: str = None,
                          background_theme: str = None) -> Dict[str, Any]:
        """Generate character's basic identity"""
        
        # Character class selection
        if character_class is None:
            classes = [
                'warrior', 'mage', 'rogue', 'cleric', 'ranger', 'paladin',
                'sorcerer', 'druid', 'bard', 'monk', 'necromancer', 'summoner',
                'artificer', 'psion', 'shaman', 'assassin', 'merchant', 'noble'
            ]
            character_class = self.rng.choice(classes)
        
        # Generate name based on class and theme
        name_style = self._get_name_style(character_class, background_theme)
        name = self.name_gen.generate_character_name(name_style)
        
        # Generate age and life stage
        age_ranges = {
            'warrior': (18, 60), 'mage': (20, 80), 'rogue': (16, 50),
            'cleric': (18, 70), 'ranger': (16, 55), 'paladin': (18, 65),
            'sorcerer': (15, 75), 'druid': (18, 85), 'bard': (16, 60),
            'monk': (15, 70), 'necromancer': (25, 90), 'summoner': (18, 75),
            'artificer': (20, 70), 'psion': (18, 80), 'shaman': (18, 85),
            'assassin': (16, 50), 'merchant': (20, 65), 'noble': (18, 80)
        }
        
        min_age, max_age = age_ranges.get(character_class, (18, 60))
        age = self.rng.randint(min_age, max_age)
        
        life_stage = self._determine_life_stage(age)
        
        # Generate gender and cultural background
        gender = self.rng.choice(['male', 'female', 'non-binary', 'fluid'])
        culture = self._generate_cultural_background(character_class, background_theme)
        
        return {
            'name': name,
            'character_class': character_class,
            'age': age,
            'life_stage': life_stage,
            'gender': gender,
            'culture': culture,
            'name_style': name_style
        }
    
    def _get_name_style(self, character_class: str, background_theme: str = None) -> str:
        """Determine appropriate name style for character"""
        
        class_styles = {
            'warrior': ['strong', 'traditional', 'military'],
            'mage': ['mystical', 'ancient', 'scholarly'],
            'rogue': ['street', 'exotic', 'common'],
            'cleric': ['devout', 'traditional', 'noble'],
            'ranger': ['natural', 'rugged', 'exotic'],
            'paladin': ['noble', 'traditional', 'heroic'],
            'sorcerer': ['mystical', 'exotic', 'noble'],
            'druid': ['natural', 'ancient', 'mystical'],
            'bard': ['artistic', 'exotic', 'noble'],
            'monk': ['spiritual', 'exotic', 'simple'],
            'necromancer': ['dark', 'ancient', 'mystical'],
            'summoner': ['mystical', 'exotic', 'scholarly'],
            'artificer': ['technical', 'scholarly', 'common'],
            'psion': ['mystical', 'scholarly', 'exotic'],
            'shaman': ['spiritual', 'natural', 'ancient'],
            'assassin': ['shadow', 'exotic', 'street'],
            'merchant': ['common', 'noble', 'exotic'],
            'noble': ['noble', 'traditional', 'elegant']
        }
        
        styles = class_styles.get(character_class, ['common'])
        if background_theme:
            theme_styles = {
                'military': ['military', 'strong'],
                'academic': ['scholarly', 'ancient'],
                'criminal': ['street', 'shadow'],
                'religious': ['devout', 'spiritual'],
                'wilderness': ['natural', 'rugged'],
                'nobility': ['noble', 'elegant'],
                'mystical': ['mystical', 'ancient']
            }
            styles.extend(theme_styles.get(background_theme, []))
        
        return self.rng.choice(styles)
    
    def _determine_life_stage(self, age: int) -> str:
        """Determine character's life stage based on age"""
        
        if age < 18:
            return 'youth'
        elif age < 30:
            return 'young_adult'
        elif age < 50:
            return 'adult'
        elif age < 70:
            return 'middle_aged'
        else:
            return 'elder'
    
    def _generate_cultural_background(self, 
                                     character_class: str,
                                     background_theme: str = None) -> Dict[str, Any]:
        """Generate character's cultural and ethnic background"""
        
        cultures = [
            {'name': 'imperial', 'traits': ['disciplined', 'hierarchical', 'technological']},
            {'name': 'nomadic', 'traits': ['adaptable', 'spiritual', 'resourceful']},
            {'name': 'scholarly', 'traits': ['knowledgeable', 'curious', 'traditional']},
            {'name': 'warrior', 'traits': ['brave', 'honorable', 'competitive']},
            {'name': 'mystical', 'traits': ['spiritual', 'intuitive', 'mysterious']},
            {'name': 'merchant', 'traits': ['ambitious', 'persuasive', 'practical']},
            {'name': 'rebel', 'traits': ['independent', 'idealistic', 'resourceful']},
            {'name': 'ancient', 'traits': ['wise', 'traditional', 'mysterious']}
        ]
        
        # Bias culture selection based on class
        culture_weights = {}
        for culture in cultures:
            weight = 1.0
            if character_class in ['warrior', 'paladin'] and 'warrior' in culture['traits']:
                weight *= 2.0
            elif character_class in ['mage', 'sorcerer', 'psion'] and 'scholarly' in culture['traits']:
                weight *= 2.0
            elif character_class in ['rogue', 'assassin'] and 'rebel' in culture['traits']:
                weight *= 2.0
            elif character_class in ['cleric', 'monk', 'shaman'] and 'mystical' in culture['traits']:
                weight *= 2.0
            culture_weights[culture['name']] = weight
        
        culture = self.rng.choices(cultures, 
                                 weights=[culture_weights[c['name']] for c in cultures])[0]
        
        return culture
    
    def _generate_personality_profile(self) -> Dict[str, Any]:
        """Generate detailed personality profile"""
        
        profile = {}
        
        # Generate Big Five personality scores
        for dimension, props in self.personality_dimensions.items():
            min_val, max_val = props['range']
            score = self.rng.randint(min_val, max_val)
            profile[dimension] = score
            
            # Determine trait intensity
            if score < 20:
                intensity = 'very_low'
            elif score < 40:
                intensity = 'low'
            elif score < 60:
                intensity = 'moderate'
            elif score < 80:
                intensity = 'high'
            else:
                intensity = 'very_high'
            
            profile[f'{dimension}_intensity'] = intensity
        
        # Generate dominant traits
        dominant_traits = []
        for dimension, props in self.personality_dimensions.items():
            score = profile[dimension]
            traits = props['traits']
            
            if score < 30:
                trait = traits[0]  # Low score trait
            elif score < 70:
                trait = traits[1] if len(traits) > 1 else traits[0]
            else:
                trait = traits[2] if len(traits) > 2 else traits[0]
            
            dominant_traits.append({
                'dimension': dimension,
                'name': trait,
                'intensity': profile[f'{dimension}_intensity'],
                'score': score
            })
        
        profile['dominant_traits'] = dominant_traits
        
        # Generate behavioral tendencies
        profile['behavioral_tendencies'] = self._generate_behavioral_tendencies(profile)
        
        return profile
    
    def _generate_behavioral_tendencies(self, personality_profile: Dict) -> List[str]:
        """Generate behavioral tendencies based on personality"""
        
        tendencies = []
        
        # Openness tendencies
        if personality_profile['openness'] > 70:
            tendencies.extend(['seeks new experiences', 'appreciates art and beauty', 'open to unconventional ideas'])
        elif personality_profile['openness'] < 30:
            tendencies.extend(['prefers routine', 'values tradition', 'practical and conventional'])
        
        # Conscientiousness tendencies
        if personality_profile['conscientiousness'] > 70:
            tendencies.extend(['highly organized', 'reliable and punctual', 'plans ahead'])
        elif personality_profile['conscientiousness'] < 30:
            tendencies.extend(['spontaneous', 'dislikes routine', 'flexible but sometimes careless'])
        
        # Extraversion tendencies
        if personality_profile['extraversion'] > 70:
            tendencies.extend(['outgoing and energetic', 'enjoys social gatherings', 'assertive'])
        elif personality_profile['extraversion'] < 30:
            tendencies.extend(['prefers solitude', 'quiet and reserved', 'thinks before speaking'])
        
        # Agreeableness tendencies
        if personality_profile['agreeableness'] > 70:
            tendencies.extend(['compassionate', 'cooperative', 'trusting'])
        elif personality_profile['agreeableness'] < 30:
            tendencies.extend(['competitive', 'suspicious', 'prioritizes self-interest'])
        
        # Neuroticism tendencies
        if personality_profile['neuroticism'] > 70:
            tendencies.extend(['experiences strong emotions', 'worries frequently', 'sensitive to stress'])
        elif personality_profile['neuroticism'] < 30:
            tendencies.extend(['calm and composed', 'handles stress well', 'emotionally stable'])
        
        return tendencies[:5]  # Limit to 5 tendencies
    
    def _generate_background(self, 
                            identity: Dict,
                            background_theme: str = None) -> Dict[str, Any]:
        """Generate character's background and history"""
        
        # Generate family background
        family_situation = self._generate_family_background(identity)
        
        # Generate early life events
        early_life = self._generate_early_life(identity, family_situation)
        
        # Generate key life events
        key_events = self._generate_key_events(identity, early_life)
        
        # Generate current circumstances
        current_situation = self._generate_current_situation(identity, key_events)
        
        # Compile detailed history
        detailed_history = self._compile_history(identity, family_situation, 
                                                early_life, key_events, current_situation)
        
        # Generate summary
        summary = self._generate_background_summary(identity, detailed_history)
        
        return {
            'family_situation': family_situation,
            'early_life': early_life,
            'key_events': key_events,
            'current_situation': current_situation,
            'detailed_history': detailed_history,
            'summary': summary,
            'theme': background_theme
        }
    
    def _generate_family_background(self, identity: Dict) -> Dict[str, Any]:
        """Generate character's family situation"""
        
        family_types = [
            'nuclear_family', 'single_parent', 'orphaned', 'large_family',
            'noble_house', 'merchant_family', 'criminal_family', 'religious_family',
            'military_family', 'scholarly_family', 'mystical_family', 'nomadic_tribe'
        ]
        
        # Bias family type based on character class
        class_family_bias = {
            'noble': ['noble_house', 'large_family'],
            'merchant': ['merchant_family', 'nuclear_family'],
            'warrior': ['military_family', 'large_family'],
            'cleric': ['religious_family', 'nuclear_family'],
            'rogue': ['criminal_family', 'single_parent', 'orphaned'],
            'mage': ['scholarly_family', 'noble_house'],
            'ranger': ['nomadic_tribe', 'single_parent']
        }
        
        family_type = self.rng.choice(family_types)
        if identity['character_class'] in class_family_bias:
            biased_types = class_family_bias[identity['character_class']]
            if self.rng.random() < 0.6:  # 60% chance to use biased type
                family_type = self.rng.choice(biased_types)
        
        family_status = self.rng.choice(['supportive', 'distant', 'estranged', 'deceased', 'unknown'])
        
        return {
            'family_type': family_type,
            'family_status': family_status,
            'socioeconomic_status': self.rng.choice(['poor', 'middle_class', 'wealthy', 'noble']),
            'family_influence': self.rng.choice(['strong', 'moderate', 'weak', 'negative'])
        }
    
    def _generate_early_life(self, 
                            identity: Dict,
                            family_situation: Dict) -> List[Dict]:
        """Generate early life events"""
        
        events = []
        num_events = self.rng.randint(2, 5)
        
        event_types = [
            'education', 'first_love', 'loss_of_loved_one', 'great_achievement',
            'terrible_failure', 'mysterious_encounter', 'travel', 'training',
            'betrayal', 'discovery', 'accident', 'miracle'
        ]
        
        for i in range(num_events):
            event_type = self.rng.choice(event_types)
            age = self.rng.randint(5, identity['age'] - 5)
            
            event = {
                'age': age,
                'type': event_type,
                'description': f"At age {age}, {self._generate_event_description(event_type, identity)}",
                'impact': self.rng.choice(['positive', 'negative', 'transformative', 'neutral']),
                'lasting_effect': self._generate_lasting_effect(event_type)
            }
            events.append(event)
        
        return sorted(events, key=lambda x: x['age'])
    
    def _generate_event_description(self, event_type: str, identity: Dict) -> str:
        """Generate description for a life event"""
        
        descriptions = {
            'education': [
                f"began formal education in {self.rng.choice(['magic', 'combat', 'scholarship', 'craftsmanship'])}",
                f"studied under a master {self.rng.choice(['teacher', 'mentor', 'guru'])}",
                "discovered a natural aptitude for learning"
            ],
            'first_love': [
                "experienced their first romantic encounter",
                "formed a deep emotional bond with someone special",
                "learned about the complexities of relationships"
            ],
            'loss_of_loved_one': [
                "lost a close family member or friend",
                "experienced a profound sense of grief",
                "learned about mortality and loss"
            ],
            'great_achievement': [
                "accomplished something remarkable for their age",
                "received recognition for exceptional talent",
                "overcame significant obstacles to succeed"
            ],
            'terrible_failure': [
                "experienced a devastating setback",
                "made a mistake with serious consequences",
                "faced humiliation and disappointment"
            ],
            'mysterious_encounter': [
                "encountered something supernatural or unexplained",
                "had an experience that defied rational explanation",
                "discovered hidden knowledge or abilities"
            ],
            'travel': [
                "journeyed to distant lands",
                "explored unknown territories",
                "experienced different cultures and peoples"
            ],
            'training': [
                f"began intensive training in {identity['character_class']} skills",
                "underwent rigorous physical or mental conditioning",
                "learned discipline and self-control"
            ],
            'betrayal': [
                "was betrayed by someone they trusted",
                "experienced deception and broken promises",
                "learned about the darker side of human nature"
            ],
            'discovery': [
                "made an important personal discovery",
                "uncovered a hidden truth about themselves",
                "found something that changed their worldview"
            ],
            'accident': [
                "survived a dangerous accident",
                "experienced a life-threatening situation",
                "learned about vulnerability and resilience"
            ],
            'miracle': [
                "witnessed or experienced something miraculous",
                "received divine intervention or extraordinary luck",
                "had their faith or beliefs profoundly affected"
            ]
        }
        
        event_descriptions = descriptions.get(event_type, ["experienced a significant life event"])
        return self.rng.choice(event_descriptions)
    
    def _generate_lasting_effect(self, event_type: str) -> str:
        """Generate lasting psychological effect from an event"""
        
        effects = {
            'education': ['increased knowledge', 'confidence in abilities', 'love of learning'],
            'first_love': ['understanding of emotions', 'capacity for intimacy', 'relationship wisdom'],
            'loss_of_loved_one': ['emotional resilience', 'appreciation for life', 'protective instincts'],
            'great_achievement': ['self-confidence', 'ambition', 'leadership qualities'],
            'terrible_failure': ['caution', 'perfectionism', 'fear of failure'],
            'mysterious_encounter': ['openness to the unknown', 'spiritual awareness', 'curiosity'],
            'travel': ['cultural awareness', 'adaptability', 'independence'],
            'training': ['discipline', 'physical/mental conditioning', 'professional skills'],
            'betrayal': ['trust issues', 'independence', 'cynicism'],
            'discovery': ['self-awareness', 'personal growth', 'new perspectives'],
            'accident': ['survival instincts', 'caution', 'resilience'],
            'miracle': ['faith', 'optimism', 'sense of destiny']
        }
        
        event_effects = effects.get(event_type, ['personal growth'])
        return self.rng.choice(event_effects)
    
    def _generate_key_events(self, 
                            identity: Dict,
                            early_life: List[Dict]) -> List[Dict]:
        """Generate key transformative events in character's life"""
        
        events = []
        num_events = self.rng.randint(1, 4)
        
        # Start from after early life
        start_age = max([e['age'] for e in early_life]) + self.rng.randint(2, 10)
        
        for i in range(num_events):
            age = start_age + i * self.rng.randint(3, 15)
            if age >= identity['age']:
                break
                
            event = self._generate_major_event(identity, age)
            events.append(event)
        
        return events
    
    def _generate_major_event(self, identity: Dict, age: int) -> Dict:
        """Generate a major life-changing event"""
        
        event_types = [
            'career_breakthrough', 'personal_loss', 'great_discovery', 'betrayal',
            'victory', 'defeat', 'transformation', 'enlightenment', 'crisis',
            'triumph', 'tragedy', 'revelation'
        ]
        
        event_type = self.rng.choice(event_types)
        
        return {
            'age': age,
            'type': event_type,
            'description': f"At age {age}, {self._generate_major_event_description(event_type, identity)}",
            'consequences': self._generate_event_consequences(event_type),
            'emotional_impact': self.rng.choice(['devastating', 'transformative', 'empowering', 'humbling'])
        }
    
    def _generate_major_event_description(self, event_type: str, identity: Dict) -> str:
        """Generate description for a major life event"""
        
        descriptions = {
            'career_breakthrough': [
                f"achieved a major breakthrough in their {identity['character_class']} career",
                "received recognition for exceptional skill or achievement",
                "rose to prominence in their field of expertise"
            ],
            'personal_loss': [
                "suffered the loss of someone deeply important",
                "experienced a profound personal tragedy",
                "lost something irreplaceable"
            ],
            'great_discovery': [
                "made a groundbreaking discovery",
                "uncovered ancient knowledge or artifacts",
                "solved a mystery that had eluded others"
            ],
            'betrayal': [
                "was betrayed by a close ally or loved one",
                "discovered a devastating secret",
                "experienced profound disloyalty"
            ],
            'victory': [
                "achieved a decisive victory against overwhelming odds",
                "defeated a powerful enemy or rival",
                "accomplished an impossible task"
            ],
            'defeat': [
                "suffered a crushing defeat",
                "failed at something critically important",
                "experienced humiliation and loss"
            ],
            'transformation': [
                "underwent a profound personal transformation",
                "experienced a complete change in perspective",
                "emerged as a different person"
            ],
            'enlightenment': [
                "achieved spiritual or intellectual enlightenment",
                "gained profound wisdom or understanding",
                "experienced a moment of perfect clarity"
            ],
            'crisis': [
                "faced a life-threatening crisis",
                "confronted their deepest fears",
                "experienced a moment of ultimate testing"
            ],
            'triumph': [
                "achieved a personal triumph",
                "overcame impossible obstacles",
                "realized a lifelong dream"
            ],
            'tragedy': [
                "experienced a devastating tragedy",
                "witnessed unspeakable horror",
                "suffered irreparable loss"
            ],
            'revelation': [
                "received a profound revelation",
                "discovered a fundamental truth",
                "had their worldview completely shattered"
            ]
        }
        
        event_descriptions = descriptions.get(event_type, ["experienced a life-changing event"])
        return self.rng.choice(event_descriptions)
    
    def _generate_event_consequences(self, event_type: str) -> List[str]:
        """Generate consequences of a major event"""
        
        consequences = {
            'career_breakthrough': ['gained fame', 'increased responsibilities', 'new opportunities'],
            'personal_loss': ['emotional scars', 'changed priorities', 'search for meaning'],
            'great_discovery': ['new knowledge', 'dangerous secrets', 'responsibility'],
            'betrayal': ['trust issues', 'isolation', 'quest for justice'],
            'victory': ['confidence', 'enemies', 'leadership role'],
            'defeat': ['humility', 'determination', 'strategic thinking'],
            'transformation': ['new outlook', 'changed relationships', 'personal growth'],
            'enlightenment': ['wisdom', 'spiritual power', 'teaching others'],
            'crisis': ['resilience', 'new fears', 'appreciation for life'],
            'triumph': ['pride', 'new goals', 'inspiration to others'],
            'tragedy': ['grief', 'purpose', 'protective instincts'],
            'revelation': ['new beliefs', 'changed behavior', 'mission']
        }
        
        event_consequences = consequences.get(event_type, ['personal change'])
        return self.rng.sample(event_consequences, min(2, len(event_consequences)))
    
    def _generate_current_situation(self, 
                                   identity: Dict,
                                   key_events: List[Dict]) -> Dict[str, Any]:
        """Generate character's current life situation"""
        
        return {
            'occupation': self._generate_current_occupation(identity),
            'social_status': self.rng.choice(['respected', 'controversial', 'outcast', 'influential', 'ordinary']),
            'living_situation': self.rng.choice(['comfortable_home', 'modest_living', 'nomadic', 'luxurious', 'precarious']),
            'current_challenges': self._generate_current_challenges(identity),
            'recent_developments': self._generate_recent_developments(identity)
        }
    
    def _generate_current_occupation(self, identity: Dict) -> str:
        """Generate character's current occupation"""
        
        class_occupations = {
            'warrior': ['mercenary', 'guard', 'instructor', 'champion', 'commander'],
            'mage': ['researcher', 'teacher', 'adviser', 'enchanter', 'scholar'],
            'rogue': ['thief', 'investigator', 'smuggler', 'spy', 'adventurer'],
            'cleric': ['priest', 'healer', 'counselor', 'missionary', 'guardian'],
            'ranger': ['scout', 'hunter', 'guide', 'warden', 'explorer'],
            'paladin': ['protector', 'crusader', 'judge', 'leader', 'champion'],
            'sorcerer': ['noble', 'adviser', 'researcher', 'diplomat', 'enchanter'],
            'druid': ['guardian', 'healer', 'teacher', 'warden', 'shaman'],
            'bard': ['performer', 'chronicler', 'diplomat', 'spy', 'teacher'],
            'monk': ['teacher', 'guardian', 'healer', 'adviser', 'warrior'],
            'necromancer': ['researcher', 'adviser', 'guardian', 'teacher', 'summoner'],
            'summoner': ['researcher', 'guardian', 'adviser', 'teacher', 'diplomat'],
            'artificer': ['inventor', 'engineer', 'merchant', 'adviser', 'teacher'],
            'psion': ['researcher', 'teacher', 'adviser', 'diplomat', 'healer'],
            'shaman': ['spiritual_leader', 'healer', 'guardian', 'teacher', 'adviser'],
            'assassin': ['operative', 'guardian', 'adviser', 'enforcer', 'investigator'],
            'merchant': ['trader', 'negotiator', 'banker', 'adviser', 'leader'],
            'noble': ['ruler', 'diplomat', 'patron', 'judge', 'leader']
        }
        
        occupations = class_occupations.get(identity['character_class'], ['adventurer'])
        return self.rng.choice(occupations)
    
    def _generate_current_challenges(self, identity: Dict) -> List[str]:
        """Generate current challenges facing the character"""
        
        challenges = [
            'financial_difficulties', 'political_intrigue', 'personal_doubts',
            'external_threats', 'moral_dilemmas', 'health_issues', 'relationship_problems',
            'professional_rivalries', 'mysterious_curses', 'unfulfilled_promises'
        ]
        
        num_challenges = self.rng.randint(1, 3)
        return self.rng.sample(challenges, num_challenges)
    
    def _generate_recent_developments(self, identity: Dict) -> List[str]:
        """Generate recent developments in character's life"""
        
        developments = [
            'new_ally', 'new_enemy', 'personal_discovery', 'professional_opportunity',
            'health_change', 'relationship_change', 'financial_change', 'power_change',
            'knowledge_gain', 'skill_improvement'
        ]
        
        num_developments = self.rng.randint(1, 3)
        return self.rng.sample(developments, num_developments)
    
    def _compile_history(self, identity: Dict, family_situation: Dict, 
                        early_life: List[Dict], key_events: List[Dict],
                        current_situation: Dict) -> str:
        """Compile detailed character history"""
        
        history_parts = []
        
        # Family background
        family_desc = f"Born into a {family_situation['family_type'].replace('_', ' ')} "
        family_desc += f"with {family_situation['family_status']} family relations, "
        family_desc += f"{identity['name']} grew up in {family_situation['socioeconomic_status']} circumstances."
        history_parts.append(family_desc)
        
        # Early life events
        if early_life:
            early_desc = "Early life was marked by "
            event_summaries = [f"{e['type'].replace('_', ' ')} at age {e['age']}" for e in early_life[:3]]
            early_desc += ", ".join(event_summaries) + "."
            history_parts.append(early_desc)
        
        # Key events
        if key_events:
            key_desc = "Major turning points included "
            event_summaries = [f"{e['type'].replace('_', ' ')} at age {e['age']}" for e in key_events]
            key_desc += ", ".join(event_summaries) + "."
            history_parts.append(key_desc)
        
        # Current situation
        current_desc = f"Currently working as a {current_situation['occupation']}, "
        current_desc += f"{identity['name']} faces {', '.join(current_situation['current_challenges'])} "
        current_desc += f"while experiencing {', '.join(current_situation['recent_developments'])}."
        history_parts.append(current_desc)
        
        return " ".join(history_parts)
    
    def _generate_background_summary(self, identity: Dict, detailed_history: str) -> str:
        """Generate concise background summary"""
        
        summary_templates = [
            f"A {identity['life_stage']} {identity['character_class']} with a complex past.",
            f"A {identity['character_class']} shaped by {self.rng.choice(['loss', 'victory', 'betrayal', 'discovery'])}.",
            f"A seasoned {identity['character_class']} seeking {self.rng.choice(['redemption', 'power', 'knowledge', 'justice'])}.",
            f"A {identity['character_class']} whose past continues to influence their present."
        ]
        
        return self.rng.choice(summary_templates)
    
    def _generate_skills(self, 
                        power_level: str,
                        identity: Dict,
                        background: Dict) -> Dict[str, Any]:
        """Generate character's skills and abilities"""
        
        # Base skill allocation based on power level
        skill_points = {
            'weak': (10, 20),
            'standard': (20, 40),
            'powerful': (40, 80),
            'legendary': (80, 150)
        }
        
        min_points, max_points = skill_points.get(power_level, skill_points['standard'])
        total_points = self.rng.randint(min_points, max_points)
        
        # Allocate points to skill categories
        skill_allocation = {}
        categories = list(self.skill_categories.keys())
        
        # Bias allocation based on character class
        class_biases = {
            'warrior': {'combat': 0.4, 'exploration': 0.2, 'social': 0.1, 'technical': 0.1, 'knowledge': 0.2},
            'mage': {'technical': 0.3, 'knowledge': 0.4, 'social': 0.1, 'combat': 0.1, 'exploration': 0.1},
            'rogue': {'combat': 0.2, 'social': 0.2, 'exploration': 0.3, 'technical': 0.2, 'knowledge': 0.1},
            'cleric': {'technical': 0.3, 'social': 0.2, 'knowledge': 0.2, 'combat': 0.2, 'exploration': 0.1},
            'ranger': {'exploration': 0.4, 'combat': 0.3, 'knowledge': 0.2, 'social': 0.05, 'technical': 0.05}
        }
        
        bias = class_biases.get(identity['character_class'], 
                               {cat: 1.0/len(categories) for cat in categories})
        
        # Allocate points
        remaining_points = total_points
        for category in categories:
            if category == categories[-1]:  # Last category gets remaining points
                allocation = remaining_points
            else:
                max_allocation = int(remaining_points * (bias.get(category, 0.2) * 2))
                allocation = min(self.rng.randint(1, max_allocation), remaining_points)
            
            skill_allocation[category] = allocation
            remaining_points -= allocation
        
        # Generate specific skills within categories
        skill_list = []
        for category, points in skill_allocation.items():
            category_skills = self.skill_categories[category]
            num_skills = min(len(category_skills), max(1, points // 5))
            
            selected_skills = self.rng.sample(category_skills, num_skills)
            for skill in selected_skills:
                level = min(100, self.rng.randint(10, 20) + (points // num_skills))
                skill_list.append({
                    'name': skill,
                    'category': category,
                    'level': level,
                    'experience': self.rng.choice(['novice', 'competent', 'skilled', 'expert', 'master'])
                })
        
        return {
            'total_skill_points': total_points,
            'skill_allocation': skill_allocation,
            'skill_list': skill_list,
            'specializations': self._generate_specializations(identity, skill_list)
        }
    
    def _generate_specializations(self, identity: Dict, skills: List[Dict]) -> List[str]:
        """Generate character specializations"""
        
        specializations = []
        
        # Find highest skill
        if skills:
            top_skill = max(skills, key=lambda x: x['level'])
            specializations.append(f"Specialist in {top_skill['name']}")
        
        # Class-based specializations
        class_specs = {
            'warrior': ['Weapon Master', 'Tactical Commander', 'Battle-hardened Veteran'],
            'mage': ['Arcane Theorist', 'Spell Weaver', 'Mystical Scholar'],
            'rogue': ['Shadow Operative', 'Master Thief', 'Infiltration Expert'],
            'cleric': ['Divine Channeler', 'Spiritual Guide', 'Healing Adept'],
            'ranger': ['Wilderness Survivor', 'Beast Master', 'Pathfinder']
        }
        
        if identity['character_class'] in class_specs:
            specializations.append(self.rng.choice(class_specs[identity['character_class']]))
        
        return specializations[:2]  # Limit to 2 specializations
    
    def _generate_motivations(self, identity: Dict, background: Dict) -> Dict[str, Any]:
        """Generate character's motivations and goals"""
        
        # Primary motivations
        num_motivations = self.rng.randint(1, 3)
        primary_motivations = self.rng.sample(self.motivation_types, num_motivations)
        
        # Generate specific goals
        goals = []
        for motivation in primary_motivations:
            goal = self._generate_specific_goal(motivation, identity, background)
            goals.append(goal)
        
        # Flaws
        num_flaws = self.rng.randint(1, 2)
        flaws = self.rng.sample(self.flaw_types, num_flaws)
        
        # Internal conflicts
        conflicts = self._generate_internal_conflicts(primary_motivations, flaws)
        
        return {
            'primary_motivations': primary_motivations,
            'specific_goals': goals,
            'flaws': flaws,
            'internal_conflicts': conflicts,
            'driving_force': self._determine_driving_force(primary_motivations, background)
        }
    
    def _generate_specific_goal(self, 
                               motivation: str,
                               identity: Dict,
                               background: Dict) -> str:
        """Generate specific goal based on motivation type"""
        
        goal_templates = {
            'revenge': [
                f"Avenge the death of a loved one",
                f"Destroy those responsible for a past betrayal",
                f"Bring justice to those who wronged them"
            ],
            'redemption': [
                f"Atone for past mistakes and failures",
                f"Prove themselves worthy of forgiveness",
                f"Rebuild their reputation and honor"
            ],
            'discovery': [
                f"Uncover ancient secrets and lost knowledge",
                f"Explore unknown territories and mysteries",
                f"Find answers to profound questions"
            ],
            'power': [
                f"Gain ultimate power and influence",
                f"Control their own destiny completely",
                f"Become the most powerful in their field"
            ],
            'wealth': [
                f"Accumulate vast riches and resources",
                f"Secure financial independence forever",
                f"Build a legacy of prosperity"
            ],
            'love': [
                f"Find true love and companionship",
                f"Reunite with a lost loved one",
                f"Protect and care for those they love"
            ],
            'justice': [
                f"Create a fair and just world",
                f"Punish the guilty and protect the innocent",
                f"Right the wrongs of society"
            ],
            'freedom': [
                f"Break free from all constraints and obligations",
                f"Live life on their own terms",
                f"Help others achieve freedom"
            ],
            'knowledge': [
                f"Master all forms of knowledge and wisdom",
                f"Understand the fundamental truths of existence",
                f"Share knowledge with the world"
            ],
            'survival': [
                f"Ensure their own survival at all costs",
                f"Protect themselves from all threats",
                f"Build an impregnable sanctuary"
            ],
            'honor': [
                f"Uphold the highest standards of honor",
                f"Live a life of integrity and virtue",
                f"Restore honor to their family or organization"
            ],
            'legacy': [
                f"Create a lasting legacy that will be remembered",
                f"Ensure their achievements endure through time",
                f"Leave the world better than they found it"
            ]
        }
        
        templates = goal_templates.get(motivation, [f"Pursue the goal of {motivation}"])
        return self.rng.choice(templates)
    
    def _generate_internal_conflicts(self, 
                                    motivations: List[str],
                                    flaws: List[str]) -> List[str]:
        """Generate internal conflicts based on motivations and flaws"""
        
        conflicts = []
        
        # Motivation vs motivation conflicts
        if len(motivations) > 1:
            for i in range(len(motivations)):
                for j in range(i+1, len(motivations)):
                    if self.rng.random() < 0.3:  # 30% chance of conflict
                        conflict = f"Conflict between {motivations[i]} and {motivations[j]}"
                        conflicts.append(conflict)
        
        # Motivation vs flaw conflicts
        for motivation in motivations:
            for flaw in flaws:
                if self.rng.random() < 0.4:  # 40% chance of conflict
                    conflict = f"{flaw.capitalize()} undermines pursuit of {motivation}"
                    conflicts.append(conflict)
        
        return conflicts[:3]  # Limit to 3 conflicts
    
    def _determine_driving_force(self, 
                                motivations: List[str],
                                background: Dict) -> str:
        """Determine the primary driving force in character's life"""
        
        if motivations:
            return f"Driven primarily by {motivations[0]}"
        else:
            return "Driven by complex and evolving motivations"
    
    def _generate_character_arc(self, 
                               identity: Dict,
                               background: Dict,
                               motivations: Dict) -> Dict[str, Any]:
        """Generate character's development arc"""
        
        arc_type = self.rng.choice(self.arc_types)
        
        # Determine arc stages based on current age and life stage
        current_stage = self._determine_arc_stage(identity, background)
        
        # Generate arc progression
        progression = self._generate_arc_progression(arc_type, current_stage, motivations)
        
        return {
            'arc_type': arc_type,
            'current_stage': current_stage,
            'progression': progression,
            'potential_endings': self._generate_potential_endings(arc_type),
            'key_moments': self._generate_key_arc_moments(arc_type, identity)
        }
    
    def _determine_arc_stage(self, identity: Dict, background: Dict) -> str:
        """Determine current stage in character's arc"""
        
        life_stage = identity['life_stage']
        age = identity['age']
        
        # Age-based stage determination
        if age < 25:
            return 'beginning'
        elif age < 40:
            return 'rising_action'
        elif age < 60:
            return 'climax_approaching'
        else:
            return 'resolution'
        
        # Could also factor in background events, but keeping simple for now
    
    def _generate_arc_progression(self, 
                                 arc_type: str,
                                 current_stage: str,
                                 motivations: Dict) -> List[str]:
        """Generate progression steps for the character arc"""
        
        progressions = {
            'redemption': [
                'Recognize past wrongs and seek forgiveness',
                'Make amends and prove changed character',
                'Find inner peace and self-acceptance',
                'Help others avoid similar mistakes'
            ],
            'corruption': [
                'Experience initial moral compromise',
                'Justify increasingly unethical actions',
                'Lose connection with former values',
                'Embrace new dark identity completely'
            ],
            'enlightenment': [
                'Question existing beliefs and assumptions',
                'Seek knowledge and new perspectives',
                'Experience profound realization',
                'Share wisdom with others'
            ],
            'redemption': [
                'Recognize past wrongs and seek forgiveness',
                'Make amends and prove changed character',
                'Find inner peace and self-acceptance',
                'Help others avoid similar mistakes'
            ]
        }
        
        base_progression = progressions.get(arc_type, [
            'Face initial challenges',
            'Experience growth and change',
            'Confront major obstacles',
            'Achieve transformation'
        ])
        
        # Adjust based on current stage
        stage_index = ['beginning', 'rising_action', 'climax_approaching', 'resolution'].index(current_stage)
        
        return base_progression[stage_index:]
    
    def _generate_potential_endings(self, arc_type: str) -> List[str]:
        """Generate potential endings for the character arc"""
        
        endings = {
            'redemption': [
                'Complete redemption and inner peace',
                'Forgiveness from others and self',
                'New life free from past burdens',
                'Tragic failure despite best efforts'
            ],
            'corruption': [
                'Complete descent into darkness',
                'Power achieved at great personal cost',
                'Redemption arc begins from rock bottom',
                'Destruction by own corrupted actions'
            ],
            'enlightenment': [
                'Profound wisdom and understanding',
                'Teaching and guiding others',
                'Transcendent state of being',
                'Wisdom leads to isolation'
            ],
            'redemption': [
                'Complete redemption and inner peace',
                'Forgiveness from others and self',
                'New life free from past burdens',
                'Tragic failure despite best efforts'
            ]
        }
        
        return endings.get(arc_type, ['Transformation achieved', 'Tragic failure', 'Bittersweet resolution'])
    
    def _generate_key_arc_moments(self, arc_type: str, identity: Dict) -> List[str]:
        """Generate key moments in the character arc"""
        
        moments = [
            f"Initial catalyst that begins the {arc_type} journey",
            f"Major turning point in {identity['name']}'s development",
            f"Moment of crisis that tests their resolve",
            f"Final confrontation with their inner demons"
        ]
        
        return moments
    
    def _generate_relationships(self, identity: Dict, background: Dict) -> List[Dict]:
        """Generate character's relationships"""
        
        relationships = []
        num_relationships = self.rng.randint(3, 8)
        
        relationship_types = [
            'mentor', 'student', 'ally', 'rival', 'friend', 'enemy',
            'lover', 'family_member', 'colleague', 'patron', 'protege'
        ]
        
        for _ in range(num_relationships):
            rel_type = self.rng.choice(relationship_types)
            relationship = {
                'id': str(uuid.uuid4()),
                'type': rel_type,
                'name': self.name_gen.generate_character_name('common'),
                'status': self.rng.choice(['active', 'strained', 'distant', 'deceased', 'unknown']),
                'importance': self.rng.choice(['minor', 'significant', 'crucial']),
                'history': f"A {rel_type} from {identity['name']}'s past",
                'current_feeling': self.rng.choice(['positive', 'negative', 'complex', 'ambivalent'])
            }
            relationships.append(relationship)
        
        return relationships
    
    def _generate_appearance(self, identity: Dict) -> Dict[str, Any]:
        """Generate character's physical appearance"""
        
        appearance = {
            'description': self.desc_gen.generate_character_description(
                identity['character_class'],
                identity['age'],
                identity['gender']
            ),
            'distinguishing_features': self._generate_distinguishing_features(identity),
            'clothing_style': self._generate_clothing_style(identity),
            'mannerisms': self._generate_mannerisms(identity),
            'voice': self._generate_voice_description(identity)
        }
        
        return appearance
    
    def _generate_distinguishing_features(self, identity: Dict) -> List[str]:
        """Generate distinguishing physical features"""
        
        features = [
            'scarred_face', 'tattooed_body', 'unusual_eye_color', 'missing_limb',
            'prosthetic_device', 'magical_aura', 'animal_features', 'glowing_marks',
            'crystalline_growths', 'ethereal_quality', 'metallic_implants'
        ]
        
        num_features = self.rng.randint(0, 3)
        return self.rng.sample(features, num_features)
    
    def _generate_clothing_style(self, identity: Dict) -> str:
        """Generate character's clothing style"""
        
        styles = {
            'warrior': ['practical_armor', 'battle_worn_leather', 'ceremonial_plate'],
            'mage': ['flowing_robes', 'scholarly_attire', 'mystical_vestments'],
            'rogue': ['dark_leathers', 'concealable_clothing', 'exotic_attire'],
            'cleric': ['religious_garb', 'healing_attire', 'ceremonial_robes'],
            'noble': ['elegant_clothing', 'formal_attire', 'luxurious_garb']
        }
        
        class_styles = styles.get(identity['character_class'], ['practical_clothing'])
        return self.rng.choice(class_styles)
    
    def _generate_mannerisms(self, identity: Dict) -> List[str]:
        """Generate character's mannerisms"""
        
        mannerisms = [
            'speaks_deliberately', 'paces_when_thinking', 'avoids_eye_contact',
            'gestures_wildly', 'maintains_stoic_posture', 'fidgets_constantly',
            'smiles_rarely', 'laughs_boisterously', 'whispers_secrets'
        ]
        
        num_mannerisms = self.rng.randint(1, 3)
        return self.rng.sample(mannerisms, num_mannerisms)
    
    def _generate_voice_description(self, identity: Dict) -> str:
        """Generate character's voice description"""
        
        voice_qualities = [
            'deep_resonant', 'soft_whispering', 'commanding_booming',
            'gentle_soothing', 'harsh_grating', 'melodic_singing',
            'raspy_rough', 'clear_crisp', 'mumbling_uncertain'
        ]
        
        return self.rng.choice(voice_qualities)
    
    def _calculate_level(self, power_level: str, background: Dict) -> int:
        """Calculate character level based on power level and background"""
        
        base_levels = {
            'weak': (1, 5),
            'standard': (5, 15),
            'powerful': (15, 25),
            'legendary': (25, 40)
        }
        
        min_level, max_level = base_levels.get(power_level, base_levels['standard'])
        
        # Adjust based on background events
        background_modifier = len(background.get('key_events', [])) // 2
        
        level = self.rng.randint(min_level, max_level) + background_modifier
        return min(level, 50)  # Cap at 50
    
    def _calculate_derived_stats(self, 
                                personality: Dict,
                                skills: Dict,
                                background: Dict) -> Dict[str, Any]:
        """Calculate derived statistics"""
        
        # Calculate overall power level
        skill_total = sum(s['level'] for s in skills['skill_list'])
        power_score = skill_total / len(skills['skill_list']) if skills['skill_list'] else 0
        
        # Calculate social influence
        social_skills = [s for s in skills['skill_list'] if s['category'] == 'social']
        social_score = sum(s['level'] for s in social_skills) / len(social_skills) if social_skills else 0
        
        # Calculate combat effectiveness
        combat_skills = [s for s in skills['skill_list'] if s['category'] == 'combat']
        combat_score = sum(s['level'] for s in combat_skills) / len(combat_skills) if combat_skills else 0
        
        # Calculate stability (inverse of neuroticism)
        stability_score = 100 - personality.get('neuroticism', 50)
        
        return {
            'power_score': round(power_score, 1),
            'social_influence': round(social_score, 1),
            'combat_effectiveness': round(combat_score, 1),
            'emotional_stability': stability_score,
            'overall_threat_level': self._calculate_threat_level(power_score, combat_score, background),
            'leadership_potential': self._calculate_leadership_potential(personality, social_score)
        }
    
    def _calculate_threat_level(self, 
                               power_score: float,
                               combat_score: float,
                               background: Dict) -> str:
        """Calculate character's threat level"""
        
        combined_score = (power_score + combat_score) / 2
        
        if combined_score > 80:
            return 'catastrophic'
        elif combined_score > 60:
            return 'severe'
        elif combined_score > 40:
            return 'moderate'
        elif combined_score > 20:
            return 'minor'
        else:
            return 'negligible'
    
    def _calculate_leadership_potential(self, 
                                       personality: Dict,
                                       social_score: float) -> str:
        """Calculate leadership potential"""
        
        extraversion = personality.get('extraversion', 50)
        conscientiousness = personality.get('conscientiousness', 50)
        
        leadership_score = (extraversion + conscientiousness + social_score) / 3
        
        if leadership_score > 75:
            return 'exceptional'
        elif leadership_score > 60:
            return 'strong'
        elif leadership_score > 45:
            return 'moderate'
        elif leadership_score > 30:
            return 'limited'
        else:
            return 'poor'
    
    def generate_batch(self, 
                      count: int,
                      character_class: str = None,
                      power_level: str = 'standard') -> List[Dict]:
        """Generate multiple characters"""
        
        characters = []
        for i in range(count):
            self.rng = random.Random(self.seed + i)
            self.name_gen = ProceduralNameGenerator(self.seed + i)
            self.desc_gen = ProceduralDescriptionGenerator(self.seed + i)
            
            character = self.generate_character(
                character_class=character_class,
                power_level=power_level
            )
            characters.append(character)
        
        return characters
        skills = ['combat', 'stealth', 'magic', 'diplomacy', 'crafting']
        return {skill: self.rng.randint(1, 10) for skill in skills}
    
    def _generate_visual_descriptors(self) -> Dict[str, str]:
        """Generate visual appearance"""
        return {
            'hair': self.rng.choice(['black', 'brown', 'blonde', 'red']),
            'eyes': self.rng.choice(['blue', 'green', 'brown', 'gray']),
            'build': self.rng.choice(['slender', 'muscular', 'stocky', 'tall'])
        }