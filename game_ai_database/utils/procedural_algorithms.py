# Procedural Generation Seeds - Original Content Building Blocks
PHONEME_SETS = {
    "harsh": ["kr", "gr", "th", "sk", "dr", "br", "tr"],
    "soft": ["li", "mi", "na", "su", "ra", "we", "ya"],
    "mystical": ["ae", "yl", "th", "or", "an", "el", "is"],
    "technological": ["ex", "ax", "ix", "oz", "uz", "ek", "on"],
    "natural": ["ri", "ve", "lo", "wa", "fo", "mo", "te"]
}

NARRATIVE_ARCHETYPES = [
    "redemption_journey",
    "power_corruption",
    "identity_discovery", 
    "survival_against_odds",
    "forbidden_knowledge",
    "broken_world_restoration",
    "cycle_breaking",
    "legacy_burden",
    "unity_through_conflict",
    "sacrifice_meaning"
]

WORLD_FOUNDATIONS = [
    "fractured_reality",
    "layered_existence",
    "cyclical_time",
    "emergent_consciousness",
    "entropic_decay",
    "synthetic_nature",
    "dream_manifested",
    "memory_architecture",
    "probability_collapse",
    "void_between"
]

"""
Procedural Generation Algorithms - No External Dependencies
Original content generation through mathematical and linguistic procedures
"""
import random
import math
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import itertools

class ProceduralNameGenerator:
    """Generates unique names using phonetic rules and markov-like processes"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Original phoneme combinations - not from any existing work
        self.consonant_initials = [
            'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
            'n', 'p', 'r', 's', 't', 'v', 'w', 'z',
            'br', 'cr', 'dr', 'fr', 'gr', 'pr', 'tr',
            'bl', 'cl', 'fl', 'gl', 'pl', 'sl',
            'sc', 'sk', 'sm', 'sn', 'sp', 'st', 'sw',
            'th', 'sh', 'ch', 'wh', 'qu'
        ]
        
        self.vowel_cores = [
            'a', 'e', 'i', 'o', 'u',
            'ae', 'ai', 'ao', 'au',
            'ea', 'ei', 'eo', 'eu',
            'ia', 'ie', 'io', 'iu',
            'oa', 'oe', 'oi', 'ou',
            'ua', 'ue', 'ui', 'uo'
        ]
        
        self.consonant_finals = [
            'b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'x', 'z',
            'ck', 'ct', 'ft', 'ld', 'lf', 'lk', 'lm', 'ln', 'lp', 'lt',
            'mb', 'mp', 'nd', 'ng', 'nk', 'nt', 'pt', 'rb', 'rd', 'rf',
            'rk', 'rm', 'rn', 'rp', 'rt', 'sk', 'sp', 'st'
        ]
        
        # Cultural flavor sets for different civilization types
        self.flavor_profiles = {
            'ancient': {
                'preferred_vowels': ['a', 'o', 'u', 'ae', 'ou'],
                'preferred_initials': ['th', 'k', 'r', 's', 'm', 'n'],
                'preferred_finals': ['s', 'n', 'r', 'th', 'x'],
                'syllable_range': (2, 4)
            },
            'technological': {
                'preferred_vowels': ['e', 'i', 'o', 'ei', 'io'],
                'preferred_initials': ['z', 'x', 'v', 'cr', 'tr', 'pr'],
                'preferred_finals': ['x', 'ck', 'ct', 'pt', 'sk'],
                'syllable_range': (1, 3)
            },
            'natural': {
                'preferred_vowels': ['a', 'e', 'i', 'ea', 'ia'],
                'preferred_initials': ['l', 'w', 'f', 'fl', 'gl', 'bl'],
                'preferred_finals': ['n', 'l', 'm', 'nd', 'lf'],
                'syllable_range': (2, 4)
            },
            'mystical': {
                'preferred_vowels': ['ae', 'ei', 'oa', 'ue', 'ia'],
                'preferred_initials': ['th', 'sh', 'wh', 'ch', 'qu'],
                'preferred_finals': ['th', 'ng', 'rm', 'rn', 'lt'],
                'syllable_range': (2, 5)
            },
            'harsh': {
                'preferred_vowels': ['a', 'o', 'u', 'au', 'ou'],
                'preferred_initials': ['gr', 'kr', 'dr', 'br', 'sk', 'st'],
                'preferred_finals': ['rk', 'rt', 'sk', 'ng', 'ck'],
                'syllable_range': (1, 3)
            }
        }
    
    def generate_syllable(self, profile: str = 'natural', position: str = 'middle') -> str:
        """Generate a single syllable based on flavor profile"""
        flavor = self.flavor_profiles.get(profile, self.flavor_profiles['natural'])
        
        parts = []
        
        # Initial consonant (more likely at start)
        if position == 'start' or self.rng.random() > 0.3:
            if self.rng.random() > 0.5:
                parts.append(self.rng.choice(flavor['preferred_initials']))
            else:
                parts.append(self.rng.choice(self.consonant_initials))
        
        # Vowel core (always present)
        if self.rng.random() > 0.4:
            parts.append(self.rng.choice(flavor['preferred_vowels']))
        else:
            parts.append(self.rng.choice(self.vowel_cores))
        
        # Final consonant (more likely at end)
        if position == 'end' or self.rng.random() > 0.5:
            if self.rng.random() > 0.5:
                parts.append(self.rng.choice(flavor['preferred_finals']))
            else:
                parts.append(self.rng.choice(self.consonant_finals))
        
        return ''.join(parts)
    
    def generate_name(self, profile: str = 'natural', 
                      min_syllables: int = None, 
                      max_syllables: int = None) -> str:
        """Generate a complete name"""
        flavor = self.flavor_profiles.get(profile, self.flavor_profiles['natural'])
        
        if min_syllables is None:
            min_syllables = flavor['syllable_range'][0]
        if max_syllables is None:
            max_syllables = flavor['syllable_range'][1]
        
        num_syllables = self.rng.randint(min_syllables, max_syllables)
        
        syllables = []
        for i in range(num_syllables):
            if i == 0:
                position = 'start'
            elif i == num_syllables - 1:
                position = 'end'
            else:
                position = 'middle'
            
            syllables.append(self.generate_syllable(profile, position))
        
        name = ''.join(syllables)
        return name.capitalize()
    
    def generate_place_name(self, terrain_type: str, profile: str = 'natural') -> str:
        """Generate place names with terrain-appropriate suffixes"""
        base = self.generate_name(profile, 1, 2)
        
        terrain_suffixes = {
            'mountains': ['peak', 'horn', 'spire', 'crag', 'heights', 'ridge'],
            'forest': ['wood', 'grove', 'glade', 'wilds', 'thicket', 'shade'],
            'plains': ['field', 'reach', 'expanse', 'stretch', 'vale', 'mead'],
            'desert': ['waste', 'dunes', 'sands', 'barrens', 'flats', 'scorch'],
            'aquatic': ['depths', 'tide', 'waters', 'mere', 'bay', 'flow'],
            'urban': ['hold', 'keep', 'haven', 'port', 'gate', 'spire'],
            'subterranean': ['deep', 'hollow', 'cavern', 'dark', 'below', 'pit'],
            'aerial': ['sky', 'drift', 'float', 'wind', 'cloud', 'aerie'],
            'void': ['rift', 'null', 'breach', 'tear', 'between', 'empty']
        }
        
        suffixes = terrain_suffixes.get(terrain_type, terrain_suffixes['plains'])
        suffix = self.rng.choice(suffixes)
        
        # Various naming patterns
        patterns = [
            f"{base}{suffix}",
            f"{base}'s {suffix.capitalize()}",
            f"The {base} {suffix.capitalize()}",
            f"{suffix.capitalize()} of {base}",
            f"{base}-{suffix}"
        ]
        
        return self.rng.choice(patterns)


class ProceduralDescriptionGenerator:
    """Generates unique descriptions through template composition and variation"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Original descriptive elements
        self.sensory_details = {
            'visual': {
                'light': ['gleaming', 'shadowed', 'luminescent', 'dim', 'radiant', 'muted', 'stark', 'soft'],
                'color': ['deep crimson', 'pale azure', 'burnished gold', 'dark emerald', 'silver-touched', 
                         'ochre-stained', 'midnight blue', 'frost-white', 'amber-hued', 'obsidian black'],
                'texture': ['weathered', 'smooth', 'rough-hewn', 'polished', 'cracked', 'seamless', 
                           'patterned', 'worn', 'pristine', 'scarred'],
                'form': ['towering', 'sprawling', 'compact', 'twisted', 'geometric', 'organic',
                        'angular', 'curved', 'fractured', 'unified']
            },
            'auditory': {
                'volume': ['thundering', 'whispering', 'silent', 'humming', 'roaring', 'murmuring'],
                'quality': ['melodic', 'discordant', 'rhythmic', 'chaotic', 'resonant', 'hollow'],
                'source': ['echoing', 'distant', 'immediate', 'surrounding', 'fading', 'persistent']
            },
            'atmospheric': {
                'temperature': ['scorching', 'frigid', 'mild', 'oppressive', 'crisp', 'stifling'],
                'moisture': ['arid', 'humid', 'misty', 'parched', 'damp', 'saturated'],
                'air': ['stale', 'fresh', 'charged', 'heavy', 'thin', 'thick']
            }
        }
        
        self.emotional_tones = {
            'ominous': ['foreboding', 'menacing', 'unsettling', 'threatening', 'dire'],
            'peaceful': ['serene', 'tranquil', 'calming', 'restful', 'harmonious'],
            'mysterious': ['enigmatic', 'cryptic', 'arcane', 'veiled', 'obscure'],
            'majestic': ['grand', 'imposing', 'magnificent', 'awe-inspiring', 'regal'],
            'desolate': ['barren', 'forsaken', 'abandoned', 'empty', 'lifeless'],
            'vibrant': ['thriving', 'bustling', 'energetic', 'lively', 'dynamic']
        }
        
        self.action_verbs = {
            'environment': ['stretches', 'rises', 'descends', 'spreads', 'dominates', 'emerges',
                           'crumbles', 'towers', 'winds', 'plunges'],
            'elements': ['swirl', 'cascade', 'drift', 'surge', 'shimmer', 'pulse',
                        'flicker', 'flow', 'gather', 'disperse'],
            'presence': ['lingers', 'permeates', 'haunts', 'fills', 'saturates', 
                        'emanates', 'radiates', 'seeps', 'overwhelms', 'touches']
        }
    
    def generate_environment_description(self, 
                                         terrain_type: str,
                                         mood: str = 'mysterious',
                                         detail_level: int = 3) -> str:
        """Generate a rich environment description"""
        
        sentences = []
        
        # Opening - establish the space
        visual = self.sensory_details['visual']
        light = self.rng.choice(visual['light'])
        form = self.rng.choice(visual['form'])
        action = self.rng.choice(self.action_verbs['environment'])
        
        openings = [
            f"The {terrain_type} {action} before you, {light} and {form}.",
            f"A {form} expanse of {terrain_type} {action} in {light} grandeur.",
            f"{light.capitalize()} shadows play across the {form} {terrain_type} that {action} endlessly."
        ]
        sentences.append(self.rng.choice(openings))
        
        # Add atmospheric details based on detail level
        if detail_level >= 2:
            atmo = self.sensory_details['atmospheric']
            temp = self.rng.choice(atmo['temperature'])
            air = self.rng.choice(atmo['air'])
            
            atmo_sentences = [
                f"The air hangs {temp} and {air}.",
                f"A {temp} presence fills the {air} atmosphere.",
                f"You sense the {temp} touch of {air} stillness."
            ]
            sentences.append(self.rng.choice(atmo_sentences))
        
        # Add sensory elements
        if detail_level >= 3:
            color = self.rng.choice(visual['color'])
            texture = self.rng.choice(visual['texture'])
            element_action = self.rng.choice(self.action_verbs['elements'])
            
            detail_sentences = [
                f"Surfaces of {color} and {texture} stone {element_action} in the distance.",
                f"The {texture} ground bears traces of {color} formations that {element_action} subtly.",
                f"{color.capitalize()} light catches on {texture} features, where shadows {element_action}."
            ]
            sentences.append(self.rng.choice(detail_sentences))
        
        # Emotional/mood closer
        tone_word = self.rng.choice(self.emotional_tones.get(mood, self.emotional_tones['mysterious']))
        presence = self.rng.choice(self.action_verbs['presence'])
        
        closers = [
            f"A {tone_word} quality {presence} throughout this place.",
            f"Something {tone_word} {presence} in the very essence of this {terrain_type}.",
            f"The {tone_word} nature of this realm {presence} all who enter."
        ]
        sentences.append(self.rng.choice(closers))
        
        return ' '.join(sentences)
    
    def generate_character_description(self, 
                                       role: str,
                                       personality_traits: List[str],
                                       visual_style: str = 'realistic') -> str:
        """Generate character visual and presence description"""
        
        physical_builds = ['imposing', 'lithe', 'weathered', 'youthful', 'scarred', 
                          'graceful', 'powerful', 'wiry', 'statuesque', 'unassuming']
        
        presence_qualities = ['commanding', 'subtle', 'magnetic', 'unsettling', 
                             'warm', 'cold', 'intense', 'measured', 'volatile', 'serene']
        
        eye_descriptions = ['piercing', 'knowing', 'haunted', 'bright', 'shadowed',
                           'calculating', 'kind', 'fierce', 'distant', 'searching']
        
        build = self.rng.choice(physical_builds)
        presence = self.rng.choice(presence_qualities)
        eyes = self.rng.choice(eye_descriptions)
        
        trait_text = ' and '.join(personality_traits[:2]) if personality_traits else 'enigmatic'
        
        descriptions = [
            f"A {build} figure with a {presence} presence. Their {eyes} eyes betray a {trait_text} nature.",
            f"There is something {presence} about this {build} individual. One notices their {eyes} gaze, "
            f"hinting at depths both {trait_text}.",
            f"Standing {build} and {presence}, they regard the world through {eyes} eyes. "
            f"Every movement suggests someone {trait_text}."
        ]
        
        return self.rng.choice(descriptions)


class ProceduralNumberGenerator:
    """Generates balanced numerical values for game systems"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def generate_stat_block(self, 
                            stat_names: List[str],
                            total_points: int = 100,
                            min_value: int = 5,
                            max_value: int = 30) -> Dict[str, int]:
        """Generate balanced stat distribution"""
        
        num_stats = len(stat_names)
        if num_stats == 0:
            return {}
        
        # Start with minimum values
        stats = {name: min_value for name in stat_names}
        remaining = total_points - (min_value * num_stats)
        
        # Distribute remaining points with weighted randomness
        while remaining > 0:
            stat = self.rng.choice(stat_names)
            if stats[stat] < max_value:
                add = min(self.rng.randint(1, 5), remaining, max_value - stats[stat])
                stats[stat] += add
                remaining -= add
        
        return stats
    
    def generate_curve(self,
                       start: float,
                       end: float,
                       points: int,
                       curve_type: str = 'linear') -> List[float]:
        """Generate progression curves"""
        
        if points <= 1:
            return [start]
        
        values = []
        
        for i in range(points):
            t = i / (points - 1)
            
            if curve_type == 'linear':
                value = start + (end - start) * t
            elif curve_type == 'ease_in':
                value = start + (end - start) * (t * t)
            elif curve_type == 'ease_out':
                value = start + (end - start) * (1 - (1 - t) ** 2)
            elif curve_type == 'ease_in_out':
                if t < 0.5:
                    value = start + (end - start) * (2 * t * t)
                else:
                    value = start + (end - start) * (1 - (-2 * t + 2) ** 2 / 2)
            elif curve_type == 'exponential':
                value = start * ((end / start) ** t) if start > 0 else end * t
            elif curve_type == 'logarithmic':
                value = start + (end - start) * (math.log(1 + t * 9) / math.log(10))
            else:
                value = start + (end - start) * t
            
            values.append(round(value, 2))
        
        return values
    
    def generate_loot_table(self,
                            item_count: int,
                            rarity_weights: Dict[str, float] = None) -> List[Dict]:
        """Generate weighted loot distribution"""
        
        if rarity_weights is None:
            rarity_weights = {
                'common': 0.50,
                'uncommon': 0.25,
                'rare': 0.15,
                'epic': 0.07,
                'legendary': 0.025,
                'mythic': 0.005
            }
        
        loot_table = []
        
        for i in range(item_count):
            # Weighted random selection
            roll = self.rng.random()
            cumulative = 0
            selected_rarity = 'common'
            
            for rarity, weight in rarity_weights.items():
                cumulative += weight
                if roll <= cumulative:
                    selected_rarity = rarity
                    break
            
            # Generate drop chance based on rarity
            rarity_drop_ranges = {
                'common': (0.10, 0.50),
                'uncommon': (0.05, 0.20),
                'rare': (0.02, 0.10),
                'epic': (0.005, 0.05),
                'legendary': (0.001, 0.02),
                'mythic': (0.0001, 0.005)
            }
            
            drop_range = rarity_drop_ranges[selected_rarity]
            drop_chance = self.rng.uniform(drop_range[0], drop_range[1])
            
            loot_table.append({
                'slot': i,
                'rarity': selected_rarity,
                'drop_chance': round(drop_chance, 4),
                'quantity_range': [1, max(1, 5 - list(rarity_weights.keys()).index(selected_rarity))]
            })
        
        return loot_table


class ProceduralGraphGenerator:
    """Generates graph structures for quests, dialogues, and world connections"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def generate_branching_structure(self,
                                     depth: int,
                                     min_branches: int = 1,
                                     max_branches: int = 3,
                                     convergence_chance: float = 0.3) -> Dict[str, Any]:
        """Generate branching narrative/quest structures"""
        
        nodes = {}
        edges = []
        
        def create_node(node_id: str, level: int, parent_id: str = None):
            nodes[node_id] = {
                'id': node_id,
                'level': level,
                'type': 'branch' if level < depth else 'leaf',
                'parent': parent_id
            }
            
            if parent_id:
                edges.append({'from': parent_id, 'to': node_id})
        
        # Create root
        create_node('root', 0)
        
        # Track nodes at each level for potential convergence
        levels = {0: ['root']}
        
        for level in range(1, depth + 1):
            levels[level] = []
            
            for parent_id in levels[level - 1]:
                num_branches = self.rng.randint(min_branches, max_branches)
                
                for b in range(num_branches):
                    # Check for convergence with existing node at this level
                    if levels[level] and self.rng.random() < convergence_chance:
                        # Connect to existing node
                        target = self.rng.choice(levels[level])
                        edges.append({'from': parent_id, 'to': target})
                    else:
                        # Create new node
                        node_id = f"node_{level}_{len(levels[level])}"
                        create_node(node_id, level, parent_id)
                        levels[level].append(node_id)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'depth': depth,
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }
    
    def generate_world_graph(self,
                             num_regions: int,
                             connectivity: float = 0.3) -> Dict[str, Any]:
        """Generate world region connectivity graph"""
        
        regions = [f"region_{i}" for i in range(num_regions)]
        connections = []
        
        # Ensure connected graph with spanning tree
        connected = [regions[0]]
        unconnected = regions[1:]
        
        while unconnected:
            from_region = self.rng.choice(connected)
            to_region = self.rng.choice(unconnected)
            
            connections.append({
                'from': from_region,
                'to': to_region,
                'type': 'path',
                'difficulty': self.rng.uniform(0.1, 1.0)
            })
            
            connected.append(to_region)
            unconnected.remove(to_region)
        
        # Add additional connections based on connectivity parameter
        max_additional = int(num_regions * (num_regions - 1) / 2 * connectivity)
        
        for _ in range(max_additional):
            from_region = self.rng.choice(regions)
            to_region = self.rng.choice(regions)
            
            if from_region != to_region:
                # Check if connection doesn't exist
                existing = any(
                    (c['from'] == from_region and c['to'] == to_region) or
                    (c['from'] == to_region and c['to'] == from_region)
                    for c in connections
                )
                
                if not existing:
                    connection_types = ['path', 'hidden', 'dangerous', 'one_way', 'conditional']
                    connections.append({
                        'from': from_region,
                        'to': to_region,
                        'type': self.rng.choice(connection_types),
                        'difficulty': self.rng.uniform(0.1, 1.0)
                    })
        
        return {
            'regions': regions,
            'connections': connections,
            'total_regions': num_regions,
            'total_connections': len(connections)
        }