"""
Level Design Generator
Generates level layouts, encounters, and spatial designs
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralGraphGenerator,
    ProceduralNumberGenerator
)
from schemas.data_schemas import LevelLayout

class LevelDesignGenerator:
    """Generates comprehensive level designs and layouts for AAA games"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.graph_gen = ProceduralGraphGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)
        
        # Level design building blocks
        self.level_types = {
            'dungeon': {
                'themes': ['ancient_ruins', 'underground_lair', 'magical_tower', 'forgotten_temple'],
                'room_types': ['entrance', 'corridor', 'chamber', 'junction', 'dead_end', 'boss_room'],
                'features': ['traps', 'treasures', 'puzzles', 'encounters', 'hazards']
            },
            'outdoor': {
                'themes': ['forest_clearing', 'mountain_pass', 'desert_oasis', 'swamp_village'],
                'room_types': ['clearing', 'path', 'camp', 'ruins', 'water_source', 'overlook'],
                'features': ['landmarks', 'resources', 'ambushes', 'shelters', 'viewpoints']
            },
            'urban': {
                'themes': ['city_streets', 'noble_district', 'slums', 'market_district'],
                'room_types': ['street', 'building', 'alley', 'square', 'gate', 'tower'],
                'features': ['shops', 'crowds', 'guards', 'secrets', 'events']
            },
            'mixed': {
                'themes': ['transitional_zone', 'border_region', 'contested_area', 'neutral_ground'],
                'room_types': ['checkpoint', 'no_mans_land', 'refuge', 'battlefield', 'sanctuary'],
                'features': ['objectives', 'hazards', 'opportunities', 'threats', 'resources']
            }
        }
        
        self.encounter_types = [
            'combat', 'puzzle', 'exploration', 'social', 'environmental',
            'stealth', 'survival', 'racing', 'collection', 'defense'
        ]
        
        self.difficulty_progression = [
            'tutorial', 'easy', 'normal', 'hard', 'expert', 'nightmare'
        ]
        
        self.pacing_elements = [
            'build_up', 'action_peak', 'breather', 'twist', 'climax', 'resolution'
        ]
    
    def generate_level(self,
                       level_type: str = 'dungeon',
                       complexity: int = 5,
                       target_difficulty: str = 'normal') -> Dict[str, Any]:
        """Generate a complete level layout"""
        
        level_id = str(uuid.uuid4())
        
        # Generate basic structure
        structure = self._generate_level_structure(level_type, complexity)
        
        # Generate layout graph
        layout_graph = self.graph_gen.generate_level_graph(
            num_rooms=structure['room_count'],
            connectivity=0.3 + complexity * 0.1,
            branching_factor=2 + complexity // 2
        )
        
        # Generate rooms
        rooms = []
        for i in range(structure['room_count']):
            room = self._generate_room(
                i, level_type, structure['theme'],
                layout_graph['nodes'].get(str(i), {})
            )
            rooms.append(room)
        
        # Generate encounters and events
        encounters = self._generate_encounters(
            rooms, complexity, target_difficulty
        )
        
        # Generate navigation and flow
        navigation = self._generate_navigation(rooms, layout_graph)
        
        # Generate environmental storytelling
        environmental_storytelling = self._generate_environmental_storytelling(
            rooms, structure['theme']
        )
        
        # Generate difficulty curve
        difficulty_curve = self.num_gen.generate_curve(
            start=0.1,
            end=0.9,
            points=len(rooms),
            curve_type='ease_in_out'
        )
        
        # Generate pacing
        pacing = self._generate_pacing(len(rooms), complexity)
        
        level_data = {
            'id': level_id,
            'name': self._generate_level_name(level_type, structure['theme']),
            'level_type': level_type,
            'theme': structure['theme'],
            'complexity_rating': complexity,
            'target_difficulty': target_difficulty,
            'dimensions': structure['dimensions'],
            'rooms': rooms,
            'layout_graph': layout_graph,
            'encounters': encounters,
            'navigation': navigation,
            'environmental_storytelling': environmental_storytelling,
            'difficulty_curve': difficulty_curve,
            'pacing': pacing,
            'estimated_completion_time': self._estimate_completion_time(complexity, len(rooms)),
            'accessibility_features': self._generate_accessibility_features(),
            'replayability_elements': self._generate_replayability_elements(complexity)
        }
        
        return level_data
    
    def _generate_level_structure(self, level_type: str, complexity: int) -> Dict[str, Any]:
        """Generate the basic structure of the level"""
        
        type_info = self.level_types.get(level_type, self.level_types['dungeon'])
        
        # Determine room count based on complexity
        room_counts = {
            1: (3, 5),    # Tutorial
            2: (5, 8),    # Simple
            3: (8, 12),   # Basic
            4: (12, 18),  # Intermediate
            5: (18, 25),  # Standard
            6: (25, 35),  # Complex
            7: (35, 50),  # Advanced
            8: (50, 70),  # Expert
            9: (70, 100), # Master
            10: (100, 150) # Legendary
        }
        
        min_rooms, max_rooms = room_counts.get(complexity, room_counts[5])
        room_count = self.rng.randint(min_rooms, max_rooms)
        
        # Select theme
        theme = self.rng.choice(type_info['themes'])
        
        # Generate dimensions
        size_multiplier = 1 + (complexity - 1) * 0.5
        dimensions = {
            'width': round(self.rng.uniform(50, 200) * size_multiplier, 1),
            'height': round(self.rng.uniform(20, 100) * size_multiplier, 1),
            'depth': round(self.rng.uniform(10, 50) * size_multiplier, 1)
        }
        
        return {
            'room_count': room_count,
            'theme': theme,
            'dimensions': dimensions,
            'type_info': type_info
        }
    
    def _generate_room(self,
                       room_index: int,
                       level_type: str,
                       theme: str,
                       node_data: Dict) -> Dict[str, Any]:
        """Generate a single room"""
        
        type_info = self.level_types[level_type]
        room_type = self.rng.choice(type_info['room_types'])
        
        # Determine room size and shape
        size_category = self.rng.choice(['small', 'medium', 'large', 'huge'])
        size_multipliers = {
            'small': 0.5,
            'medium': 1.0,
            'large': 1.5,
            'huge': 2.0
        }
        
        base_size = size_multipliers[size_category]
        dimensions = {
            'width': round(self.rng.uniform(5, 15) * base_size, 1),
            'height': round(self.rng.uniform(3, 8) * base_size, 1),
            'depth': round(self.rng.uniform(2, 6) * base_size, 1)
        }
        
        # Generate room features
        features = self._generate_room_features(room_type, level_type, theme)
        
        # Generate interactive elements
        interactive_elements = self._generate_interactive_elements(room_type, features)
        
        # Generate ambiance
        ambiance = self._generate_room_ambiance(room_type, theme)
        
        # Determine connectivity
        connections = node_data.get('connections', [])
        
        room = {
            'id': str(uuid.uuid4()),
            'index': room_index,
            'name': self._generate_room_name(room_type, theme),
            'type': room_type,
            'size_category': size_category,
            'dimensions': dimensions,
            'features': features,
            'interactive_elements': interactive_elements,
            'ambiance': ambiance,
            'connections': connections,
            'difficulty_modifier': self.rng.uniform(0.5, 1.5),
            'exploration_value': self.rng.uniform(0.1, 1.0),
            'narrative_significance': self.rng.randint(1, 10)
        }
        
        return room
    
    def _generate_room_features(self,
                                room_type: str,
                                level_type: str,
                                theme: str) -> List[Dict]:
        """Generate features for a room"""
        
        features = []
        type_info = self.level_types[level_type]
        
        # Base features based on room type
        base_features = {
            'entrance': ['welcome_mat', 'entryway_decorations', 'orientation_clues'],
            'corridor': ['lighting', 'wall_decorations', 'floor_patterns'],
            'chamber': ['central_feature', 'side_rooms', 'architectural_details'],
            'junction': ['directional_signs', 'rest_area', 'multiple_paths'],
            'dead_end': ['hidden_secrets', 'treasure_cache', 'environmental_hazard'],
            'boss_room': ['impressive_architecture', 'boss_mechanic_triggers', 'spectacle_elements']
        }
        
        base_feature_list = base_features.get(room_type, ['basic_architecture'])
        
        for feature_name in base_feature_list:
            feature = {
                'name': feature_name,
                'type': self.rng.choice(['architectural', 'environmental', 'interactive', 'narrative']),
                'description': f"A {feature_name.replace('_', ' ')} that enhances the room's atmosphere",
                'interactivity': self.rng.choice(['none', 'examinable', 'usable', 'destructible']),
                'importance': self.rng.choice(['cosmetic', 'functional', 'narrative', 'gameplay'])
            }
            features.append(feature)
        
        # Add random additional features
        extra_features = self.rng.randint(0, 3)
        for _ in range(extra_features):
            feature = {
                'name': self.rng.choice(type_info['features']),
                'type': 'dynamic',
                'description': f"A dynamic {self.rng.choice(type_info['features'])} element",
                'interactivity': 'high',
                'importance': 'gameplay'
            }
            features.append(feature)
        
        return features
    
    def _generate_interactive_elements(self,
                                       room_type: str,
                                       features: List[Dict]) -> List[Dict]:
        """Generate interactive elements for the room"""
        
        elements = []
        
        # Base interactive elements
        base_elements = {
            'entrance': ['door', 'lever', 'switch'],
            'corridor': ['chest', 'lever', 'hidden_button'],
            'chamber': ['altar', 'pedestal', 'mechanism'],
            'junction': ['signpost', 'map', 'rest_bench'],
            'dead_end': ['secret_door', 'treasure_chest', 'puzzle_lock'],
            'boss_room': ['boss_summoner', 'arena_controls', 'spectator_seats']
        }
        
        element_names = base_elements.get(room_type, ['generic_interactive'])
        
        for element_name in element_names:
            element = {
                'name': element_name,
                'type': self.rng.choice(['switch', 'container', 'device', 'decoration']),
                'function': self.rng.choice(['unlock', 'reveal', 'activate', 'deactivate', 'reward']),
                'requirements': self._generate_requirements(),
                'consequences': self._generate_consequences(),
                'cooldown': self.rng.randint(0, 30),  # seconds
                'durability': self.rng.randint(1, 10)
            }
            elements.append(element)
        
        return elements
    
    def _generate_room_ambiance(self, room_type: str, theme: str) -> Dict[str, Any]:
        """Generate room ambiance settings"""
        
        return {
            'lighting': {
                'primary_source': self.rng.choice(['torch', 'crystal', 'sunlight', 'magic', 'bioluminescent']),
                'intensity': self.rng.choice(['dim', 'normal', 'bright', 'dramatic']),
                'color_temperature': self.rng.choice(['warm', 'cool', 'neutral', 'colored']),
                'flickering': self.rng.random() > 0.7
            },
            'soundscape': {
                'background': self.rng.choice(['echoing', 'muffled', 'resonant', 'dead']),
                'ambient_sounds': self.rng.sample([
                    'dripping_water', 'creaking_wood', 'wind_howl', 'distant_echoes',
                    'machinery_hum', 'creature_calls', 'magical_hum', 'silence'
                ], self.rng.randint(1, 3)),
                'volume': self.rng.choice(['quiet', 'normal', 'loud', 'deafening'])
            },
            'atmosphere': {
                'temperature': self.rng.choice(['freezing', 'cold', 'normal', 'warm', 'hot']),
                'humidity': self.rng.choice(['dry', 'normal', 'humid', 'oppressive']),
                'air_quality': self.rng.choice(['fresh', 'stale', 'toxic', 'magical']),
                'special_effects': self.rng.sample([
                    'dust_motes', 'floating_particles', 'color_shifts', 'shadows',
                    'aurora', 'mist', 'sparks', 'echoes'
                ], self.rng.randint(0, 2))
            },
            'emotional_tone': self.rng.choice([
                'foreboding', 'peaceful', 'mysterious', 'ominous', 'majestic',
                'claustrophobic', 'open', 'intimate', 'grand', 'intimidating'
            ])
        }
    
    def _generate_encounters(self,
                             rooms: List[Dict],
                             complexity: int,
                             target_difficulty: str) -> List[Dict]:
        """Generate encounters for the level"""
        
        encounters = []
        
        # Determine number of encounters
        encounter_count = max(1, len(rooms) // 3 + complexity // 2)
        
        # Select rooms for encounters
        encounter_rooms = self.rng.sample(rooms, min(encounter_count, len(rooms)))
        
        for i, room in enumerate(encounter_rooms):
            encounter = self._generate_encounter(
                i, room, complexity, target_difficulty
            )
            encounters.append(encounter)
        
        return encounters
    
    def _generate_encounter(self,
                            encounter_index: int,
                            room: Dict,
                            complexity: int,
                            target_difficulty: str) -> Dict[str, Any]:
        """Generate a single encounter"""
        
        encounter_type = self.rng.choice(self.encounter_types)
        
        # Scale difficulty
        difficulty_modifier = self.difficulty_progression.index(target_difficulty) / 5.0
        encounter_difficulty = min(1.0, difficulty_modifier + self.rng.uniform(-0.2, 0.2))
        
        # Generate encounter elements
        elements = self._generate_encounter_elements(encounter_type, complexity)
        
        # Generate objectives
        objectives = self._generate_encounter_objectives(encounter_type)
        
        # Generate rewards
        rewards = self._generate_encounter_rewards(encounter_difficulty, complexity)
        
        return {
            'id': str(uuid.uuid4()),
            'index': encounter_index,
            'type': encounter_type,
            'room_id': room['id'],
            'difficulty': round(encounter_difficulty, 2),
            'elements': elements,
            'objectives': objectives,
            'rewards': rewards,
            'time_limit': self.rng.randint(30, 300),  # seconds
            'failure_conditions': self._generate_failure_conditions(encounter_type),
            'success_conditions': self._generate_success_conditions(encounter_type),
            'scaling_factors': self._generate_scaling_factors()
        }
    
    def _generate_encounter_elements(self,
                                     encounter_type: str,
                                     complexity: int) -> List[Dict]:
        """Generate elements for an encounter"""
        
        elements = []
        
        element_counts = {
            'combat': {'enemies': complexity + 1, 'obstacles': complexity // 2},
            'puzzle': {'puzzle_pieces': complexity + 2, 'hints': complexity},
            'exploration': {'secrets': complexity, 'hazards': complexity // 2},
            'social': {'npcs': 2 + complexity // 2, 'topics': complexity},
            'environmental': {'hazards': complexity + 1, 'resources': complexity // 2}
        }
        
        counts = element_counts.get(encounter_type, {'generic': complexity})
        
        for element_type, count in counts.items():
            for _ in range(count):
                element = {
                    'type': element_type,
                    'name': f"{element_type}_{self.rng.randint(1, 1000)}",
                    'properties': self._generate_element_properties(element_type),
                    'behavior': self._generate_element_behavior(element_type)
                }
                elements.append(element)
        
        return elements
    
    def _generate_navigation(self,
                             rooms: List[Dict],
                             layout_graph: Dict) -> Dict[str, Any]:
        """Generate navigation and flow information"""
        
        return {
            'main_path': self._find_main_path(layout_graph),
            'shortcuts': self._find_shortcuts(layout_graph),
            'secrets': self._find_secret_areas(layout_graph),
            'choke_points': self._identify_choke_points(rooms),
            'fast_travel_points': self._generate_fast_travel_points(rooms),
            'navigation_hints': self._generate_navigation_hints(rooms)
        }
    
    def _generate_environmental_storytelling(self,
                                             rooms: List[Dict],
                                             theme: str) -> List[Dict]:
        """Generate environmental storytelling elements"""
        
        storytelling_elements = []
        
        for room in rooms:
            if self.rng.random() > 0.6:  # 40% chance per room
                element = {
                    'room_id': room['id'],
                    'type': self.rng.choice(['lore_fragment', 'character_echo', 'world_building', 'foreshadowing']),
                    'content': f"A {self.rng.choice(['faded', 'carved', 'whispered', 'projected'])} message about {theme}",
                    'delivery_method': self.rng.choice(['inscription', 'recording', 'vision', 'ambient_sound']),
                    'significance': self.rng.choice(['minor', 'moderate', 'major', 'critical'])
                }
                storytelling_elements.append(element)
        
        return storytelling_elements
    
    def _generate_pacing(self, room_count: int, complexity: int) -> List[Dict]:
        """Generate pacing structure for the level"""
        
        pacing_segments = []
        segment_count = max(3, room_count // 5)
        
        for i in range(segment_count):
            segment = {
                'segment_number': i + 1,
                'rooms': [],  # Would be populated with actual room IDs
                'pacing_type': self.rng.choice(self.pacing_elements),
                'intensity_level': round(self.rng.uniform(0.1, 1.0), 2),
                'duration_estimate': self.rng.randint(5, 20),  # minutes
                'key_moments': self._generate_key_moments()
            }
            pacing_segments.append(segment)
        
        return pacing_segments
    
    def _generate_level_name(self, level_type: str, theme: str) -> str:
        """Generate evocative level name"""
        
        name_patterns = [
            "The {theme} {type}",
            "{theme} of {concept}",
            "Chamber of {theme}",
            "Hall of {concept}",
            "{adjective} {theme}",
            "The {concept} {type}"
        ]
        
        pattern = self.rng.choice(name_patterns)
        
        concepts = ['Shadows', 'Light', 'Echoes', 'Whispers', 'Forgotten', 'Ancient', 'Lost', 'Hidden']
        adjectives = ['Dark', 'Forgotten', 'Ancient', 'Mysterious', 'Dangerous', 'Sacred', 'Cursed', 'Blessed']
        
        return pattern.format(
            theme=theme.replace('_', ' ').title(),
            type=level_type.title(),
            concept=self.rng.choice(concepts),
            adjective=self.rng.choice(adjectives)
        )
    
    def _generate_room_name(self, room_type: str, theme: str) -> str:
        """Generate room name"""
        
        return f"{room_type.replace('_', ' ').title()} of {theme.replace('_', ' ').title()}"
    
    def _generate_requirements(self) -> List[str]:
        """Generate requirements for interactive elements"""
        
        requirements = [
            'no_requirements',
            'key_required',
            'skill_check',
            'item_needed',
            'puzzle_solved',
            'quest_progress'
        ]
        
        return self.rng.sample(requirements, self.rng.randint(1, 2))
    
    def _generate_consequences(self) -> List[str]:
        """Generate consequences for interactive elements"""
        
        consequences = [
            'door_opens',
            'trap_activates',
            'reward_granted',
            'enemy_spawns',
            'area_changes',
            'narrative_progresses'
        ]
        
        return self.rng.sample(consequences, self.rng.randint(1, 3))
    
    def _generate_encounter_objectives(self, encounter_type: str) -> List[str]:
        """Generate objectives for encounters"""
        
        objectives = {
            'combat': ['defeat_all_enemies', 'survive_duration', 'protect_target'],
            'puzzle': ['solve_puzzle', 'find_solution', 'unlock_mechanism'],
            'exploration': ['discover_secret', 'map_area', 'find_item'],
            'social': ['convince_npc', 'gather_information', 'resolve_conflict'],
            'environmental': ['survive_hazard', 'reach_safety', 'utilize_environment']
        }
        
        return objectives.get(encounter_type, ['complete_task'])
    
    def _generate_encounter_rewards(self, difficulty: float, complexity: int) -> Dict[str, Any]:
        """Generate rewards for encounters"""
        
        base_reward = difficulty * complexity * 10
        
        return {
            'experience': int(base_reward),
            'currency': int(base_reward * 0.5),
            'items': self.rng.sample(['weapon', 'armor', 'consumable', 'material'], self.rng.randint(0, 2)),
            'unlocks': [] if self.rng.random() > 0.8 else ['new_ability'],
            'narrative_progress': self.rng.random() > 0.7
        }
    
    def _generate_failure_conditions(self, encounter_type: str) -> List[str]:
        """Generate failure conditions"""
        
        conditions = {
            'combat': ['player_death', 'time_expires', 'target_destroyed'],
            'puzzle': ['time_expires', 'too_many_failures', 'wrong_solution'],
            'exploration': ['time_expires', 'detection_failed', 'area_collapsed'],
            'social': ['npc_killed', 'relationship_broken', 'time_expires'],
            'environmental': ['player_death', 'environmental_failure', 'time_expires']
        }
        
        return conditions.get(encounter_type, ['task_failed'])
    
    def _generate_success_conditions(self, encounter_type: str) -> List[str]:
        """Generate success conditions"""
        
        conditions = {
            'combat': ['all_enemies_defeated', 'survival_duration_met'],
            'puzzle': ['correct_solution_found', 'mechanism_unlocked'],
            'exploration': ['secret_discovered', 'area_mapped'],
            'social': ['npc_convinced', 'information_gathered'],
            'environmental': ['hazard_survived', 'safety_reached']
        }
        
        return conditions.get(encounter_type, ['task_completed'])
    
    def _generate_scaling_factors(self) -> Dict[str, float]:
        """Generate scaling factors for difficulty adjustment"""
        
        return {
            'player_count': 1.0 + self.rng.uniform(-0.2, 0.5),
            'difficulty_level': 1.0 + self.rng.uniform(-0.3, 0.7),
            'time_pressure': 1.0 + self.rng.uniform(-0.1, 0.3),
            'resource_availability': 1.0 + self.rng.uniform(-0.2, 0.2)
        }
    
    def _generate_element_properties(self, element_type: str) -> Dict[str, Any]:
        """Generate properties for encounter elements"""
        
        properties = {
            'enemies': {
                'health': self.rng.randint(50, 200),
                'damage': self.rng.randint(10, 50),
                'behavior': self.rng.choice(['aggressive', 'defensive', 'tactical'])
            },
            'puzzle_pieces': {
                'complexity': self.rng.randint(1, 5),
                'hints_available': self.rng.randint(0, 3),
                'solution_type': self.rng.choice(['pattern', 'logic', 'memory'])
            },
            'secrets': {
                'discovery_difficulty': self.rng.uniform(0.1, 1.0),
                'reward_value': self.rng.uniform(0.1, 1.0),
                'type': self.rng.choice(['treasure', 'information', 'shortcut'])
            }
        }
        
        return properties.get(element_type, {'generic': True})
    
    def _generate_element_behavior(self, element_type: str) -> str:
        """Generate behavior for encounter elements"""
        
        behaviors = {
            'enemies': 'hostile',
            'puzzle_pieces': 'static',
            'secrets': 'hidden',
            'hazards': 'dangerous',
            'resources': 'valuable'
        }
        
        return behaviors.get(element_type, 'neutral')
    
    def _find_main_path(self, layout_graph: Dict) -> List[str]:
        """Find the main path through the level"""
        
        # Simplified pathfinding - in practice would use proper algorithms
        nodes = list(layout_graph.get('nodes', {}).keys())
        if len(nodes) > 1:
            return [nodes[0], nodes[-1]]  # Start to end
        return nodes
    
    def _find_shortcuts(self, layout_graph: Dict) -> List[Dict]:
        """Find shortcut paths"""
        
        shortcuts = []
        connections = layout_graph.get('connections', [])
        
        for conn in connections:
            if self.rng.random() > 0.8:  # 20% of connections are shortcuts
                shortcuts.append({
                    'start': conn['from'],
                    'end': conn['to'],
                    'difficulty': self.rng.choice(['easy', 'medium', 'hard']),
                    'risk': self.rng.choice(['low', 'medium', 'high'])
                })
        
        return shortcuts
    
    def _find_secret_areas(self, layout_graph: Dict) -> List[Dict]:
        """Find secret areas"""
        
        secrets = []
        nodes = layout_graph.get('nodes', {})
        
        for node_id, node_data in nodes.items():
            if self.rng.random() > 0.9:  # 10% chance per node
                secrets.append({
                    'location': node_id,
                    'type': self.rng.choice(['hidden_room', 'secret_passage', 'concealed_cache']),
                    'discovery_method': self.rng.choice(['search', 'interaction', 'skill_check']),
                    'reward': self.rng.choice(['treasure', 'shortcut', 'information'])
                })
        
        return secrets
    
    def _identify_choke_points(self, rooms: List[Dict]) -> List[Dict]:
        """Identify choke points in the level"""
        
        choke_points = []
        
        for room in rooms:
            if len(room.get('connections', [])) <= 2 and self.rng.random() > 0.7:
                choke_points.append({
                    'room_id': room['id'],
                    'type': 'narrow_passage',
                    'strategic_value': self.rng.choice(['high', 'medium', 'low']),
                    'defensibility': self.rng.uniform(0.3, 1.0)
                })
        
        return choke_points
    
    def _generate_fast_travel_points(self, rooms: List[Dict]) -> List[Dict]:
        """Generate fast travel points"""
        
        points = []
        
        for room in rooms:
            if self.rng.random() > 0.85:  # 15% chance per room
                points.append({
                    'room_id': room['id'],
                    'type': self.rng.choice(['waypoint', 'teleporter', 'portal']),
                    'activation_cost': self.rng.randint(0, 100),
                    'cooldown': self.rng.randint(0, 300)
                })
        
        return points
    
    def _generate_navigation_hints(self, rooms: List[Dict]) -> List[Dict]:
        """Generate navigation hints"""
        
        hints = []
        
        for room in rooms:
            if self.rng.random() > 0.6:  # 40% chance per room
                hints.append({
                    'room_id': room['id'],
                    'type': self.rng.choice(['signpost', 'map_fragment', 'npc_direction', 'environmental_clue']),
                    'accuracy': self.rng.uniform(0.5, 1.0),
                    'clarity': self.rng.choice(['clear', 'vague', 'misleading'])
                })
        
        return hints
    
    def _generate_key_moments(self) -> List[str]:
        """Generate key moments for pacing segments"""
        
        return self.rng.sample([
            'enemy_encounter', 'puzzle_reveal', 'treasure_discovery',
            'narrative_twist', 'boss_fight', 'rest_area', 'checkpoint'
        ], self.rng.randint(1, 3))
    
    def _estimate_completion_time(self, complexity: int, room_count: int) -> str:
        """Estimate level completion time"""
        
        base_time = room_count * 5  # 5 minutes per room average
        complexity_multiplier = 1 + (complexity - 1) * 0.3
        
        total_minutes = int(base_time * complexity_multiplier)
        
        if total_minutes < 30:
            return "Short (15-30 minutes)"
        elif total_minutes < 60:
            return "Medium (30-60 minutes)"
        elif total_minutes < 120:
            return "Long (1-2 hours)"
        else:
            return "Epic (2+ hours)"
    
    def _generate_accessibility_features(self) -> List[str]:
        """Generate accessibility features"""
        
        features = [
            'colorblind_friendly_indicators',
            'adjustable_difficulty_scaling',
            'multiple_input_options',
            'clear_navigation_cues',
            'lenient_time_limits',
            'alternative_puzzle_solutions'
        ]
        
        return self.rng.sample(features, self.rng.randint(2, 4))
    
    def _generate_replayability_elements(self, complexity: int) -> List[Dict]:
        """Generate replayability elements"""
        
        elements = []
        element_count = complexity // 2
        
        for _ in range(element_count):
            element = {
                'type': self.rng.choice(['secret_area', 'alternate_path', 'bonus_objective', 'speed_run']),
                'unlock_condition': self.rng.choice(['first_completion', 'specific_action', 'time_limit', 'no_damage']),
                'reward': self.rng.choice(['achievement', 'cosmetic', 'gameplay_bonus', 'narrative_variant'])
            }
            elements.append(element)
        
        return elements
    
    def generate_batch(self,
                       count: int,
                       level_types: List[str] = None) -> List[Dict]:
        """Generate multiple levels"""
        
        if level_types is None:
            level_types = list(self.level_types.keys())
        
        levels = []
        
        for i in range(count):
            level_type = self.rng.choice(level_types)
            complexity = self.rng.randint(3, 10)
            difficulty = self.rng.choice(self.difficulty_progression)
            
            level = self.generate_level(
                level_type=level_type,
                complexity=complexity,
                target_difficulty=difficulty
            )
            levels.append(level)
        
        return levels