"""
World Building Generator
Generates original world structures, regions, and environmental systems
"""
import random
import math
from typing import List, Dict, Any, Tuple
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralDescriptionGenerator,
    ProceduralGraphGenerator,
    ProceduralNumberGenerator
)
from schemas.data_schemas import WorldRegion, TerrainType

class WorldBuilder:
    """Generates complete world systems and geographical structures"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.graph_gen = ProceduralGraphGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)
        
        # World generation parameters
        self.climate_factors = {
            'temperature': {'range': (-40, 50), 'unit': 'celsius'},
            'humidity': {'range': (0, 100), 'unit': 'percent'},
            'precipitation': {'range': (0, 5000), 'unit': 'mm_annual'},
            'wind_intensity': {'range': (0, 100), 'unit': 'severity'},
            'magical_saturation': {'range': (0, 100), 'unit': 'percent'},
            'temporal_stability': {'range': (0, 100), 'unit': 'percent'}
        }
        
        self.resource_types = [
            'mineral_wealth', 'fertile_soil', 'fresh_water', 'timber',
            'magical_essence', 'rare_flora', 'rare_fauna', 'ancient_artifacts',
            'energy_sources', 'strategic_position'
        ]
        
        self.hazard_types = {
            'natural': ['earthquakes', 'volcanic_activity', 'floods', 'storms', 
                       'wildfires', 'avalanches', 'sinkholes'],
            'magical': ['wild_magic_zones', 'cursed_grounds', 'planar_rifts',
                       'temporal_anomalies', 'essence_storms'],
            'creature': ['predator_territories', 'swarm_regions', 'corrupted_beasts',
                        'territorial_entities'],
            'environmental': ['toxic_atmosphere', 'extreme_temperatures', 'radiation',
                             'corrosive_elements', 'void_exposure']
        }
        
        self.poi_types = [
            'ancient_ruins', 'natural_wonder', 'settlement', 'dungeon',
            'sacred_site', 'resource_deposit', 'strategic_point', 'anomaly',
            'creature_lair', 'hidden_sanctuary', 'battleground', 'monument'
        ]
    
    def generate_world(self, 
                       scale: str = 'continental',
                       num_regions: int = None) -> Dict[str, Any]:
        """Generate a complete world structure"""
        
        world_id = str(uuid.uuid4())
        
        # Determine region count based on scale
        scale_regions = {
            'room': (1, 3),
            'building': (3, 8),
            'city': (8, 20),
            'regional': (15, 40),
            'continental': (30, 80),
            'planetary': (60, 150)
        }
        
        if num_regions is None:
            min_r, max_r = scale_regions.get(scale, scale_regions['regional'])
            num_regions = self.rng.randint(min_r, max_r)
        
        # Generate world foundation
        world_concept = self._generate_world_concept()
        
        # Generate regions
        regions = []
        for i in range(num_regions):
            region = self.generate_region(world_concept, i, num_regions)
            regions.append(region)
        
        # Generate connectivity
        connectivity = self.graph_gen.generate_world_graph(
            num_regions,
            connectivity=0.2 + 0.1 * (num_regions / 50)
        )
        
        # Map connectivity to regions
        for conn in connectivity['connections']:
            from_idx = int(conn['from'].split('_')[1])
            to_idx = int(conn['to'].split('_')[1])
            
            if from_idx < len(regions) and to_idx < len(regions):
                regions[from_idx]['connected_regions'].append(regions[to_idx]['id'])
                if conn['type'] != 'one_way':
                    regions[to_idx]['connected_regions'].append(regions[from_idx]['id'])
        
        # Generate world systems
        world_systems = self._generate_world_systems(world_concept, regions)
        
        return {
            'id': world_id,
            'name': self.name_gen.generate_name('mystical', 2, 4),
            'scale': scale,
            'concept': world_concept,
            'regions': [r for r in regions],
            'region_count': num_regions,
            'connectivity': connectivity,
            'world_systems': world_systems,
            'creation_seed': self.seed
        }
    
    def _generate_world_concept(self) -> Dict[str, Any]:
        """Generate the foundational concept of the world"""
        
        world_foundations = [
            {
                'type': 'fractured_reality',
                'description': 'A world broken into floating fragments, each with its own rules',
                'unique_properties': ['variable_gravity', 'isolated_ecosystems', 'inter-fragment_travel']
            },
            {
                'type': 'layered_existence',
                'description': 'Multiple overlapping planes of reality occupying the same space',
                'unique_properties': ['planar_bleeding', 'phase_shifting', 'echo_locations']
            },
            {
                'type': 'cyclical_time',
                'description': 'A world caught in temporal loops of varying scales',
                'unique_properties': ['time_echoes', 'loop_memory', 'temporal_landmarks']
            },
            {
                'type': 'living_world',
                'description': 'The world itself is a vast conscious entity',
                'unique_properties': ['responsive_terrain', 'dream_manifestation', 'world_will']
            },
            {
                'type': 'constructed_reality',
                'description': 'An artificial world created by unknown architects',
                'unique_properties': ['hidden_mechanisms', 'purpose_fragments', 'design_patterns']
            },
            {
                'type': 'dying_world',
                'description': 'A world in the final stages of existence',
                'unique_properties': ['entropy_zones', 'last_bastions', 'memory_preservation']
            },
            {
                'type': 'nascent_world',
                'description': 'A young world still in the process of formation',
                'unique_properties': ['unstable_geography', 'primordial_forces', 'undefined_regions']
            },
            {
                'type': 'merged_worlds',
                'description': 'Multiple worlds forcibly combined into one',
                'unique_properties': ['cultural_collision', 'physics_boundaries', 'merged_histories']
            }
        ]
        
        foundation = self.rng.choice(world_foundations)
        
        # Add cosmological elements
        cosmology = {
            'origin_myth': self._generate_origin_myth(foundation['type']),
            'cosmic_forces': self._generate_cosmic_forces(),
            'world_laws': self._generate_world_laws(foundation['type'])
        }
        
        return {
            'foundation': foundation,
            'cosmology': cosmology,
            'age_category': self.rng.choice(['primordial', 'ancient', 'old', 'mature', 'young', 'new']),
            'dominant_element': self.rng.choice(['fire', 'water', 'earth', 'air', 'void', 'life', 'death', 'time', 'mind'])
        }
    
    def _generate_origin_myth(self, world_type: str) -> str:
        """Generate world origin mythology"""
        
        origins = {
            'fractured_reality': [
                "The world was once whole, shattered by a catastrophic event of immense power.",
                "Creation itself was an explosion, and the fragments are still finding their places.",
                "A great entity was divided, each fragment a piece of its being."
            ],
            'layered_existence': [
                "The planes were once one, separated by the first great conflict.",
                "Each layer is a different dream of the sleeping creator.",
                "Reality folded upon itself, creating endless reflections."
            ],
            'cyclical_time': [
                "Time broke and looped back upon itself, trapping the world.",
                "The gods disagreed on how time should flow, and all solutions exist at once.",
                "A moment of perfect significance repeats eternally with variations."
            ],
            'living_world': [
                "The world is the body of a sleeping titan, whose dreams shape reality.",
                "All that exists grew from a single seed of consciousness.",
                "The world awakened from the void, and still remembers nothing."
            ],
            'default': [
                "The world emerged from chaos through unknown processes.",
                "Creation was an accident with profound consequences.",
                "The origin is lost, known only in fragments and contradictions."
            ]
        }
        
        myths = origins.get(world_type, origins['default'])
        return self.rng.choice(myths)
    
    def _generate_cosmic_forces(self) -> List[Dict]:
        """Generate the fundamental forces of the world"""
        
        force_types = ['creation', 'destruction', 'order', 'chaos', 'change', 
                      'stasis', 'connection', 'isolation', 'truth', 'illusion']
        
        num_forces = self.rng.randint(2, 5)
        selected_forces = self.rng.sample(force_types, num_forces)
        
        forces = []
        for force_name in selected_forces:
            force = {
                'name': force_name.capitalize(),
                'aspect_name': self.name_gen.generate_name('mystical', 2, 3),
                'influence_level': self.rng.uniform(0.3, 1.0),
                'manifestation': f"The force of {force_name} manifests as {self.rng.choice(['visible phenomena', 'subtle influence', 'rare events', 'constant presence'])}",
                'opposing_force': self.rng.choice([f for f in force_types if f != force_name])
            }
            forces.append(force)
        
        return forces
    
    def _generate_world_laws(self, world_type: str) -> List[str]:
        """Generate the fundamental laws/rules of the world"""
        
        universal_laws = [
            "Energy cannot be created or destroyed, only transformed",
            "Cause precedes effect in most circumstances",
            "Consciousness shapes local reality to a limited degree",
            "All things seek equilibrium over sufficient time"
        ]
        
        type_specific_laws = {
            'fractured_reality': [
                "Each fragment maintains its own gravitational center",
                "Inter-fragment travel requires specific conditions",
                "Fragment proximity affects shared properties"
            ],
            'layered_existence': [
                "Layers interact based on resonance",
                "Physical laws may vary between planes",
                "Strong emotions can cause planar bleeding"
            ],
            'cyclical_time': [
                "Loops may be broken under specific conditions",
                "Memory persists across cycles with increasing difficulty",
                "Actions have consequences across iterations"
            ],
            'living_world': [
                "The world responds to collective belief",
                "Geographic features may shift based on need",
                "Harming the world has direct consequences"
            ]
        }
        
        laws = universal_laws.copy()
        laws.extend(type_specific_laws.get(world_type, []))
        
        return self.rng.sample(laws, min(len(laws), 5))
    
    def generate_region(self, 
                        world_concept: Dict,
                        region_index: int,
                        total_regions: int) -> Dict[str, Any]:
        """Generate a single region"""
        
        # Determine terrain type
        terrain_type = self.rng.choice(list(TerrainType))
        
        # Generate climate profile
        climate_profile = {}
        for factor, props in self.climate_factors.items():
            min_val, max_val = props['range']
            climate_profile[factor] = round(self.rng.uniform(min_val, max_val), 2)
        
        # Adjust climate based on terrain
        climate_profile = self._adjust_climate_for_terrain(climate_profile, terrain_type)
        
        # Generate resources
        num_resources = self.rng.randint(2, 6)
        selected_resources = self.rng.sample(self.resource_types, num_resources)
        resource_distribution = {
            res: round(self.rng.uniform(0.1, 1.0), 2) 
            for res in selected_resources
        }
        
        # Generate hazards
        hazard_level = self.rng.uniform(0.0, 1.0)
        hazards = self._generate_hazards(hazard_level)
        
        # Generate points of interest
        num_pois = self.rng.randint(1, 5 + int(hazard_level * 3))
        pois = [self._generate_poi(terrain_type) for _ in range(num_pois)]
        
        # Generate ambient properties
        ambient = self._generate_ambient_properties(terrain_type, climate_profile)
        
        # Determine narrative significance
        narrative_significance = self.rng.randint(1, 10)
        
        region = WorldRegion(
            name=self.name_gen.generate_place_name(terrain_type.value, 
                                                   self._get_cultural_profile(world_concept)),
            description=self.desc_gen.generate_environment_description(
                terrain_type.value,
                self._get_mood_for_region(hazard_level, narrative_significance),
                detail_level=3
            ),
            terrain_type=terrain_type,
            climate_profile=climate_profile,
            resource_distribution=resource_distribution,
            hazard_level=round(hazard_level, 2),
            discovery_state=self.rng.choice(['hidden', 'rumored', 'known', 'explored', 'mapped']),
            connected_regions=[],  # Will be populated later
            points_of_interest=pois,
            ambient_properties=ambient,
            narrative_significance=narrative_significance
        )
        
        region_dict = region.to_dict()
        region_dict['hazards'] = hazards
        region_dict['region_index'] = region_index
        
        return region_dict
    
    def _adjust_climate_for_terrain(self, climate: Dict, terrain: TerrainType) -> Dict:
        """Adjust climate values based on terrain type"""
        
        terrain_adjustments = {
            TerrainType.DESERT: {'humidity': -50, 'precipitation': -3000, 'temperature': 15},
            TerrainType.AQUATIC: {'humidity': 30, 'precipitation': 1000},
            TerrainType.MOUNTAINS: {'temperature': -15, 'wind_intensity': 20},
            TerrainType.FOREST: {'humidity': 20, 'precipitation': 500},
            TerrainType.SUBTERRANEAN: {'wind_intensity': -50, 'temperature': -10},
            TerrainType.AERIAL: {'wind_intensity': 40, 'temperature': -20},
            TerrainType.VOID: {'temporal_stability': -50, 'magical_saturation': 30}
        }
        
        adjustments = terrain_adjustments.get(terrain, {})
        adjusted = climate.copy()
        
        for factor, adjustment in adjustments.items():
            if factor in adjusted:
                min_val, max_val = self.climate_factors[factor]['range']
                adjusted[factor] = max(min_val, min(max_val, adjusted[factor] + adjustment))
        
        return adjusted
    
    def _generate_hazards(self, hazard_level: float) -> List[Dict]:
        """Generate region hazards"""
        
        hazards = []
        num_hazards = int(hazard_level * 5) + self.rng.randint(0, 2)
        
        hazard_categories = list(self.hazard_types.keys())
        
        for _ in range(num_hazards):
            category = self.rng.choice(hazard_categories)
            hazard_name = self.rng.choice(self.hazard_types[category])
            
            hazards.append({
                'name': hazard_name,
                'category': category,
                'severity': round(self.rng.uniform(0.3, 1.0), 2),
                'frequency': self.rng.choice(['constant', 'frequent', 'occasional', 'rare', 'triggered']),
                'affected_area': self.rng.choice(['localized', 'regional', 'widespread'])
            })
        
        return hazards
    
    def _generate_poi(self, terrain_type: TerrainType) -> Dict:
        """Generate a point of interest"""
        
        poi_type = self.rng.choice(self.poi_types)
        
        return {
            'id': str(uuid.uuid4()),
            'type': poi_type,
            'name': self.name_gen.generate_place_name(terrain_type.value, 'ancient'),
            'description': f"A {poi_type.replace('_', ' ')} of mysterious origin",
            'discovery_difficulty': round(self.rng.uniform(0.1, 1.0), 2),
            'danger_level': round(self.rng.uniform(0.0, 1.0), 2),
            'reward_potential': round(self.rng.uniform(0.1, 1.0), 2),
            'secrets': self.rng.randint(0, 5),
            'connected_quests': []  # To be populated
        }
    
    def _generate_ambient_properties(self, 
                                     terrain_type: TerrainType,
                                     climate: Dict) -> Dict:
        """Generate ambient sensory properties"""
        
        return {
            'lighting': {
                'primary_source': self.rng.choice(['sun', 'moons', 'bioluminescence', 
                                                   'magical', 'artificial', 'none']),
                'quality': self.rng.choice(['bright', 'dim', 'dappled', 'harsh', 
                                           'soft', 'colorful', 'monochrome']),
                'day_night_cycle': self.rng.choice(['normal', 'extended_day', 
                                                    'extended_night', 'none', 'irregular'])
            },
            'soundscape': {
                'dominant_sounds': self.rng.sample(['wind', 'water', 'wildlife', 
                                                   'silence', 'machinery', 'voices',
                                                   'unknown', 'music'], 2),
                'volume_level': self.rng.choice(['silent', 'quiet', 'moderate', 'loud', 'deafening']),
                'rhythm': self.rng.choice(['constant', 'rhythmic', 'irregular', 'building'])
            },
            'atmosphere': {
                'visibility': self.rng.choice(['crystal_clear', 'normal', 'hazy', 
                                              'foggy', 'obscured', 'variable']),
                'air_quality': self.rng.choice(['pure', 'fresh', 'stale', 
                                               'toxic', 'magical', 'charged']),
                'special_effects': self.rng.sample(['none', 'aurora', 'floating_particles',
                                                   'color_shifts', 'echoes', 'shadows'], 
                                                  self.rng.randint(0, 2))
            }
        }
    
    def _get_cultural_profile(self, world_concept: Dict) -> str:
        """Determine naming culture based on world concept"""
        
        element_to_culture = {
            'fire': 'harsh',
            'water': 'soft',
            'earth': 'natural',
            'air': 'mystical',
            'void': 'technological',
            'life': 'natural',
            'death': 'ancient',
            'time': 'mystical',
            'mind': 'technological'
        }
        
        dominant = world_concept.get('dominant_element', 'earth')
        return element_to_culture.get(dominant, 'natural')
    
    def _get_mood_for_region(self, hazard_level: float, narrative_significance: int) -> str:
        """Determine mood based on region properties"""
        
        if hazard_level > 0.7:
            return 'ominous'
        elif hazard_level < 0.3 and narrative_significance < 4:
            return 'peaceful'
        elif narrative_significance > 7:
            return 'majestic'
        elif hazard_level < 0.3:
            return 'vibrant'
        else:
            return 'mysterious'
    
    def _generate_world_systems(self, 
                                world_concept: Dict,
                                regions: List[Dict]) -> Dict:
        """Generate world-level systems and mechanics"""
        
        return {
            'magic_system': self._generate_magic_system(world_concept),
            'faction_territories': self._generate_faction_territories(regions),
            'trade_routes': self._generate_trade_routes(regions),
            'migration_patterns': self._generate_migration_patterns(regions),
            'historical_events': self._generate_world_history(world_concept, regions)
        }
    
    def _generate_magic_system(self, world_concept: Dict) -> Dict:
        """Generate world magic system rules"""
        
        return {
            'source': self.rng.choice(['innate', 'learned', 'granted', 'environmental', 
                                      'technological', 'emotional', 'sacrificial']),
            'limitation_type': self.rng.choice(['energy_pool', 'time_based', 'material_components',
                                               'emotional_cost', 'physical_cost', 'karma_based']),
            'schools': [
                {
                    'name': self.name_gen.generate_name('mystical', 2, 3),
                    'focus': self.rng.choice(['destruction', 'creation', 'transformation',
                                             'divination', 'control', 'protection', 'healing'])
                }
                for _ in range(self.rng.randint(4, 8))
            ],
            'wild_magic_chance': round(self.rng.uniform(0.0, 0.3), 2),
            'power_ceiling': self.rng.choice(['limited', 'moderate', 'high', 'unlimited'])
        }
    
    def _generate_faction_territories(self, regions: List[Dict]) -> List[Dict]:
        """Generate faction control over regions"""
        
        num_factions = max(2, len(regions) // 5)
        factions = []
        
        for i in range(num_factions):
            faction = {
                'id': str(uuid.uuid4()),
                'name': f"The {self.name_gen.generate_name('ancient', 2, 3)}",
                'controlled_regions': [],
                'influence_type': self.rng.choice(['military', 'economic', 'religious', 
                                                  'magical', 'cultural', 'political'])
            }
            factions.append(faction)
        
        # Assign regions to factions
        for region in regions:
            if self.rng.random() > 0.3:  # 70% of regions have faction presence
                faction = self.rng.choice(factions)
                faction['controlled_regions'].append(region['id'])
        
        return factions
    
    def _generate_trade_routes(self, regions: List[Dict]) -> List[Dict]:
        """Generate trade routes between regions"""
        
        routes = []
        num_routes = len(regions) // 2
        
        for _ in range(num_routes):
            if len(regions) >= 2:
                endpoints = self.rng.sample(regions, 2)
                route = {
                    'id': str(uuid.uuid4()),
                    'start_region': endpoints[0]['id'],
                    'end_region': endpoints[1]['id'],
                    'primary_goods': self.rng.sample(self.resource_types, self.rng.randint(1, 3)),
                    'danger_level': round(self.rng.uniform(0.0, 1.0), 2),
                    'traffic_level': self.rng.choice(['sparse', 'light', 'moderate', 'heavy', 'congested'])
                }
                routes.append(route)
        
        return routes
    
    def _generate_migration_patterns(self, regions: List[Dict]) -> List[Dict]:
        """Generate creature/people migration patterns"""
        
        patterns = []
        num_patterns = len(regions) // 4
        
        migration_types = ['seasonal', 'resource_following', 'predator_avoidance',
                          'breeding', 'magical_cycle', 'historical_tradition']
        
        for _ in range(num_patterns):
            if len(regions) >= 3:
                route_regions = self.rng.sample(regions, self.rng.randint(2, min(5, len(regions))))
                pattern = {
                    'id': str(uuid.uuid4()),
                    'type': self.rng.choice(migration_types),
                    'species': self.name_gen.generate_name('natural', 2, 3),
                    'route': [r['id'] for r in route_regions],
                    'cycle_duration': self.rng.choice(['monthly', 'seasonal', 'annual', 
                                                       'multi_year', 'irregular']),
                    'population_size': self.rng.choice(['small', 'medium', 'large', 'massive'])
                }
                patterns.append(pattern)
        
        return patterns
    
    def _generate_world_history(self, 
                                world_concept: Dict,
                                regions: List[Dict]) -> List[Dict]:
        """Generate major historical events"""
        
        event_types = ['war', 'cataclysm', 'discovery', 'founding', 'extinction',
                      'transformation', 'arrival', 'departure', 'union', 'schism']
        
        num_events = self.rng.randint(5, 15)
        events = []
        
        current_era = 0
        for i in range(num_events):
            era_jump = self.rng.randint(100, 1000)
            current_era -= era_jump
            
            event = {
                'id': str(uuid.uuid4()),
                'era': current_era,
                'era_name': f"Year {abs(current_era)} Before Present",
                'event_type': self.rng.choice(event_types),
                'name': f"The {self.name_gen.generate_name('ancient', 2, 3)} {self.rng.choice(['Event', 'Era', 'War', 'Age', 'Time'])}",
                'description': f"A {self.rng.choice(['pivotal', 'devastating', 'miraculous', 'mysterious'])} {self.rng.choice(event_types)} that shaped the world",
                'affected_regions': [r['id'] for r in self.rng.sample(regions, 
                                    min(self.rng.randint(1, 5), len(regions)))],
                'lasting_effects': self.rng.sample([
                    'changed geography', 'cultural shift', 'new species',
                    'lost knowledge', 'new power', 'faction formed',
                    'faction destroyed', 'magical change', 'population shift'
                ], self.rng.randint(1, 3))
            }
            events.append(event)
        
        return sorted(events, key=lambda x: x['era'])
    
    def generate_batch(self, count: int, scale: str = 'regional') -> List[Dict]:
        """Generate multiple worlds"""
        
        worlds = []
        for i in range(count):
            self.rng = random.Random(self.seed + i)
            self.name_gen = ProceduralNameGenerator(self.seed + i)
            self.desc_gen = ProceduralDescriptionGenerator(self.seed + i)
            
            world = self.generate_world(scale)
            worlds.append(world)
        
        return worlds