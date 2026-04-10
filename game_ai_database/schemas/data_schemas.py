"""
Enterprise Data Schemas for Game AI Training
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json

class Rarity(Enum):
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5
    MYTHIC = 6

class CharacterRole(Enum):
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    MENTOR = "mentor"
    ALLY = "ally"
    NEUTRAL = "neutral"
    WILD_CARD = "wild_card"

class TerrainType(Enum):
    PLAINS = "plains"
    MOUNTAINS = "mountains"
    FOREST = "forest"
    DESERT = "desert"
    AQUATIC = "aquatic"
    URBAN = "urban"
    SUBTERRANEAN = "subterranean"
    AERIAL = "aerial"
    VOID = "void"
    HYBRID = "hybrid"

@dataclass
class WorldRegion:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    terrain_type: TerrainType = TerrainType.PLAINS
    climate_profile: Dict[str, float] = field(default_factory=dict)
    resource_distribution: Dict[str, float] = field(default_factory=dict)
    hazard_level: float = 0.0
    discovery_state: str = "hidden"
    connected_regions: List[str] = field(default_factory=list)
    points_of_interest: List[Dict] = field(default_factory=list)
    ambient_properties: Dict[str, Any] = field(default_factory=dict)
    narrative_significance: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "terrain_type": self.terrain_type.value,
            "climate_profile": self.climate_profile,
            "resource_distribution": self.resource_distribution,
            "hazard_level": self.hazard_level,
            "discovery_state": self.discovery_state,
            "connected_regions": self.connected_regions,
            "points_of_interest": self.points_of_interest,
            "ambient_properties": self.ambient_properties,
            "narrative_significance": self.narrative_significance
        }

@dataclass
class Character:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    title: str = ""
    role: CharacterRole = CharacterRole.NEUTRAL
    backstory: str = ""
    personality_traits: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    skills: Dict[str, int] = field(default_factory=dict)
    relationships: Dict[str, str] = field(default_factory=dict)
    dialogue_style: Dict[str, Any] = field(default_factory=dict)
    visual_descriptors: Dict[str, str] = field(default_factory=dict)
    arc_progression: List[Dict] = field(default_factory=list)
    combat_style: str = ""
    faction_affiliations: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "role": self.role.value,
            "backstory": self.backstory,
            "personality_traits": self.personality_traits,
            "motivations": self.motivations,
            "fears": self.fears,
            "skills": self.skills,
            "relationships": self.relationships,
            "dialogue_style": self.dialogue_style,
            "visual_descriptors": self.visual_descriptors,
            "arc_progression": self.arc_progression,
            "combat_style": self.combat_style,
            "faction_affiliations": self.faction_affiliations,
            "secrets": self.secrets
        }

@dataclass
class Quest:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    quest_type: str = ""
    objectives: List[Dict] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    rewards: Dict[str, Any] = field(default_factory=dict)
    narrative_branches: List[Dict] = field(default_factory=list)
    involved_characters: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    difficulty_curve: List[float] = field(default_factory=list)
    time_constraints: Optional[Dict] = None
    moral_choices: List[Dict] = field(default_factory=list)
    hidden_outcomes: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "quest_type": self.quest_type,
            "objectives": self.objectives,
            "prerequisites": self.prerequisites,
            "rewards": self.rewards,
            "narrative_branches": self.narrative_branches,
            "involved_characters": self.involved_characters,
            "locations": self.locations,
            "difficulty_curve": self.difficulty_curve,
            "time_constraints": self.time_constraints,
            "moral_choices": self.moral_choices,
            "hidden_outcomes": self.hidden_outcomes
        }

@dataclass
class GameMechanic:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""
    description: str = ""
    rules: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    interactions: List[Dict] = field(default_factory=list)
    progression_hooks: List[str] = field(default_factory=list)
    balance_considerations: Dict[str, float] = field(default_factory=dict)
    player_feedback: Dict[str, str] = field(default_factory=dict)
    edge_cases: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "rules": self.rules,
            "parameters": self.parameters,
            "interactions": self.interactions,
            "progression_hooks": self.progression_hooks,
            "balance_considerations": self.balance_considerations,
            "player_feedback": self.player_feedback,
            "edge_cases": self.edge_cases
        }

@dataclass
class DialogueNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speaker_id: str = ""
    text: str = ""
    emotion: str = ""
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    responses: List[Dict] = field(default_factory=list)
    effects: List[Dict] = field(default_factory=list)
    voice_direction: str = ""
    animation_cues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "emotion": self.emotion,
            "context_requirements": self.context_requirements,
            "responses": self.responses,
            "effects": self.effects,
            "voice_direction": self.voice_direction,
            "animation_cues": self.animation_cues
        }

@dataclass
class AssetDescriptor:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_type: str = ""
    name: str = ""
    description: str = ""
    visual_properties: Dict[str, Any] = field(default_factory=dict)
    material_properties: Dict[str, Any] = field(default_factory=dict)
    animation_requirements: List[Dict] = field(default_factory=list)
    lod_specifications: List[Dict] = field(default_factory=list)
    collision_type: str = ""
    interaction_points: List[Dict] = field(default_factory=list)
    variants: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "asset_type": self.asset_type,
            "name": self.name,
            "description": self.description,
            "visual_properties": self.visual_properties,
            "material_properties": self.material_properties,
            "animation_requirements": self.animation_requirements,
            "lod_specifications": self.lod_specifications,
            "collision_type": self.collision_type,
            "interaction_points": self.interaction_points,
            "variants": self.variants
        }

@dataclass 
class LevelLayout:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    level_type: str = ""
    dimensions: Dict[str, float] = field(default_factory=dict)
    zones: List[Dict] = field(default_factory=list)
    spawn_points: List[Dict] = field(default_factory=list)
    navigation_mesh: Dict[str, Any] = field(default_factory=dict)
    lighting_scheme: Dict[str, Any] = field(default_factory=dict)
    encounter_placements: List[Dict] = field(default_factory=list)
    secret_areas: List[Dict] = field(default_factory=list)
    environmental_hazards: List[Dict] = field(default_factory=list)
    pacing_markers: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "level_type": self.level_type,
            "dimensions": self.dimensions,
            "zones": self.zones,
            "spawn_points": self.spawn_points,
            "navigation_mesh": self.navigation_mesh,
            "lighting_scheme": self.lighting_scheme,
            "encounter_placements": self.encounter_placements,
            "secret_areas": self.secret_areas,
            "environmental_hazards": self.environmental_hazards,
            "pacing_markers": self.pacing_markers
        }

@dataclass
class EconomySystem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    currency_types: List[Dict] = field(default_factory=list)
    exchange_rates: Dict[str, float] = field(default_factory=dict)
    item_valuations: Dict[str, Dict] = field(default_factory=dict)
    inflation_model: Dict[str, Any] = field(default_factory=dict)
    sink_sources: List[Dict] = field(default_factory=list)
    market_dynamics: Dict[str, Any] = field(default_factory=dict)
    trade_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "currency_types": self.currency_types,
            "exchange_rates": self.exchange_rates,
            "item_valuations": self.item_valuations,
            "inflation_model": self.inflation_model,
            "sink_sources": self.sink_sources,
            "market_dynamics": self.market_dynamics,
            "trade_rules": self.trade_rules
        }

@dataclass
class NarrativeTheme:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    core_conflict: str = ""
    philosophical_question: str = ""
    symbolic_elements: List[str] = field(default_factory=list)
    tone: str = ""
    resolution_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "core_conflict": self.core_conflict,
            "philosophical_question": self.philosophical_question,
            "symbolic_elements": self.symbolic_elements,
            "tone": self.tone,
            "resolution_types": self.resolution_types
        }

@dataclass
class PlotBeat:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    beat_type: str = ""
    description: str = ""
    emotional_shift: str = ""
    required_elements: List[str] = field(default_factory=list)
    optional_branches: List[str] = field(default_factory=list)
    tension_level: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "beat_type": self.beat_type,
            "description": self.description,
            "emotional_shift": self.emotional_shift,
            "required_elements": self.required_elements,
            "optional_branches": self.optional_branches,
            "tension_level": self.tension_level
        }