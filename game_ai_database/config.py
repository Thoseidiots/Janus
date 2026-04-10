"""
Enterprise Configuration for AAA Game Database Generator
"""
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class GameGenre(Enum):
    RPG = "rpg"
    FPS = "fps"
    STRATEGY = "strategy"
    ADVENTURE = "adventure"
    SIMULATION = "simulation"
    HORROR = "horror"
    RACING = "racing"
    FIGHTING = "fighting"

class ArtStyle(Enum):
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    PIXEL = "pixel"
    CEL_SHADED = "cel_shaded"
    LOW_POLY = "low_poly"

@dataclass
class DatabaseConfig:
    db_path: str = "game_ai_training.db"
    batch_size: int = 1000
    max_records_per_table: int = 100000
    enable_compression: bool = True
    validate_entries: bool = True

@dataclass
class GenerationConfig:
    seed: int = 42
    narrative_complexity: int = 5  # 1-10
    world_scale: str = "continental"  # room, building, city, regional, continental, planetary
    character_depth: int = 7  # 1-10
    mechanics_innovation: int = 6  # 1-10
    target_genres: List[GameGenre] = None
    art_styles: List[ArtStyle] = None
    
    def __post_init__(self):
        if self.target_genres is None:
            self.target_genres = [GameGenre.RPG, GameGenre.ADVENTURE]
        if self.art_styles is None:
            self.art_styles = [ArtStyle.REALISTIC, ArtStyle.STYLIZED]