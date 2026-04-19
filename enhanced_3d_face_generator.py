"""
Enhanced 3D Face Generator for Janus Game AI System
Generates AAA-quality 3D faces for games with full integration to training pipelines
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import hashlib
from datetime import datetime
import sqlite3
import random

@dataclass
class Vector3:
    x: float
    y: float
    z: float
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

@dataclass
class GameCharacterFeatures:
    """Enhanced facial features for game characters"""
    # Basic structure
    head_width: float = 1.0
    head_height: float = 1.0
    head_depth: float = 1.0
    jaw_width: float = 1.0
    chin_prominence: float = 0.5
    cheekbone_height: float = 0.5
    forehead_slope: float = 0.5
    
    # Eyes (critical for character emotion)
    eye_size: float = 1.0
    eye_spacing: float = 1.0
    eye_tilt: float = 0.0
    eye_depth: float = 0.5
    iris_size: float = 0.5
    pupil_dilation: float = 0.5
    eyelash_length: float = 0.5
    eyebrow_thickness: float = 0.5
    eyebrow_arch: float = 0.5
    
    # Nose
    nose_width: float = 1.0
    nose_length: float = 1.0
    nose_bridge_height: float = 0.5
    nose_tip_rotation: float = 0.5
    nostril_flare: float = 0.5
    
    # Mouth (crucial for dialogue)
    mouth_width: float = 1.0
    lip_thickness: float = 0.5
    lip_asymmetry: float = 0.0
    mouth_corner_height: float = 0.5
    teeth_visibility: float = 0.5
    
    # Ears
    ear_size: float = 1.0
    ear_rotation: float = 0.0
    ear_pointedness: float = 0.0  # For fantasy characters
    
    # Character type modifiers
    character_archetype: str = "human"  # human, elf, orc, dwarf, etc.
    heroic_proportions: float = 0.0  # 0 = realistic, 1 = heroic
    stylization_level: float = 0.0  # 0 = realistic, 1 = stylized
    
    # Skin and materials
    skin_tone_r: float = 0.9
    skin_tone_g: float = 0.7
    skin_tone_b: float = 0.6
    skin_metallic: float = 0.0  # For robot characters
    skin_roughness: float = 0.5
    subsurface_scattering: float = 0.5
    
    # Age and wear
    age: float = 0.5
    battle_scars: float = 0.0
    weathering: float = 0.0
    
    # Fantasy/Sci-fi features
    horn_size: float = 0.0
    fang_length: float = 0.0
    cybernetic_implants: float = 0.0
    magical_markings: float = 0.0

@dataclass
class GameFaceAsset:
    """Complete game-ready face asset"""
    character_id: str
    features: GameCharacterFeatures
    mesh_data: Dict
    texture_maps: Dict  # diffuse, normal, roughness, etc.
    blend_shapes: List[Dict]
    bone_weights: Dict
    lod_levels: List[Dict]  # Level of detail meshes
    performance_stats: Dict
    game_metadata: Dict

class GameAIFaceDatabase:
    """Database for storing and managing game face assets"""
    
    def __init__(self, db_path: str = "game_faces.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize face asset database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Face assets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_assets (
                id TEXT PRIMARY KEY,
                character_name TEXT,
                archetype TEXT,
                features_json TEXT,
                mesh_data_json TEXT,
                texture_maps_json TEXT,
                performance_rating REAL,
                polygon_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_genre TEXT,
                art_style TEXT
            )
        ''')
        
        # Training data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_faces (
                id TEXT PRIMARY KEY,
                face_asset_id TEXT,
                training_context TEXT,
                emotion_labels TEXT,
                dialogue_phonemes TEXT,
                animation_data TEXT,
                quality_score REAL,
                FOREIGN KEY (face_asset_id) REFERENCES face_assets (id)
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_performance (
                face_id TEXT,
                render_time_ms REAL,
                memory_usage_mb REAL,
                triangle_count INTEGER,
                texture_memory_mb REAL,
                animation_fps REAL,
                FOREIGN KEY (face_id) REFERENCES face_assets (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_face_asset(self, asset: GameFaceAsset):
        """Store face asset in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO face_assets 
            (id, character_name, archetype, features_json, mesh_data_json, 
             texture_maps_json, performance_rating, polygon_count, game_genre, art_style)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            asset.character_id,
            asset.game_metadata.get('character_name', 'Unknown'),
            asset.features.character_archetype,
            json.dumps(asdict(asset.features)),
            json.dumps(asset.mesh_data),
            json.dumps(asset.texture_maps),
            asset.performance_stats.get('overall_rating', 0.0),
            asset.performance_stats.get('polygon_count', 0),
            asset.game_metadata.get('genre', 'Unknown'),
            asset.game_metadata.get('art_style', 'Realistic')
        ))
        
        conn.commit()
        conn.close()
    
    def get_faces_by_archetype(self, archetype: str) -> List[Dict]:
        """Get all faces of a specific character type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM face_assets WHERE archetype = ?
        ''', (archetype,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]

class AAA_FaceGenerator:
    """AAA-quality face generator for game development"""
    
    def __init__(self):
        self.database = GameAIFaceDatabase()
        self.archetype_templates = self._load_archetype_templates()
        self.performance_targets = {
            'mobile': {'max_triangles': 2000, 'max_texture_size': 512},
            'console': {'max_triangles': 8000, 'max_texture_size': 2048},
            'pc_high': {'max_triangles': 20000, 'max_texture_size': 4096},
            'cinematic': {'max_triangles': 100000, 'max_texture_size': 8192}
        }
    
    def _load_archetype_templates(self) -> Dict:
        """Load character archetype templates"""
        return {
            'human_hero': GameCharacterFeatures(
                heroic_proportions=0.7,
                jaw_width=1.2,
                cheekbone_height=0.8,
                eye_size=1.1
            ),
            'human_villain': GameCharacterFeatures(
                jaw_width=1.3,
                eyebrow_arch=0.8,
                eye_tilt=0.3,
                battle_scars=0.4
            ),
            'elf': GameCharacterFeatures(
                character_archetype='elf',
                ear_pointedness=0.8,
                eye_size=1.2,
                face_length=1.1,
                cheekbone_height=0.9
            ),
            'orc': GameCharacterFeatures(
                character_archetype='orc',
                jaw_width=1.5,
                fang_length=0.6,
                skin_tone_g=0.8,
                battle_scars=0.7,
                nose_width=1.3
            ),
            'dwarf': GameCharacterFeatures(
                character_archetype='dwarf',
                head_width=1.2,
                jaw_width=1.4,
                nose_width=1.2,
                eyebrow_thickness=0.8
            ),
            'cyborg': GameCharacterFeatures(
                character_archetype='cyborg',
                cybernetic_implants=0.8,
                skin_metallic=0.3,
                eye_size=0.9,
                battle_scars=0.5
            )
        }
    
    def generate_character_face(self, 
                               archetype: str = 'human',
                               performance_target: str = 'console',
                               customization: Optional[Dict] = None) -> GameFaceAsset:
        """Generate a complete game character face"""
        
        # Start with archetype template
        if archetype in self.archetype_templates:
            features = self.archetype_templates[archetype]
        else:
            features = GameCharacterFeatures()
            features.character_archetype = archetype
        
        # Apply customization
        if customization:
            for key, value in customization.items():
                if hasattr(features, key):
                    setattr(features, key, value)
        
        # Generate mesh data
        mesh_data = self._generate_mesh(features, performance_target)
        
        # Generate texture maps
        texture_maps = self._generate_texture_maps(features, performance_target)
        
        # Generate blend shapes for animation
        blend_shapes = self._generate_blend_shapes(features)
        
        # Generate bone weights for rigging
        bone_weights = self._generate_bone_weights(mesh_data)
        
        # Generate LOD levels
        lod_levels = self._generate_lod_levels(mesh_data, performance_target)
        
        # Calculate performance stats
        performance_stats = self._calculate_performance_stats(mesh_data, texture_maps)
        
        # Create game metadata
        game_metadata = {
            'character_name': f"{archetype}_character_{random.randint(1000, 9999)}",
            'genre': 'RPG',
            'art_style': 'Realistic' if features.stylization_level < 0.5 else 'Stylized',
            'target_platform': performance_target,
            'animation_ready': True,
            'dialogue_ready': True
        }
        
        # Create asset
        character_id = hashlib.md5(
            f"{archetype}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        asset = GameFaceAsset(
            character_id=character_id,
            features=features,
            mesh_data=mesh_data,
            texture_maps=texture_maps,
            blend_shapes=blend_shapes,
            bone_weights=bone_weights,
            lod_levels=lod_levels,
            performance_stats=performance_stats,
            game_metadata=game_metadata
        )
        
        # Store in database
        self.database.store_face_asset(asset)
        
        return asset
    
    def _generate_mesh(self, features: GameCharacterFeatures, target: str) -> Dict:
        """Generate optimized mesh based on performance target"""
        target_specs = self.performance_targets[target]
        
        # Adjust resolution based on target
        if target == 'mobile':
            segments, rings = 24, 18
        elif target == 'console':
            segments, rings = 32, 24
        elif target == 'pc_high':
            segments, rings = 48, 36
        else:  # cinematic
            segments, rings = 64, 48
        
        vertices = []
        faces = []
        uv_coords = []
        
        # Generate base head mesh with archetype modifications
        for ring in range(rings + 1):
            ring_factor = ring / rings
            
            # Adjust for character archetype
            if features.character_archetype == 'elf':
                # Elongated face
                theta = ring_factor * np.pi * 1.1
            elif features.character_archetype == 'dwarf':
                # Compressed face
                theta = ring_factor * np.pi * 0.9
            else:
                theta = ring_factor * np.pi
            
            for segment in range(segments):
                phi = (segment / segments) * 2 * np.pi
                
                x = np.sin(theta) * np.cos(phi)
                y = np.cos(theta)
                z = np.sin(theta) * np.sin(phi)
                
                # Apply archetype deformations
                x, y, z = self._apply_archetype_deformations(x, y, z, features, ring_factor, phi)
                
                vertices.append([x, y, z])
                uv_coords.append([segment / segments, ring / rings])
        
        # Generate faces
        for ring in range(rings):
            for segment in range(segments):
                current = ring * segments + segment
                next_ring = (ring + 1) * segments + segment
                next_segment = ring * segments + ((segment + 1) % segments)
                next_both = (ring + 1) * segments + ((segment + 1) % segments)
                
                faces.append([current, next_ring, next_both])
                faces.append([current, next_both, next_segment])
        
        return {
            'vertices': vertices,
            'faces': faces,
            'uv_coords': uv_coords,
            'vertex_count': len(vertices),
            'triangle_count': len(faces),
            'target_platform': target
        }
    
    def _apply_archetype_deformations(self, x: float, y: float, z: float, 
                                     features: GameCharacterFeatures, 
                                     ring_factor: float, phi: float) -> Tuple[float, float, float]:
        """Apply character archetype-specific deformations"""
        
        # Orc modifications
        if features.character_archetype == 'orc':
            # Larger jaw
            if ring_factor > 0.6:
                x *= 1.0 + features.jaw_width * 0.3
            # Protruding brow
            if 0.2 < ring_factor < 0.4:
                z *= 1.0 + 0.2
            # Larger nose
            if 0.4 < ring_factor < 0.7 and abs(phi) < np.pi/6:
                z *= 1.0 + features.nose_width * 0.3
        
        # Elf modifications
        elif features.character_archetype == 'elf':
            # Pointed ears
            if abs(phi - np.pi/2) < np.pi/8 or abs(phi - 3*np.pi/2) < np.pi/8:
                if 0.3 < ring_factor < 0.6:
                    ear_side = 1 if abs(phi - np.pi/2) < np.pi/8 else -1
                    x += ear_side * features.ear_pointedness * 0.3
                    z += features.ear_pointedness * 0.2
            # Refined features
            if 0.4 < ring_factor < 0.8:
                x *= 0.95  # Narrower face
        
        # Dwarf modifications
        elif features.character_archetype == 'dwarf':
            # Broader, shorter proportions
            x *= 1.0 + features.head_width * 0.2
            y *= 0.9  # Shorter head
            # Prominent nose
            if 0.4 < ring_factor < 0.7 and abs(phi) < np.pi/8:
                z *= 1.0 + 0.4
        
        # Cyborg modifications
        elif features.character_archetype == 'cyborg':
            # Mechanical implants (simplified as surface modifications)
            if features.cybernetic_implants > 0.5:
                # Add angular modifications for tech look
                if 0.3 < ring_factor < 0.6 and abs(phi - np.pi/4) < np.pi/12:
                    z *= 1.0 + 0.1  # Implant protrusion
        
        # Apply heroic proportions
        if features.heroic_proportions > 0:
            # Stronger jaw
            if ring_factor > 0.6:
                x *= 1.0 + features.heroic_proportions * 0.15
                z *= 1.0 + features.heroic_proportions * 0.1
            # More defined cheekbones
            if 0.4 < ring_factor < 0.6:
                x *= 1.0 + features.heroic_proportions * 0.1
        
        return x, y, z
    
    def _generate_texture_maps(self, features: GameCharacterFeatures, target: str) -> Dict:
        """Generate PBR texture maps for the character"""
        target_specs = self.performance_targets[target]
        resolution = target_specs['max_texture_size']
        
        # Base diffuse color
        base_color = np.array([
            features.skin_tone_r,
            features.skin_tone_g,
            features.skin_tone_b
        ])
        
        # Generate diffuse map
        diffuse_map = np.full((resolution, resolution, 3), base_color, dtype=np.float32)
        
        # Add skin variation
        noise = np.random.normal(0, 0.05, (resolution, resolution, 3))
        diffuse_map = np.clip(diffuse_map + noise, 0, 1)
        
        # Add archetype-specific coloring
        if features.character_archetype == 'orc':
            # Greenish tint
            diffuse_map[:, :, 1] *= 1.2
            diffuse_map[:, :, 0] *= 0.8
        elif features.character_archetype == 'elf':
            # Slight luminous quality
            diffuse_map *= 1.05
        
        # Generate normal map (simplified)
        normal_map = np.full((resolution, resolution, 3), [0.5, 0.5, 1.0], dtype=np.float32)
        
        # Add skin texture to normal map
        skin_noise = np.random.normal(0.5, features.skin_roughness * 0.1, (resolution, resolution, 2))
        normal_map[:, :, :2] = np.clip(skin_noise, 0, 1)
        
        # Generate roughness map
        roughness_map = np.full((resolution, resolution), features.skin_roughness, dtype=np.float32)
        
        # Generate metallic map (for cyborg parts)
        metallic_map = np.full((resolution, resolution), features.skin_metallic, dtype=np.float32)
        
        # Add battle scars and weathering
        if features.battle_scars > 0:
            scar_mask = np.random.random((resolution, resolution)) < (features.battle_scars * 0.01)
            diffuse_map[scar_mask] *= 0.7  # Darker scars
            roughness_map[scar_mask] *= 1.3  # Rougher scars
        
        return {
            'diffuse': (diffuse_map * 255).astype(np.uint8).tolist(),
            'normal': (normal_map * 255).astype(np.uint8).tolist(),
            'roughness': (roughness_map * 255).astype(np.uint8).tolist(),
            'metallic': (metallic_map * 255).astype(np.uint8).tolist(),
            'resolution': resolution,
            'format': 'RGB'
        }
    
    def _generate_blend_shapes(self, features: GameCharacterFeatures) -> List[Dict]:
        """Generate facial animation blend shapes"""
        blend_shapes = []
        
        # Standard expressions
        expressions = [
            'neutral', 'smile', 'frown', 'surprise', 'anger', 'fear', 'disgust', 'sadness',
            'blink_left', 'blink_right', 'blink_both',
            'eyebrow_raise_left', 'eyebrow_raise_right', 'eyebrow_furrow',
            'mouth_open', 'mouth_pucker', 'cheek_puff'
        ]
        
        # Phoneme shapes for dialogue
        phonemes = [
            'A', 'E', 'I', 'O', 'U', 'M', 'B', 'P', 'F', 'V', 'T', 'D', 'S', 'Z'
        ]
        
        for expression in expressions + phonemes:
            blend_shapes.append({
                'name': expression,
                'vertex_deltas': self._calculate_expression_deltas(expression, features),
                'weight': 0.0,
                'category': 'expression' if expression in expressions else 'phoneme'
            })
        
        return blend_shapes
    
    def _calculate_expression_deltas(self, expression: str, features: GameCharacterFeatures) -> List:
        """Calculate vertex deltas for facial expressions"""
        # Simplified - in real implementation, this would be much more complex
        deltas = []
        vertex_count = 1000  # Approximate based on mesh resolution
        
        for i in range(vertex_count):
            # Default no movement
            delta = [0.0, 0.0, 0.0]
            
            # Apply expression-specific movements (simplified)
            if expression == 'smile':
                # Move mouth corners up
                if i % 100 < 10:  # Approximate mouth region
                    delta = [0.1 if i % 2 == 0 else -0.1, 0.05, 0.02]
            elif expression == 'frown':
                if i % 100 < 10:
                    delta = [0.05 if i % 2 == 0 else -0.05, -0.03, 0.01]
            
            deltas.append(delta)
        
        return deltas
    
    def _generate_bone_weights(self, mesh_data: Dict) -> Dict:
        """Generate bone weights for facial rigging"""
        vertex_count = mesh_data['vertex_count']
        
        # Define facial bones
        bones = [
            'head', 'jaw', 'left_eye', 'right_eye', 'nose',
            'left_cheek', 'right_cheek', 'upper_lip', 'lower_lip',
            'left_eyebrow', 'right_eyebrow', 'left_ear', 'right_ear'
        ]
        
        # Generate weights (simplified)
        bone_weights = {}
        for bone in bones:
            weights = []
            for i in range(vertex_count):
                # Simplified weight assignment based on vertex position
                weight = random.uniform(0.0, 1.0) if random.random() < 0.3 else 0.0
                weights.append(weight)
            bone_weights[bone] = weights
        
        return bone_weights
    
    def _generate_lod_levels(self, mesh_data: Dict, target: str) -> List[Dict]:
        """Generate Level of Detail meshes"""
        base_triangles = mesh_data['triangle_count']
        
        lod_levels = []
        
        # LOD 0 (highest quality)
        lod_levels.append({
            'level': 0,
            'triangle_count': base_triangles,
            'distance_threshold': 0.0,
            'quality': 1.0
        })
        
        # LOD 1 (medium quality)
        lod_levels.append({
            'level': 1,
            'triangle_count': int(base_triangles * 0.6),
            'distance_threshold': 10.0,
            'quality': 0.8
        })
        
        # LOD 2 (low quality)
        lod_levels.append({
            'level': 2,
            'triangle_count': int(base_triangles * 0.3),
            'distance_threshold': 25.0,
            'quality': 0.5
        })
        
        # LOD 3 (very low quality)
        if target in ['console', 'pc_high', 'cinematic']:
            lod_levels.append({
                'level': 3,
                'triangle_count': int(base_triangles * 0.1),
                'distance_threshold': 50.0,
                'quality': 0.2
            })
        
        return lod_levels
    
    def _calculate_performance_stats(self, mesh_data: Dict, texture_maps: Dict) -> Dict:
        """Calculate performance statistics"""
        triangle_count = mesh_data['triangle_count']
        texture_resolution = texture_maps['resolution']
        
        # Estimate memory usage
        vertex_memory = mesh_data['vertex_count'] * 32  # bytes per vertex (pos + normal + uv)
        texture_memory = texture_resolution * texture_resolution * 4 * 4  # 4 maps, 4 bytes per pixel
        
        # Estimate render performance
        render_complexity = triangle_count / 1000.0  # Simplified metric
        
        return {
            'polygon_count': triangle_count,
            'vertex_count': mesh_data['vertex_count'],
            'memory_usage_mb': (vertex_memory + texture_memory) / (1024 * 1024),
            'render_complexity': render_complexity,
            'texture_memory_mb': texture_memory / (1024 * 1024),
            'overall_rating': min(10.0, max(1.0, 10.0 - render_complexity * 0.5))
        }
    
    def generate_character_batch(self, count: int, archetype: str = 'human', 
                                target: str = 'console') -> List[GameFaceAsset]:
        """Generate a batch of character faces for training data"""
        characters = []
        
        for i in range(count):
            # Add variation to each character
            variation = {
                'head_width': random.uniform(0.8, 1.2),
                'head_height': random.uniform(0.9, 1.1),
                'eye_size': random.uniform(0.8, 1.2),
                'nose_width': random.uniform(0.7, 1.3),
                'mouth_width': random.uniform(0.8, 1.2),
                'age': random.uniform(0.2, 0.8),
                'skin_tone_r': random.uniform(0.6, 1.0),
                'skin_tone_g': random.uniform(0.4, 0.9),
                'skin_tone_b': random.uniform(0.3, 0.8)
            }
            
            character = self.generate_character_face(archetype, target, variation)
            characters.append(character)
        
        return characters

def main():
    """Demo the enhanced face generator"""
    print("="*60)
    print("AAA GAME FACE GENERATOR")
    print("="*60)
    
    generator = AAA_FaceGenerator()
    
    # Generate different character archetypes
    archetypes = ['human_hero', 'human_villain', 'elf', 'orc', 'dwarf', 'cyborg']
    
    for archetype in archetypes:
        print(f"\nGenerating {archetype} character...")
        character = generator.generate_character_face(
            archetype=archetype,
            performance_target='console'
        )
        
        print(f"  Character ID: {character.character_id}")
        print(f"  Triangles: {character.performance_stats['polygon_count']}")
        print(f"  Memory: {character.performance_stats['memory_usage_mb']:.1f} MB")
        print(f"  Rating: {character.performance_stats['overall_rating']:.1f}/10")
    
    # Generate training batch
    print(f"\nGenerating training batch...")
    batch = generator.generate_character_batch(count=10, archetype='human')
    print(f"Generated {len(batch)} training characters")
    
    # Show database stats
    human_faces = generator.database.get_faces_by_archetype('human')
    print(f"\nDatabase contains {len(human_faces)} human faces")
    
    print("\nAAA Face Generator ready for game development!")

if __name__ == "__main__":
    main()