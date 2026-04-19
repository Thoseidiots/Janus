"""
Game AI Training Pipeline
Integrates 3D face generation, hardware awareness, and game development
Trains AI models to understand and generate AAA game content
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
import random
import threading
import queue
import time

# Import Janus systems
try:
    from enhanced_3d_face_generator import AAA_FaceGenerator, GameCharacterFeatures
    from hardware_sense import HardwareAwareness
    from avus_brain import AvusBrain
    from avus_inference import AvusInference
    JANUS_AVAILABLE = True
except ImportError as e:
    print(f"Janus systems not available: {e}")
    JANUS_AVAILABLE = False

@dataclass
class GameTrainingData:
    """Training data for game AI"""
    data_id: str
    data_type: str  # 'character', 'environment', 'dialogue', 'animation'
    content: Dict
    labels: List[str]
    quality_score: float
    hardware_context: Dict
    timestamp: str

@dataclass
class TrainingSession:
    """A complete training session"""
    session_id: str
    model_type: str
    training_data: List[GameTrainingData]
    hardware_profile: Dict
    performance_metrics: Dict
    start_time: str
    end_time: Optional[str] = None

class GameAITrainingPipeline:
    """
    Complete training pipeline for game AI development
    Integrates face generation, hardware awareness, and model training
    """
    
    def __init__(self):
        self.face_generator = AAA_FaceGenerator() if JANUS_AVAILABLE else None
        self.hardware_awareness = HardwareAwareness() if JANUS_AVAILABLE else None
        self.avus_brain = None  # Lazy load
        
        # Training database
        self.db_path = "game_ai_training.db"
        self.init_training_database()
        
        # Training queues
        self.training_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Training state
        self.active_sessions = {}
        self.training_active = False
        
        print("Game AI Training Pipeline initialized")
    
    def init_training_database(self):
        """Initialize training database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Training sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                model_type TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                hardware_profile TEXT,
                performance_metrics TEXT,
                data_count INTEGER,
                success_rate REAL
            )
        ''')
        
        # Training data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                data_id TEXT PRIMARY KEY,
                session_id TEXT,
                data_type TEXT,
                content_json TEXT,
                labels_json TEXT,
                quality_score REAL,
                hardware_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
            )
        ''')
        
        # Model performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_id TEXT,
                session_id TEXT,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                hardware_utilization REAL,
                memory_usage_mb REAL,
                training_time_seconds REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Game asset generation logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_generation_log (
                generation_id TEXT PRIMARY KEY,
                asset_type TEXT,
                generation_params TEXT,
                output_quality REAL,
                generation_time_seconds REAL,
                hardware_state TEXT,
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def ensure_avus_loaded(self) -> bool:
        """Lazy load Avus brain"""
        if self.avus_brain is None and JANUS_AVAILABLE:
            try:
                self.avus_brain = AvusBrain()
                return self.avus_brain.ensure_loaded()
            except:
                return False
        return self.avus_brain is not None
    
    def generate_character_training_data(self, count: int = 100, 
                                       archetypes: List[str] = None) -> List[GameTrainingData]:
        """Generate character training data using 3D face generator"""
        if not self.face_generator:
            return []
        
        if archetypes is None:
            archetypes = ['human_hero', 'human_villain', 'elf', 'orc', 'dwarf', 'cyborg']
        
        training_data = []
        
        for i in range(count):
            # Select random archetype
            archetype = random.choice(archetypes)
            
            # Generate character with random variations
            variation = self._generate_random_character_params()
            
            # Get current hardware state
            hardware_context = {}
            if self.hardware_awareness:
                hardware_context = {
                    'feeling': self.hardware_awareness.sense.feel(),
                    'cpu_usage': self.hardware_awareness.sense.sense().cpu_usage,
                    'memory_pressure': self.hardware_awareness.sense.sense().memory_pressure,
                    'temperature': self.hardware_awareness.sense.sense().cpu_temp
                }
            
            # Generate character
            character = self.face_generator.generate_character_face(
                archetype=archetype,
                performance_target='console',
                customization=variation
            )
            
            # Create training labels
            labels = self._generate_character_labels(character, archetype)
            
            # Calculate quality score
            quality_score = self._assess_character_quality(character)
            
            # Create training data entry
            data_entry = GameTrainingData(
                data_id=f"char_{i:04d}_{archetype}",
                data_type='character',
                content={
                    'character_data': asdict(character),
                    'archetype': archetype,
                    'variation_params': variation
                },
                labels=labels,
                quality_score=quality_score,
                hardware_context=hardware_context,
                timestamp=datetime.now().isoformat()
            )
            
            training_data.append(data_entry)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} character training samples")
        
        return training_data
    
    def _generate_random_character_params(self) -> Dict:
        """Generate random character variation parameters"""
        return {
            'head_width': random.uniform(0.7, 1.3),
            'head_height': random.uniform(0.8, 1.2),
            'eye_size': random.uniform(0.7, 1.3),
            'eye_spacing': random.uniform(0.8, 1.2),
            'nose_width': random.uniform(0.6, 1.4),
            'nose_length': random.uniform(0.7, 1.3),
            'mouth_width': random.uniform(0.7, 1.3),
            'jaw_width': random.uniform(0.8, 1.4),
            'age': random.uniform(0.1, 0.9),
            'skin_tone_r': random.uniform(0.5, 1.0),
            'skin_tone_g': random.uniform(0.3, 0.9),
            'skin_tone_b': random.uniform(0.2, 0.8),
            'battle_scars': random.uniform(0.0, 0.8),
            'heroic_proportions': random.uniform(0.0, 1.0),
            'stylization_level': random.uniform(0.0, 0.7)
        }
    
    def _generate_character_labels(self, character, archetype: str) -> List[str]:
        """Generate training labels for character"""
        labels = [archetype]
        
        features = character.features
        
        # Age labels
        if features.age < 0.3:
            labels.append('young')
        elif features.age > 0.7:
            labels.append('old')
        else:
            labels.append('adult')
        
        # Style labels
        if features.stylization_level > 0.5:
            labels.append('stylized')
        else:
            labels.append('realistic')
        
        # Heroic labels
        if features.heroic_proportions > 0.6:
            labels.append('heroic')
        
        # Battle-worn labels
        if features.battle_scars > 0.4:
            labels.append('battle_worn')
        
        # Archetype-specific labels
        if archetype == 'human_hero':
            labels.extend(['protagonist', 'noble', 'strong'])
        elif archetype == 'human_villain':
            labels.extend(['antagonist', 'menacing', 'dark'])
        elif archetype == 'elf':
            labels.extend(['fantasy', 'elegant', 'mystical'])
        elif archetype == 'orc':
            labels.extend(['fantasy', 'brutal', 'warrior'])
        elif archetype == 'dwarf':
            labels.extend(['fantasy', 'sturdy', 'craftsman'])
        elif archetype == 'cyborg':
            labels.extend(['sci-fi', 'technological', 'enhanced'])
        
        return labels
    
    def _assess_character_quality(self, character) -> float:
        """Assess the quality of generated character"""
        score = 0.0
        
        # Performance score (30%)
        perf_rating = character.performance_stats.get('overall_rating', 5.0)
        score += (perf_rating / 10.0) * 0.3
        
        # Mesh quality (25%)
        triangle_count = character.performance_stats.get('polygon_count', 0)
        if 2000 <= triangle_count <= 15000:  # Good range
            score += 0.25
        elif triangle_count > 0:
            score += 0.15
        
        # Texture quality (20%)
        texture_res = character.texture_maps.get('resolution', 0)
        if texture_res >= 1024:
            score += 0.20
        elif texture_res >= 512:
            score += 0.15
        elif texture_res > 0:
            score += 0.10
        
        # Feature completeness (25%)
        if len(character.blend_shapes) >= 10:
            score += 0.15
        if character.bone_weights:
            score += 0.10
        
        return min(1.0, score)
    
    def generate_dialogue_training_data(self, character_count: int = 50) -> List[GameTrainingData]:
        """Generate dialogue training data with character context"""
        if not self.ensure_avus_loaded():
            return []
        
        training_data = []
        
        # Generate characters first
        characters = []
        if self.face_generator:
            for i in range(character_count):
                archetype = random.choice(['human_hero', 'human_villain', 'elf', 'orc'])
                char = self.face_generator.generate_character_face(archetype=archetype)
                characters.append((char, archetype))
        
        # Generate dialogue for each character
        for i, (character, archetype) in enumerate(characters):
            # Generate context-appropriate dialogue
            dialogue_prompt = f"""
Generate dialogue for a {archetype} character in a fantasy RPG game.

Character traits based on appearance:
- Age: {character.features.age:.1f}
- Battle experience: {character.features.battle_scars:.1f}
- Heroic nature: {character.features.heroic_proportions:.1f}

Generate 3 different dialogue lines that this character might say:
1. A greeting
2. A quest-related statement
3. A combat taunt or encouragement

Make the dialogue match the character archetype and traits.

DIALOGUE:
"""
            
            try:
                dialogue_response = self.avus_brain.ask(dialogue_prompt, max_tokens=200)
                
                # Parse dialogue lines
                dialogue_lines = self._parse_dialogue_response(dialogue_response)
                
                # Create training data
                data_entry = GameTrainingData(
                    data_id=f"dialogue_{i:04d}_{archetype}",
                    data_type='dialogue',
                    content={
                        'character_id': character.character_id,
                        'archetype': archetype,
                        'character_features': asdict(character.features),
                        'dialogue_lines': dialogue_lines,
                        'context': 'RPG_fantasy'
                    },
                    labels=[archetype, 'dialogue', 'RPG', 'fantasy'],
                    quality_score=self._assess_dialogue_quality(dialogue_lines),
                    hardware_context=self._get_current_hardware_context(),
                    timestamp=datetime.now().isoformat()
                )
                
                training_data.append(data_entry)
                
            except Exception as e:
                print(f"Error generating dialogue for character {i}: {e}")
        
        return training_data
    
    def _parse_dialogue_response(self, response: str) -> List[str]:
        """Parse dialogue response into individual lines"""
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('DIALOGUE:'):
                # Remove numbering and clean up
                if line[0].isdigit() and '.' in line[:5]:
                    line = line.split('.', 1)[1].strip()
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                lines.append(line)
        return lines[:3]  # Limit to 3 lines
    
    def _assess_dialogue_quality(self, dialogue_lines: List[str]) -> float:
        """Assess quality of generated dialogue"""
        if not dialogue_lines:
            return 0.0
        
        score = 0.0
        
        # Check if we have expected number of lines
        if len(dialogue_lines) >= 3:
            score += 0.3
        
        # Check average length (good dialogue should be substantial but not too long)
        avg_length = sum(len(line) for line in dialogue_lines) / len(dialogue_lines)
        if 20 <= avg_length <= 100:
            score += 0.3
        
        # Check for variety (different starting words)
        starting_words = [line.split()[0].lower() if line.split() else '' for line in dialogue_lines]
        if len(set(starting_words)) == len(starting_words):
            score += 0.2
        
        # Check for appropriate content (no empty lines)
        if all(len(line.strip()) > 5 for line in dialogue_lines):
            score += 0.2
        
        return min(1.0, score)
    
    def _get_current_hardware_context(self) -> Dict:
        """Get current hardware context"""
        if self.hardware_awareness:
            return {
                'feeling': self.hardware_awareness.sense.feel(),
                'cpu_usage': self.hardware_awareness.sense.sense().cpu_usage,
                'memory_pressure': self.hardware_awareness.sense.sense().memory_pressure
            }
        return {}
    
    def generate_animation_training_data(self, count: int = 30) -> List[GameTrainingData]:
        """Generate animation training data"""
        training_data = []
        
        # Animation types for games
        animation_types = [
            'idle', 'walk', 'run', 'jump', 'attack', 'defend', 'cast_spell',
            'talk', 'emote_happy', 'emote_sad', 'emote_angry', 'death'
        ]
        
        for i in range(count):
            anim_type = random.choice(animation_types)
            
            # Generate animation parameters
            animation_data = {
                'type': anim_type,
                'duration': random.uniform(0.5, 3.0),
                'loop': anim_type in ['idle', 'walk', 'run'],
                'blend_shapes': self._generate_animation_blend_shapes(anim_type),
                'bone_rotations': self._generate_bone_rotations(anim_type),
                'root_motion': anim_type in ['walk', 'run', 'jump']
            }
            
            # Create labels
            labels = [anim_type, 'animation', 'game_ready']
            if animation_data['loop']:
                labels.append('looping')
            if animation_data['root_motion']:
                labels.append('root_motion')
            
            data_entry = GameTrainingData(
                data_id=f"anim_{i:04d}_{anim_type}",
                data_type='animation',
                content=animation_data,
                labels=labels,
                quality_score=random.uniform(0.7, 1.0),  # Simplified
                hardware_context=self._get_current_hardware_context(),
                timestamp=datetime.now().isoformat()
            )
            
            training_data.append(data_entry)
        
        return training_data
    
    def _generate_animation_blend_shapes(self, anim_type: str) -> Dict:
        """Generate blend shape weights for animation type"""
        blend_shapes = {}
        
        if anim_type == 'talk':
            # Mouth shapes for dialogue
            blend_shapes = {
                'mouth_open': random.uniform(0.3, 0.8),
                'A': random.uniform(0.0, 0.7),
                'E': random.uniform(0.0, 0.6),
                'O': random.uniform(0.0, 0.8)
            }
        elif anim_type == 'emote_happy':
            blend_shapes = {
                'smile': random.uniform(0.6, 1.0),
                'eyebrow_raise_left': random.uniform(0.2, 0.5),
                'eyebrow_raise_right': random.uniform(0.2, 0.5)
            }
        elif anim_type == 'emote_angry':
            blend_shapes = {
                'frown': random.uniform(0.5, 0.9),
                'eyebrow_furrow': random.uniform(0.6, 1.0)
            }
        elif anim_type == 'emote_sad':
            blend_shapes = {
                'frown': random.uniform(0.3, 0.7),
                'eyebrow_raise_left': random.uniform(0.1, 0.3),
                'eyebrow_raise_right': random.uniform(0.1, 0.3)
            }
        
        return blend_shapes
    
    def _generate_bone_rotations(self, anim_type: str) -> Dict:
        """Generate bone rotation data for animation type"""
        rotations = {}
        
        if anim_type == 'walk':
            rotations = {
                'left_leg': [random.uniform(-30, 30), 0, 0],
                'right_leg': [random.uniform(-30, 30), 0, 0],
                'spine': [random.uniform(-5, 5), random.uniform(-10, 10), 0]
            }
        elif anim_type == 'attack':
            rotations = {
                'right_arm': [random.uniform(-90, 45), random.uniform(-45, 45), 0],
                'spine': [random.uniform(-15, 15), random.uniform(-30, 30), 0]
            }
        elif anim_type == 'cast_spell':
            rotations = {
                'left_arm': [random.uniform(30, 90), random.uniform(-30, 30), 0],
                'right_arm': [random.uniform(30, 90), random.uniform(-30, 30), 0],
                'head': [random.uniform(-15, 15), 0, 0]
            }
        
        return rotations
    
    def start_training_session(self, model_type: str, data_types: List[str], 
                              data_count: int = 100) -> str:
        """Start a new training session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Starting training session: {session_id}")
        print(f"Model type: {model_type}")
        print(f"Data types: {data_types}")
        
        # Generate training data
        all_training_data = []
        
        if 'character' in data_types:
            print("Generating character training data...")
            char_data = self.generate_character_training_data(count=data_count//len(data_types))
            all_training_data.extend(char_data)
        
        if 'dialogue' in data_types:
            print("Generating dialogue training data...")
            dialogue_data = self.generate_dialogue_training_data(count=data_count//len(data_types))
            all_training_data.extend(dialogue_data)
        
        if 'animation' in data_types:
            print("Generating animation training data...")
            anim_data = self.generate_animation_training_data(count=data_count//len(data_types))
            all_training_data.extend(anim_data)
        
        # Get hardware profile
        hardware_profile = {}
        if self.hardware_awareness:
            hardware_profile = self.hardware_awareness.sense.get_hardware_profile()
        
        # Create training session
        session = TrainingSession(
            session_id=session_id,
            model_type=model_type,
            training_data=all_training_data,
            hardware_profile=hardware_profile,
            performance_metrics={},
            start_time=datetime.now().isoformat()
        )
        
        self.active_sessions[session_id] = session
        
        # Store in database
        self._store_training_session(session)
        
        print(f"Training session {session_id} started with {len(all_training_data)} samples")
        return session_id
    
    def _store_training_session(self, session: TrainingSession):
        """Store training session in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store session
        cursor.execute('''
            INSERT INTO training_sessions 
            (session_id, model_type, start_time, hardware_profile, data_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session.session_id,
            session.model_type,
            session.start_time,
            json.dumps(session.hardware_profile),
            len(session.training_data)
        ))
        
        # Store training data
        for data in session.training_data:
            cursor.execute('''
                INSERT INTO training_data 
                (data_id, session_id, data_type, content_json, labels_json, 
                 quality_score, hardware_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.data_id,
                session.session_id,
                data.data_type,
                json.dumps(data.content),
                json.dumps(data.labels),
                data.quality_score,
                json.dumps(data.hardware_context)
            ))
        
        conn.commit()
        conn.close()
    
    def get_training_statistics(self) -> Dict:
        """Get training pipeline statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Session stats
        cursor.execute('SELECT COUNT(*) FROM training_sessions')
        total_sessions = cursor.fetchone()[0]
        
        # Data stats
        cursor.execute('SELECT data_type, COUNT(*) FROM training_data GROUP BY data_type')
        data_by_type = dict(cursor.fetchall())
        
        # Quality stats
        cursor.execute('SELECT AVG(quality_score) FROM training_data')
        avg_quality = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_sessions': total_sessions,
            'data_by_type': data_by_type,
            'average_quality': avg_quality,
            'active_sessions': len(self.active_sessions)
        }

def main():
    """Demo the training pipeline"""
    print("="*60)
    print("GAME AI TRAINING PIPELINE")
    print("="*60)
    
    pipeline = GameAITrainingPipeline()
    
    # Start a comprehensive training session
    session_id = pipeline.start_training_session(
        model_type='game_character_ai',
        data_types=['character', 'dialogue', 'animation'],
        data_count=30
    )
    
    # Show statistics
    stats = pipeline.get_training_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Data by type: {stats['data_by_type']}")
    print(f"  Average quality: {stats['average_quality']:.2f}")
    
    print(f"\nTraining pipeline ready for AAA game development!")

if __name__ == "__main__":
    main()