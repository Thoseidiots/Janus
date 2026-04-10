"""
Audio Descriptor Generator
Generates sound design specifications and audio assets
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralDescriptionGenerator,
    ProceduralNumberGenerator
)

class AudioDescriptorGenerator:
    """Generates comprehensive audio descriptions and specifications for AAA games"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)
        
        # Audio building blocks
        self.audio_types = {
            'music': {
                'genres': ['orchestral', 'electronic', 'ambient', 'rock', 'folk', 'industrial'],
                'moods': ['epic', 'tense', 'peaceful', 'mysterious', 'triumphant', 'foreboding'],
                'tempos': ['largo', 'adagio', 'andante', 'moderato', 'allegro', 'presto'],
                'instruments': ['strings', 'brass', 'woodwinds', 'percussion', 'synthesizers', 'vocals']
            },
            'sfx': {
                'categories': ['combat', 'environment', 'ui', 'character', 'mechanical', 'magical'],
                'qualities': ['impact', 'continuous', 'one_shot', 'layered', 'procedural', 'adaptive'],
                'variations': ['pitch_shifted', 'filtered', 'distorted', 'reverberated', 'panned']
            },
            'ambience': {
                'environments': ['forest', 'urban', 'dungeon', 'space', 'underwater', 'mountain'],
                'weather': ['clear', 'rain', 'storm', 'wind', 'fog', 'snow'],
                'times': ['dawn', 'day', 'dusk', 'night', 'eternal']
            },
            'voice': {
                'types': ['character', 'narrator', 'monster', 'ambient', 'ui_feedback'],
                'emotions': ['neutral', 'angry', 'happy', 'sad', 'surprised', 'determined'],
                'languages': ['english', 'fantasy', 'alien', 'ancient', 'mechanical']
            }
        }
        
        self.audio_formats = [
            'wav', 'mp3', 'ogg', 'flac', 'aac', 'wma'
        ]
        
        self.sample_rates = [
            22050, 44100, 48000, 96000
        ]
        
        self.bit_depths = [
            16, 24, 32
        ]
        
        self.channel_configs = [
            'mono', 'stereo', '5.1', '7.1', 'ambisonic'
        ]
    
    def generate_audio_asset(self,
                            audio_type: str = 'music',
                            complexity: int = 5,
                            target_mood: str = None) -> Dict[str, Any]:
        """Generate a complete audio asset specification"""
        
        asset_id = str(uuid.uuid4())
        
        # Generate basic properties
        properties = self._generate_audio_properties(audio_type, complexity)
        
        # Generate technical specifications
        technical_specs = self._generate_technical_specs(audio_type, complexity)
        
        # Generate content description
        content = self._generate_content_description(audio_type, target_mood or properties.get('mood', 'neutral'))
        
        # Generate implementation details
        implementation = self._generate_implementation_details(audio_type, complexity)
        
        # Generate variations and layers
        variations = self._generate_variations(audio_type, complexity)
        
        # Generate usage contexts
        usage_contexts = self._generate_usage_contexts(audio_type)
        
        # Generate performance requirements
        performance = self._generate_performance_requirements(complexity)
        
        asset_data = {
            'id': asset_id,
            'name': self._generate_asset_name(audio_type, content['theme']),
            'type': audio_type,
            'complexity_rating': complexity,
            'properties': properties,
            'technical_specs': technical_specs,
            'content': content,
            'implementation': implementation,
            'variations': variations,
            'usage_contexts': usage_contexts,
            'performance_requirements': performance,
            'estimated_file_size': self._estimate_file_size(technical_specs, complexity),
            'accessibility_features': self._generate_accessibility_features(),
            'localization_notes': self._generate_localization_notes(audio_type)
        }
        
        return asset_data
    
    def _generate_audio_properties(self, audio_type: str, complexity: int) -> Dict[str, Any]:
        """Generate audio properties based on type"""
        
        type_info = self.audio_types.get(audio_type, self.audio_types['music'])
        
        if audio_type == 'music':
            return {
                'genre': self.rng.choice(type_info['genres']),
                'mood': self.rng.choice(type_info['moods']),
                'tempo': self.rng.choice(type_info['tempos']),
                'key': self.rng.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B']) + self.rng.choice(['', 'm', '7', 'maj7']),
                'time_signature': self.rng.choice(['4/4', '3/4', '6/8', '5/4']),
                'instruments': self.rng.sample(type_info['instruments'], self.rng.randint(2, 5)),
                'dynamics': self.rng.choice(['static', 'building', 'alternating', 'crescendo'])
            }
        elif audio_type == 'sfx':
            return {
                'category': self.rng.choice(type_info['categories']),
                'quality': self.rng.choice(type_info['qualities']),
                'duration': self.rng.uniform(0.1, 5.0),
                'variations': self.rng.sample(type_info['variations'], self.rng.randint(1, 3)),
                'impact': self.rng.choice(['subtle', 'moderate', 'strong', 'overwhelming']),
                'frequency_range': self.rng.choice(['low', 'mid', 'high', 'full_spectrum'])
            }
        elif audio_type == 'ambience':
            return {
                'environment': self.rng.choice(type_info['environments']),
                'weather': self.rng.choice(type_info['weather']),
                'time_of_day': self.rng.choice(type_info['times']),
                'density': self.rng.choice(['sparse', 'moderate', 'dense', 'overwhelming']),
                'loopable': self.rng.random() > 0.3,
                'interactive': self.rng.random() > 0.7
            }
        elif audio_type == 'voice':
            return {
                'voice_type': self.rng.choice(type_info['types']),
                'emotion': self.rng.choice(type_info['emotions']),
                'language': self.rng.choice(type_info['languages']),
                'gender': self.rng.choice(['male', 'female', 'neutral', 'other']),
                'age_group': self.rng.choice(['child', 'young_adult', 'adult', 'elder', 'ageless']),
                'accent': self.rng.choice(['neutral', 'regional', 'foreign', 'alien', 'robotic'])
            }
        
        return {'generic': True}
    
    def _generate_technical_specs(self, audio_type: str, complexity: int) -> Dict[str, Any]:
        """Generate technical specifications"""
        
        # Base specs
        specs = {
            'format': self.rng.choice(self.audio_formats),
            'sample_rate': self.rng.choice(self.sample_rates),
            'bit_depth': self.rng.choice(self.bit_depths),
            'channels': self.rng.choice(self.channel_configs)
        }
        
        # Adjust for complexity
        if complexity > 7:
            specs['sample_rate'] = max(specs['sample_rate'], 48000)
            specs['bit_depth'] = max(specs['bit_depth'], 24)
        
        # Type-specific adjustments
        if audio_type == 'music':
            specs['channels'] = self.rng.choice(['stereo', '5.1', '7.1'])
        elif audio_type == 'sfx':
            specs['channels'] = self.rng.choice(['mono', 'stereo'])
        elif audio_type == 'ambience':
            specs['channels'] = self.rng.choice(['stereo', '5.1', 'ambisonic'])
        
        # Add compression settings
        specs['compression'] = {
            'type': self.rng.choice(['lossless', 'lossy', 'adaptive']),
            'quality': self.rng.choice(['high', 'medium', 'low']) if specs['compression']['type'] == 'lossy' else 'original'
        }
        
        return specs
    
    def _generate_content_description(self, audio_type: str, target_mood: str) -> Dict[str, Any]:
        """Generate detailed content description"""
        
        themes = [
            'adventure', 'mystery', 'conflict', 'triumph', 'loss', 'discovery',
            'transformation', 'peril', 'wonder', 'despair', 'hope', 'betrayal'
        ]
        
        content = {
            'theme': self.rng.choice(themes),
            'mood': target_mood,
            'narrative_context': self._generate_narrative_context(audio_type),
            'emotional_arc': self._generate_emotional_arc(audio_type),
            'cultural_influence': self.rng.choice(['western', 'eastern', 'fantasy', 'sci-fi', 'historical', 'modern'])
        }
        
        if audio_type == 'music':
            content.update({
                'structure': self._generate_musical_structure(),
                'melodic_motifs': self._generate_melodic_motifs(),
                'harmonic_progression': self._generate_harmonic_progression()
            })
        elif audio_type == 'voice':
            content.update({
                'dialogue_type': self.rng.choice(['narrative', 'instruction', 'reaction', 'ambient']),
                'delivery_style': self.rng.choice(['formal', 'casual', 'urgent', 'calm', 'emotional'])
            })
        
        return content
    
    def _generate_implementation_details(self, audio_type: str, complexity: int) -> Dict[str, Any]:
        """Generate implementation details"""
        
        details = {
            'creation_method': self.rng.choice(['recorded', 'synthesized', 'hybrid', 'procedural']),
            'tools_used': self._generate_tools_list(audio_type),
            'processing_chain': self._generate_processing_chain(audio_type),
            'layering_approach': self._generate_layering_approach(complexity)
        }
        
        if audio_type == 'music':
            details.update({
                'arrangement_style': self.rng.choice(['full_orchestra', 'chamber', 'electronic', 'folk_ensemble']),
                'mixing_approach': self.rng.choice(['immersive', 'focused', 'dynamic', 'adaptive'])
            })
        elif audio_type == 'sfx':
            details.update({
                'recording_setup': self.rng.choice(['studio', 'field', 'synthesized', 'hybrid']),
                'post_processing': self._generate_post_processing_chain()
            })
        
        return details
    
    def _generate_variations(self, audio_type: str, complexity: int) -> List[Dict]:
        """Generate variations and layers"""
        
        variations = []
        variation_count = max(1, complexity // 2)
        
        for i in range(variation_count):
            variation = {
                'name': f"variation_{i+1}",
                'type': self.rng.choice(['intensity', 'mood', 'context', 'length']),
                'trigger_condition': self.rng.choice(['time', 'event', 'player_action', 'game_state']),
                'transition_type': self.rng.choice(['crossfade', 'immediate', 'stinger', 'morph'])
            }
            
            if audio_type == 'music':
                variation.update({
                    'tempo_change': self.rng.choice(['slower', 'faster', 'same']),
                    'instrumentation_change': self.rng.choice(['reduced', 'enhanced', 'different'])
                })
            elif audio_type == 'ambience':
                variation.update({
                    'density_change': self.rng.choice(['quieter', 'louder', 'different_sources'])
                })
            
            variations.append(variation)
        
        return variations
    
    def _generate_usage_contexts(self, audio_type: str) -> List[Dict]:
        """Generate usage contexts"""
        
        contexts = []
        
        context_types = {
            'music': ['menu', 'exploration', 'combat', 'cinematic', 'victory', 'defeat'],
            'sfx': ['ui_interaction', 'combat_action', 'environmental_interaction', 'character_action'],
            'ambience': ['background', 'zone_specific', 'weather_based', 'time_based'],
            'voice': ['dialogue', 'narration', 'ui_feedback', 'character_interaction']
        }
        
        for context_type in self.rng.sample(context_types.get(audio_type, ['general']), 3):
            context = {
                'type': context_type,
                'priority': self.rng.choice(['high', 'medium', 'low']),
                'loop_behavior': self.rng.choice(['loop', 'one_shot', 'conditional_loop']),
                'volume_range': [self.rng.uniform(0.1, 0.5), self.rng.uniform(0.5, 1.0)],
                'spatialization': self.rng.choice(['2d', '3d', 'positional'])
            }
            contexts.append(context)
        
        return contexts
    
    def _generate_performance_requirements(self, complexity: int) -> Dict[str, Any]:
        """Generate performance requirements"""
        
        return {
            'memory_usage': f"{self.rng.randint(1, 10) * complexity}MB",
            'cpu_usage': self.rng.choice(['low', 'medium', 'high']),
            'load_time': f"{self.rng.uniform(0.1, 2.0):.1f}s",
            'streaming_priority': self.rng.choice(['high', 'medium', 'low']),
            'platform_compatibility': ['pc', 'console', 'mobile'] if complexity < 8 else ['pc', 'console']
        }
    
    def _generate_asset_name(self, audio_type: str, theme: str) -> str:
        """Generate descriptive asset name"""
        
        name_patterns = [
            "{theme}_{type}_{mood}",
            "{type}_{theme}_v{version}",
            "AUD_{theme}_{type}",
            "{mood}_{theme}_{type}"
        ]
        
        pattern = self.rng.choice(name_patterns)
        version = self.rng.randint(1, 99)
        mood = self.rng.choice(['epic', 'tense', 'calm', 'dark', 'bright'])
        
        return pattern.format(
            theme=theme,
            type=audio_type,
            mood=mood,
            version=version
        )
    
    def _generate_narrative_context(self, audio_type: str) -> str:
        """Generate narrative context"""
        
        contexts = [
            "During a critical mission objective",
            "In the heat of battle",
            "Exploring ancient ruins",
            "Making a difficult moral choice",
            "Discovering a long-lost secret",
            "Facing an overwhelming threat",
            "Celebrating a hard-won victory",
            "Mourning a significant loss"
        ]
        
        return self.rng.choice(contexts)
    
    def _generate_emotional_arc(self, audio_type: str) -> List[str]:
        """Generate emotional arc"""
        
        arcs = [
            ['calm', 'building_tension', 'climax', 'resolution'],
            ['mysterious', 'revealing', 'surprising', 'satisfying'],
            ['hopeful', 'challenging', 'triumphant', 'reflective'],
            ['peaceful', 'disruptive', 'chaotic', 'harmonious']
        ]
        
        return self.rng.choice(arcs)
    
    def _generate_musical_structure(self) -> List[str]:
        """Generate musical structure"""
        
        structures = [
            ['introduction', 'development', 'climax', 'coda'],
            ['verse', 'chorus', 'bridge', 'outro'],
            ['exposition', 'rising_action', 'climax', 'falling_action'],
            ['theme', 'variation', 'development', 'recapitulation']
        ]
        
        return self.rng.choice(structures)
    
    def _generate_melodic_motifs(self) -> List[Dict]:
        """Generate melodic motifs"""
        
        motifs = []
        motif_count = self.rng.randint(1, 3)
        
        for _ in range(motif_count):
            motif = {
                'name': f"motif_{self.rng.randint(1, 100)}",
                'length': self.rng.randint(2, 8),  # notes
                'instrument': self.rng.choice(['piano', 'strings', 'brass', 'woodwinds', 'voice']),
                'character': self.rng.choice(['heroic', 'mysterious', 'menacing', 'hopeful', 'melancholic'])
            }
            motifs.append(motif)
        
        return motifs
    
    def _generate_harmonic_progression(self) -> List[str]:
        """Generate harmonic progression"""
        
        progressions = [
            ['I', 'IV', 'V', 'I'],
            ['i', 'III', 'VII', 'i'],
            ['I', 'vi', 'IV', 'V'],
            ['i', 'iv', 'V', 'i']
        ]
        
        return self.rng.choice(progressions)
    
    def _generate_tools_list(self, audio_type: str) -> List[str]:
        """Generate tools list"""
        
        tool_sets = {
            'music': ['DAW', 'virtual_instruments', 'MIDI_controller', 'audio_interface'],
            'sfx': ['microphone', 'field_recorder', 'audio_editor', 'effects_processor'],
            'ambience': ['multi-track_recorder', 'ambisonic_mic', 'noise_generator'],
            'voice': ['microphone', 'voice_processor', 'ADR_suite', 'dialogue_editor']
        }
        
        tools = tool_sets.get(audio_type, ['audio_workstation'])
        return self.rng.sample(tools, min(len(tools), 3))
    
    def _generate_processing_chain(self, audio_type: str) -> List[str]:
        """Generate processing chain"""
        
        chains = {
            'music': ['recording', 'editing', 'mixing', 'mastering'],
            'sfx': ['recording', 'editing', 'processing', 'normalization'],
            'ambience': ['capture', 'editing', 'spatialization', 'optimization'],
            'voice': ['recording', 'editing', 'processing', 'delivery']
        }
        
        return chains.get(audio_type, ['processing'])
    
    def _generate_layering_approach(self, complexity: int) -> str:
        """Generate layering approach"""
        
        approaches = [
            'additive_synthesis',
            'sample_layering',
            'multi-track_recording',
            'procedural_generation'
        ]
        
        return self.rng.choice(approaches)
    
    def _generate_post_processing_chain(self) -> List[str]:
        """Generate post-processing chain"""
        
        return self.rng.sample([
            'equalization', 'compression', 'reverb', 'delay',
            'distortion', 'modulation', 'spatialization', 'normalization'
        ], self.rng.randint(2, 5))
    
    def _estimate_file_size(self, technical_specs: Dict, complexity: int) -> str:
        """Estimate file size"""
        
        base_size = 1  # MB
        
        # Adjust for specs
        if technical_specs.get('sample_rate', 44100) > 44100:
            base_size *= 2
        if technical_specs.get('bit_depth', 16) > 16:
            base_size *= 1.5
        if technical_specs.get('channels') in ['5.1', '7.1']:
            base_size *= 6
        elif technical_specs.get('channels') == 'stereo':
            base_size *= 2
        
        # Adjust for complexity
        base_size *= (1 + complexity * 0.2)
        
        return f"{base_size:.1f}MB"
    
    def _generate_accessibility_features(self) -> List[str]:
        """Generate accessibility features"""
        
        features = [
            'subtitles_for_voice',
            'visual_audio_cues',
            'adjustable_volume_levels',
            'mono_audio_option',
            'high_contrast_audio_markers',
            'reduced_motion_audio_alternatives'
        ]
        
        return self.rng.sample(features, self.rng.randint(2, 4))
    
    def _generate_localization_notes(self, audio_type: str) -> Dict[str, Any]:
        """Generate localization notes"""
        
        if audio_type == 'voice':
            return {
                'requires_translation': True,
                'requires_voicing': True,
                'cultural_adaptations': self.rng.sample(['idioms', 'pronunciation', 'tone'], self.rng.randint(1, 3)),
                'estimated_languages': self.rng.randint(5, 15)
            }
        else:
            return {
                'requires_translation': False,
                'requires_voicing': False,
                'cultural_adaptations': [],
                'estimated_languages': 1
            }
    
    def generate_batch(self,
                       count: int,
                       audio_types: List[str] = None) -> List[Dict]:
        """Generate multiple audio assets"""
        
        if audio_types is None:
            audio_types = list(self.audio_types.keys())
        
        assets = []
        
        for i in range(count):
            audio_type = self.rng.choice(audio_types)
            complexity = self.rng.randint(3, 10)
            mood = self.rng.choice(['epic', 'tense', 'calm', 'mysterious', 'triumphant'])
            
            asset = self.generate_audio_asset(
                audio_type=audio_type,
                complexity=complexity,
                target_mood=mood
            )
            assets.append(asset)
        
        return assets