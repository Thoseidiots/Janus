"""
Janus Training Connector
Connects 3D face generation to training pipelines
Generates training data that includes hardware awareness and 3D assets
"""

import json
import random
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict

try:
    from janus_integration_hub import JanusIntegrationHub
    from advanced_3d_face_generator import FacialFeatures
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Integration not available: {e}")
    INTEGRATION_AVAILABLE = False


class JanusTrainingConnector:
    """
    Connects Janus capabilities to training data generation
    """
    
    def __init__(self):
        self.hub = JanusIntegrationHub() if INTEGRATION_AVAILABLE else None
        self.training_data = []
        
    def generate_character_training_batch(self, count: int = 20) -> List[Dict]:
        """Generate a batch of character training data"""
        if not self.hub:
            return []
        
        print(f"Generating {count} character training samples...")
        
        # Character archetypes for training
        character_types = [
            "heroic knight with strong jaw",
            "wise wizard with aged features", 
            "elven archer with delicate features",
            "battle-scarred warrior",
            "young apprentice with soft features",
            "mysterious rogue with sharp eyes",
            "noble princess with refined features",
            "gruff dwarf with broad face",
            "ancient sage with weathered skin",
            "fierce barbarian with wild features"
        ]
        
        training_batch = []
        
        for i in range(count):
            # Select character type
            char_type = random.choice(character_types)
            
            # Generate character using integrated system
            result = self.hub.generate_character_with_context(char_type)
            
            if result['success']:
                # Create training data entry
                training_entry = {
                    'id': f"train_char_{i:04d}",
                    'type': 'character_generation',
                    'input': {
                        'description': char_type,
                        'hardware_context': result['hardware_state']
                    },
                    'output': {
                        'character_data': result['character_data'],
                        'generation_success': True
                    },
                    'metadata': {
                        'ai_interpretation': result.get('ai_interpretation'),
                        'generation_context': result['generation_context'],
                        'timestamp': datetime.now().isoformat(),
                        'capabilities_used': self.hub.active_capabilities
                    },
                    'labels': self._generate_labels(char_type, result['character_data'])
                }
                
                training_batch.append(training_entry)
                
                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1}/{count} samples")
        
        self.training_data.extend(training_batch)
        print(f"Training batch complete: {len(training_batch)} samples generated")
        
        return training_batch
    
    def _generate_labels(self, description: str, character_data: Dict) -> List[str]:
        """Generate training labels from character description and data"""
        labels = []
        
        # Extract labels from description
        desc_lower = description.lower()
        
        if 'hero' in desc_lower or 'knight' in desc_lower:
            labels.extend(['heroic', 'noble', 'protagonist'])
        if 'wizard' in desc_lower or 'sage' in desc_lower:
            labels.extend(['magical', 'wise', 'elderly'])
        if 'elf' in desc_lower:
            labels.extend(['fantasy', 'elegant', 'pointed_ears'])
        if 'warrior' in desc_lower or 'barbarian' in desc_lower:
            labels.extend(['combat', 'strong', 'battle_ready'])
        if 'young' in desc_lower or 'apprentice' in desc_lower:
            labels.extend(['youthful', 'inexperienced'])
        if 'dwarf' in desc_lower:
            labels.extend(['fantasy', 'sturdy', 'bearded'])
        if 'rogue' in desc_lower:
            labels.extend(['stealthy', 'cunning', 'shadowy'])
        if 'princess' in desc_lower or 'noble' in desc_lower:
            labels.extend(['royal', 'refined', 'elegant'])
        
        # Add technical labels based on character data
        vertex_count = character_data['metadata']['vertex_count']
        if vertex_count > 1500:
            labels.append('high_detail')
        elif vertex_count > 800:
            labels.append('medium_detail')
        else:
            labels.append('low_detail')
        
        # Add feature-based labels
        features = character_data.get('features', {})
        if features.get('age', 0.5) > 0.7:
            labels.append('elderly')
        elif features.get('age', 0.5) < 0.3:
            labels.append('young')
        
        if features.get('jaw_width', 1.0) > 1.2:
            labels.append('strong_jaw')
        if features.get('eye_size', 1.0) > 1.1:
            labels.append('large_eyes')
        
        return list(set(labels))  # Remove duplicates
    
    def generate_hardware_aware_training_data(self, count: int = 10) -> List[Dict]:
        """Generate training data that includes hardware awareness"""
        if not self.hub or not self.hub.hardware_awareness:
            return []
        
        print(f"Generating {count} hardware-aware training samples...")
        
        training_batch = []
        
        for i in range(count):
            # Get current hardware state
            hardware_state = {
                'feeling': self.hub.hardware_awareness.sense.feel(),
                'cpu_usage': self.hub.hardware_awareness.sense.sense().cpu_usage,
                'memory_pressure': self.hub.hardware_awareness.sense.sense().memory_pressure,
                'temperature': self.hub.hardware_awareness.sense.sense().cpu_temp
            }
            
            # Generate character based on hardware state
            if hardware_state['cpu_usage'] > 70:
                char_desc = "intense warrior with fierce expression"
            elif hardware_state['memory_pressure'] > 0.8:
                char_desc = "wise elder with thoughtful features"
            elif hardware_state.get('temperature', 0) and hardware_state['temperature'] > 70:
                char_desc = "fire mage with passionate features"
            else:
                char_desc = "calm scholar with serene expression"
            
            # Generate character
            result = self.hub.generate_character_with_context(char_desc)
            
            if result['success']:
                training_entry = {
                    'id': f"hw_aware_{i:04d}",
                    'type': 'hardware_aware_generation',
                    'input': {
                        'hardware_state': hardware_state,
                        'derived_description': char_desc
                    },
                    'output': {
                        'character_data': result['character_data']
                    },
                    'metadata': {
                        'hardware_influence': True,
                        'timestamp': datetime.now().isoformat()
                    },
                    'labels': ['hardware_influenced'] + self._generate_labels(char_desc, result['character_data'])
                }
                
                training_batch.append(training_entry)
        
        print(f"Hardware-aware training batch complete: {len(training_batch)} samples")
        return training_batch
    
    def export_training_data(self, filename: str = None) -> str:
        """Export training data to JSON file"""
        if filename is None:
            filename = f"janus_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'metadata': {
                'generator': 'Janus Training Connector',
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(self.training_data),
                'capabilities_used': self.hub.active_capabilities if self.hub else []
            },
            'training_data': self.training_data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Training data exported to {filename}")
        print(f"  Total samples: {len(self.training_data)}")
        
        # Show sample distribution
        sample_types = {}
        for sample in self.training_data:
            sample_type = sample['type']
            sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
        
        print("  Sample distribution:")
        for stype, count in sample_types.items():
            print(f"    {stype}: {count}")
        
        return filename
    
    def get_training_statistics(self) -> Dict:
        """Get statistics about generated training data"""
        if not self.training_data:
            return {'total_samples': 0}
        
        stats = {
            'total_samples': len(self.training_data),
            'sample_types': {},
            'label_distribution': {},
            'success_rate': 0.0,
            'average_vertex_count': 0.0
        }
        
        # Count sample types
        for sample in self.training_data:
            stype = sample['type']
            stats['sample_types'][stype] = stats['sample_types'].get(stype, 0) + 1
        
        # Count labels
        for sample in self.training_data:
            for label in sample.get('labels', []):
                stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + 1
        
        # Calculate success rate
        successful = sum(1 for sample in self.training_data 
                        if sample['output'].get('generation_success', False))
        stats['success_rate'] = successful / len(self.training_data) * 100
        
        # Calculate average vertex count
        vertex_counts = []
        for sample in self.training_data:
            char_data = sample['output'].get('character_data', {})
            if 'metadata' in char_data:
                vertex_counts.append(char_data['metadata'].get('vertex_count', 0))
        
        if vertex_counts:
            stats['average_vertex_count'] = sum(vertex_counts) / len(vertex_counts)
        
        return stats


def main():
    """Demo the training connector"""
    print("="*60)
    print("JANUS TRAINING CONNECTOR")
    print("="*60)
    
    connector = JanusTrainingConnector()
    
    if not connector.hub:
        print("Integration hub not available - cannot generate training data")
        return
    
    # Generate character training batch
    print("\n1. GENERATING CHARACTER TRAINING BATCH")
    print("-" * 40)
    char_batch = connector.generate_character_training_batch(count=5)
    
    # Generate hardware-aware training data
    print("\n2. GENERATING HARDWARE-AWARE TRAINING DATA")
    print("-" * 40)
    hw_batch = connector.generate_hardware_aware_training_data(count=3)
    
    # Show statistics
    print("\n3. TRAINING DATA STATISTICS")
    print("-" * 40)
    stats = connector.get_training_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Average vertex count: {stats['average_vertex_count']:.0f}")
    
    print("\n  Sample types:")
    for stype, count in stats['sample_types'].items():
        print(f"    {stype}: {count}")
    
    print("\n  Top labels:")
    sorted_labels = sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels[:10]:
        print(f"    {label}: {count}")
    
    # Export training data
    print("\n4. EXPORTING TRAINING DATA")
    print("-" * 40)
    filename = connector.export_training_data()
    
    print(f"\nTraining connector ready!")
    print(f"Generated {stats['total_samples']} training samples")


if __name__ == "__main__":
    main()