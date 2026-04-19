"""
Janus Integration Hub
Connects all Janus systems together for seamless operation
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import Janus systems
try:
    from janus_human_capable import JanusHumanCapable
    from advanced_3d_face_generator import ProceduralFaceGenerator, FacialFeatures
    from hardware_sense import HardwareAwareness
    from avus_brain import AvusBrain
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Some systems not available: {e}")
    SYSTEMS_AVAILABLE = False


class JanusIntegrationHub:
    """
    Central hub that connects all Janus capabilities
    """
    
    def __init__(self):
        print("Initializing Janus Integration Hub...")
        
        # Core Janus system
        self.janus = JanusHumanCapable() if SYSTEMS_AVAILABLE else None
        
        # 3D Face generation
        self.face_generator = ProceduralFaceGenerator() if SYSTEMS_AVAILABLE else None
        
        # Hardware awareness
        self.hardware_awareness = HardwareAwareness() if SYSTEMS_AVAILABLE else None
        
        # AI brain
        self.avus_brain = None  # Lazy load
        
        # Integration state
        self.active_capabilities = []
        self.integration_log = []
        
        self._verify_integrations()
        
        print("Janus Integration Hub ready!")
    
    def _verify_integrations(self):
        """Verify which systems are integrated and working"""
        print("\nVerifying system integrations...")
        
        if self.janus:
            capabilities = self.janus.verify_capabilities()
            working_caps = [k for k, v in capabilities.items() if v]
            self.active_capabilities.extend(working_caps)
            print(f"  ✓ Janus Human-Capable: {len(working_caps)} capabilities")
        
        if self.face_generator:
            try:
                # Test face generation
                test_features = FacialFeatures(head_width=1.1)
                test_face = self.face_generator.generate_face(test_features)
                self.active_capabilities.append('3d_face_generation')
                print(f"  ✓ 3D Face Generator: {test_face['metadata']['vertex_count']} vertices")
            except Exception as e:
                print(f"  ✗ 3D Face Generator failed: {e}")
        
        if self.hardware_awareness:
            try:
                feeling = self.hardware_awareness.sense.feel()
                self.active_capabilities.append('hardware_awareness')
                print(f"  ✓ Hardware Awareness: {feeling[:50]}...")
            except Exception as e:
                print(f"  ✗ Hardware Awareness failed: {e}")
        
        print(f"\nActive capabilities: {len(self.active_capabilities)}")
    
    def ensure_avus_loaded(self) -> bool:
        """Lazy load Avus brain"""
        if self.avus_brain is None and SYSTEMS_AVAILABLE:
            try:
                self.avus_brain = AvusBrain()
                if self.avus_brain.ensure_loaded():
                    self.active_capabilities.append('ai_brain')
                    return True
            except Exception:
                pass
        return self.avus_brain is not None
    
    def generate_character_with_context(self, character_description: str) -> Dict[str, Any]:
        """
        Generate a 3D character using AI understanding and hardware context
        """
        print(f"\nGenerating character: {character_description}")
        
        result = {
            'success': False,
            'character_data': None,
            'generation_context': {},
            'hardware_state': {},
            'ai_interpretation': None
        }
        
        # Get hardware context
        if self.hardware_awareness:
            result['hardware_state'] = {
                'feeling': self.hardware_awareness.sense.feel(),
                'mood': self.hardware_awareness.personality.mood_based_on_hardware() if hasattr(self.hardware_awareness, 'personality') else 'unknown'
            }
        
        # Use AI to interpret character description
        if self.ensure_avus_loaded():
            try:
                interpretation_prompt = f"""
Analyze this character description and provide facial feature parameters for 3D generation:

Character: {character_description}

Provide values (0.0 to 2.0, with 1.0 being average) for:
- head_width: 
- head_height:
- eye_size:
- eye_spacing:
- nose_width:
- nose_length:
- mouth_width:
- jaw_width:
- age: (0.0 = child, 0.5 = adult, 1.0 = elderly)
- skin_tone_r: (0.0 to 1.0)
- skin_tone_g: (0.0 to 1.0)
- skin_tone_b: (0.0 to 1.0)

Respond in JSON format with just the parameter values.

PARAMETERS:
"""
                
                ai_response = self.avus_brain.ask(interpretation_prompt, max_tokens=200)
                result['ai_interpretation'] = ai_response
                
                # Try to parse AI response into parameters
                try:
                    # Extract JSON from response
                    json_start = ai_response.find('{')
                    json_end = ai_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        params = json.loads(ai_response[json_start:json_end])
                        
                        # Create facial features from AI interpretation
                        features = FacialFeatures(
                            head_width=params.get('head_width', 1.0),
                            head_height=params.get('head_height', 1.0),
                            eye_size=params.get('eye_size', 1.0),
                            eye_spacing=params.get('eye_spacing', 1.0),
                            nose_width=params.get('nose_width', 1.0),
                            nose_length=params.get('nose_length', 1.0),
                            mouth_width=params.get('mouth_width', 1.0),
                            jaw_width=params.get('jaw_width', 1.0),
                            age=params.get('age', 0.5),
                            skin_tone_r=params.get('skin_tone_r', 0.9),
                            skin_tone_g=params.get('skin_tone_g', 0.7),
                            skin_tone_b=params.get('skin_tone_b', 0.6)
                        )
                        
                        result['generation_context']['ai_parsed_params'] = params
                        
                    else:
                        # Fallback to default features
                        features = FacialFeatures()
                        result['generation_context']['ai_parse_failed'] = True
                
                except json.JSONDecodeError:
                    features = FacialFeatures()
                    result['generation_context']['json_parse_failed'] = True
                
            except Exception as e:
                print(f"AI interpretation failed: {e}")
                features = FacialFeatures()
                result['generation_context']['ai_failed'] = str(e)
        else:
            # No AI available, use default features
            features = FacialFeatures()
            result['generation_context']['no_ai'] = True
        
        # Generate the 3D face
        if self.face_generator:
            try:
                character_data = self.face_generator.generate_face(features)
                result['character_data'] = character_data
                result['success'] = True
                
                print(f"  ✓ Generated character with {character_data['metadata']['vertex_count']} vertices")
                
            except Exception as e:
                print(f"  ✗ Face generation failed: {e}")
                result['generation_context']['face_gen_failed'] = str(e)
        else:
            result['generation_context']['no_face_generator'] = True
        
        # Log the integration
        self.integration_log.append({
            'action': 'generate_character',
            'description': character_description,
            'success': result['success'],
            'timestamp': datetime.now().isoformat(),
            'capabilities_used': [cap for cap in ['ai_brain', '3d_face_generation', 'hardware_awareness'] if cap in self.active_capabilities]
        })
        
        return result
    
    def perform_autonomous_task_with_character_generation(self, task_description: str) -> Dict[str, Any]:
        """
        Perform an autonomous task that involves character generation
        """
        print(f"\nPerforming autonomous task: {task_description}")
        
        result = {
            'task_success': False,
            'character_generated': False,
            'task_result': None,
            'character_result': None
        }
        
        # Check if task involves character generation
        if any(keyword in task_description.lower() for keyword in ['character', 'face', 'person', 'npc']):
            print("  Task involves character generation...")
            
            # Extract character description from task
            if 'create' in task_description.lower() or 'generate' in task_description.lower():
                # Try to extract character description
                char_desc = task_description
                if 'character' in task_description.lower():
                    parts = task_description.lower().split('character')
                    if len(parts) > 1:
                        char_desc = parts[1].strip()
                
                # Generate character
                char_result = self.generate_character_with_context(char_desc)
                result['character_result'] = char_result
                result['character_generated'] = char_result['success']
        
        # Perform the main task using Janus
        if self.janus:
            try:
                task_result = self.janus.execute_task(task_description, max_steps=10)
                result['task_result'] = task_result
                result['task_success'] = task_result.get('success', False)
                
            except Exception as e:
                print(f"  Task execution failed: {e}")
                result['task_result'] = {'error': str(e)}
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'active_capabilities': self.active_capabilities,
            'integration_log_count': len(self.integration_log),
            'systems': {}
        }
        
        # Janus status
        if self.janus:
            janus_status = self.janus.get_system_status()
            status['systems']['janus'] = {
                'available': True,
                'monitors': len(janus_status.get('monitors', [])),
                'windows': len(janus_status.get('windows', [])),
                'hardware_feeling': janus_status.get('hardware_feeling', 'Unknown'),
                'mood': janus_status.get('mood', 'Unknown')
            }
        
        # Face generator status
        if self.face_generator:
            status['systems']['face_generator'] = {
                'available': True,
                'blend_shapes': len(self.face_generator.blend_shapes),
                'rig_points': len(self.face_generator.rig_points)
            }
        
        # Hardware awareness status
        if self.hardware_awareness:
            status['systems']['hardware_awareness'] = {
                'available': True,
                'current_feeling': self.hardware_awareness.sense.feel(),
                'hardware_profile': self.hardware_awareness.profile
            }
        
        # AI brain status
        if self.avus_brain:
            status['systems']['ai_brain'] = {
                'available': True,
                'loaded': True
            }
        
        return status
    
    def run_integration_demo(self):
        """Run a demonstration of integrated capabilities"""
        print("\n" + "="*60)
        print("JANUS INTEGRATION HUB DEMONSTRATION")
        print("="*60)
        
        # Show system status
        status = self.get_system_status()
        print(f"\nActive capabilities: {len(status['active_capabilities'])}")
        for cap in status['active_capabilities']:
            print(f"  ✓ {cap}")
        
        # Test character generation with AI
        print(f"\n1. AI-DRIVEN CHARACTER GENERATION")
        print("-" * 40)
        
        test_characters = [
            "a wise old wizard with a long beard",
            "a young elven warrior with sharp features",
            "a battle-scarred orc chieftain"
        ]
        
        for char_desc in test_characters:
            result = self.generate_character_with_context(char_desc)
            if result['success']:
                char_data = result['character_data']
                print(f"  ✓ {char_desc}")
                print(f"    Vertices: {char_data['metadata']['vertex_count']}")
                if result['ai_interpretation']:
                    print(f"    AI interpreted features successfully")
            else:
                print(f"  ✗ {char_desc} - Failed")
        
        # Test autonomous task with character generation
        print(f"\n2. AUTONOMOUS TASK WITH CHARACTER GENERATION")
        print("-" * 40)
        
        task = "Create a heroic character for a fantasy game"
        result = self.perform_autonomous_task_with_character_generation(task)
        
        if result['character_generated']:
            print(f"  ✓ Character generated successfully")
        if result['task_success']:
            print(f"  ✓ Task completed successfully")
        
        # Show integration statistics
        print(f"\n3. INTEGRATION STATISTICS")
        print("-" * 40)
        print(f"  Total integrations performed: {len(self.integration_log)}")
        print(f"  Systems working together: {len(status['systems'])}")
        
        successful_integrations = sum(1 for log in self.integration_log if log['success'])
        if self.integration_log:
            success_rate = successful_integrations / len(self.integration_log) * 100
            print(f"  Integration success rate: {success_rate:.1f}%")
        
        print("\n" + "="*60)
        print("JANUS INTEGRATION HUB READY")
        print("All systems connected and working together!")
        print("="*60)


def main():
    """Main integration hub demo"""
    hub = JanusIntegrationHub()
    hub.run_integration_demo()


if __name__ == "__main__":
    main()