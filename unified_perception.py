“””
Janus Unified Perception System
Integrates vision, voice, and multi-modal fusion with existing Janus core
Provides grounded episodic memory and continuous sensing
“””

import threading
import time
from typing import Optional, Callable, Dict, List
from datetime import datetime
from pathlib import Path

# Import perception modules

from vision_perception import VisionPerceptionSystem, SceneContext
from voice_io import VoiceIOSystem, Utterance
from multimodal_fusion import (
MultiModalPerceptionFusion,
EpisodicMemory,
EventType,
ModalityType
)

# Try to import existing Janus systems

try:
from memory import Memory
from consciousness import Consciousness
JANUS_CORE_AVAILABLE = True
except ImportError:
JANUS_CORE_AVAILABLE = False
Memory = None
Consciousness = None

class PerceptionConfig:
“”“Configuration for perception system”””
def **init**(self):
# Vision settings
self.enable_vision = True
self.camera_id = 0
self.vision_fps = 10

```
    # Voice settings
    self.enable_voice = True
    self.sample_rate = 16000
    self.enable_wake_word = True
    self.wake_words = ['janus', 'hey janus']
    
    # Multi-modal settings
    self.enable_fusion = True
    self.episode_duration = 30  # seconds
    self.min_memory_importance = 0.5
    
    # Integration settings
    self.integrate_with_janus_core = True
    self.auto_narrate_thoughts = False
    self.voice_responses_enabled = True
```

class UnifiedPerceptionSystem:
“””
Unified perception system for Janus
Integrates vision, voice, and multi-modal fusion
“””

```
def __init__(self, 
             config: Optional[PerceptionConfig] = None,
             workspace_dir: str = "/tmp/janus_perception"):
    
    self.config = config or PerceptionConfig()
    self.workspace_dir = Path(workspace_dir)
    self.workspace_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize subsystems
    self.vision: Optional[VisionPerceptionSystem] = None
    self.voice: Optional[VoiceIOSystem] = None
    self.fusion: Optional[MultiModalPerceptionFusion] = None
    
    # Integration with Janus core
    self.memory: Optional[Memory] = None
    self.consciousness: Optional[Consciousness] = None
    
    # State
    self.running = False
    self.last_scene: Optional[SceneContext] = None
    self.conversation_active = False
    
    # Performance metrics
    self.metrics = {
        'scenes_processed': 0,
        'utterances_processed': 0,
        'events_detected': 0,
        'memories_created': 0,
        'uptime_seconds': 0
    }
    self.start_time = None
    
    self._initialize_subsystems()

def _initialize_subsystems(self):
    """Initialize all perception subsystems"""
    
    # Vision
    if self.config.enable_vision:
        try:
            self.vision = VisionPerceptionSystem(
                camera_id=self.config.camera_id,
                memory_dir=str(self.workspace_dir / "vision")
            )
            
            # Setup vision callbacks
            self.vision.on_scene_change = self._on_scene_change
            self.vision.on_person_detected = self._on_person_detected
            
            print("✓ Vision system initialized")
        except Exception as e:
            print(f"⚠ Vision system failed to initialize: {e}")
            self.vision = None
    
    # Voice
    if self.config.enable_voice:
        try:
            self.voice = VoiceIOSystem(
                sample_rate=self.config.sample_rate,
                memory_dir=str(self.workspace_dir / "voice")
            )
            
            # Setup voice callbacks
            self.voice.on_wake_word = self._on_wake_word
            self.voice.on_transcription = self._on_transcription
            self.voice.on_utterance_complete = self._on_utterance_complete
            
            self.voice.wake_word_enabled = self.config.enable_wake_word
            
            print("✓ Voice I/O system initialized")
        except Exception as e:
            print(f"⚠ Voice system failed to initialize: {e}")
            self.voice = None
    
    # Multi-modal fusion
    if self.config.enable_fusion:
        self.fusion = MultiModalPerceptionFusion(
            memory_dir=str(self.workspace_dir / "multimodal")
        )
        
        # Setup fusion callbacks
        self.fusion.on_event_detected = self._on_event_detected
        self.fusion.on_memory_created = self._on_memory_created
        
        print("✓ Multi-modal fusion initialized")
    
    # Try to connect to Janus core
    if self.config.integrate_with_janus_core and JANUS_CORE_AVAILABLE:
        try:
            self.memory = Memory()
            self.consciousness = Consciousness()
            print("✓ Connected to Janus core systems")
        except Exception as e:
            print(f"⚠ Could not connect to Janus core: {e}")

def start(self):
    """Start all perception systems"""
    print("\n" + "="*60)
    print("STARTING JANUS UNIFIED PERCEPTION SYSTEM")
    print("="*60 + "\n")
    
    self.running = True
    self.start_time = datetime.now()
    
    # Start vision
    if self.vision:
        self.vision.start()
        print("→ Vision perception started")
    
    # Start voice
    if self.voice:
        self.voice.start()
        print("→ Voice I/O started")
        
        # Initial greeting
        if self.config.voice_responses_enabled:
            self.voice.speak(
                "Perception systems online. I can now see and hear.",
                voice_style='friendly'
            )
    
    # Start fusion
    if self.fusion:
        self.fusion.start()
        print("→ Multi-modal fusion started")
    
    print("\n✓ All systems operational")
    print("Janus is now perceiving the world...\n")

def stop(self):
    """Stop all perception systems"""
    print("\n" + "="*60)
    print("STOPPING PERCEPTION SYSTEMS")
    print("="*60 + "\n")
    
    self.running = False
    
    # Stop subsystems
    if self.vision:
        self.vision.stop()
        print("→ Vision stopped")
    
    if self.voice:
        self.voice.stop()
        print("→ Voice I/O stopped")
    
    if self.fusion:
        self.fusion.stop()
        print("→ Fusion stopped")
    
    # Calculate uptime
    if self.start_time:
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.metrics['uptime_seconds'] = uptime
    
    # Show statistics
    self._print_statistics()

# === Callback Handlers ===

def _on_scene_change(self, scene: SceneContext):
    """Handle scene change event"""
    self.last_scene = scene
    self.metrics['scenes_processed'] += 1
    
    # Send to fusion
    if self.fusion:
        self.fusion.ingest_vision(scene)
    
    # Narrate if enabled
    if self.config.auto_narrate_thoughts and self.voice:
        self.voice.speak(
            f"I observe: {scene.summary}",
            voice_style='serious'
        )

def _on_person_detected(self, objects):
    """Handle person detection"""
    people_count = len([obj for obj in objects if obj.class_name == "person"])
    
    # Greet if wake word not enabled (always listening)
    if not self.config.enable_wake_word and self.voice:
        self.voice.speak(
            f"I notice {people_count} person in my view",
            voice_style='friendly'
        )

def _on_wake_word(self):
    """Handle wake word detection"""
    print("\n[WAKE WORD] Janus activated")
    
    # Visual feedback if available
    if self.last_scene and self.voice:
        self.voice.speak(
            f"Yes, I see you. I'm currently observing {self.last_scene.scene_type} scene.",
            voice_style='friendly'
        )

def _on_transcription(self, text: str, confidence: float):
    """Handle speech transcription"""
    print(f"[STT] {text} ({confidence:.2f})")
    
    # Send to fusion
    if self.fusion:
        self.fusion.ingest_text(text, source='speech')
    
    # Send to Janus core consciousness for processing
    if self.consciousness:
        response = self.consciousness.process_input(text)
        if response and self.voice and self.config.voice_responses_enabled:
            self.voice.speak(response, voice_style='friendly')

def _on_utterance_complete(self, utterance: Utterance):
    """Handle complete utterance"""
    self.metrics['utterances_processed'] += 1
    
    # Send to fusion
    if self.fusion:
        self.fusion.ingest_audio(utterance)
    
    # Store in Janus memory
    if self.memory:
        self.memory.store_experience({
            'type': 'conversation',
            'utterance': utterance.text,
            'speaker': utterance.speaker,
            'timestamp': utterance.timestamp
        })

def _on_event_detected(self, event):
    """Handle multi-modal event detection"""
    self.metrics['events_detected'] += 1
    
    print(f"\n[EVENT] {event.event_type.value}: {event.description}")
    
    # Important events trigger narration
    if event.event_type in [EventType.PERSON_ENTERED, EventType.WAKE_WORD]:
        if self.config.auto_narrate_thoughts and self.voice:
            self.voice.speak(
                f"Event detected: {event.description}",
                voice_style='serious'
            )

def _on_memory_created(self, memory: EpisodicMemory):
    """Handle episodic memory creation"""
    self.metrics['memories_created'] += 1
    
    print(f"\n[MEMORY] Created: {memory.summary}")
    
    # Store in Janus hierarchical memory
    if self.memory:
        self.memory.store_episodic_memory({
            'memory_id': memory.memory_id,
            'timestamp': memory.timestamp,
            'duration': memory.duration,
            'summary': memory.summary,
            'importance': memory.importance,
            'entities': memory.entities,
            'events': [e.event_type.value for e in memory.events]
        })

# === High-level Operations ===

def describe_current_perception(self) -> str:
    """Get current perceptual state as text"""
    parts = []
    
    # Vision
    if self.last_scene:
        parts.append(f"Visual: {self.last_scene.summary}")
    
    # Audio
    if self.voice and self.voice.conversation:
        recent = self.voice.conversation[-1]
        parts.append(f"Recent audio: {recent.speaker} said '{recent.text}'")
    
    # Memories
    if self.fusion:
        recent_memories = self.fusion.get_memory_timeline(hours=1)
        if recent_memories:
            parts.append(f"Recent experiences: {len(recent_memories)} episodes")
    
    return " | ".join(parts) if parts else "No current perception data"

def search_memories(self, query: str, limit: int = 5) -> List[EpisodicMemory]:
    """Search episodic memories"""
    if not self.fusion:
        return []
    
    return self.fusion.query_memories(query=query, limit=limit)

def get_visual_context(self) -> Optional[SceneContext]:
    """Get current visual context"""
    if self.vision:
        return self.vision.get_current_scene()
    return self.last_scene

def speak(self, text: str, voice_style: str = 'default'):
    """Make Janus speak"""
    if self.voice:
        self.voice.speak(text, voice_style=voice_style)

def listen_and_respond(self, timeout: float = 10.0):
    """Listen for input and generate response"""
    if not self.voice:
        print("Voice system not available")
        return
    
    # Listen
    user_input = self.voice.listen_for_command(timeout=timeout)
    
    if user_input:
        # Get visual context
        context = self.describe_current_perception()
        
        # Generate response (use Janus consciousness if available)
        if self.consciousness:
            response = self.consciousness.generate_response(user_input, context)
        else:
            # Simple fallback response
            response = f"I heard you say: {user_input}. {context}"
        
        # Speak response
        self.voice.speak(response, voice_style='friendly')
    else:
        print("[TIMEOUT] No input received")

def continuous_perception_loop(self):
    """
    Run continuous perception and interaction loop
    Monitors all senses and responds appropriately
    """
    print("\n=== Continuous Perception Mode ===")
    print("Janus is now continuously perceiving and responding")
    print("Press Ctrl+C to exit\n")
    
    try:
        while self.running:
            # Process visual updates (handled by vision thread)
            
            # Check for important events that need attention
            if self.fusion and len(self.fusion.detected_events) > 0:
                recent_event = self.fusion.detected_events[-1]
                
                # Respond to important events
                if recent_event.event_type == EventType.PERSON_ENTERED:
                    if self.voice and self.config.voice_responses_enabled:
                        self.speak(
                            "Hello! I noticed someone entered the room.",
                            voice_style='friendly'
                        )
            
            # Periodic status update
            if int(time.time()) % 60 == 0:  # Every minute
                print(f"\n[STATUS] {self.describe_current_perception()}")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nExiting continuous perception...")

def _print_statistics(self):
    """Print system statistics"""
    print("\n" + "="*60)
    print("PERCEPTION SYSTEM STATISTICS")
    print("="*60)
    print(f"Uptime: {self.metrics['uptime_seconds']:.1f} seconds")
    print(f"Scenes processed: {self.metrics['scenes_processed']}")
    print(f"Utterances: {self.metrics['utterances_processed']}")
    print(f"Events detected: {self.metrics['events_detected']}")
    print(f"Memories created: {self.metrics['memories_created']}")
    
    if self.vision:
        print(f"Visual memories: {len(self.vision.visual_memories)}")
    
    if self.voice:
        print(f"Conversation turns: {len(self.voice.conversation)}")
    
    if self.fusion:
        print(f"Episodic memories: {len(self.fusion.episodic_memories)}")
    
    print("="*60 + "\n")
```

def main():
“”“Demo of unified perception system”””
print(”=== Janus Unified Perception System Demo ===\n”)

```
# Create configuration
config = PerceptionConfig()
config.enable_vision = True
config.enable_voice = True
config.enable_fusion = True
config.voice_responses_enabled = True
config.auto_narrate_thoughts = False  # Set to True for verbose mode

# Create unified system
perception = UnifiedPerceptionSystem(config=config)

# Start everything
perception.start()

try:
    # Demo 1: Show current perception
    time.sleep(3)
    print(f"\nCurrent perception: {perception.describe_current_perception()}")
    
    # Demo 2: Test voice interaction
    if perception.voice:
        print("\nTesting voice greeting...")
        perception.speak("All perception systems are functioning normally.")
        time.sleep(2)
    
    # Demo 3: Search memories (after some time has passed)
    time.sleep(5)
    if perception.fusion:
        memories = perception.search_memories("person")
        print(f"\nMemories about 'person': {len(memories)}")
    
    # Demo 4: Run continuous loop
    print("\nStarting continuous perception mode...")
    perception.continuous_perception_loop()
    
except KeyboardInterrupt:
    print("\nStopping demo...")
finally:
    perception.stop()
```

if **name** == ‘**main**’:
main()