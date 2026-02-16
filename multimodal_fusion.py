“””
Janus Multi-Modal Perception Fusion System
Combines vision, audio, sensors into unified episodic memories
Binds multi-modal inputs into hierarchical memory structure
Real-time sensor streams and event detection
“””

import json
import threading
import queue
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

class ModalityType(Enum):
“”“Types of sensory input”””
VISION = “vision”
AUDIO = “audio”
TEXT = “text”
SENSOR = “sensor”
PROPRIOCEPTION = “proprioception”

class EventType(Enum):
“”“Types of detected events”””
PERSON_ENTERED = “person_entered”
PERSON_LEFT = “person_left”
OBJECT_DETECTED = “object_detected”
SPEECH_DETECTED = “speech_detected”
WAKE_WORD = “wake_word”
SCENE_CHANGE = “scene_change”
MOTION = “motion”
SOUND_EVENT = “sound_event”
ANOMALY = “anomaly”

@dataclass
class ModalityInput:
“”“Input from a specific sensory modality”””
modality: ModalityType
timestamp: str
data: Any
confidence: float
metadata: Dict = field(default_factory=dict)

@dataclass
class MultiModalEvent:
“”“An event detected across multiple modalities”””
event_id: str
event_type: EventType
timestamp: str
primary_modality: ModalityType
modalities: List[ModalityInput]
confidence: float
description: str
entities: List[str] = field(default_factory=list)
location: Optional[Tuple[float, float]] = None  # For spatial events

@dataclass
class EpisodicMemory:
“”“Grounded episodic memory trace binding multiple modalities”””
memory_id: str
timestamp: str
duration: float  # seconds
events: List[MultiModalEvent]
modalities: Dict[ModalityType, List[ModalityInput]]

```
# Hierarchical memory properties
importance: float  # 0-1
emotional_valence: float  # -1 to 1
arousal: float  # 0-1

# Relationships
parent_memory: Optional[str] = None
child_memories: List[str] = field(default_factory=list)
related_memories: List[str] = field(default_factory=list)

# Semantic content
summary: str = ""
entities: List[str] = field(default_factory=list)
tags: List[str] = field(default_factory=list)

# Grounding
visual_snapshot: Optional[str] = None  # Path to image
audio_recording: Optional[str] = None  # Path to audio
```

@dataclass
class SensorReading:
“”“Reading from a sensor”””
sensor_id: str
sensor_type: str  # ‘temperature’, ‘motion’, ‘light’, etc.
value: float
unit: str
timestamp: str

class MultiModalPerceptionFusion:
“”“Fuses multi-modal inputs into unified perception”””

```
def __init__(self, memory_dir: str = "/tmp/janus_multimodal"):
    self.memory_dir = Path(memory_dir)
    self.memory_dir.mkdir(exist_ok=True, parents=True)
    
    # Input queues for each modality
    self.vision_queue = queue.Queue()
    self.audio_queue = queue.Queue()
    self.sensor_queue = queue.Queue()
    self.text_queue = queue.Queue()
    
    # Event detection
    self.event_queue = queue.Queue()
    self.detected_events: List[MultiModalEvent] = []
    
    # Episodic memories
    self.episodic_memories: List[EpisodicMemory] = []
    self.current_episode: Optional[EpisodicMemory] = None
    self.episode_start_time: Optional[datetime] = None
    
    # Running state
    self.running = False
    self.fusion_thread = None
    
    # Event detection rules
    self.event_detectors = self._initialize_event_detectors()
    
    # Callbacks
    self.on_event_detected = None
    self.on_memory_created = None

def _initialize_event_detectors(self) -> Dict:
    """Initialize event detection rules"""
    return {
        EventType.PERSON_ENTERED: self._detect_person_entered,
        EventType.PERSON_LEFT: self._detect_person_left,
        EventType.SCENE_CHANGE: self._detect_scene_change,
        EventType.SPEECH_DETECTED: self._detect_speech,
        EventType.WAKE_WORD: self._detect_wake_word,
    }

def start(self):
    """Start multi-modal fusion"""
    self.running = True
    
    self.fusion_thread = threading.Thread(target=self._fusion_loop)
    self.fusion_thread.daemon = True
    self.fusion_thread.start()
    
    print("Multi-modal perception fusion started")

def stop(self):
    """Stop fusion system"""
    self.running = False
    
    if self.fusion_thread:
        self.fusion_thread.join(timeout=2)
    
    # Finalize current episode
    if self.current_episode:
        self._finalize_episode()
    
    print("Multi-modal perception fusion stopped")

def ingest_vision(self, scene_context: Any):
    """Ingest vision input"""
    modality_input = ModalityInput(
        modality=ModalityType.VISION,
        timestamp=datetime.now().isoformat(),
        data=scene_context,
        confidence=0.9,
        metadata={'source': 'camera'}
    )
    self.vision_queue.put(modality_input)

def ingest_audio(self, utterance: Any):
    """Ingest audio input"""
    modality_input = ModalityInput(
        modality=ModalityType.AUDIO,
        timestamp=datetime.now().isoformat(),
        data=utterance,
        confidence=0.8,
        metadata={'source': 'microphone'}
    )
    self.audio_queue.put(modality_input)

def ingest_sensor(self, sensor_reading: SensorReading):
    """Ingest sensor reading"""
    modality_input = ModalityInput(
        modality=ModalityType.SENSOR,
        timestamp=sensor_reading.timestamp,
        data=sensor_reading,
        confidence=1.0,
        metadata={'sensor_type': sensor_reading.sensor_type}
    )
    self.sensor_queue.put(modality_input)

def ingest_text(self, text: str, source: str = 'input'):
    """Ingest text input"""
    modality_input = ModalityInput(
        modality=ModalityType.TEXT,
        timestamp=datetime.now().isoformat(),
        data=text,
        confidence=1.0,
        metadata={'source': source}
    )
    self.text_queue.put(modality_input)

def _fusion_loop(self):
    """Main fusion processing loop"""
    while self.running:
        # Collect inputs from all modalities
        inputs = self._collect_inputs(timeout=0.1)
        
        if inputs:
            # Detect events from multi-modal inputs
            events = self._detect_events(inputs)
            
            # Add to current episode
            self._update_episode(inputs, events)
            
            # Check if episode should end
            if self._should_end_episode():
                self._finalize_episode()
                self._start_new_episode()
        
        time.sleep(0.01)

def _collect_inputs(self, timeout: float = 0.1) -> List[ModalityInput]:
    """Collect inputs from all modality queues"""
    inputs = []
    
    # Vision
    try:
        while True:
            inp = self.vision_queue.get_nowait()
            inputs.append(inp)
    except queue.Empty:
        pass
    
    # Audio
    try:
        while True:
            inp = self.audio_queue.get_nowait()
            inputs.append(inp)
    except queue.Empty:
        pass
    
    # Sensors
    try:
        while True:
            inp = self.sensor_queue.get_nowait()
            inputs.append(inp)
    except queue.Empty:
        pass
    
    # Text
    try:
        while True:
            inp = self.text_queue.get_nowait()
            inputs.append(inp)
    except queue.Empty:
        pass
    
    return inputs

def _detect_events(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect events from multi-modal inputs"""
    events = []
    
    # Run event detectors
    for event_type, detector in self.event_detectors.items():
        detected = detector(inputs)
        if detected:
            events.extend(detected)
    
    # Notify callbacks
    for event in events:
        self.detected_events.append(event)
        if self.on_event_detected:
            self.on_event_detected(event)
    
    return events

def _detect_person_entered(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect when a person enters the scene"""
    events = []
    
    # Check vision inputs
    vision_inputs = [inp for inp in inputs if inp.modality == ModalityType.VISION]
    
    for inp in vision_inputs:
        scene = inp.data
        if hasattr(scene, 'people_count') and scene.people_count > 0:
            # Check if this is a new person (compare with previous)
            # Simplified - in production, track person IDs
            event = MultiModalEvent(
                event_id=f"event_{len(self.detected_events)}",
                event_type=EventType.PERSON_ENTERED,
                timestamp=inp.timestamp,
                primary_modality=ModalityType.VISION,
                modalities=[inp],
                confidence=0.8,
                description=f"{scene.people_count} person(s) detected in scene",
                entities=['person']
            )
            events.append(event)
    
    return events

def _detect_person_left(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect when a person leaves"""
    # Similar to person_entered, but checking for count decrease
    return []

def _detect_scene_change(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect significant scene changes"""
    events = []
    
    vision_inputs = [inp for inp in inputs if inp.modality == ModalityType.VISION]
    
    for inp in vision_inputs:
        scene = inp.data
        if hasattr(scene, 'scene_type'):
            # Check if scene type is notable
            event = MultiModalEvent(
                event_id=f"event_{len(self.detected_events)}",
                event_type=EventType.SCENE_CHANGE,
                timestamp=inp.timestamp,
                primary_modality=ModalityType.VISION,
                modalities=[inp],
                confidence=0.7,
                description=f"Scene changed to {scene.scene_type}",
                entities=[]
            )
            events.append(event)
    
    return events

def _detect_speech(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect speech events"""
    events = []
    
    audio_inputs = [inp for inp in inputs if inp.modality == ModalityType.AUDIO]
    
    for inp in audio_inputs:
        utterance = inp.data
        if hasattr(utterance, 'text') and utterance.text:
            event = MultiModalEvent(
                event_id=f"event_{len(self.detected_events)}",
                event_type=EventType.SPEECH_DETECTED,
                timestamp=inp.timestamp,
                primary_modality=ModalityType.AUDIO,
                modalities=[inp],
                confidence=0.9,
                description=f"Speech: {utterance.text}",
                entities=[]
            )
            events.append(event)
    
    return events

def _detect_wake_word(self, inputs: List[ModalityInput]) -> List[MultiModalEvent]:
    """Detect wake word"""
    events = []
    
    audio_inputs = [inp for inp in inputs if inp.modality == ModalityType.AUDIO]
    
    for inp in audio_inputs:
        utterance = inp.data
        if hasattr(utterance, 'text') and 'janus' in utterance.text.lower():
            event = MultiModalEvent(
                event_id=f"event_{len(self.detected_events)}",
                event_type=EventType.WAKE_WORD,
                timestamp=inp.timestamp,
                primary_modality=ModalityType.AUDIO,
                modalities=[inp],
                confidence=0.95,
                description="Wake word detected",
                entities=['janus']
            )
            events.append(event)
    
    return events

def _start_new_episode(self):
    """Start a new episodic memory"""
    self.current_episode = EpisodicMemory(
        memory_id=f"episode_{len(self.episodic_memories)}",
        timestamp=datetime.now().isoformat(),
        duration=0.0,
        events=[],
        modalities={},
        importance=0.5,
        emotional_valence=0.0,
        arousal=0.5
    )
    self.episode_start_time = datetime.now()

def _update_episode(self, inputs: List[ModalityInput], events: List[MultiModalEvent]):
    """Update current episode with new inputs and events"""
    if not self.current_episode:
        self._start_new_episode()
    
    # Add events
    self.current_episode.events.extend(events)
    
    # Organize inputs by modality
    for inp in inputs:
        if inp.modality not in self.current_episode.modalities:
            self.current_episode.modalities[inp.modality] = []
        self.current_episode.modalities[inp.modality].append(inp)
    
    # Update duration
    if self.episode_start_time:
        duration = (datetime.now() - self.episode_start_time).total_seconds()
        self.current_episode.duration = duration

def _should_end_episode(self) -> bool:
    """Determine if current episode should end"""
    if not self.current_episode or not self.episode_start_time:
        return False
    
    # End after 30 seconds of activity
    duration = (datetime.now() - self.episode_start_time).total_seconds()
    if duration > 30:
        return True
    
    # End if significant scene change
    recent_events = self.current_episode.events[-5:] if self.current_episode.events else []
    scene_changes = sum(1 for e in recent_events if e.event_type == EventType.SCENE_CHANGE)
    if scene_changes > 2:
        return True
    
    return False

def _finalize_episode(self):
    """Finalize and store current episode"""
    if not self.current_episode:
        return
    
    # Calculate importance
    importance = self._calculate_episode_importance(self.current_episode)
    self.current_episode.importance = importance
    
    # Generate summary
    summary = self._generate_episode_summary(self.current_episode)
    self.current_episode.summary = summary
    
    # Extract entities
    entities = self._extract_episode_entities(self.current_episode)
    self.current_episode.entities = entities
    
    # Store episode
    self.episodic_memories.append(self.current_episode)
    
    # Save to disk
    self._save_episode(self.current_episode)
    
    # Notify callback
    if self.on_memory_created:
        self.on_memory_created(self.current_episode)
    
    print(f"\n[EPISODE FINALIZED] {self.current_episode.memory_id}")
    print(f"  Duration: {self.current_episode.duration:.1f}s")
    print(f"  Events: {len(self.current_episode.events)}")
    print(f"  Importance: {importance:.2f}")
    print(f"  Summary: {summary}")
    
    self.current_episode = None
    self.episode_start_time = None

def _calculate_episode_importance(self, episode: EpisodicMemory) -> float:
    """Calculate importance score for episode"""
    score = 0.0
    
    # More events = more important
    score += min(len(episode.events) * 0.1, 0.4)
    
    # Multiple modalities = richer experience
    score += len(episode.modalities) * 0.15
    
    # Specific event types boost importance
    for event in episode.events:
        if event.event_type in [EventType.PERSON_ENTERED, EventType.WAKE_WORD]:
            score += 0.2
        elif event.event_type == EventType.SPEECH_DETECTED:
            score += 0.15
    
    # Longer episodes (but not too long) are important
    if 5 < episode.duration < 60:
        score += 0.1
    
    return min(score, 1.0)

def _generate_episode_summary(self, episode: EpisodicMemory) -> str:
    """Generate natural language summary of episode"""
    parts = []
    
    # Duration
    parts.append(f"{episode.duration:.1f} second episode")
    
    # Event summary
    if episode.events:
        event_types = {}
        for event in episode.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        event_desc = ", ".join([f"{count} {etype.value}" for etype, count in event_types.items()])
        parts.append(f"with {event_desc}")
    
    # Modality summary
    modality_names = [m.value for m in episode.modalities.keys()]
    if modality_names:
        parts.append(f"across {', '.join(modality_names)}")
    
    return " ".join(parts)

def _extract_episode_entities(self, episode: EpisodicMemory) -> List[str]:
    """Extract entities from episode"""
    entities = set()
    
    for event in episode.events:
        entities.update(event.entities)
    
    # Extract from vision
    if ModalityType.VISION in episode.modalities:
        for inp in episode.modalities[ModalityType.VISION]:
            scene = inp.data
            if hasattr(scene, 'objects'):
                for obj in scene.objects:
                    entities.add(obj.class_name)
    
    return list(entities)

def _save_episode(self, episode: EpisodicMemory):
    """Save episode to disk"""
    filepath = self.memory_dir / f"{episode.memory_id}.json"
    
    # Convert to serializable format
    data = asdict(episode)
    
    # Remove non-serializable data
    if 'modalities' in data:
        for modality in data['modalities']:
            data['modalities'][modality] = [
                {**asdict(inp), 'data': str(inp.data)[:200]}  # Truncate data
                for inp in episode.modalities[modality]
            ]
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def query_memories(self, 
                  query: str = None,
                  modality: ModalityType = None,
                  event_type: EventType = None,
                  min_importance: float = 0.0,
                  limit: int = 10) -> List[EpisodicMemory]:
    """Query episodic memories"""
    results = []
    
    for memory in self.episodic_memories:
        # Filter by importance
        if memory.importance < min_importance:
            continue
        
        # Filter by modality
        if modality and modality not in memory.modalities:
            continue
        
        # Filter by event type
        if event_type:
            has_event = any(e.event_type == event_type for e in memory.events)
            if not has_event:
                continue
        
        # Text search
        if query:
            query_lower = query.lower()
            if query_lower in memory.summary.lower():
                results.append(memory)
            elif any(query_lower in entity.lower() for entity in memory.entities):
                results.append(memory)
        else:
            results.append(memory)
    
    # Sort by importance and recency
    results.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
    
    return results[:limit]

def get_memory_timeline(self, hours: int = 24) -> List[EpisodicMemory]:
    """Get memories from recent timeline"""
    cutoff = datetime.now() - timedelta(hours=hours)
    
    recent = []
    for memory in self.episodic_memories:
        memory_time = datetime.fromisoformat(memory.timestamp)
        if memory_time >= cutoff:
            recent.append(memory)
    
    return sorted(recent, key=lambda m: m.timestamp)

def build_memory_hierarchy(self):
    """Build hierarchical relationships between memories"""
    # Group memories by time proximity and shared entities
    for i, memory in enumerate(self.episodic_memories):
        # Find temporally close memories
        memory_time = datetime.fromisoformat(memory.timestamp)
        
        for j, other in enumerate(self.episodic_memories):
            if i == j:
                continue
            
            other_time = datetime.fromisoformat(other.timestamp)
            time_diff = abs((memory_time - other_time).total_seconds())
            
            # If within 5 minutes
            if time_diff < 300:
                # Check for shared entities
                shared = set(memory.entities) & set(other.entities)
                if shared:
                    memory.related_memories.append(other.memory_id)
```

def main():
“”“Demo of multi-modal fusion”””
print(”=== Janus Multi-Modal Perception Fusion Demo ===\n”)

```
# Create fusion system
fusion = MultiModalPerceptionFusion()

# Setup callbacks
def on_event(event):
    print(f"\n[EVENT] {event.event_type.value}: {event.description}")

def on_memory(memory):
    print(f"\n[MEMORY CREATED] {memory.memory_id}")
    print(f"  Summary: {memory.summary}")
    print(f"  Entities: {', '.join(memory.entities)}")

fusion.on_event_detected = on_event
fusion.on_memory_created = on_memory

# Start fusion
fusion.start()

try:
    # Simulate multi-modal inputs
    print("Simulating multi-modal inputs...\n")
    
    # Simulate vision input
    from vision_perception import SceneContext, DetectedObject
    scene = SceneContext(
        scene_id="scene_1",
        timestamp=datetime.now().isoformat(),
        objects=[
            DetectedObject(
                object_id="obj_1",
                class_name="person",
                confidence=0.9,
                bbox=(100, 100, 200, 300),
                center=(200, 250),
                timestamp=datetime.now().isoformat()
            )
        ],
        scene_type="workspace",
        lighting="bright",
        dominant_colors=[(200, 200, 200)],
        motion_detected=False,
        people_count=1,
        text_detected=[],
        summary="Workspace scene with 1 person"
    )
    fusion.ingest_vision(scene)
    
    time.sleep(1)
    
    # Simulate audio input
    from voice_io import Utterance
    utterance = Utterance(
        utterance_id="utt_1",
        timestamp=datetime.now().isoformat(),
        speaker="user",
        text="Hey Janus, what do you see?",
        audio_data=None
    )
    fusion.ingest_audio(utterance)
    
    time.sleep(1)
    
    # Simulate sensor input
    sensor = SensorReading(
        sensor_id="temp_1",
        sensor_type="temperature",
        value=22.5,
        unit="celsius",
        timestamp=datetime.now().isoformat()
    )
    fusion.ingest_sensor(sensor)
    
    # Wait for episode to form
    time.sleep(5)
    
    # Query memories
    print("\n" + "="*60)
    print("Querying memories...")
    print("="*60)
    
    memories = fusion.query_memories(min_importance=0.3)
    print(f"\nFound {len(memories)} important memories:")
    for mem in memories:
        print(f"\n{mem.memory_id}:")
        print(f"  {mem.summary}")
        print(f"  Importance: {mem.importance:.2f}")
        print(f"  Events: {len(mem.events)}")
    
    # Build hierarchy
    fusion.build_memory_hierarchy()
    
    # Wait a bit more
    time.sleep(5)
    
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    fusion.stop()

print(f"\nTotal episodic memories: {len(fusion.episodic_memories)}")
print(f"Total events detected: {len(fusion.detected_events)}")
```

if **name** == ‘**main**’:
main()