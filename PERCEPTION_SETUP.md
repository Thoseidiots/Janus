# Multi-Modal Perception System Setup Guide

## 🎯 Overview

This adds complete multi-modal perception to Janus:

- **Vision Perception**: Real-time camera input, object detection, scene understanding
- **Voice I/O**: Speech recognition (STT), text-to-speech (TTS), continuous dialogue
- **Multi-Modal Fusion**: Combines all senses into grounded episodic memories
- **Unified System**: Integrates with existing Janus consciousness and memory

## 📦 Files Added

```
vision_perception.py          # Camera input & object detection
voice_io.py                   # Speech recognition & synthesis
multimodal_fusion.py          # Sensor fusion & episodic memory
unified_perception.py         # Integration with Janus core
perception_requirements.txt   # Dependencies
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Core requirements (minimum)
pip install opencv-python numpy pyaudio --break-system-packages

# Or install all from requirements file
pip install -r perception_requirements.txt --break-system-packages
```

### Step 2: Test Individual Systems

#### A) Test Vision

```bash
python3 vision_perception.py
```

- Should open camera feed with detection overlay
- Press ‘q’ to quit

#### B) Test Voice

```bash
python3 voice_io.py
```

- Should start audio I/O
- Will speak a greeting

#### C) Test Fusion

```bash
python3 multimodal_fusion.py
```

- Combines inputs into episodes

#### D) Test Unified System

```bash
python3 unified_perception.py
```

- Starts all systems together

### Step 3: Integrate with Janus

Add to your `main.py`:

```python
from unified_perception import UnifiedPerceptionSystem, PerceptionConfig

class Janus:
    def __init__(self):
        # Your existing initialization...
        
        # Add perception
        perception_config = PerceptionConfig()
        perception_config.enable_vision = True
        perception_config.enable_voice = True
        perception_config.enable_wake_word = True
        
        self.perception = UnifiedPerceptionSystem(config=perception_config)
        self.perception.start()
    
    def shutdown(self):
        # Your existing shutdown...
        self.perception.stop()
```

## 🔧 Configuration Options

### PerceptionConfig Settings

```python
config = PerceptionConfig()

# Vision
config.enable_vision = True        # Enable camera input
config.camera_id = 0               # Camera device ID
config.vision_fps = 10             # Processing frame rate

# Voice
config.enable_voice = True         # Enable audio I/O
config.sample_rate = 16000         # Audio sample rate
config.enable_wake_word = True     # Wake word detection
config.wake_words = ['janus']      # Custom wake words

# Multi-modal
config.enable_fusion = True        # Enable sensor fusion
config.episode_duration = 30       # Episode length (seconds)
config.min_memory_importance = 0.5 # Memory threshold

# Integration
config.integrate_with_janus_core = True    # Use existing memory/consciousness
config.auto_narrate_thoughts = False       # Speak thoughts aloud
config.voice_responses_enabled = True      # Voice responses
```

## 📖 Detailed Usage

### Vision Perception

```python
from vision_perception import VisionPerceptionSystem

# Initialize
vision = VisionPerceptionSystem(camera_id=0)

# Setup callbacks
def on_scene_change(scene):
    print(f"Scene: {scene.summary}")
    print(f"People: {scene.people_count}")
    print(f"Objects: {len(scene.objects)}")

vision.on_scene_change = on_scene_change

# Start perception
vision.start()

# Get current scene
current = vision.get_current_scene()
print(f"Current scene type: {current.scene_type}")

# Query visual memories
memories = vision.query_memories("person")
print(f"Found {len(memories)} memories with people")

# Display live feed (blocking)
vision.display_live_feed()

# Stop when done
vision.stop()
```

### Voice I/O

```python
from voice_io import VoiceIOSystem

# Initialize
voice = VoiceIOSystem()

# Setup callbacks
def on_wake_word():
    voice.speak("Yes, I'm listening")

def on_transcription(text, confidence):
    print(f"Heard: {text}")

voice.on_wake_word = on_wake_word
voice.on_transcription = on_transcription

# Start voice I/O
voice.start()

# Speak
voice.speak("Hello! I am Janus.", voice_style='friendly')

# Listen for command
user_input = voice.listen_for_command(timeout=10.0)
if user_input:
    print(f"User said: {user_input}")

# Self-narrate thoughts
thoughts = [
    "I am analyzing the visual input",
    "I detect a person in the scene",
    "They appear to be working"
]
voice.self_narrate(thoughts, interval=2.0)

# Continuous conversation
def response_callback(user_input):
    return f"I heard: {user_input}"

voice.continuous_conversation(response_callback)

# Stop
voice.stop()
```

### Multi-Modal Fusion

```python
from multimodal_fusion import MultiModalPerceptionFusion
from vision_perception import SceneContext
from voice_io import Utterance

# Initialize
fusion = MultiModalPerceptionFusion()

# Setup callbacks
def on_event(event):
    print(f"Event: {event.event_type.value} - {event.description}")

def on_memory(memory):
    print(f"Memory: {memory.summary}")
    print(f"Importance: {memory.importance}")

fusion.on_event_detected = on_event
fusion.on_memory_created = on_memory

# Start fusion
fusion.start()

# Ingest from different modalities
fusion.ingest_vision(scene_context)
fusion.ingest_audio(utterance)
fusion.ingest_text("Hello Janus", source='user')

# Query memories
memories = fusion.query_memories(
    query="person",
    min_importance=0.5,
    limit=10
)

# Get timeline
recent = fusion.get_memory_timeline(hours=24)

# Stop
fusion.stop()
```

### Unified System

```python
from unified_perception import UnifiedPerceptionSystem, PerceptionConfig

# Configure
config = PerceptionConfig()
config.enable_vision = True
config.enable_voice = True

# Create system
perception = UnifiedPerceptionSystem(config=config)

# Start all subsystems
perception.start()

# Get current state
status = perception.describe_current_perception()
print(status)

# Get visual context
scene = perception.get_visual_context()

# Search memories
memories = perception.search_memories("conversation")

# Speak
perception.speak("All systems operational")

# Listen and respond
perception.listen_and_respond(timeout=10.0)

# Run continuous loop
perception.continuous_perception_loop()

# Stop
perception.stop()
```

## 🔗 Integration with Existing Janus

### Method 1: Direct Integration

In your existing `consciousness.py`:

```python
from unified_perception import UnifiedPerceptionSystem, PerceptionConfig

class Consciousness:
    def __init__(self):
        # Your existing init...
        
        # Add perception
        config = PerceptionConfig()
        self.perception = UnifiedPerceptionSystem(config=config)
        self.perception.start()
    
    def process_input(self, text_input):
        # Your existing processing...
        
        # Add visual context
        visual_context = self.perception.get_visual_context()
        if visual_context:
            # Use visual info in decision making
            context = f"Visual: {visual_context.summary}. Input: {text_input}"
        
        return self.generate_response(context)
```

### Method 2: Memory Integration

In your existing `memory.py`:

```python
class Memory:
    def __init__(self):
        # Your existing init...
        
        # Store reference to perception system
        self.perception = None
    
    def set_perception(self, perception):
        self.perception = perception
        
        # Connect perception memories to hierarchical memory
        perception.fusion.on_memory_created = self._integrate_episodic_memory
    
    def _integrate_episodic_memory(self, episode):
        # Store episodic memory in your hierarchical structure
        self.store_experience({
            'type': 'episodic',
            'memory_id': episode.memory_id,
            'summary': episode.summary,
            'importance': episode.importance,
            'timestamp': episode.timestamp,
            'modalities': list(episode.modalities.keys()),
            'entities': episode.entities
        })
```

### Method 3: Event-Driven Architecture

```python
class JanusEventSystem:
    def __init__(self):
        self.perception = UnifiedPerceptionSystem()
        self.event_handlers = {}
    
    def register_handler(self, event_type, handler):
        self.event_handlers[event_type] = handler
    
    def start(self):
        # Connect perception events to handlers
        self.perception.fusion.on_event_detected = self._handle_event
        self.perception.start()
    
    def _handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)

# Usage
janus = JanusEventSystem()

def on_person_entered(event):
    janus.perception.speak("Hello! I noticed someone entered.")

janus.register_handler(EventType.PERSON_ENTERED, on_person_entered)
janus.start()
```

## 🎯 Common Use Cases

### Use Case 1: Continuous Monitoring

```python
config = PerceptionConfig()
config.enable_vision = True
config.enable_voice = True
config.auto_narrate_thoughts = True  # Narrate observations

perception = UnifiedPerceptionSystem(config=config)
perception.start()

# Janus will now continuously:
# - Monitor camera for changes
# - Listen for wake word
# - Create episodic memories
# - Narrate important observations

perception.continuous_perception_loop()
```

### Use Case 2: Voice-Activated Assistant

```python
config = PerceptionConfig()
config.enable_voice = True
config.enable_wake_word = True
config.wake_words = ['hey janus', 'janus']

perception = UnifiedPerceptionSystem(config=config)
perception.start()

def handle_command(user_input):
    # Process command with your custom LLM
    response = your_llm.generate(user_input)
    perception.speak(response)
    return response

perception.voice.continuous_conversation(handle_command)
```

### Use Case 3: Multi-Modal Analysis

```python
perception = UnifiedPerceptionSystem()
perception.start()

# After some time, analyze what Janus experienced
memories = perception.search_memories("person", limit=10)

for memory in memories:
    print(f"\n{memory.timestamp}")
    print(f"Summary: {memory.summary}")
    print(f"Entities: {', '.join(memory.entities)}")
    print(f"Modalities: {', '.join(m.value for m in memory.modalities.keys())}")
    print(f"Importance: {memory.importance:.2f}")
```

### Use Case 4: Context-Aware Responses

```python
perception = UnifiedPerceptionSystem()
perception.start()

# Wait for user to speak
user_input = perception.voice.listen_for_command()

# Get multi-modal context
visual = perception.get_visual_context()
recent_memories = perception.fusion.get_memory_timeline(hours=1)

# Generate context-aware response
context = f"""
Visual: {visual.summary if visual else 'No visual data'}
Recent experiences: {len(recent_memories)} episodes
User said: {user_input}
"""

response = your_llm.generate(context)
perception.speak(response)
```

## 🔧 Advanced Configuration

### Custom Object Detection

Replace the simple detector in `vision_perception.py`:

```python
# Install: pip install ultralytics --break-system-packages
from ultralytics import YOLO

class YOLODetector(ObjectDetector):
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Nano model
    
    def detect(self, frame):
        results = self.model(frame)
        objects = []
        
        for result in results:
            for box in result.boxes:
                obj = DetectedObject(
                    object_id=f"obj_{int(time.time())}",
                    class_name=result.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=box.xyxy[0].tolist(),
                    center=box.xywh[0][:2].tolist(),
                    timestamp=datetime.now().isoformat()
                )
                objects.append(obj)
        
        return objects
```

### Advanced Speech Recognition

Replace SimpleSTT in `voice_io.py`:

```python
# Install: pip install vosk --break-system-packages
from vosk import Model, KaldiRecognizer

class VoskSTT(SimpleSTT):
    def __init__(self):
        self.model = Model("model")  # Download Vosk model
        self.recognizer = KaldiRecognizer(self.model, 16000)
    
    def transcribe(self, audio_data):
        self.recognizer.AcceptWaveform(audio_data)
        result = json.loads(self.recognizer.Result())
        return result.get('text', ''), result.get('confidence', 0.0)
```

### Neural TTS

Replace SimpleTTS in `voice_io.py`:

```python
# Install: pip install TTS --break-system-packages
from TTS.api import TTS

class NeuralTTS(SimpleTTS):
    def __init__(self):
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    
    def synthesize(self, text, voice_style='default', emotion='neutral'):
        wav = self.tts.tts(text)
        # Convert to bytes
        return np.array(wav * 32767, dtype=np.int16).tobytes()
```

## 🐛 Troubleshooting

### Camera Not Found

```bash
# List available cameras
ls /dev/video*

# Try different camera ID
config.camera_id = 1  # or 2, 3, etc.
```

### Audio Issues

```bash
# Test PyAudio
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count())"

# On Linux, may need ALSA/PulseAudio
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Import Errors

```bash
# Make sure all files are in same directory
ls -la *.py

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:/path/to/janus"
```

### Performance Issues

```python
# Reduce vision processing rate
config.vision_fps = 5  # Lower FPS

# Disable auto-narration
config.auto_narrate_thoughts = False

# Increase episode duration
config.episode_duration = 60  # Longer episodes = fewer saves
```

## 📊 System Requirements

**Minimum:**

- Python 3.7+
- 2GB RAM
- Webcam (for vision)
- Microphone/speakers (for voice)
- CPU: Any modern processor

**Recommended:**

- Python 3.9+
- 4GB RAM
- HD Webcam
- Good quality microphone
- GPU (for neural models)

## 🎓 Next Steps

1. **Test each system individually** - Verify camera, mic, speakers work
1. **Start with voice only** - Test wake word and dialogue
1. **Add vision gradually** - Enable camera once voice works
1. **Integrate with your LLM** - Connect to custom Janus reasoning
1. **Train custom models** - Fine-tune detection for your use case
1. **Extend sensors** - Add temperature, motion, or other IoT sensors

## 📚 API Reference

See individual file docstrings for complete API documentation:

- `vision_perception.py` - Vision system classes and methods
- `voice_io.py` - Voice I/O classes and methods
- `multimodal_fusion.py` - Fusion and memory classes
- `unified_perception.py` - Integrated system interface

## 💡 Tips

- Start with lower camera resolution for better performance
- Use wake word detection to save power when not in use
- Adjust importance threshold to control memory storage
- Enable auto-narration for debugging
- Use continuous loop for always-on assistant mode
- Query memories periodically to build long-term understanding

## 🔒 Privacy Notes

- All processing is **local** - no cloud APIs
- Camera/audio data stays on your machine
- Memories stored in `/tmp` by default (ephemeral)
- Change `workspace_dir` for persistent storage
- Disable systems you don’t need

## 📄 License

Same as Janus project.