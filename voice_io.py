“””
Janus Voice I/O System
Real-time speech recognition (STT) and synthesis (TTS)
Continuous voice conversations, self-narration, wake word detection
No external API dependencies - uses local models
“””

import numpy as np
import wave
import pyaudio
import threading
import queue
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable, Dict
from datetime import datetime
from pathlib import Path
import time
import struct

@dataclass
class VoiceInput:
“”“Captured voice input”””
input_id: str
timestamp: str
audio_data: bytes
duration: float
transcription: Optional[str] = None
confidence: float = 0.0
language: str = “en”
speaker_id: Optional[str] = None

@dataclass
class VoiceOutput:
“”“Generated voice output”””
output_id: str
timestamp: str
text: str
audio_data: Optional[bytes] = None
voice_style: str = “default”
emotion: str = “neutral”
speed: float = 1.0

@dataclass
class Utterance:
“”“A conversational utterance”””
utterance_id: str
timestamp: str
speaker: str  # ‘user’ or ‘janus’
text: str
audio_data: Optional[bytes] = None
intent: Optional[str] = None
entities: Dict = None

class SimpleSTT:
“”“Simple Speech-to-Text using basic speech recognition”””

```
def __init__(self):
    self.sample_rate = 16000
    self.chunk_size = 1024
    
    # Simple wake words
    self.wake_words = ['janus', 'hey janus', 'ok janus']
    
def transcribe(self, audio_data: bytes) -> tuple[str, float]:
    """
    Transcribe audio to text
    In production, use Vosk, Whisper, or similar
    This is a placeholder that returns basic patterns
    """
    
    # For demo purposes, detect simple patterns
    # In production, integrate actual STT model
    
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Simple energy-based voice activity detection
    energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    
    if energy > 1000:  # Threshold for voice activity
        # Placeholder transcription
        # In production, this would use actual STT
        return "[Speech detected - integrate STT model for transcription]", 0.8
    else:
        return "", 0.0

def detect_wake_word(self, audio_data: bytes) -> bool:
    """Detect wake word in audio"""
    # Placeholder - in production, use keyword spotting
    text, confidence = self.transcribe(audio_data)
    
    if confidence > 0.5:
        text_lower = text.lower()
        return any(wake_word in text_lower for wake_word in self.wake_words)
    
    return False
```

class SimpleTTS:
“”“Simple Text-to-Speech system”””

```
def __init__(self):
    self.sample_rate = 22050
    self.voices = {
        'default': {'pitch': 1.0, 'speed': 1.0},
        'friendly': {'pitch': 1.1, 'speed': 0.95},
        'serious': {'pitch': 0.9, 'speed': 0.9},
        'excited': {'pitch': 1.2, 'speed': 1.1}
    }

def synthesize(self, text: str, voice_style: str = 'default', 
               emotion: str = 'neutral') -> bytes:
    """
    Synthesize speech from text
    In production, use pyttsx3, Coqui TTS, or similar
    This generates a simple tone placeholder
    """
    
    # Get voice parameters
    voice_params = self.voices.get(voice_style, self.voices['default'])
    
    # For demo, generate a simple tone
    # In production, integrate actual TTS engine
    duration = len(text) * 0.05  # Rough duration estimate
    t = np.linspace(0, duration, int(self.sample_rate * duration))
    
    # Generate varying frequencies based on text length
    frequency = 440 * voice_params['pitch']
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation for "speech-like" quality
    modulation = np.sin(2 * np.pi * 3 * t) * 0.3
    audio = audio * (1 + modulation)
    
    # Normalize and convert to int16
    audio = (audio * 32767).astype(np.int16)
    
    return audio.tobytes()
```

class VoiceActivityDetector:
“”“Detects when someone is speaking”””

```
def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
    self.sample_rate = sample_rate
    self.chunk_size = chunk_size
    self.energy_threshold = 1000
    self.silence_threshold = 1.5  # seconds

def is_speech(self, audio_chunk: bytes) -> bool:
    """Detect if audio chunk contains speech"""
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    return energy > self.energy_threshold

def find_speech_boundaries(self, audio_data: bytes) -> List[tuple[int, int]]:
    """Find start and end of speech segments"""
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Calculate energy in windows
    window_size = self.chunk_size
    energies = []
    
    for i in range(0, len(audio_array), window_size):
        window = audio_array[i:i+window_size]
        if len(window) > 0:
            energy = np.sqrt(np.mean(window.astype(np.float32) ** 2))
            energies.append(energy)
    
    # Find speech segments
    segments = []
    in_speech = False
    start_idx = 0
    
    for i, energy in enumerate(energies):
        if energy > self.energy_threshold and not in_speech:
            start_idx = i
            in_speech = True
        elif energy < self.energy_threshold and in_speech:
            segments.append((start_idx * window_size, i * window_size))
            in_speech = False
    
    return segments
```

class VoiceIOSystem:
“”“Real-time voice input/output system”””

```
def __init__(self, 
             sample_rate: int = 16000,
             chunk_size: int = 1024,
             memory_dir: str = "/tmp/janus_voice"):
    
    self.sample_rate = sample_rate
    self.chunk_size = chunk_size
    self.memory_dir = Path(memory_dir)
    self.memory_dir.mkdir(exist_ok=True, parents=True)
    
    # Audio interface
    self.audio = pyaudio.PyAudio()
    self.input_stream = None
    self.output_stream = None
    
    # Processing components
    self.stt = SimpleSTT()
    self.tts = SimpleTTS()
    self.vad = VoiceActivityDetector(sample_rate, chunk_size)
    
    # State
    self.running = False
    self.listening = False
    self.speaking = False
    
    # Queues
    self.audio_queue = queue.Queue()
    self.speech_queue = queue.Queue()
    self.output_queue = queue.Queue()
    
    # Threads
    self.input_thread = None
    self.processing_thread = None
    self.output_thread = None
    
    # Conversation history
    self.conversation: List[Utterance] = []
    self.wake_word_enabled = True
    self.wake_word_detected = False
    
    # Callbacks
    self.on_wake_word: Optional[Callable] = None
    self.on_speech_detected: Optional[Callable[[str], None]] = None
    self.on_transcription: Optional[Callable[[str, float], None]] = None
    self.on_utterance_complete: Optional[Callable[[Utterance], None]] = None
    
    # Voice memories
    self.voice_memories: List[VoiceInput] = []

def start(self):
    """Start voice I/O system"""
    # Open audio streams
    self.input_stream = self.audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_size,
        stream_callback=self._audio_input_callback
    )
    
    self.output_stream = self.audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=self.sample_rate,
        output=True,
        frames_per_buffer=self.chunk_size
    )
    
    self.running = True
    self.listening = True
    
    # Start processing threads
    self.processing_thread = threading.Thread(target=self._processing_loop)
    self.processing_thread.daemon = True
    self.processing_thread.start()
    
    self.output_thread = threading.Thread(target=self._output_loop)
    self.output_thread.daemon = True
    self.output_thread.start()
    
    self.input_stream.start_stream()
    
    print("Voice I/O system started")
    print(f"Sample rate: {self.sample_rate} Hz")
    print(f"Wake word detection: {'enabled' if self.wake_word_enabled else 'disabled'}")

def stop(self):
    """Stop voice I/O system"""
    self.running = False
    self.listening = False
    
    if self.input_stream:
        self.input_stream.stop_stream()
        self.input_stream.close()
    
    if self.output_stream:
        self.output_stream.close()
    
    self.audio.terminate()
    
    print("Voice I/O system stopped")

def _audio_input_callback(self, in_data, frame_count, time_info, status):
    """Callback for audio input"""
    if self.listening:
        self.audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def _processing_loop(self):
    """Process incoming audio"""
    audio_buffer = b''
    speech_buffer = b''
    silence_duration = 0
    last_speech_time = time.time()
    
    while self.running:
        try:
            # Get audio chunk
            chunk = self.audio_queue.get(timeout=0.1)
            audio_buffer += chunk
            
            # Check for speech
            if self.vad.is_speech(chunk):
                speech_buffer += chunk
                last_speech_time = time.time()
                silence_duration = 0
                
                if self.on_speech_detected:
                    self.on_speech_detected("Speech activity detected")
            else:
                silence_duration = time.time() - last_speech_time
            
            # Process on silence after speech
            if speech_buffer and silence_duration > 1.0:
                self._process_speech_segment(speech_buffer)
                speech_buffer = b''
            
            # Check for wake word if enabled
            if self.wake_word_enabled and not self.wake_word_detected:
                if len(audio_buffer) > self.sample_rate * 2:  # 2 seconds
                    if self.stt.detect_wake_word(audio_buffer):
                        self.wake_word_detected = True
                        if self.on_wake_word:
                            self.on_wake_word()
                        print("\n[WAKE WORD DETECTED]")
                    audio_buffer = b''
            
        except queue.Empty:
            continue

def _process_speech_segment(self, audio_data: bytes):
    """Process a segment of speech"""
    # Transcribe
    text, confidence = self.stt.transcribe(audio_data)
    
    if text and confidence > 0.3:
        print(f"\n[TRANSCRIPTION] {text} (confidence: {confidence:.2f})")
        
        if self.on_transcription:
            self.on_transcription(text, confidence)
        
        # Create utterance
        utterance = Utterance(
            utterance_id=f"utt_{len(self.conversation)}",
            timestamp=datetime.now().isoformat(),
            speaker="user",
            text=text,
            audio_data=audio_data,
            intent=None,
            entities={}
        )
        
        self.conversation.append(utterance)
        
        if self.on_utterance_complete:
            self.on_utterance_complete(utterance)
        
        # Store as voice memory
        voice_input = VoiceInput(
            input_id=f"vin_{len(self.voice_memories)}",
            timestamp=datetime.now().isoformat(),
            audio_data=audio_data,
            duration=len(audio_data) / (self.sample_rate * 2),
            transcription=text,
            confidence=confidence
        )
        
        self.voice_memories.append(voice_input)
        
        # Save audio file
        self._save_audio(audio_data, f"input_{voice_input.input_id}.wav")

def _output_loop(self):
    """Handle speech output"""
    while self.running:
        try:
            output = self.output_queue.get(timeout=0.1)
            self._play_audio(output.audio_data)
            
            # Add to conversation
            utterance = Utterance(
                utterance_id=f"utt_{len(self.conversation)}",
                timestamp=output.timestamp,
                speaker="janus",
                text=output.text,
                audio_data=output.audio_data
            )
            
            self.conversation.append(utterance)
            
        except queue.Empty:
            continue

def speak(self, text: str, voice_style: str = 'default', 
         emotion: str = 'neutral', block: bool = False):
    """Make Janus speak"""
    print(f"\n[JANUS SPEAKING] {text}")
    
    self.speaking = True
    
    # Synthesize speech
    audio_data = self.tts.synthesize(text, voice_style, emotion)
    
    output = VoiceOutput(
        output_id=f"vout_{len(self.conversation)}",
        timestamp=datetime.now().isoformat(),
        text=text,
        audio_data=audio_data,
        voice_style=voice_style,
        emotion=emotion
    )
    
    if block:
        self._play_audio(audio_data)
    else:
        self.output_queue.put(output)
    
    # Save audio
    self._save_audio(audio_data, f"output_{output.output_id}.wav")
    
    self.speaking = False

def _play_audio(self, audio_data: bytes):
    """Play audio through output stream"""
    if self.output_stream:
        self.output_stream.write(audio_data)

def _save_audio(self, audio_data: bytes, filename: str):
    """Save audio to WAV file"""
    filepath = self.memory_dir / filename
    
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(self.sample_rate)
        wf.writeframes(audio_data)

def listen_for_command(self, timeout: float = 5.0) -> Optional[str]:
    """Listen for a voice command with timeout"""
    print(f"\n[LISTENING] Waiting for command (timeout: {timeout}s)...")
    
    self.listening = True
    start_time = time.time()
    
    initial_count = len(self.conversation)
    
    while time.time() - start_time < timeout:
        if len(self.conversation) > initial_count:
            latest = self.conversation[-1]
            if latest.speaker == "user":
                return latest.text
        time.sleep(0.1)
    
    print("[TIMEOUT] No command received")
    return None

def continuous_conversation(self, response_callback: Callable[[str], str]):
    """
    Run continuous conversation loop
    response_callback: function that takes user input and returns Janus response
    """
    print("\n=== Starting continuous conversation ===")
    print("Say a wake word to begin, or press Ctrl+C to exit\n")
    
    try:
        while self.running:
            # Wait for wake word
            if self.wake_word_enabled and not self.wake_word_detected:
                time.sleep(0.1)
                continue
            
            # Listen for user input
            user_input = self.listen_for_command(timeout=10.0)
            
            if user_input:
                # Get response from callback
                response = response_callback(user_input)
                
                # Speak response
                self.speak(response, voice_style='friendly')
                
                # Reset wake word
                self.wake_word_detected = False
            else:
                # Timeout - reset
                self.wake_word_detected = False
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nExiting conversation...")

def self_narrate(self, thoughts: List[str], interval: float = 2.0):
    """
    Narrate internal thoughts aloud
    Useful for debugging or explaining Janus's reasoning
    """
    print("\n=== Self-narration mode ===")
    
    for i, thought in enumerate(thoughts):
        print(f"\n[THOUGHT {i+1}] {thought}")
        self.speak(thought, voice_style='serious', emotion='thoughtful')
        time.sleep(interval)

def get_conversation_history(self, limit: int = 10) -> List[Utterance]:
    """Get recent conversation history"""
    return self.conversation[-limit:]

def get_conversation_transcript(self) -> str:
    """Get full conversation as text transcript"""
    lines = []
    for utt in self.conversation:
        speaker = "User" if utt.speaker == "user" else "Janus"
        lines.append(f"{speaker}: {utt.text}")
    return "\n".join(lines)

def save_conversation(self, filename: str = "conversation.json"):
    """Save conversation history to file"""
    filepath = self.memory_dir / filename
    
    data = {
        'conversation': [asdict(utt) for utt in self.conversation],
        'voice_memories': [asdict(mem) for mem in self.voice_memories],
        'saved_at': datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Conversation saved to {filepath}")
```

def main():
“”“Demo of voice I/O system”””
print(”=== Janus Voice I/O System Demo ===\n”)

```
# Create voice system
voice = VoiceIOSystem()

# Setup callbacks
def on_wake_word():
    voice.speak("Yes, I'm listening", voice_style='friendly')

def on_transcription(text, confidence):
    print(f"[DEBUG] Heard: {text} ({confidence:.2f})")

voice.on_wake_word = on_wake_word
voice.on_transcription = on_transcription

# Start system
voice.start()

try:
    # Demo 1: Simple greeting
    print("\nDemo 1: Simple greeting")
    voice.speak("Hello! I am Janus, your autonomous AI assistant.")
    time.sleep(3)
    
    # Demo 2: Self-narration
    print("\nDemo 2: Self-narration")
    thoughts = [
        "I am processing visual input from my camera",
        "I detect a person in my field of view",
        "I am analyzing their facial expression",
        "They appear to be focused on their work"
    ]
    voice.self_narrate(thoughts, interval=3)
    
    # Demo 3: Response function
    def simple_response(user_input: str) -> str:
        user_lower = user_input.lower()
        
        if 'hello' in user_lower or 'hi' in user_lower:
            return "Hello! How can I help you today?"
        elif 'how are you' in user_lower:
            return "I'm functioning optimally, thank you for asking!"
        elif 'time' in user_lower:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        else:
            return "I understand. Please continue."
    
    # Demo 4: Continuous conversation
    print("\nDemo 3: Continuous conversation mode")
    print("(In production, this would use actual STT/wake word detection)")
    
    # Simulate some conversation
    voice.speak("I'm ready for conversation. What would you like to talk about?")
    
except KeyboardInterrupt:
    print("\nStopping demo...")
finally:
    voice.stop()

# Show statistics
print(f"\nConversation turns: {len(voice.conversation)}")
print(f"Voice memories: {len(voice.voice_memories)}")

# Show transcript
if voice.conversation:
    print("\nConversation transcript:")
    print(voice.get_conversation_transcript())

# Save conversation
voice.save_conversation()
```

if **name** == ‘**main**’:
main()