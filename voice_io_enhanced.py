"""
voice_io_enhanced.py
────────────────────────────────────────────────────────────
Always-on wake-word + conversation loop using local STT (Whisper.cpp) 
and TTS (Piper). Zero API keys required.

Features:
- Wake word detection ("Hey Janus", "OK Janus", "Janus")
- Continuous conversation loop
- Local Whisper.cpp integration for STT
- Local Piper integration for TTS
- Voice activity detection (VAD)
- Conversation memory and context
"""

import numpy as np
import wave
import pyaudio
import threading
import queue
import json
import re
import subprocess
import tempfile
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime
from pathlib import Path
import time
import struct
import ctypes
from collections import deque

@dataclass
class VoiceInput:
    """Captured voice input"""
    input_id: str
    timestamp: str
    audio_data: bytes
    duration: float
    transcription: Optional[str] = None
    confidence: float = 0.0
    language: str = "en"
    speaker_id: Optional[str] = None

@dataclass
class VoiceOutput:
    """Generated voice output"""
    output_id: str
    timestamp: str
    text: str
    audio_data: Optional[bytes] = None
    voice_style: str = "default"
    emotion: str = "neutral"
    speed: float = 1.0

@dataclass
class Utterance:
    """A conversational utterance"""
    utterance_id: str
    timestamp: str
    speaker: str  # 'user' or 'janus'
    text: str
    audio_data: Optional[bytes] = None
    intent: Optional[str] = None
    entities: Dict = None


class WhisperSTT:
    """
    Speech-to-Text using local Whisper.cpp
    Requires: whisper.cpp compiled with libwhisper.so
    """
    
    def __init__(self, model_path: str = "models/ggml-base.en.bin", 
                 lib_path: str = "libwhisper.so",
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path)
        self.lib_path = Path(lib_path)
        self.ctx = None
        self._init_whisper()
        
        # Wake words - expanded list
        self.wake_words = [
            'janus', 'hey janus', 'ok janus', 'hi janus',
            'janice', 'hey janice', 'yo janus'
        ]
        
        # Command patterns for intent detection
        self.command_patterns = {
            'email': r'\b(email|mail|inbox|message)\b',
            'file': r'\b(file|folder|directory|open|save)\b',
            'search': r'\b(search|find|look for|google)\b',
            'code': r'\b(code|program|script|function|write)\b',
            'calendar': r'\b(calendar|schedule|appointment|meeting)\b',
            'reminder': r'\b(remind|reminder|remember|don\'t forget)\b',
            'call': r'\b(call|phone|dial|ring)\b',
            'sms': r'\b(text|sms|message|send)\b',
        }
        
    def _init_whisper(self):
        """Initialize whisper.cpp library"""
        try:
            if self.lib_path.exists():
                self.whisper = ctypes.CDLL(str(self.lib_path))
                # Load model if available
                if self.model_path.exists():
                    self.ctx = self._load_model()
                    print(f"[WhisperSTT] Loaded model: {self.model_path}")
                else:
                    print(f"[WhisperSTT] Model not found at {self.model_path}, using fallback")
            else:
                print(f"[WhisperSTT] Library not found at {self.lib_path}, using fallback")
                self.whisper = None
        except Exception as e:
            print(f"[WhisperSTT] Initialization error: {e}")
            self.whisper = None
    
    def _load_model(self):
        """Load whisper model"""
        # Placeholder - actual implementation depends on whisper.cpp bindings
        return None
    
    def transcribe(self, audio_data: bytes, language: str = "en") -> tuple[str, float]:
        """
        Transcribe audio to text using Whisper.cpp
        Falls back to command-line whisper if library not available
        """
        if not audio_data:
            return "", 0.0
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            self._save_wav(audio_data, temp_path)
        
        try:
            # Try command-line whisper.cpp
            if self.model_path.exists():
                result = subprocess.run(
                    ['whisper-cli', '-m', str(self.model_path), 
                     '-f', temp_path, '-l', language, '--no-timestamps'],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    text = result.stdout.strip()
                    confidence = 0.85  # Whisper is generally confident
                    return text, confidence
            
            # Fallback: use energy-based detection with placeholder
            return self._fallback_transcribe(audio_data)
            
        except Exception as e:
            print(f"[WhisperSTT] Transcription error: {e}")
            return self._fallback_transcribe(audio_data)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _fallback_transcribe(self, audio_data: bytes) -> tuple[str, float]:
        """Fallback transcription using energy detection"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        if energy > 1000:
            return "[Speech detected - Whisper model loading...]", 0.5
        return "", 0.0
    
    def _save_wav(self, audio_data: bytes, path: str):
        """Save audio data to WAV file"""
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
    
    def detect_wake_word(self, audio_data: bytes) -> tuple[bool, str]:
        """
        Detect wake word in audio
        Returns: (detected, transcription)
        """
        text, confidence = self.transcribe(audio_data)
        
        if confidence > 0.4 and text:
            text_lower = text.lower()
            for wake_word in self.wake_words:
                if wake_word in text_lower:
                    # Remove wake word from text
                    remaining = text_lower.replace(wake_word, '').strip()
                    return True, remaining if remaining else text
        
        return False, text
    
    def detect_intent(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Detect intent from transcribed text"""
        text_lower = text.lower()
        
        for intent, pattern in self.command_patterns.items():
            if re.search(pattern, text_lower):
                # Extract entities (simple approach)
                entities = {'query': text}
                return intent, entities
        
        return 'general', {'query': text}


class PiperTTS:
    """
    Text-to-Speech using local Piper
    Requires: piper-tts installed
    """
    
    def __init__(self, model_path: str = "models/en_US-lessac-medium.onnx",
                 config_path: Optional[str] = None,
                 sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Voice styles with Piper parameters
        self.voices = {
            'default': {'speaker_id': 0, 'length_scale': 1.0, 'noise_scale': 0.667},
            'friendly': {'speaker_id': 0, 'length_scale': 1.1, 'noise_scale': 0.6},
            'serious': {'speaker_id': 0, 'length_scale': 0.9, 'noise_scale': 0.7},
            'excited': {'speaker_id': 0, 'length_scale': 1.2, 'noise_scale': 0.5},
            'calm': {'speaker_id': 0, 'length_scale': 1.3, 'noise_scale': 0.4},
        }
        
        self._check_piper()
    
    def _check_piper(self):
        """Check if Piper is available"""
        try:
            result = subprocess.run(['piper', '--version'], 
                                    capture_output=True, text=True)
            self.piper_available = result.returncode == 0
            print(f"[PiperTTS] Available: {self.piper_available}")
        except:
            self.piper_available = False
            print("[PiperTTS] Piper not found, using fallback synthesis")
    
    def synthesize(self, text: str, voice_style: str = 'default',
                   emotion: str = 'neutral') -> bytes:
        """
        Synthesize speech using Piper
        Falls back to tone generation if Piper unavailable
        """
        if not text:
            return b''
        
        if self.piper_available and self.model_path.exists():
            return self._piper_synthesize(text, voice_style)
        else:
            return self._fallback_synthesize(text, voice_style)
    
    def _piper_synthesize(self, text: str, voice_style: str) -> bytes:
        """Synthesize using Piper"""
        voice_params = self.voices.get(voice_style, self.voices['default'])
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name
        
        try:
            cmd = [
                'piper',
                '--model', str(self.model_path),
                '--output_file', output_path,
                '--length_scale', str(voice_params['length_scale']),
                '--noise_scale', str(voice_params['noise_scale']),
            ]
            
            if self.config_path and self.config_path.exists():
                cmd.extend(['--config', str(self.config_path)])
            
            result = subprocess.run(
                cmd, input=text, text=True, capture_output=True, timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    return f.read()
            else:
                print(f"[PiperTTS] Error: {result.stderr}")
                return self._fallback_synthesize(text, voice_style)
                
        except Exception as e:
            print(f"[PiperTTS] Synthesis error: {e}")
            return self._fallback_synthesize(text, voice_style)
        finally:
            try:
                os.unlink(output_path)
            except:
                pass
    
    def _fallback_synthesize(self, text: str, voice_style: str) -> bytes:
        """Fallback TTS using tone generation"""
        voice_params = self.voices.get(voice_style, self.voices['default'])
        
        # Estimate duration
        duration = len(text) * 0.06 * voice_params['length_scale']
        duration = min(duration, 30)  # Cap at 30 seconds
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate base tone
        base_freq = 440 * (0.9 + voice_params['length_scale'] * 0.1)
        audio = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics for more voice-like quality
        audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Add modulation
        modulation = np.sin(2 * np.pi * 4 * t) * 0.2
        audio = audio * (1 + modulation)
        
        # Apply envelope
        envelope = np.ones_like(t)
        attack = int(0.05 * self.sample_rate)
        release = int(0.1 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32767 * 0.7).astype(np.int16)
        
        return audio.tobytes()


class WebRTCVAD:
    """
    Voice Activity Detection using WebRTC VAD
    More accurate than simple energy-based detection
    """
    
    def __init__(self, sample_rate: int = 16000, mode: int = 2):
        """
        mode: 0=quality, 1=low bitrate, 2=aggressive, 3=very aggressive
        """
        self.sample_rate = sample_rate
        self.mode = mode
        self.frame_duration = 30  # ms
        self.frame_samples = int(sample_rate * self.frame_duration / 1000)
        
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(mode)
            self.available = True
        except ImportError:
            print("[WebRTCVAD] webrtcvad not available, using energy-based VAD")
            self.vad = None
            self.available = False
        
        # Energy-based fallback thresholds
        self.energy_threshold = 500
        self.silence_threshold = 1.5  # seconds
        
        # State
        self.speech_buffer = deque(maxlen=100)
        self.is_speaking = False
        self.silence_start = None
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech"""
        if self.available and len(audio_chunk) >= self.frame_samples * 2:
            # Use WebRTC VAD
            try:
                return self.vad.is_speech(audio_chunk[:self.frame_samples * 2], self.sample_rate)
            except:
                pass
        
        # Fallback to energy-based detection
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        if len(audio_array) == 0:
            return False
        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        return energy > self.energy_threshold
    
    def process_frame(self, audio_chunk: bytes) -> tuple[bool, bool]:
        """
        Process a frame and detect speech start/end
        Returns: (is_speech, speech_ended)
        """
        speech = self.is_speech(audio_chunk)
        self.speech_buffer.append(speech)
        
        # Require multiple consecutive frames for speech start
        if speech and not self.is_speaking:
            if sum(self.speech_buffer) >= 3:  # 3 consecutive speech frames
                self.is_speaking = True
                self.silence_start = None
                return True, False
        
        # Detect speech end (silence)
        if not speech and self.is_speaking:
            if self.silence_start is None:
                self.silence_start = time.time()
            elif time.time() - self.silence_start > self.silence_threshold:
                self.is_speaking = False
                self.silence_start = None
                return False, True
        
        return speech, False
    
    def find_speech_boundaries(self, audio_data: bytes) -> List[tuple[int, int]]:
        """Find start and end indices of speech segments"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        window_size = self.frame_samples
        
        segments = []
        in_speech = False
        start_idx = 0
        
        for i in range(0, len(audio_array), window_size):
            window = audio_array[i:i+window_size]
            if len(window) == 0:
                continue
            
            window_bytes = window.tobytes()
            speech = self.is_speech(window_bytes)
            
            if speech and not in_speech:
                start_idx = i
                in_speech = True
            elif not speech and in_speech:
                # Check for sustained silence
                silence_count = 0
                for j in range(i, min(i + 10 * window_size, len(audio_array)), window_size):
                    check_window = audio_array[j:j+window_size]
                    if len(check_window) > 0:
                        if not self.is_speech(check_window.tobytes()):
                            silence_count += 1
                        else:
                            break
                
                if silence_count >= 5:  # ~150ms of silence
                    segments.append((start_idx * 2, i * 2))
                    in_speech = False
        
        # Close open segment
        if in_speech:
            segments.append((start_idx * 2, len(audio_data)))
        
        return segments


class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 50):
        self.history: List[Utterance] = []
        self.max_history = max_history
        self.current_topic: Optional[str] = None
        self.pending_intent: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}
    
    def add_utterance(self, utterance: Utterance):
        """Add an utterance to history"""
        self.history.append(utterance)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent_context(self, n: int = 5) -> List[Utterance]:
        """Get recent conversation context"""
        return self.history[-n:] if self.history else []
    
    def get_context_string(self, n: int = 5) -> str:
        """Get context as a formatted string"""
        recent = self.get_recent_context(n)
        lines = []
        for utt in recent:
            speaker = "User" if utt.speaker == "user" else "Janus"
            lines.append(f"{speaker}: {utt.text}")
        return "\n".join(lines)
    
    def set_topic(self, topic: str):
        """Set current conversation topic"""
        self.current_topic = topic
    
    def clear(self):
        """Clear conversation context"""
        self.history.clear()
        self.current_topic = None
        self.pending_intent = None


class EnhancedVoiceIOSystem:
    """
    Enhanced real-time voice I/O system with always-on wake word
    and continuous conversation loop
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 memory_dir: str = "/tmp/janus_voice",
                 whisper_model: str = "models/ggml-base.en.bin",
                 piper_model: str = "models/en_US-lessac-medium.onnx"):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        # Processing components
        self.stt = WhisperSTT(model_path=whisper_model, sample_rate=sample_rate)
        self.tts = PiperTTS(model_path=piper_model, sample_rate=22050)
        self.vad = WebRTCVAD(sample_rate, mode=2)
        
        # Conversation context
        self.context = ConversationContext()
        
        # State
        self.running = False
        self.listening = False
        self.speaking = False
        self.conversation_active = False
        
        # Wake word settings
        self.wake_word_enabled = True
        self.wake_word_detected = False
        self.post_wake_timeout = 30  # seconds to keep listening after wake word
        self.last_wake_time = 0
        
        # Audio buffers
        self.audio_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Pre-buffer for wake word detection
        self.pre_buffer = deque(maxlen=int(sample_rate * 2 / chunk_size))  # 2 seconds
        
        # Threads
        self.input_thread = None
        self.processing_thread = None
        self.output_thread = None
        self.conversation_thread = None
        
        # Voice memories
        self.voice_memories: List[VoiceInput] = []
        
        # Callbacks
        self.on_wake_word: Optional[Callable] = None
        self.on_speech_detected: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str, float], None]] = None
        self.on_utterance_complete: Optional[Callable[[Utterance], None]] = None
        self.on_intent_detected: Optional[Callable[[str, Dict], None]] = None
        self.on_response_needed: Optional[Callable[[str], str]] = None
        
        # Response handler (set by external system)
        self.response_handler: Optional[Callable[[str, str], str]] = None
    
    def start(self):
        """Start voice I/O system"""
        print("\n" + "═" * 56)
        print("  JANUS VOICE — Always-On Wake Word + Conversation")
        print("═" * 56)
        
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
            rate=22050,  # Piper sample rate
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.running = True
        self.listening = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self.output_thread.start()
        
        self.conversation_thread = threading.Thread(target=self._conversation_loop, daemon=True)
        self.conversation_thread.start()
        
        self.input_stream.start_stream()
        
        print("\n[VoiceIO] System started")
        print(f"[VoiceIO] Sample rate: {self.sample_rate} Hz")
        print(f"[VoiceIO] Wake word detection: ENABLED")
        print(f"[VoiceIO] Say 'Hey Janus' to begin...\n")
    
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
        
        print("[VoiceIO] System stopped")
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        if self.listening:
            self.audio_queue.put(in_data)
            self.pre_buffer.append(in_data)
        return (None, pyaudio.paContinue)
    
    def _processing_loop(self):
        """Main audio processing loop"""
        speech_buffer = b''
        
        while self.running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.05)
                
                # Process with VAD
                is_speech, speech_ended = self.vad.process_frame(chunk)
                
                if is_speech:
                    speech_buffer += chunk
                
                # Check for speech end
                if speech_ended and speech_buffer:
                    self._process_speech_segment(speech_buffer)
                    speech_buffer = b''
                
                # Wake word detection when not in conversation
                if (self.wake_word_enabled and 
                    not self.conversation_active and 
                    not self.wake_word_detected):
                    self._check_wake_word()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceIO] Processing error: {e}")
    
    def _check_wake_word(self):
        """Check for wake word in pre-buffer"""
        if len(self.pre_buffer) < self.pre_buffer.maxlen:
            return
        
        # Concatenate pre-buffer
        audio_data = b''.join(self.pre_buffer)
        
        # Try to detect wake word
        detected, remaining = self.stt.detect_wake_word(audio_data)
        
        if detected:
            self.wake_word_detected = True
            self.last_wake_time = time.time()
            self.conversation_active = True
            
            print("\n[Wake Word] 'Hey Janus' detected!")
            
            if self.on_wake_word:
                self.on_wake_word()
            
            # Play acknowledgment
            self.speak("Yes? I'm listening.", voice_style='friendly', block=False)
            
            # Clear pre-buffer
            self.pre_buffer.clear()
    
    def _process_speech_segment(self, audio_data: bytes):
        """Process a complete speech segment"""
        duration = len(audio_data) / (self.sample_rate * 2)
        
        if duration < 0.3:  # Too short
            return
        
        # Transcribe
        text, confidence = self.stt.transcribe(audio_data)
        
        if not text or confidence < 0.3:
            return
        
        print(f"\n[Transcription] '{text}' (confidence: {confidence:.2f})")
        
        if self.on_transcription:
            self.on_transcription(text, confidence)
        
        # Detect intent
        intent, entities = self.stt.detect_intent(text)
        
        if self.on_intent_detected:
            self.on_intent_detected(intent, entities)
        
        # Create utterance
        utterance = Utterance(
            utterance_id=f"utt_{len(self.context.history)}",
            timestamp=datetime.now().isoformat(),
            speaker="user",
            text=text,
            audio_data=audio_data,
            intent=intent,
            entities=entities
        )
        
        self.context.add_utterance(utterance)
        
        if self.on_utterance_complete:
            self.on_utterance_complete(utterance)
        
        # Store voice memory
        voice_input = VoiceInput(
            input_id=f"vin_{len(self.voice_memories)}",
            timestamp=datetime.now().isoformat(),
            audio_data=audio_data,
            duration=duration,
            transcription=text,
            confidence=confidence
        )
        self.voice_memories.append(voice_input)
        
        # Save audio file
        self._save_audio(audio_data, f"input_{voice_input.input_id}.wav")
        
        # Add to speech queue for response
        self.speech_queue.put(utterance)
    
    def _conversation_loop(self):
        """Handle conversation responses"""
        while self.running:
            try:
                utterance = self.speech_queue.get(timeout=0.5)
                
                # Get response from handler
                if self.response_handler:
                    context = self.context.get_context_string(3)
                    response = self.response_handler(utterance.text, context)
                    
                    if response:
                        self.speak(response, voice_style='friendly')
                
                elif self.on_response_needed:
                    response = self.on_response_needed(utterance.text)
                    if response:
                        self.speak(response, voice_style='friendly')
                
            except queue.Empty:
                # Check for conversation timeout
                if self.conversation_active:
                    elapsed = time.time() - self.last_wake_time
                    if elapsed > self.post_wake_timeout:
                        print("\n[Conversation] Timeout - returning to wake word mode")
                        self.conversation_active = False
                        self.wake_word_detected = False
                continue
    
    def _output_loop(self):
        """Handle speech output"""
        while self.running:
            try:
                output = self.output_queue.get(timeout=0.1)
                self._play_audio(output.audio_data)
                
                # Add to conversation
                utterance = Utterance(
                    utterance_id=f"utt_{len(self.context.history)}",
                    timestamp=output.timestamp,
                    speaker="janus",
                    text=output.text,
                    audio_data=output.audio_data
                )
                self.context.add_utterance(utterance)
                
            except queue.Empty:
                continue
    
    def speak(self, text: str, voice_style: str = 'default',
              emotion: str = 'neutral', block: bool = False):
        """Make Janus speak"""
        print(f"\n[Janus] {text}")
        
        self.speaking = True
        
        # Synthesize speech
        audio_data = self.tts.synthesize(text, voice_style, emotion)
        
        output = VoiceOutput(
            output_id=f"vout_{len(self.context.history)}",
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
        
        # Update last activity
        if self.conversation_active:
            self.last_wake_time = time.time()
    
    def _play_audio(self, audio_data: bytes):
        """Play audio through output stream"""
        if self.output_stream and audio_data:
            self.output_stream.write(audio_data)
    
    def _save_audio(self, audio_data: bytes, filename: str):
        """Save audio to WAV file"""
        filepath = self.memory_dir / filename
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
    
    def listen_for_command(self, timeout: float = 10.0) -> Optional[str]:
        """Listen for a voice command with timeout"""
        print(f"\n[Listening] Waiting for command (timeout: {timeout}s)...")
        
        start_time = time.time()
        initial_count = len(self.context.history)
        
        while time.time() - start_time < timeout:
            if len(self.context.history) > initial_count:
                latest = self.context.history[-1]
                if latest.speaker == "user":
                    return latest.text
            time.sleep(0.1)
        
        print("[Timeout] No command received")
        return None
    
    def start_conversation(self):
        """Manually start a conversation"""
        self.wake_word_detected = True
        self.conversation_active = True
        self.last_wake_time = time.time()
        self.speak("I'm here. What can I do for you?", voice_style='friendly')
    
    def end_conversation(self):
        """End current conversation"""
        self.conversation_active = False
        self.wake_word_detected = False
        self.speak("Let me know if you need anything else.", voice_style='calm')
    
    def get_conversation_history(self, limit: int = 10) -> List[Utterance]:
        """Get recent conversation history"""
        return self.context.get_recent_context(limit)
    
    def get_conversation_transcript(self) -> str:
        """Get full conversation as text transcript"""
        lines = []
        for utt in self.context.history:
            speaker = "User" if utt.speaker == "user" else "Janus"
            lines.append(f"[{utt.timestamp}] {speaker}: {utt.text}")
        return "\n".join(lines)
    
    def save_conversation(self, filename: str = "conversation.json"):
        """Save conversation history to file"""
        filepath = self.memory_dir / filename
        
        data = {
            'conversation': [asdict(utt) for utt in self.context.history],
            'voice_memories': [asdict(mem) for mem in self.voice_memories],
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[VoiceIO] Conversation saved to {filepath}")
    
    def get_status(self) -> dict:
        """Get current system status"""
        return {
            'running': self.running,
            'listening': self.listening,
            'speaking': self.speaking,
            'conversation_active': self.conversation_active,
            'wake_word_detected': self.wake_word_detected,
            'conversation_turns': len(self.context.history),
            'voice_memories': len(self.voice_memories),
        }


def main():
    """Demo of enhanced voice I/O system"""
    print("=== Janus Enhanced Voice I/O Demo ===\n")
    
    # Create voice system
    voice = EnhancedVoiceIOSystem()
    
    # Setup response handler
    def response_handler(text: str, context: str) -> str:
        """Simple response handler"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif 'how are you' in text_lower:
            return "I'm functioning optimally, thank you for asking!"
        elif 'time' in text_lower:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif 'weather' in text_lower:
            return "I don't have access to weather data yet, but I can help you check a weather website."
        elif any(w in text_lower for w in ['bye', 'goodbye', 'see you']):
            voice.end_conversation()
            return "Goodbye! Have a great day!"
        else:
            return f"I heard you say: {text}. I'm still learning to handle more commands."
    
    voice.response_handler = response_handler
    
    # Setup callbacks
    def on_wake():
        print("[Callback] Wake word detected!")
    
    def on_intent(intent: str, entities: dict):
        print(f"[Callback] Intent: {intent}, Entities: {entities}")
    
    voice.on_wake_word = on_wake
    voice.on_intent_detected = on_intent
    
    # Start system
    voice.start()
    
    try:
        print("\nSystem is running. Say 'Hey Janus' to start a conversation.")
        print("Press Ctrl+C to stop.\n")
        
        while True:
            time.sleep(1)
            status = voice.get_status()
            if status['conversation_active']:
                print(f"[Active] Turns: {status['conversation_turns']}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nStopping demo...")
    finally:
        voice.stop()
        voice.save_conversation()
        
        # Show transcript
        print("\n=== Conversation Transcript ===")
        print(voice.get_conversation_transcript())


if __name__ == "__main__":
    main()
