"""
janus_voice.py
==============
Janus voice interface — fully wired brain + voice.

Say "Hey Janus" → she wakes up → you talk → she thinks → she responds.

Pipeline:
  Microphone → faster-whisper STT → JanusGPT brain → Kokoro TTS → Speakers

Run:
    python janus_voice.py

Requirements:
    pip install pyaudio numpy faster-whisper kokoro soundfile
"""

import sys
import time
import queue
import threading
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("janus.voice")


# ─────────────────────────────────────────────────────────────────────────────
# STT — faster-whisper (local, no internet)
# ─────────────────────────────────────────────────────────────────────────────

class JanusSTT:
    """Speech-to-text using faster-whisper (tiny model, fast on CPU)."""

    WAKE_WORDS = ["hey janus", "ok janus", "janus", "hi janus"]
    SAMPLE_RATE = 16000
    CHUNK = 1024
    SILENCE_THRESHOLD = 500    # RMS below this = silence
    SILENCE_DURATION  = 1.5    # seconds of silence to end utterance
    MAX_RECORD_SEC    = 15     # max recording length

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("[STT] faster-whisper base loaded")
        except Exception as e:
            logger.warning(f"[STT] faster-whisper unavailable: {e}")

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribe float32 audio array at 16kHz → text."""
        if self._model is None or len(audio_np) < 1600:
            return ""
        try:
            segments, _ = self._model.transcribe(
                audio_np, language="en", beam_size=1, vad_filter=True
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            logger.error(f"[STT] Transcription error: {e}")
            return ""

    def is_wake_word(self, text: str) -> bool:
        t = text.lower().strip()
        return any(w in t for w in self.WAKE_WORDS)

    def record_until_silence(self, stream) -> np.ndarray:
        """Record from pyaudio stream until silence detected."""
        frames = []
        silent_chunks = 0
        max_chunks = int(self.MAX_RECORD_SEC * self.SAMPLE_RATE / self.CHUNK)
        silence_chunks_needed = int(self.SILENCE_DURATION * self.SAMPLE_RATE / self.CHUNK)

        for _ in range(max_chunks):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float32) ** 2))
            if rms < self.SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    break
            else:
                silent_chunks = 0

        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        return audio


# ─────────────────────────────────────────────────────────────────────────────
# TTS — Kokoro (local, no internet)
# ─────────────────────────────────────────────────────────────────────────────

class JanusTTS:
    """Text-to-speech using Kokoro af_heart voice."""

    SAMPLE_RATE = 24000

    def __init__(self):
        self._pipeline = None
        self._load()

    def _load(self):
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
            logger.info("[TTS] Kokoro af_heart loaded")
        except Exception as e:
            logger.warning(f"[TTS] Kokoro unavailable: {e}")

    def speak(self, text: str, speed: float = 1.0):
        """Synthesize and play text immediately."""
        if not text.strip():
            return

        # Fix pronunciation
        text = self._fix_names(text)

        audio = self._synthesize(text, speed)
        if audio is not None:
            self._play(audio)
        else:
            # Fallback: print only
            print(f"[Janus] {text}")

    def _fix_names(self, text: str) -> str:
        import re
        return re.sub(r'\bJanus\b', 'Yanus', text)

    def _synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        if self._pipeline is None:
            return None
        try:
            chunks = []
            for _, _, audio in self._pipeline(text, voice="af_heart", speed=speed):
                chunks.append(audio)
            if not chunks:
                return None
            return np.concatenate(chunks)
        except Exception as e:
            logger.error(f"[TTS] Synthesis error: {e}")
            return None

    def _play(self, audio: np.ndarray):
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.SAMPLE_RATE,
                output=True,
            )
            stream.write(audio.astype(np.float32).tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logger.error(f"[TTS] Playback error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Brain
# ─────────────────────────────────────────────────────────────────────────────

def _load_brain():
    try:
        from janus_gpt import load_janus_brain
        brain = load_janus_brain()
        logger.info("[brain] JanusGPT loaded")
        return brain
    except Exception as e:
        logger.warning(f"[brain] JanusGPT unavailable: {e}")

    try:
        from avus_brain import AvusBrain
        brain = AvusBrain()
        logger.info("[brain] AvusBrain loaded")
        return brain
    except Exception as e:
        logger.warning(f"[brain] AvusBrain unavailable: {e}")

    logger.warning("[brain] No AI brain — using rule-based fallback")
    return None


def _load_human_core():
    try:
        from janus_human_core import HumanCore
        core = HumanCore(auto_load_mood=True)
        logger.info("[human] HumanCore loaded")
        return core
    except Exception as e:
        logger.warning(f"[human] HumanCore unavailable: {e}")
        return None


_FALLBACKS = [
    "I heard you.",
    "Interesting. Tell me more.",
    "I am still learning, but I am here.",
    "Got it. What else is on your mind?",
    "I am listening.",
]
_fb_idx = 0

def _fallback(text: str) -> str:
    global _fb_idx
    r = _FALLBACKS[_fb_idx % len(_FALLBACKS)]
    _fb_idx += 1
    return r


def generate_response(brain, human, user_text: str, history: list) -> str:
    """Generate Janus's response to user input."""
    if human:
        human.social.observe(user_text)
        human.mood.drift_toward_neutral()

    # Build prompt with recent history
    context_lines = []
    for role, text in history[-6:]:  # last 3 turns
        label = "User" if role == "user" else "Janus"
        context_lines.append(f"{label}: {text}")
    context_lines.append(f"User: {user_text}")
    context_lines.append("Janus:")
    prompt = "\n".join(context_lines)

    response = None
    if brain is not None:
        try:
            if hasattr(brain, "generate"):
                response = brain.generate(prompt, max_new=120, temperature=0.7)
                # Strip any continuation past the first newline
                response = response.split("\n")[0].strip()
            elif hasattr(brain, "chat"):
                response = brain.chat(user_text)
            elif hasattr(brain, "ask"):
                response = brain.ask(prompt)
            elif callable(brain):
                response = brain(prompt)
        except Exception as e:
            logger.error(f"[brain] Error: {e}")

    if not response or not response.strip():
        response = _fallback(user_text)

    # Apply HumanCore personality
    if human:
        try:
            response = human.social.apply_tone(response)
            if human.fatigue.state.energy < 0.3:
                response = human._trim_response(response)
            human.fatigue.work(minutes=len(response.split()) / 150)
            human.mood.save()
        except Exception:
            pass

    logger.info(f"[janus] → {response[:80]}")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Main conversation loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║          JANUS VOICE INTERFACE           ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # Load components
    stt   = JanusSTT()
    tts   = JanusTTS()
    brain = _load_brain()
    human = _load_human_core()

    # Check microphone
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        p.terminate()
    except Exception as e:
        print(f"[error] pyaudio not available: {e}")
        print("Install with: pip install pyaudio")
        sys.exit(1)

    # Greeting based on mood
    mood = human.mood.mood.label if human else "neutral"
    greetings = {
        "excited":  "Hey! I am here. What is up?",
        "content":  "Hello. Good to hear from you.",
        "positive": "Hi there. I am listening.",
        "neutral":  "Hello. Say hey Janus whenever you need me.",
        "low":      "I am here. A bit tired, but I am listening.",
    }
    greeting = greetings.get(mood, "Hello. I am Janus. Say hey Janus to talk.")

    print(f"[Janus] {greeting}")
    tts.speak(greeting)

    print()
    print("  Say 'Hey Janus' to wake her up.")
    print("  Press Ctrl+C to stop.")
    print()

    # Conversation state
    awake = False
    last_activity = time.time()
    SLEEP_AFTER = 30.0  # seconds of silence before going back to sleep
    history = []

    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=stt.SAMPLE_RATE,
        input=True,
        frames_per_buffer=stt.CHUNK,
    )

    try:
        while True:
            # Always listening — record a chunk
            audio = stt.record_until_silence(stream)

            if len(audio) < 3200:  # too short, skip
                continue

            text = stt.transcribe(audio)
            if not text:
                continue

            logger.info(f"[heard] '{text}'")

            if not awake:
                # Waiting for wake word
                if stt.is_wake_word(text):
                    awake = True
                    last_activity = time.time()
                    ack = "Yes? I am listening."
                    print(f"[Janus] {ack}")
                    tts.speak(ack, speed=1.05)
                continue

            # Awake — process as conversation
            last_activity = time.time()

            # Check for sleep command
            if any(w in text.lower() for w in ["go to sleep", "goodbye", "stop listening"]):
                awake = False
                bye = "Okay. I will be here when you need me."
                print(f"[Janus] {bye}")
                tts.speak(bye)
                continue

            # Generate and speak response
            print(f"[User] {text}")
            response = generate_response(brain, human, text, history)
            history.append(("user", text))
            history.append(("janus", response))
            if len(history) > 20:
                history = history[-20:]

            print(f"[Janus] {response}")
            tts.speak(response)

            # Auto-sleep after inactivity
            if time.time() - last_activity > SLEEP_AFTER:
                awake = False
                logger.info("[voice] Returning to wake word mode")

    except KeyboardInterrupt:
        print("\n[shutdown] Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        if human:
            human.mood.save()
        print("[shutdown] Done.")


if __name__ == "__main__":
    main()
