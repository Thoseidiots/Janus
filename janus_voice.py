"""
janus_voice.py
==============
Wires Janus's voice I/O to her brain.

Say "Hey Janus" → she wakes up → you talk → she thinks → she responds.

Uses:
  - voice_io_enhanced.py  (microphone, wake word, STT, TTS)
  - janus_gpt.py / avus_brain.py  (thinking)
  - janus_human_core.py  (mood, personality, fatigue)

Run:
    python janus_voice.py

Requirements (install once):
    pip install pyaudio numpy
    # For real STT (optional, falls back to energy detection without it):
    # Install whisper.cpp and place ggml-base.en.bin in models/
    # For real TTS (optional, falls back to tone synthesis without it):
    # Install piper-tts and place en_US-lessac-medium.onnx in models/
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("janus.voice")


# ─────────────────────────────────────────────────────────────────────────────
# Brain — what Janus thinks with
# ─────────────────────────────────────────────────────────────────────────────

def _load_brain():
    """Load the best available brain backend."""
    # Try JanusGPT first (uses local Avus weights)
    try:
        from janus_gpt import JanusGPT
        brain = JanusGPT()
        logger.info("[brain] JanusGPT loaded")
        return brain
    except Exception as e:
        logger.warning(f"[brain] JanusGPT unavailable: {e}")

    # Try AvusBrain directly
    try:
        from avus_brain import AvusBrain
        brain = AvusBrain()
        logger.info("[brain] AvusBrain loaded")
        return brain
    except Exception as e:
        logger.warning(f"[brain] AvusBrain unavailable: {e}")

    # Fallback: simple rule-based responses so voice still works
    logger.warning("[brain] No AI brain available — using rule-based fallback")
    return None


def _load_human_core():
    """Load HumanCore for personality and mood."""
    try:
        from janus_human_core import HumanCore
        core = HumanCore(auto_load_mood=True)
        logger.info("[human] HumanCore loaded")
        return core
    except Exception as e:
        logger.warning(f"[human] HumanCore unavailable: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Response generation
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_RESPONSES = [
    "I heard you. Let me think about that.",
    "Interesting. Tell me more.",
    "I'm still learning, but I'm here.",
    "Got it. What else is on your mind?",
    "I'm listening.",
]

_fallback_idx = 0

def _fallback_response(text: str) -> str:
    global _fallback_idx
    resp = FALLBACK_RESPONSES[_fallback_idx % len(FALLBACK_RESPONSES)]
    _fallback_idx += 1
    return resp


def build_response_handler(brain, human):
    """
    Returns a function: (user_text, context) -> janus_response_text
    This is what gets passed to EnhancedVoiceIOSystem.response_handler
    """

    def handle(user_text: str, context: str = "") -> str:
        # Let HumanCore shape the response style
        if human:
            human.social.observe(user_text)
            human.mood.drift_toward_neutral()

        # Build prompt
        prompt = user_text
        if context:
            prompt = f"{context}\nUser: {user_text}\nJanus:"

        # Generate response
        response = None

        if brain is not None:
            try:
                if hasattr(brain, "chat"):
                    response = brain.chat(user_text)
                elif hasattr(brain, "generate_response"):
                    response = brain.generate_response(prompt)
                elif hasattr(brain, "generate"):
                    import asyncio
                    response = asyncio.get_event_loop().run_until_complete(
                        brain.generate(prompt, max_tokens=120)
                    )
                elif hasattr(brain, "ask"):
                    response = brain.ask(prompt)
                elif callable(brain):
                    response = brain(prompt)
            except Exception as e:
                logger.error(f"[brain] Generation error: {e}")

        if not response or not response.strip():
            response = _fallback_response(user_text)

        # Apply HumanCore personality layer
        if human:
            try:
                response = human.social.apply_tone(response)
                # Trim if tired
                if human.fatigue.state.energy < 0.3:
                    response = human._trim_response(response)
                # Simulate cognitive cost
                human.fatigue.work(minutes=len(response.split()) / 150)
                human.mood.save()
            except Exception:
                pass

        logger.info(f"[janus] → {response[:80]}{'...' if len(response) > 80 else ''}")
        # Attach current mood so TTS can use it for prosody
        handle._last_mood = human.mood.mood if human else None
        return response

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║          JANUS VOICE INTERFACE           ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # Load brain and personality
    brain = _load_brain()
    human = _load_human_core()

    # Build response handler
    response_handler = build_response_handler(brain, human)

    # Load voice system
    try:
        from voice_io_enhanced import EnhancedVoiceIOSystem
    except ImportError as e:
        print(f"[error] voice_io_enhanced.py not importable: {e}")
        print("Make sure pyaudio and numpy are installed:")
        print("  pip install pyaudio numpy")
        sys.exit(1)

    # Patch PiperTTS to use Janus's own voice engine (Edge TTS + optional Voicemod)
    try:
        from janus_voicemod import JanusVoicemodTTS
        _janus_vm_tts = JanusVoicemodTTS(auto_connect=True)
        print("[voice] Using Edge TTS + Voicemod pipeline")

        from voice_io_enhanced import PiperTTS as _PiperTTS

        def _janus_synth(self, text, voice_style="default", emotion="neutral"):
            speed = {"calm": 0.85, "serious": 0.9, "excited": 1.15}.get(voice_style, 1.0)
            mood = getattr(_janus_vm_tts.tts, "_last_mood", None)
            return _janus_vm_tts.tts.synthesize(text, speed=speed, mood=mood)

        _PiperTTS.synthesize = _janus_synth
    except Exception as e:
        print(f"[voice] Voicemod TTS unavailable ({e}), using Edge TTS directly")
        try:
            from janus_tts import JanusTTS as _JanusTTS
            _janus_tts_engine = _JanusTTS()
            from voice_io_enhanced import PiperTTS as _PiperTTS

            def _janus_synth_fallback(self, text, voice_style="default", emotion="neutral"):
                speed = {"calm": 0.85, "serious": 0.9, "excited": 1.15}.get(voice_style, 1.0)
                return _janus_tts_engine.synthesize(text, speed=speed)

            _PiperTTS.synthesize = _janus_synth_fallback
        except Exception as e2:
            print(f"[voice] TTS fallback also failed: {e2}")

    voice = EnhancedVoiceIOSystem(
        whisper_model="models/ggml-base.en.bin",
        piper_model="models/en_US-lessac-medium.onnx",
    )

    # Wire response handler
    voice.response_handler = response_handler

    # Wake word callback — mood boost when she wakes up
    def on_wake():
        if human:
            human.mood.update("positive", intensity=0.2)
        logger.info("[wake] Janus is awake")

    voice.on_wake_word = on_wake

    # Greeting
    mood_label = human.mood.mood.label if human else "neutral"
    greetings = {
        "excited":  "Hey! I'm here. What's up?",
        "content":  "Hello. Good to hear from you.",
        "positive": "Hi there. I'm listening.",
        "neutral":  "Hello. Say 'Hey Janus' whenever you need me.",
        "low":      "I'm here. A bit tired, but I'm listening.",
        "stressed": "I'm here. Take your time.",
    }
    greeting = greetings.get(mood_label, "Hello. I'm Janus. Say 'Hey Janus' to talk.")

    # Start voice system
    voice.start()
    voice.speak(greeting, voice_style="friendly", block=True)

    print()
    print("  Say 'Hey Janus' to wake her up.")
    print("  Press Ctrl+C to stop.")
    print()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[shutdown] Stopping...")
        voice.stop()
        if human:
            human.mood.save()
        print("[shutdown] Done.")


if __name__ == "__main__":
    main()
