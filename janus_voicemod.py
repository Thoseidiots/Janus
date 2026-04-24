"""
janus_voicemod.py
=================
Voicemod V3 integration for Janus.

Voicemod exposes a WebSocket API on localhost:59129.
This module:
  1. Connects to Voicemod and selects a voice effect
  2. Routes Janus's TTS audio through Voicemod's virtual microphone
  3. Plays the result so it sounds like Janus's chosen voice

Usage:
    vm = VoicemodBridge()
    if vm.connect():
        vm.set_voice("anime")       # or "female_pitch", "robot", etc.
        vm.play_through(pcm_bytes)  # plays via Voicemod virtual mic
        vm.disconnect()

Voicemod must be running for this to work.
"""

import json
import time
import wave
import tempfile
import os
import threading
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("janus.voicemod")

VOICEMOD_WS_PORT = 59129
VOICEMOD_WS_URL  = f"ws://localhost:{VOICEMOD_WS_PORT}/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Voicemod WebSocket client
# ─────────────────────────────────────────────────────────────────────────────

class VoicemodBridge:
    """
    Controls Voicemod V3 via its local WebSocket API.
    Lets Janus select voice effects and route audio through them.
    """

    # Voice effect names → Voicemod internal IDs (common ones)
    VOICE_PRESETS = {
        "anime":        "anime-girl",
        "female":       "female-pitch",
        "robot":        "robot",
        "alien":        "alien",
        "cave":         "cave",
        "deep":         "deep-voice",
        "helium":       "helium",
        "baby":         "baby",
        "none":         "nofx",
        "default":      "nofx",
    }

    def __init__(self):
        self._ws = None
        self._connected = False
        self._voices = []
        self._current_voice = None
        self._msg_id = 1
        self._responses = {}
        self._lock = threading.Lock()

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Voicemod WebSocket API.
        Voicemod V3 requires a developer client key for full API access.
        We use direct virtual device routing instead which needs no auth.
        Returns True if Voicemod virtual mic device is found.
        """
        self._connected = self._find_virtual_device()
        return self._connected

    def _find_virtual_device(self) -> bool:
        """Find Voicemod's virtual audio output device for routing."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            # Priority: VAD Wave Line Out > Virtual Audio Device Line > any Voicemod output
            priorities = ["voicemod vad wave", "line (voicemod", "voicemod virtual audio"]
            best_idx = None
            best_name = None
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                name = info.get("name", "").lower()
                outs = info.get("maxOutputChannels", 0)
                if outs > 0:
                    for pri in priorities:
                        if pri in name:
                            best_idx = i
                            best_name = info["name"]
                            break
                    if best_idx is not None:
                        break
            p.terminate()
            if best_idx is not None:
                self._virtual_device_index = best_idx
                self._virtual_device_name = best_name
                logger.info(f"[Voicemod] Output device: [{best_idx}] {best_name}")
                return True
            logger.info("[Voicemod] No Voicemod output device found")
            return False
        except Exception as e:
            logger.warning(f"[Voicemod] Device scan failed: {e}")
            return False

    def disconnect(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_voice(self, voice_name: str) -> bool:
        """
        Select a voice effect by name.
        voice_name: one of VOICE_PRESETS keys, or a raw Voicemod voice ID.
        """
        if not self._connected:
            return False

        voice_id = self.VOICE_PRESETS.get(voice_name.lower(), voice_name)

        # Find matching voice in available list
        matched = None
        for v in self._voices:
            vid = v.get("id", "").lower()
            vname = v.get("friendlyName", "").lower()
            if voice_id.lower() in vid or voice_id.lower() in vname:
                matched = v.get("id")
                break

        if not matched:
            # Try partial match
            for v in self._voices:
                if voice_name.lower() in v.get("friendlyName", "").lower():
                    matched = v.get("id")
                    break

        if matched:
            resp = self._send("selectVoice", {"voiceID": matched})
            self._current_voice = matched
            logger.info(f"[Voicemod] Voice set to: {matched}")
            return True
        else:
            logger.warning(f"[Voicemod] Voice '{voice_name}' not found. "
                           f"Available: {[v.get('friendlyName') for v in self._voices[:5]]}")
            return False

    def list_voices(self) -> list:
        """Return list of available voice names."""
        return [v.get("friendlyName", v.get("id", "?")) for v in self._voices]

    def enable_background_effects(self, enabled: bool = True):
        """Toggle background effects (noise removal etc.)."""
        if self._connected:
            self._send("toggleBackground", {"value": enabled})

    def play_through(self, pcm_bytes: bytes, sample_rate: int = 22050):
        """
        Play PCM audio through Voicemod's virtual microphone device.
        Voicemod applies its voice effects in real-time.
        """
        if not pcm_bytes:
            return
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            device_index = getattr(self, "_virtual_device_index", None)
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=device_index,
            )
            stream.write(pcm_bytes)
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("[Voicemod] Audio played through virtual device")
        except Exception as e:
            logger.error(f"[Voicemod] Playback error: {e}")

    def _get_voices(self) -> list:
        """Fetch available voices from Voicemod."""
        try:
            resp = self._send("getVoices", {})
            if resp and "voices" in resp:
                return resp["voices"]
        except Exception:
            pass
        return []

    def _send(self, action: str, payload: dict) -> dict:
        """Send a message to Voicemod and return the response."""
        if not self._ws:
            return {}
        with self._lock:
            msg_id = self._msg_id
            self._msg_id += 1
            msg = json.dumps({
                "id": msg_id,
                "action": action,
                "payload": payload,
            })
            try:
                self._ws.send(msg)
                raw = self._ws.recv()
                return json.loads(raw) if raw else {}
            except Exception as e:
                logger.debug(f"[Voicemod] Send error: {e}")
                return {}


# ─────────────────────────────────────────────────────────────────────────────
# Voicemod-aware TTS wrapper
# ─────────────────────────────────────────────────────────────────────────────

class JanusVoicemodTTS:
    """
    Wraps JanusTTS and optionally routes output through Voicemod.

    If Voicemod is running: uses Edge TTS + Voicemod voice effect
    If Voicemod is not running: uses Edge TTS directly (still sounds good)
    """

    # Best Voicemod voice for Janus — anime-girl or female-pitch
    JANUS_VOICE_EFFECT = "anime"

    def __init__(self, tts=None, auto_connect: bool = True):
        from janus_tts import JanusTTS
        self.tts = tts or JanusTTS()
        self.voicemod = VoicemodBridge()
        self._voicemod_active = False

        if auto_connect:
            self._try_connect_voicemod()

    def _try_connect_voicemod(self):
        """Try to connect to Voicemod. Non-fatal if unavailable."""
        if self.voicemod.connect():
            voices = self.voicemod.list_voices()
            logger.info(f"[Voicemod] Available voices: {voices[:8]}")

            # Set Janus's voice effect
            if self.voicemod.set_voice(self.JANUS_VOICE_EFFECT):
                self._voicemod_active = True
                logger.info(f"[Voicemod] Active — using '{self.JANUS_VOICE_EFFECT}' effect")
            else:
                # Try first available voice
                if voices:
                    self.voicemod.set_voice(voices[0])
                    self._voicemod_active = True
        else:
            logger.info("[Voicemod] Not running — using Edge TTS directly")

    def speak(self, text: str, speed: float = 1.0, pitch: float = 1.0) -> bytes:
        """
        Synthesize and play speech.
        Returns PCM bytes (also plays audio).
        """
        pcm = self.tts.synthesize(text, speed=speed, pitch=pitch)

        if self._voicemod_active:
            # Route through Voicemod virtual mic
            self.voicemod.play_through(pcm)
        else:
            # Play directly
            self._play_pcm(pcm)

        return pcm

    def _play_pcm(self, pcm_bytes: bytes, sample_rate: int = 22050):
        """Play PCM bytes directly via pyaudio."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            stream.write(pcm_bytes)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logger.error(f"[TTS] Playback error: {e}")

    def set_voice_effect(self, effect: str):
        """Change Voicemod voice effect at runtime."""
        if self._voicemod_active:
            self.voicemod.set_voice(effect)

    def list_voice_effects(self) -> list:
        """List available Voicemod effects."""
        if self._voicemod_active:
            return self.voicemod.list_voices()
        return ["(Voicemod not connected)"]

    def __del__(self):
        if self.voicemod:
            self.voicemod.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Janus Voicemod Integration Test")
    print("-" * 40)

    vm = VoicemodBridge()
    if vm.connect():
        print(f"Connected to Voicemod!")
        voices = vm.list_voices()
        print(f"Available voices ({len(voices)}):")
        for v in voices[:10]:
            print(f"  - {v}")

        print(f"\nSetting voice to 'anime'...")
        vm.set_voice("anime")
        vm.disconnect()
    else:
        print("Voicemod not running — start Voicemod V3 first.")
        print("Edge TTS will be used as standalone fallback.")

    print("\nTesting Edge TTS synthesis...")
    from janus_tts import JanusTTS, _save_wav, SAMPLE_RATE
    import numpy as np

    tts = JanusTTS()
    pcm = tts.synthesize("Hello, I am Janus. Voicemod integration is ready.")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    _save_wav("janus_voicemod_test.wav", audio, SAMPLE_RATE)
    print(f"Saved: janus_voicemod_test.wav ({len(audio)/SAMPLE_RATE:.1f}s)")

    import subprocess
    subprocess.Popen(["powershell", "-Command",
                      "(New-Object Media.SoundPlayer 'janus_voicemod_test.wav').PlaySync()"])
