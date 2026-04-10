"""
speech_synthesis.py
Human Speech Synthesis engine for Janus.
Implements the diagnostic framework taxonomy (all 8 sections).
No API keys. Outputs 16-bit PCM WAV bytes.

Usage:
    from speech_synthesis import HumanSpeechSynthesizer, SpeechContext
    synth = HumanSpeechSynthesizer()
    ctx   = SpeechContext(emotion="friendly", fatigue=0.1)
    audio = synth.synthesize("Hello, I am Janus.", ctx)
    synth.save_wav(audio, "output.wav")
"""
from __future__ import annotations
import math, random, re, wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

SAMPLE_RATE = 22050

# ── Section 1: Anatomical constraints ────────────────────────────────────────

@dataclass
class VocalTract:
    """Models anatomical constraints (Section 1)."""
    fold_length_mm: float = 14.0       # male ~14mm, female ~10mm
    subglottal_pressure: float = 8.0   # cm H2O
    max_tongue_velocity: float = 400.0 # mm/s
    jaw_opening_mm: float = 15.0
    tidal_volume_ml: float = 900.0
    breath_group_syllables: int = 8

    @property
    def f0_base(self) -> float:
        """Fundamental frequency from fold length."""
        return 1680.0 / self.fold_length_mm

    @property
    def max_utterance_duration(self) -> float:
        """Max seconds before forced breath."""
        return self.tidal_volume_ml / 150.0


# ── Section 5: Emotional & paralinguistic state ───────────────────────────────

@dataclass
class SpeechContext:
    """Full speech context (Sections 5, 6, 8)."""
    emotion: str = "neutral"
    # neutral, happy, sad, angry, fearful, surprised, disgusted,
    # excited, bored, anxious, confident, uncertain, friendly
    formality: str = "neutral"         # formal, neutral, casual, intimate
    cognitive_load: float = 0.0        # 0=relaxed, 1=max load
    fatigue: float = 0.0               # 0=fresh, 1=exhausted
    utterance_position: str = "mid"    # start, mid, end
    allow_errors: bool = True
    room_size: str = "none"            # none, small, medium, large
    add_breath_noise: bool = True


# ── Emotion parameter table (Section 5.1) ────────────────────────────────────
# (f0_mult, f0_range, rate_mult, noise, breathiness, pressed)
EMOTION_PARAMS = {
    "neutral":   (1.00, 0.20, 1.00, 0.01, 0.02, 0.0),
    "happy":     (1.15, 0.40, 1.10, 0.01, 0.01, 0.1),
    "sad":       (0.88, 0.12, 0.85, 0.02, 0.06, 0.0),
    "angry":     (1.20, 0.45, 1.05, 0.03, 0.01, 0.3),
    "fearful":   (1.18, 0.35, 1.15, 0.04, 0.05, 0.1),
    "surprised": (1.25, 0.50, 1.20, 0.01, 0.02, 0.0),
    "disgusted": (0.90, 0.15, 0.90, 0.02, 0.04, 0.1),
    "excited":   (1.30, 0.55, 1.25, 0.01, 0.01, 0.2),
    "bored":     (0.85, 0.10, 0.80, 0.01, 0.03, 0.0),
    "anxious":   (1.10, 0.30, 1.10, 0.05, 0.04, 0.1),
    "confident": (1.00, 0.22, 0.95, 0.01, 0.01, 0.1),
    "uncertain": (1.05, 0.25, 0.95, 0.02, 0.03, 0.0),
    "friendly":  (1.08, 0.30, 1.05, 0.01, 0.02, 0.0),
}

# (rate_mult, f0_mult, articulation_precision)
FORMALITY_PARAMS = {
    "formal":   (0.90, 0.95, 1.0),
    "neutral":  (1.00, 1.00, 0.9),
    "casual":   (1.10, 1.05, 0.7),
    "intimate": (0.95, 1.02, 0.6),
}


# ── Section 2.1: Glottal source ──────────────────────────────────────────────

class GlottalSource:
    """Models glottal flow waveform with jitter, shimmer, breathiness."""

    def __init__(self, tract: VocalTract):
        self.tract = tract

    def generate(self, duration: float, f0: float, ctx: SpeechContext,
                 breathiness: float = 0.02, pressed: float = 0.0) -> np.ndarray:
        n = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n, endpoint=False)

        # Jitter: cycle-to-cycle F0 variation (Section 4.2)
        jitter_rate = 0.01 + ctx.fatigue * 0.01
        jitter = 1.0 + np.cumsum(
            np.random.normal(0, jitter_rate / SAMPLE_RATE, n)
        ).clip(-0.03, 0.03)
        phase = np.cumsum(2 * np.pi * f0 * jitter / SAMPLE_RATE)

        # Glottal flow: skewed waveform (gradual open, fast close)
        glottal = np.sin(phase)
        glottal = np.where(glottal > 0, glottal ** 0.7, glottal * 1.3)

        # Shimmer: amplitude variation (Section 4.2)
        shimmer_rate = 0.02 + ctx.fatigue * 0.02
        shimmer = 1.0 + np.random.normal(0, shimmer_rate, n)
        glottal *= shimmer

        # Breathiness: additive aspiration noise (Section 2.1)
        breathiness_total = breathiness + ctx.fatigue * 0.04
        if breathiness_total > 0:
            noise = np.random.normal(0, breathiness_total, n)
            glottal = glottal * (1 - breathiness_total) + noise

        # Pressed phonation: enhance high harmonics (Section 2.1)
        if pressed > 0:
            glottal += pressed * 0.3 * np.sin(phase * 3)

        # Vocal fry: occasional low-frequency irregularity (Section 2.1)
        if ctx.fatigue > 0.7 or (ctx.allow_errors and random.random() < 0.02):
            fry_len = int(0.05 * SAMPLE_RATE)
            fry_start = random.randint(0, max(0, n - fry_len))
            fry = np.sin(2 * np.pi * (f0 * 0.3) * t[fry_start:fry_start+fry_len])
            glottal[fry_start:fry_start+fry_len] *= 0.4
            glottal[fry_start:fry_start+fry_len] += fry * 0.3

        return glottal


# ── Section 2.2: Vocal tract filter ──────────────────────────────────────────

class VocalTractFilter:
    """Formant filter: F1-F4 with bandwidths and anti-formants."""

    DEFAULT_FORMANTS = [
        (800,  80),   # F1
        (1200, 100),  # F2
        (2500, 120),  # F3
        (3500, 150),  # F4
    ]

    def apply(self, signal: np.ndarray,
              formants: Optional[List[Tuple[float, float]]] = None,
              nasality: float = 0.0) -> np.ndarray:
        formants = formants or self.DEFAULT_FORMANTS
        result = np.zeros_like(signal)
        for freq, bw in formants:
            result += self._bandpass(signal, freq, bw)
        if nasality > 0:
            notch = self._notch(result, 1000, 200)
            result = result * (1 - nasality) + notch * nasality
        return result

    def _bandpass(self, x: np.ndarray, freq: float, bw: float) -> np.ndarray:
        r = max(0.0, min(0.999, 1.0 - (math.pi * bw / SAMPLE_RATE)))
        cos_w = math.cos(2 * math.pi * freq / SAMPLE_RATE)
        a1, a2, b0 = -2*r*cos_w, r*r, 1-r
        y = np.zeros_like(x)
        for i in range(2, len(x)):
            y[i] = b0*x[i] - a1*y[i-1] - a2*y[i-2]
        return y

    def _notch(self, x: np.ndarray, freq: float, bw: float) -> np.ndarray:
        r = 1.0 - (math.pi * bw / SAMPLE_RATE)
        cos_w = math.cos(2 * math.pi * freq / SAMPLE_RATE)
        b0, b1, b2 = 1.0, -2*cos_w, 1.0
        a1, a2 = -2*r*cos_w, r*r
        y = np.zeros_like(x)
        for i in range(2, len(x)):
            y[i] = b0*x[i]+b1*x[i-1]+b2*x[i-2] - a1*y[i-1] - a2*y[i-2]
        return y


# ── Section 3: Prosody engine ─────────────────────────────────────────────────

class ProsodyEngine:
    """F0 contours, timing, loudness (Section 3)."""

    def compute_f0_contour(self, n_syl: int, f0_base: float,
                           ctx: SpeechContext) -> np.ndarray:
        ep = EMOTION_PARAMS.get(ctx.emotion, EMOTION_PARAMS["neutral"])
        fp = FORMALITY_PARAMS.get(ctx.formality, FORMALITY_PARAMS["neutral"])
        f0 = f0_base * ep[0] * fp[1]
        contour = np.ones(n_syl) * f0
        # Declination (Section 3.1)
        contour *= np.linspace(1.0, 0.85, n_syl)
        # Initial high
        if ctx.utterance_position == "start" and n_syl > 0:
            contour[0] *= 1.15
        # Downstep
        for i in range(1, n_syl):
            if random.random() < 0.3:
                contour[i] *= 0.95
        # Final lowering
        if ctx.utterance_position == "end" and n_syl > 1:
            contour[-1] *= 0.80
            if n_syl > 2:
                contour[-2] *= 0.90
        # Continuation rise
        if ctx.utterance_position == "mid" and n_syl > 1:
            contour[-1] *= 1.05
        # Emotion: widen/narrow F0 range
        mean_f0 = contour.mean()
        contour = mean_f0 + (contour - mean_f0) * (1.0 + ep[1] * 2)
        # Cognitive load: flatten prosody (Section 8.1)
        if ctx.cognitive_load > 0:
            flat = np.ones_like(contour) * mean_f0
            contour = contour*(1-ctx.cognitive_load*0.5) + flat*(ctx.cognitive_load*0.5)
        return contour

    def compute_durations(self, n_syl: int, ctx: SpeechContext) -> np.ndarray:
        ep = EMOTION_PARAMS.get(ctx.emotion, EMOTION_PARAMS["neutral"])
        fp = FORMALITY_PARAMS.get(ctx.formality, FORMALITY_PARAMS["neutral"])
        base_dur = 0.15 / (ep[2] * fp[0])
        base_dur *= (1.0 + ctx.cognitive_load * 0.3)
        base_dur *= (1.0 + ctx.fatigue * 0.2)
        durations = np.ones(n_syl) * base_dur
        # Phrase-final lengthening (Section 3.2)
        if n_syl > 0:
            durations[-1] *= random.uniform(1.3, 1.6)
        # Natural timing variation
        durations *= np.random.normal(1.0, 0.08, n_syl).clip(0.7, 1.4)
        return durations

    def compute_amplitude_envelope(self, n: int, ctx: SpeechContext,
                                   durations: np.ndarray) -> np.ndarray:
        envelope = np.ones(n)
        envelope *= np.linspace(1.0, 0.85, n)  # declination
        envelope *= (1.0 - ctx.fatigue * 0.25)
        attack  = int(0.01 * SAMPLE_RATE)
        release = int(0.02 * SAMPLE_RATE)
        if attack < n:
            envelope[:attack] *= np.linspace(0, 1, attack)
        if release < n:
            envelope[-release:] *= np.linspace(1, 0, release)
        return envelope


# ── Section 4: Micro-variations ───────────────────────────────────────────────

class MicroVariationEngine:
    """Stochastic imperfections (Section 4)."""

    def add_hesitations(self, audio: np.ndarray, ctx: SpeechContext) -> np.ndarray:
        if not ctx.allow_errors:
            return audio
        if random.random() > (0.05 + ctx.cognitive_load * 0.15):
            return audio
        pause_len = int(random.uniform(0.05, 0.20) * SAMPLE_RATE)
        pos = random.randint(0, max(0, len(audio) - pause_len))
        return np.concatenate([audio[:pos], np.zeros(pause_len), audio[pos:]])

    def add_voice_crack(self, audio: np.ndarray, ctx: SpeechContext) -> np.ndarray:
        if not ctx.allow_errors:
            return audio
        ep = EMOTION_PARAMS.get(ctx.emotion, EMOTION_PARAMS["neutral"])
        if random.random() > (ctx.fatigue * 0.05 + ep[0] * 0.01):
            return audio
        crack_len = int(random.uniform(0.01, 0.04) * SAMPLE_RATE)
        pos = random.randint(0, max(0, len(audio) - crack_len))
        audio[pos:pos+crack_len] *= random.uniform(0.1, 0.5)
        return audio

    def add_breathiness_episode(self, audio: np.ndarray, ctx: SpeechContext) -> np.ndarray:
        if not ctx.allow_errors:
            return audio
        if random.random() > (ctx.fatigue * 0.1 + 0.02):
            return audio
        ep_len = int(random.uniform(0.05, 0.15) * SAMPLE_RATE)
        pos = random.randint(0, max(0, len(audio) - ep_len))
        noise = np.random.normal(0, 0.1, ep_len)
        audio[pos:pos+ep_len] = audio[pos:pos+ep_len] * 0.5 + noise
        return audio

    def add_lip_smack(self, audio: np.ndarray) -> np.ndarray:
        if random.random() > 0.08:
            return audio
        smack_len = int(0.015 * SAMPLE_RATE)
        smack = np.random.normal(0, 0.15, smack_len) * np.linspace(1, 0, smack_len)
        return np.concatenate([smack, audio])

    def add_breath_intake(self, audio: np.ndarray, ctx: SpeechContext) -> np.ndarray:
        if not ctx.add_breath_noise:
            return audio
        breath_len = int(random.uniform(0.08, 0.20) * SAMPLE_RATE)
        breath = np.random.normal(0, 0.04, breath_len)
        env = np.concatenate([
            np.linspace(0, 1, breath_len // 2),
            np.linspace(1, 0, breath_len - breath_len // 2)
        ])
        return np.concatenate([breath * env, audio])


# ── Section 7: Environmental artifacts ───────────────────────────────────────

class EnvironmentProcessor:
    """Room acoustics and recording artifacts (Section 7)."""

    ROOM_RT60 = {"none": 0.0, "small": 0.15, "medium": 0.35, "large": 0.80}

    def apply(self, audio: np.ndarray, ctx: SpeechContext) -> np.ndarray:
        rt60 = self.ROOM_RT60.get(ctx.room_size, 0.0)
        if rt60 > 0:
            audio = self._reverb(audio, rt60)
        return audio

    def _reverb(self, audio: np.ndarray, rt60: float) -> np.ndarray:
        delay = int(0.02 * SAMPLE_RATE)
        decay = math.exp(-6.908 * delay / (rt60 * SAMPLE_RATE))
        result = audio.copy()
        if delay < len(audio):
            result[delay:] += audio[:-delay] * decay * 0.4
            d2 = delay * 2
            if d2 < len(audio):
                result[d2:] += audio[:-d2] * decay * decay * 0.2
        return result


# ── Main synthesizer ──────────────────────────────────────────────────────────

class HumanSpeechSynthesizer:
    """
    Full human speech synthesis pipeline.
    text -> prosody -> glottal source -> formant filter -> micro-variations
    -> environment -> 16-bit PCM
    """

    def __init__(self, vocal_tract: Optional[VocalTract] = None):
        self.tract   = vocal_tract or VocalTract()
        self.source  = GlottalSource(self.tract)
        self.filter  = VocalTractFilter()
        self.prosody = ProsodyEngine()
        self.micro   = MicroVariationEngine()
        self.env     = EnvironmentProcessor()

    def synthesize(self, text: str, ctx: Optional[SpeechContext] = None) -> bytes:
        """Synthesize text to 16-bit PCM bytes at SAMPLE_RATE Hz, mono."""
        ctx = ctx or SpeechContext()
        syllables = self._segment_syllables(text)
        n_syl = max(1, len(syllables))

        f0_contour = self.prosody.compute_f0_contour(n_syl, self.tract.f0_base, ctx)
        durations  = self.prosody.compute_durations(n_syl, ctx)

        # Enforce breath group limit (Section 8.2)
        max_dur = self.tract.max_utterance_duration
        if durations.sum() > max_dur:
            durations *= max_dur / durations.sum()

        ep = EMOTION_PARAMS.get(ctx.emotion, EMOTION_PARAMS["neutral"])
        breathiness, pressed = ep[4], ep[5]

        segments = []
        for i, (syl, dur, f0) in enumerate(zip(syllables, durations, f0_contour)):
            glottal  = self.source.generate(dur, f0, ctx, breathiness, pressed)
            formants = self._syllable_formants(syl, ctx)
            voiced   = self.filter.apply(glottal, formants)
            if i < n_syl - 1:
                gap = int(random.uniform(0.005, 0.015) * SAMPLE_RATE)
                voiced = np.concatenate([voiced, np.zeros(gap)])
            segments.append(voiced)

        audio = np.concatenate(segments) if segments else np.zeros(SAMPLE_RATE)
        audio *= self.prosody.compute_amplitude_envelope(len(audio), ctx, durations)

        # Micro-variations (Section 4)
        audio = self.micro.add_breath_intake(audio, ctx)
        audio = self.micro.add_lip_smack(audio)
        audio = self.micro.add_hesitations(audio, ctx)
        audio = self.micro.add_voice_crack(audio, ctx)
        audio = self.micro.add_breathiness_episode(audio, ctx)

        # Environment (Section 7)
        audio = self.env.apply(audio, ctx)

        # Normalize and convert to 16-bit PCM
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.85
        return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

    def save_wav(self, pcm_bytes: bytes, path: str):
        """Save PCM bytes to a WAV file."""
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        print(f"[SpeechSynth] Saved {len(pcm_bytes)//2} samples -> {path}")

    def _segment_syllables(self, text: str) -> List[str]:
        """Rough syllable segmentation by vowel nuclei."""
        vowels = re.findall(r"[aeiouAEIOU]+[^aeiouAEIOU\s]*", text)
        return vowels if vowels else [text[:max(1, len(text)//3)]]

    def _syllable_formants(self, syllable: str,
                           ctx: SpeechContext) -> List[Tuple[float, float]]:
        """Approximate formant frequencies per syllable (Section 2.2)."""
        s = syllable.lower()
        if any(v in s for v in ("ee", "i", "e")):
            f1, f2 = 300, 2200
        elif any(v in s for v in ("oo", "u", "o")):
            f1, f2 = 400, 800
        elif any(v in s for v in ("ah", "a")):
            f1, f2 = 800, 1200
        else:
            f1, f2 = 600, 1500
        # Formant variation (Section 4.3)
        f1 += random.gauss(0, 30)
        f2 += random.gauss(0, 40)
        # Formality: wider vowel space for formal speech (Section 5.3)
        fp = FORMALITY_PARAMS.get(ctx.formality, FORMALITY_PARAMS["neutral"])
        f2 *= fp[2]
        return [
            (max(200, f1), 80),
            (max(600, f2), 100),
            (2500 + random.gauss(0, 50), 120),
            (3500 + random.gauss(0, 80), 150),
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

_synth: Optional[HumanSpeechSynthesizer] = None

def get_synthesizer(vocal_tract: Optional[VocalTract] = None) -> HumanSpeechSynthesizer:
    global _synth
    if _synth is None:
        _synth = HumanSpeechSynthesizer(vocal_tract)
    return _synth


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    synth = HumanSpeechSynthesizer()
    tests = [
        ("Hello, I am Janus.",           SpeechContext(emotion="neutral")),
        ("I am very excited about this!", SpeechContext(emotion="excited")),
        ("I am not sure about that...",  SpeechContext(emotion="uncertain", cognitive_load=0.6)),
        ("This is unacceptable.",        SpeechContext(emotion="angry", formality="formal")),
        ("Hey, how are you doing?",      SpeechContext(emotion="friendly", formality="casual",
                                                       room_size="small")),
    ]
    for text, ctx in tests:
        print(f"Synthesizing [{ctx.emotion}]: {text}")
        audio = synth.synthesize(text, ctx)
        fname = f"test_{ctx.emotion}.wav"
        synth.save_wav(audio, fname)
        print(f"  -> {len(audio)//2} samples ({len(audio)/2/SAMPLE_RATE:.2f}s)")