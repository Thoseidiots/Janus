"""
audio_validator.py
==================
Validates synthesized audio before it enters the training dataset.

Two-stage check:
  1. Energy / silence gate
     - Computes RMS energy of the waveform
     - If below threshold → silence or near-silence → word wasn't spoken
     - Also checks that voiced frames (energy > floor) cover enough of the clip

  2. Spectral speech check
     - Computes spectral centroid and spectral flatness
     - Human speech sits in 300-3400 Hz (telephone band), centroid typically 1000-3000 Hz
     - Spectral flatness near 1.0 = white noise; near 0.0 = tonal/speech
     - If centroid is outside speech range OR flatness is too high → noise, not speech

  3. Duration sanity check
     - Too short (< 0.3s) for a non-trivial utterance → synthesis failed
     - Too long (> 30s) → runaway synthesis

Returns a ValidationResult with pass/fail and a reason string.
Can be used standalone to audit existing WAV files.

Usage:
    from audio_validator import AudioValidator
    v = AudioValidator()
    result = v.validate_pcm(pcm_bytes, sample_rate=24000, text="hello world")
    if not result.passed:
        print(f"Rejected: {result.reason}")

    # Or validate a WAV file directly
    result = v.validate_wav("output.wav", text="hello world")
"""

from __future__ import annotations
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (tuned for 24kHz mono speech from Ana Neural / Kokoro)
# ---------------------------------------------------------------------------

# Energy gate
RMS_SILENCE_THRESHOLD   = 0.005   # below this = silence (0-1 normalised)
VOICED_FRAME_MIN_RATIO  = 0.25    # at least 25% of frames must be voiced

# Spectral speech check
SPEECH_CENTROID_MIN_HZ  = 200     # below this = rumble / DC / noise
SPEECH_CENTROID_MAX_HZ  = 4500    # above this = hiss / high-freq noise
FLATNESS_NOISE_THRESHOLD = 0.35   # above this = too flat = noise-like

# Duration sanity
MIN_DURATION_S = 0.25             # shorter than this = synthesis failure
MAX_DURATION_S = 30.0             # longer than this = runaway synthesis

# Minimum expected duration per character (very rough lower bound)
# "hi" (2 chars) → ~0.3s, "hello world" (11 chars) → ~0.8s
CHARS_PER_SECOND = 12.0           # average speaking rate


@dataclass
class ValidationResult:
    passed: bool
    reason: str
    rms: float = 0.0
    voiced_ratio: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_flatness: float = 0.0
    duration_s: float = 0.0

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"[{status}] {self.reason} | "
                f"rms={self.rms:.4f} voiced={self.voiced_ratio:.2f} "
                f"centroid={self.spectral_centroid_hz:.0f}Hz "
                f"flatness={self.spectral_flatness:.3f} "
                f"dur={self.duration_s:.2f}s")


class AudioValidator:
    """
    Validates audio waveforms for speech content.
    Pure numpy — no external dependencies beyond numpy.
    """

    def __init__(
        self,
        rms_threshold: float = RMS_SILENCE_THRESHOLD,
        voiced_ratio_min: float = VOICED_FRAME_MIN_RATIO,
        centroid_min_hz: float = SPEECH_CENTROID_MIN_HZ,
        centroid_max_hz: float = SPEECH_CENTROID_MAX_HZ,
        flatness_max: float = FLATNESS_NOISE_THRESHOLD,
        min_duration_s: float = MIN_DURATION_S,
        max_duration_s: float = MAX_DURATION_S,
    ):
        self.rms_threshold    = rms_threshold
        self.voiced_ratio_min = voiced_ratio_min
        self.centroid_min_hz  = centroid_min_hz
        self.centroid_max_hz  = centroid_max_hz
        self.flatness_max     = flatness_max
        self.min_duration_s   = min_duration_s
        self.max_duration_s   = max_duration_s

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate raw PCM int16 bytes.
        pcm_bytes: raw int16 little-endian bytes (as returned by JanusTTSv2.synthesize)
        """
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self._validate(audio, sample_rate, text)

    def validate_wav(
        self,
        wav_path: str,
        text: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a WAV file on disk."""
        try:
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels  = wf.getnchannels()
                sampwidth   = wf.getsampwidth()
                n_frames    = wf.getnframes()
                raw         = wf.readframes(n_frames)

            if sampwidth == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return ValidationResult(False, f"Unsupported sample width: {sampwidth}")

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            return self._validate(audio, sample_rate, text)

        except Exception as e:
            return ValidationResult(False, f"WAV read error: {e}")

    def validate_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a float32 numpy array (values in [-1, 1])."""
        return self._validate(audio.astype(np.float32), sample_rate, text)

    # ── Core validation logic ─────────────────────────────────────────────────

    def _validate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str],
    ) -> ValidationResult:

        duration_s = len(audio) / sample_rate

        # ── 1. Duration sanity ────────────────────────────────────────────────
        if duration_s < self.min_duration_s:
            return ValidationResult(
                False,
                f"Too short ({duration_s:.2f}s < {self.min_duration_s}s) — synthesis likely failed",
                duration_s=duration_s,
            )

        if duration_s > self.max_duration_s:
            return ValidationResult(
                False,
                f"Too long ({duration_s:.1f}s > {self.max_duration_s}s) — runaway synthesis",
                duration_s=duration_s,
            )

        # If we know the text, check duration is plausible for that many chars
        if text:
            min_expected = max(self.min_duration_s, len(text) / (CHARS_PER_SECOND * 3))
            if duration_s < min_expected:
                return ValidationResult(
                    False,
                    f"Duration {duration_s:.2f}s too short for {len(text)}-char text "
                    f"(expected ≥ {min_expected:.2f}s)",
                    duration_s=duration_s,
                )

        # ── 2. Energy / silence gate ──────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms < self.rms_threshold:
            return ValidationResult(
                False,
                f"Silent audio (RMS={rms:.5f} < threshold={self.rms_threshold})",
                rms=rms,
                duration_s=duration_s,
            )

        # Frame-level voiced ratio — how much of the clip has energy above floor
        frame_size = int(sample_rate * 0.02)   # 20ms frames
        if frame_size < 1:
            frame_size = 1
        n_frames = len(audio) // frame_size
        if n_frames > 0:
            frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            voiced_ratio = float(np.mean(frame_rms > self.rms_threshold * 0.5))
        else:
            voiced_ratio = 1.0

        if voiced_ratio < self.voiced_ratio_min:
            return ValidationResult(
                False,
                f"Too much silence ({voiced_ratio:.1%} voiced < {self.voiced_ratio_min:.0%} required)",
                rms=rms,
                voiced_ratio=voiced_ratio,
                duration_s=duration_s,
            )

        # ── 3. Spectral speech check ──────────────────────────────────────────
        centroid_hz, flatness = self._spectral_features(audio, sample_rate)

        if centroid_hz < self.centroid_min_hz:
            return ValidationResult(
                False,
                f"Spectral centroid too low ({centroid_hz:.0f}Hz < {self.centroid_min_hz}Hz) "
                f"— DC offset, rumble, or synthesis failure",
                rms=rms,
                voiced_ratio=voiced_ratio,
                spectral_centroid_hz=centroid_hz,
                spectral_flatness=flatness,
                duration_s=duration_s,
            )

        if centroid_hz > self.centroid_max_hz:
            return ValidationResult(
                False,
                f"Spectral centroid too high ({centroid_hz:.0f}Hz > {self.centroid_max_hz}Hz) "
                f"— hiss, aliasing, or high-freq noise",
                rms=rms,
                voiced_ratio=voiced_ratio,
                spectral_centroid_hz=centroid_hz,
                spectral_flatness=flatness,
                duration_s=duration_s,
            )

        if flatness > self.flatness_max:
            return ValidationResult(
                False,
                f"Spectral flatness too high ({flatness:.3f} > {self.flatness_max}) "
                f"— audio resembles noise rather than speech",
                rms=rms,
                voiced_ratio=voiced_ratio,
                spectral_centroid_hz=centroid_hz,
                spectral_flatness=flatness,
                duration_s=duration_s,
            )

        # ── All checks passed ─────────────────────────────────────────────────
        return ValidationResult(
            True,
            "OK",
            rms=rms,
            voiced_ratio=voiced_ratio,
            spectral_centroid_hz=centroid_hz,
            spectral_flatness=flatness,
            duration_s=duration_s,
        )

    # ── Signal processing helpers ─────────────────────────────────────────────

    def _spectral_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_fft: int = 2048,
    ) -> tuple[float, float]:
        """
        Compute spectral centroid and spectral flatness from the audio.

        Spectral centroid: weighted mean frequency — where the "centre of mass"
        of the spectrum is. Speech sits roughly 500-3000 Hz.

        Spectral flatness (Wiener entropy): ratio of geometric mean to
        arithmetic mean of the power spectrum.
          - Near 0 = tonal / speech-like
          - Near 1 = white noise / flat spectrum

        Uses a single FFT over a representative chunk (up to 2s) for speed.
        """
        # Use up to 2 seconds from the middle of the clip (avoids leading silence)
        max_samples = sample_rate * 2
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            chunk = audio[start:start + max_samples]
        else:
            chunk = audio

        # Apply Hann window
        window = np.hanning(len(chunk)).astype(np.float32)
        windowed = chunk * window

        # FFT magnitude spectrum
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2  # power spectrum
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Avoid log(0)
        spectrum = np.maximum(spectrum, 1e-10)

        # Spectral centroid
        total_power = spectrum.sum()
        if total_power < 1e-10:
            return 0.0, 1.0
        centroid_hz = float(np.sum(freqs * spectrum) / total_power)

        # Spectral flatness (geometric mean / arithmetic mean)
        log_mean = np.mean(np.log(spectrum))
        arith_mean = np.mean(spectrum)
        flatness = float(np.exp(log_mean) / (arith_mean + 1e-10))
        flatness = min(flatness, 1.0)

        return centroid_hz, flatness


# ---------------------------------------------------------------------------
# Batch audit utility
# ---------------------------------------------------------------------------

def audit_directory(
    audio_dir: str,
    sample_rate: int = 24000,
    pattern: str = "**/*.wav",
) -> dict:
    """
    Audit all WAV files in a directory. Returns summary stats and a list
    of rejected files with reasons.

    Usage:
        from audio_validator import audit_directory
        results = audit_directory("human_dataset/output/audio")
        print(f"Pass rate: {results['pass_rate']:.1%}")
        for r in results['rejected']:
            print(r['file'], r['reason'])
    """
    validator = AudioValidator()
    audio_path = Path(audio_dir)
    wav_files = list(audio_path.glob(pattern))

    passed = []
    rejected = []

    for wav_file in wav_files:
        result = validator.validate_wav(str(wav_file))
        entry = {
            "file": str(wav_file.relative_to(audio_path)),
            "passed": result.passed,
            "reason": result.reason,
            "rms": result.rms,
            "centroid_hz": result.spectral_centroid_hz,
            "flatness": result.spectral_flatness,
            "duration_s": result.duration_s,
        }
        if result.passed:
            passed.append(entry)
        else:
            rejected.append(entry)

    total = len(wav_files)
    return {
        "total": total,
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": len(passed) / total if total > 0 else 0.0,
        "rejected_files": rejected,
    }


# ---------------------------------------------------------------------------
# __main__ — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    validator = AudioValidator()

    if len(sys.argv) > 1:
        # Audit a directory or single file
        path = sys.argv[1]
        if Path(path).is_dir():
            print(f"Auditing directory: {path}")
            results = audit_directory(path)
            print(f"\nResults:")
            print(f"  Total files : {results['total']}")
            print(f"  Passed      : {results['passed']}")
            print(f"  Rejected    : {results['rejected']}")
            print(f"  Pass rate   : {results['pass_rate']:.1%}")
            if results['rejected_files']:
                print(f"\nRejected files:")
                for r in results['rejected_files'][:20]:
                    print(f"  {r['file']}: {r['reason']}")
        else:
            result = validator.validate_wav(path)
            print(result)
    else:
        # Self-test with synthetic signals
        sr = 24000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)

        print("=== AudioValidator self-test ===\n")

        # Test 1: silence
        silence = np.zeros(sr, dtype=np.float32)
        r = validator.validate_numpy(silence, sr, "hello world")
        print(f"Silence:          {r}")

        # Test 2: white noise
        noise = np.random.randn(sr).astype(np.float32) * 0.3
        r = validator.validate_numpy(noise, sr, "hello world")
        print(f"White noise:      {r}")

        # Test 3: pure 100Hz tone (too low — rumble)
        tone_low = (np.sin(2 * np.pi * 100 * t) * 0.5).astype(np.float32)
        r = validator.validate_numpy(tone_low, sr, "hello world")
        print(f"100Hz tone:       {r}")

        # Test 4: pure 8000Hz tone (too high — hiss)
        tone_high = (np.sin(2 * np.pi * 8000 * t) * 0.5).astype(np.float32)
        r = validator.validate_numpy(tone_high, sr, "hello world")
        print(f"8000Hz tone:      {r}")

        # Test 5: speech-like signal (mix of 200-3000Hz harmonics)
        speech_like = np.zeros(sr, dtype=np.float32)
        for f in [200, 400, 800, 1200, 1600, 2000, 2400, 2800]:
            speech_like += np.sin(2 * np.pi * f * t) * (1.0 / f * 500)
        speech_like = (speech_like / np.max(np.abs(speech_like)) * 0.7).astype(np.float32)
        r = validator.validate_numpy(speech_like, sr, "hello world")
        print(f"Speech-like mix:  {r}")

        # Test 6: too short
        short = speech_like[:int(sr * 0.1)]
        r = validator.validate_numpy(short, sr, "hello world")
        print(f"Too short (0.1s): {r}")

        print("\nSelf-test complete.")