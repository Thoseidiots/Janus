"""
janus_tts_v2.py - Janus Neural TTS Engine v2
Pure PyTorch + numpy + scipy. No external TTS libraries.
Architecture: TextEncoder → DurationPredictor → LengthRegulator → Decoder → iSTFTNet Vocoder
"""

import re
import math
import wave
import struct
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0.0
FMAX = 8000.0

# ---------------------------------------------------------------------------
# 1. Phoneme vocabulary (64 symbols)
# ---------------------------------------------------------------------------
PHONEME_SYMBOLS = [
    "<pad>", "<sos>", "<eos>", "<sp>",          # 0-3  special
    "AA", "AE", "AH", "AO", "AW", "AY",         # 4-9  vowels
    "B", "CH", "D", "DH", "EH", "ER", "EY",     # 10-16
    "F", "G", "HH", "IH", "IY", "JH", "K",      # 17-23
    "L", "M", "N", "NG", "OW", "OY", "P",       # 24-30
    "R", "S", "SH", "T", "TH", "UH", "UW",      # 31-37
    "V", "W", "Y", "Z", "ZH",                   # 38-42
    "AX", "IX", "UX", "EL", "EM", "EN",         # 43-48
    "Q", "DX", "NX", "WH", "X",                 # 49-53
    "sp", "sil", "pau",                          # 54-56
    "<r1>", "<r2>", "<r3>", "<r4>", "<r5>",     # 57-61 reserved
    "<r6>", "<r7>",                              # 62-63
]
assert len(PHONEME_SYMBOLS) == 64, f"Expected 64 symbols, got {len(PHONEME_SYMBOLS)}"
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_SYMBOLS)}
VOCAB_SIZE = 64

# ---------------------------------------------------------------------------
# CMU Dictionary (ARPAbet)
# NOTE: "janus" → ["Y","AE","N","AH","S"]  (YAH-nus, not JH)
# ---------------------------------------------------------------------------
CMU_DICT = {
    "a": ["AH"], "an": ["AH", "N"], "the": ["DH", "AH"],
    "hello": ["HH", "AH", "L", "OW"], "hi": ["HH", "AY"],
    "i": ["AY"], "am": ["AE", "M"], "is": ["IH", "Z"],
    "are": ["AA", "R"], "was": ["W", "AH", "Z"], "be": ["B", "IY"],
    "been": ["B", "IH", "N"], "have": ["HH", "AE", "V"],
    "has": ["HH", "AE", "Z"], "had": ["HH", "AE", "D"],
    "do": ["D", "UW"], "does": ["D", "AH", "Z"], "did": ["D", "IH", "D"],
    "will": ["W", "IH", "L"], "would": ["W", "UH", "D"],
    "can": ["K", "AE", "N"], "could": ["K", "UH", "D"],
    "may": ["M", "EY"], "might": ["M", "AY", "T"],
    "shall": ["SH", "AE", "L"], "should": ["SH", "UH", "D"],
    "not": ["N", "AA", "T"], "no": ["N", "OW"],
    "yes": ["Y", "EH", "S"], "ok": ["OW", "K", "EY"],
    "okay": ["OW", "K", "EY"], "please": ["P", "L", "IY", "Z"],
    "thank": ["TH", "AE", "NG", "K"], "thanks": ["TH", "AE", "NG", "K", "S"],
    "you": ["Y", "UW"], "your": ["Y", "AO", "R"],
    "my": ["M", "AY"], "me": ["M", "IY"],
    "we": ["W", "IY"], "our": ["AW", "R"],
    "he": ["HH", "IY"], "she": ["SH", "IY"],
    "it": ["IH", "T"], "its": ["IH", "T", "S"],
    "they": ["DH", "EY"], "them": ["DH", "EH", "M"],
    "this": ["DH", "IH", "S"], "that": ["DH", "AE", "T"],
    "these": ["DH", "IY", "Z"], "those": ["DH", "OW", "Z"],
    "here": ["HH", "IH", "R"], "there": ["DH", "EH", "R"],
    "where": ["W", "EH", "R"], "when": ["W", "EH", "N"],
    "what": ["W", "AH", "T"], "who": ["HH", "UW"],
    "how": ["HH", "AW"], "why": ["W", "AY"],
    "which": ["W", "IH", "CH"], "with": ["W", "IH", "DH"],
    "for": ["F", "AO", "R"], "from": ["F", "R", "AH", "M"],
    "to": ["T", "UW"], "too": ["T", "UW"], "two": ["T", "UW"],
    "of": ["AH", "V"], "in": ["IH", "N"], "on": ["AA", "N"],
    "at": ["AE", "T"], "by": ["B", "AY"], "up": ["AH", "P"],
    "out": ["AW", "T"], "if": ["IH", "F"], "so": ["S", "OW"],
    "as": ["AE", "Z"], "or": ["AO", "R"], "and": ["AE", "N", "D"],
    "but": ["B", "AH", "T"], "all": ["AO", "L"],
    "one": ["W", "AH", "N"], "three": ["TH", "R", "IY"],
    "four": ["F", "AO", "R"], "five": ["F", "AY", "V"],
    "six": ["S", "IH", "K", "S"], "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"], "nine": ["N", "AY", "N"], "ten": ["T", "EH", "N"],
    # KEY FIX: Janus says her own name as "YAH-nus"
    "janus": ["Y", "AE", "N", "AH", "S"],
    "help": ["HH", "EH", "L", "P"], "today": ["T", "AH", "D", "EY"],
    "good": ["G", "UH", "D"], "great": ["G", "R", "EY", "T"],
    "time": ["T", "AY", "M"], "day": ["D", "EY"],
    "now": ["N", "AW"], "just": ["JH", "AH", "S", "T"],
    "like": ["L", "AY", "K"], "know": ["N", "OW"],
    "think": ["TH", "IH", "NG", "K"], "see": ["S", "IY"],
    "come": ["K", "AH", "M"], "go": ["G", "OW"],
    "get": ["G", "EH", "T"], "make": ["M", "EY", "K"],
    "take": ["T", "EY", "K"], "give": ["G", "IH", "V"],
    "say": ["S", "EY"], "tell": ["T", "EH", "L"],
    "ask": ["AE", "S", "K"], "work": ["W", "ER", "K"],
    "call": ["K", "AO", "L"], "try": ["T", "R", "AY"],
    "need": ["N", "IY", "D"], "want": ["W", "AA", "N", "T"],
    "use": ["Y", "UW", "Z"], "find": ["F", "AY", "N", "D"],
    "feel": ["F", "IY", "L"], "look": ["L", "UH", "K"],
    "seem": ["S", "IY", "M"], "leave": ["L", "IY", "V"],
    "keep": ["K", "IY", "P"], "let": ["L", "EH", "T"],
    "begin": ["B", "IH", "G", "IH", "N"], "show": ["SH", "OW"],
    "hear": ["HH", "IH", "R"], "play": ["P", "L", "EY"],
    "run": ["R", "AH", "N"], "move": ["M", "UW", "V"],
    "live": ["L", "IH", "V"], "believe": ["B", "IH", "L", "IY", "V"],
    "hold": ["HH", "OW", "L", "D"], "bring": ["B", "R", "IH", "NG"],
    "happen": ["HH", "AE", "P", "AH", "N"],
    "write": ["R", "AY", "T"], "provide": ["P", "R", "AH", "V", "AY", "D"],
    "sit": ["S", "IH", "T"], "stand": ["S", "T", "AE", "N", "D"],
    "lose": ["L", "UW", "Z"], "pay": ["P", "EY"],
    "meet": ["M", "IY", "T"], "include": ["IH", "N", "K", "L", "UW", "D"],
    "continue": ["K", "AH", "N", "T", "IH", "N", "Y", "UW"],
    "set": ["S", "EH", "T"], "learn": ["L", "ER", "N"],
    "change": ["CH", "EY", "N", "JH"], "lead": ["L", "IY", "D"],
    "understand": ["AH", "N", "D", "ER", "S", "T", "AE", "N", "D"],
    "watch": ["W", "AA", "CH"], "follow": ["F", "AA", "L", "OW"],
    "stop": ["S", "T", "AA", "P"], "create": ["K", "R", "IY", "EY", "T"],
    "speak": ["S", "P", "IY", "K"], "read": ["R", "IY", "D"],
    "spend": ["S", "P", "EH", "N", "D"], "grow": ["G", "R", "OW"],
    "open": ["OW", "P", "AH", "N"], "walk": ["W", "AO", "K"],
    "win": ["W", "IH", "N"], "offer": ["AO", "F", "ER"],
    "remember": ["R", "IH", "M", "EH", "M", "B", "ER"],
    "love": ["L", "AH", "V"], "consider": ["K", "AH", "N", "S", "IH", "D", "ER"],
    "appear": ["AH", "P", "IH", "R"], "buy": ["B", "AY"],
    "wait": ["W", "EY", "T"], "serve": ["S", "ER", "V"],
    "die": ["D", "AY"], "send": ["S", "EH", "N", "D"],
    "expect": ["IH", "K", "S", "P", "EH", "K", "T"],
    "build": ["B", "IH", "L", "D"], "stay": ["S", "T", "EY"],
    "fall": ["F", "AO", "L"], "cut": ["K", "AH", "T"],
    "reach": ["R", "IY", "CH"], "kill": ["K", "IH", "L"],
    "remain": ["R", "IH", "M", "EY", "N"],
    "suggest": ["S", "AH", "G", "JH", "EH", "S", "T"],
    "raise": ["R", "EY", "Z"], "pass": ["P", "AE", "S"],
    "sell": ["S", "EH", "L"], "require": ["R", "IH", "K", "W", "AY", "R"],
    "report": ["R", "IH", "P", "AO", "R", "T"],
    "decide": ["D", "IH", "S", "AY", "D"],
    "pull": ["P", "UH", "L"], "break": ["B", "R", "EY", "K"],
    "voice": ["V", "OY", "S"], "speech": ["S", "P", "IY", "CH"],
    "sound": ["S", "AW", "N", "D"], "music": ["M", "Y", "UW", "Z", "IH", "K"],
    "word": ["W", "ER", "D"], "name": ["N", "EY", "M"],
    "place": ["P", "L", "EY", "S"], "world": ["W", "ER", "L", "D"],
    "people": ["P", "IY", "P", "AH", "L"],
    "life": ["L", "AY", "F"], "hand": ["HH", "AE", "N", "D"],
    "part": ["P", "AA", "R", "T"], "child": ["CH", "AY", "L", "D"],
    "eye": ["AY"], "woman": ["W", "UH", "M", "AH", "N"],
    "man": ["M", "AE", "N"], "year": ["Y", "IH", "R"],
    "way": ["W", "EY"], "thing": ["TH", "IH", "NG"],
    "case": ["K", "EY", "S"], "week": ["W", "IY", "K"],
    "company": ["K", "AH", "M", "P", "AH", "N", "IY"],
    "system": ["S", "IH", "S", "T", "AH", "M"],
    "program": ["P", "R", "OW", "G", "R", "AE", "M"],
    "question": ["K", "W", "EH", "S", "CH", "AH", "N"],
    "government": ["G", "AH", "V", "ER", "N", "M", "AH", "N", "T"],
    "number": ["N", "AH", "M", "B", "ER"],
    "night": ["N", "AY", "T"], "point": ["P", "OY", "N", "T"],
    "home": ["HH", "OW", "M"], "water": ["W", "AO", "T", "ER"],
    "room": ["R", "UW", "M"], "mother": ["M", "AH", "DH", "ER"],
    "area": ["EH", "R", "IY", "AH"], "money": ["M", "AH", "N", "IY"],
    "story": ["S", "T", "AO", "R", "IY"],
    "fact": ["F", "AE", "K", "T"], "month": ["M", "AH", "N", "TH"],
    "lot": ["L", "AA", "T"], "right": ["R", "AY", "T"],
    "study": ["S", "T", "AH", "D", "IY"],
    "book": ["B", "UH", "K"], "job": ["JH", "AA", "B"],
    "business": ["B", "IH", "Z", "N", "AH", "S"],
    "issue": ["IH", "SH", "UW"], "side": ["S", "AY", "D"],
    "kind": ["K", "AY", "N", "D"], "head": ["HH", "EH", "D"],
    "house": ["HH", "AW", "S"], "service": ["S", "ER", "V", "AH", "S"],
    "friend": ["F", "R", "EH", "N", "D"],
    "father": ["F", "AA", "DH", "ER"],
    "power": ["P", "AW", "ER"], "hour": ["AW", "ER"],
    "game": ["G", "EY", "M"], "line": ["L", "AY", "N"],
    "end": ["EH", "N", "D"], "among": ["AH", "M", "AH", "NG"],
    "never": ["N", "EH", "V", "ER"], "last": ["L", "AE", "S", "T"],
    "always": ["AO", "L", "W", "EY", "Z"],
    "sometimes": ["S", "AH", "M", "T", "AY", "M", "Z"],
    "together": ["T", "AH", "G", "EH", "DH", "ER"],
    "little": ["L", "IH", "T", "AH", "L"],
    "very": ["V", "EH", "R", "IY"],
    "still": ["S", "T", "IH", "L"], "own": ["OW", "N"],
    "while": ["W", "AY", "L"], "down": ["D", "AW", "N"],
    "each": ["IY", "CH"], "about": ["AH", "B", "AW", "T"],
    "after": ["AE", "F", "T", "ER"],
    "only": ["OW", "N", "L", "IY"], "over": ["OW", "V", "ER"],
    "also": ["AO", "L", "S", "OW"], "back": ["B", "AE", "K"],
    "well": ["W", "EH", "L"], "even": ["IY", "V", "AH", "N"],
    "new": ["N", "UW"], "because": ["B", "IH", "K", "AH", "Z"],
    "any": ["EH", "N", "IY"], "most": ["M", "OW", "S", "T"],
    "us": ["AH", "S"],
}

_LTS_MAP = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH",
    "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "o": "AO",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
    "u": "AH", "v": "V", "w": "W", "x": "K", "y": "Y",
    "z": "Z",
}
_LTS_DIGRAPHS = {
    "ch": "CH", "sh": "SH", "th": "TH", "ph": "F",
    "wh": "W", "ck": "K", "ng": "NG", "gh": "",
    "qu": "K W",
}


# ---------------------------------------------------------------------------
# Mel filterbank (pure numpy, no librosa)
# ---------------------------------------------------------------------------
def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _build_mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                           fmin=FMIN, fmax=FMAX) -> np.ndarray:
    """Returns mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fb[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fb[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fb

# Pre-compute filterbank once at module load
_MEL_FB = _build_mel_filterbank()


def _audio_to_mel_np(audio: np.ndarray, sr=SAMPLE_RATE) -> np.ndarray:
    """Compute log-mel spectrogram from float32 audio. Returns (T, N_MELS)."""
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    pad = N_FFT // 2
    audio_pad = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (len(audio_pad) - WIN_LENGTH) // HOP_LENGTH
    # Build frame matrix
    frames = np.lib.stride_tricks.as_strided(
        audio_pad,
        shape=(WIN_LENGTH, n_frames),
        strides=(audio_pad.strides[0], audio_pad.strides[0] * HOP_LENGTH)
    ).copy()
    frames *= window[:, np.newaxis]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=0)) ** 2  # (n_freqs, T)
    mel = _MEL_FB @ spec                                        # (N_MELS, T)
    log_mel = np.log(np.maximum(mel, 1e-8)).T                  # (T, N_MELS)
    return log_mel.astype(np.float32)


def _istft_np(magnitude: np.ndarray, phase: np.ndarray,
              n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH) -> np.ndarray:
    """Pure numpy iSTFT. magnitude/phase: (n_freqs, T). Returns float32 waveform."""
    n_freqs, n_frames = magnitude.shape
    window = np.hanning(win_length).astype(np.float32)
    # Reconstruct complex spectrogram
    complex_spec = magnitude * np.exp(1j * phase)
    # Expected output length
    expected_len = (n_frames - 1) * hop_length + win_length
    waveform = np.zeros(expected_len, dtype=np.float32)
    window_sum = np.zeros(expected_len, dtype=np.float32)

    for t in range(n_frames):
        frame = np.fft.irfft(complex_spec[:, t], n=n_fft)[:win_length]
        frame = frame * window
        start = t * hop_length
        end = start + win_length
        waveform[start:end] += frame
        window_sum[start:end] += window ** 2

    # Normalize by window overlap
    nonzero = window_sum > 1e-8
    waveform[nonzero] /= window_sum[nonzero]
    return waveform


def _save_wav(path: str, waveform: np.ndarray, sr: int = SAMPLE_RATE):
    """Save float32 waveform as 16-bit PCM WAV."""
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


# ---------------------------------------------------------------------------
# 1. Phonemizer
# ---------------------------------------------------------------------------
class Phonemizer:
    """Converts English text to ARPAbet phoneme sequences."""

    def __init__(self):
        self.dict = {k.lower(): v for k, v in CMU_DICT.items()}

    def text_to_phonemes(self, text: str) -> list:
        text = text.lower()
        text = re.sub(r"[^a-z\s'\-]", " ", text)
        words = text.split()
        phonemes = ["<sos>"]
        for word in words:
            word = word.strip("'-")
            if not word:
                continue
            if word in self.dict:
                phonemes.extend(self.dict[word])
            else:
                phonemes.extend(self._lts(word))
            phonemes.append("<sp>")
        if phonemes and phonemes[-1] == "<sp>":
            phonemes[-1] = "<eos>"
        else:
            phonemes.append("<eos>")
        return phonemes

    def _lts(self, word: str) -> list:
        """Letter-to-sound fallback for unknown words."""
        result = []
        i = 0
        while i < len(word):
            if i + 1 < len(word):
                dg = word[i:i + 2]
                if dg in _LTS_DIGRAPHS:
                    val = _LTS_DIGRAPHS[dg]
                    if val:
                        result.extend(val.split())
                    i += 2
                    continue
            ch = word[i]
            if ch in _LTS_MAP:
                result.append(_LTS_MAP[ch])
            i += 1
        return result if result else ["AH"]

    def phonemes_to_ids(self, phonemes: list) -> list:
        return [PHONEME_TO_ID.get(p, PHONEME_TO_ID["<pad>"]) for p in phonemes]


# ---------------------------------------------------------------------------
# 2. TextEncoder
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextEncoder(nn.Module):
    """
    Phoneme sequence → encoded features.
    Embedding → 2-layer bidirectional GRU.
    Fast on CPU. Upgrade to Transformer when GPU is available.
    Output: (batch, T_phoneme, 256)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = 256,
                 n_heads: int = 4, ffn_dim: int = 1024,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim // 2, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.dim = dim

    def forward(self, phoneme_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(phoneme_ids)
        x, _ = self.gru(x)
        return x


# ---------------------------------------------------------------------------
# 3. StyleEncoder
# ---------------------------------------------------------------------------
class StyleEncoder(nn.Module):
    """
    Reference mel spectrogram → 256-dim style vector.
    4-layer 1D conv stack + global average pooling.
    """

    def __init__(self, n_mels: int = N_MELS, style_dim: int = 256):
        super().__init__()
        channels = [n_mels, 128, 256, 256, 256]
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=5,
                          padding=2, stride=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
            ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(256, style_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (batch, T, n_mels)
        Returns: (batch, style_dim)
        """
        x = mel.transpose(1, 2)          # (batch, n_mels, T)
        x = self.convs(x)                # (batch, 512, T)
        x = x.mean(dim=-1)               # (batch, 512) global avg pool
        x = self.proj(x)
        # Normalize to unit sphere
        x = F.normalize(x, p=2, dim=-1)
        return x


# ---------------------------------------------------------------------------
# 4. DurationPredictor
# ---------------------------------------------------------------------------
class DurationPredictor(nn.Module):
    """Predicts duration per phoneme. 3-layer conv."""

    def __init__(self, dim: int = 256, kernel_size: int = 3, n_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Conv1d(dim, dim, kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(dim))
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, dim)
        Returns: (batch, T) durations
        """
        h = x.transpose(1, 2)  # (batch, dim, T)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h)
            h = h.transpose(1, 2)  # (batch, T, dim)
            h = norm(h)
            h = F.relu(h)
            h = h.transpose(1, 2)  # (batch, dim, T)
        h = h.transpose(1, 2)      # (batch, T, dim)
        durations = F.softplus(self.linear(h).squeeze(-1))  # (batch, T)
        return durations


# ---------------------------------------------------------------------------
# 5. LengthRegulator
# ---------------------------------------------------------------------------
def length_regulate(x: torch.Tensor, durations: torch.Tensor,
                    speed: float = 1.0) -> torch.Tensor:
    """
    Expand phoneme features to frame-level by repeating each phoneme
    by its predicted duration.
    x: (batch, T_phoneme, dim)
    durations: (batch, T_phoneme) — positive floats
    Returns: (batch, T_frames, dim)
    """
    batch_size, T_phon, dim = x.shape
    # Apply speed factor
    durations = durations / speed
    # Round to integer frame counts — clamp to [1, 15] per phoneme
    # Untrained model can predict huge values; 15 frames ≈ 160ms at 24kHz/256hop
    int_durs = torch.clamp(torch.round(durations).long(), min=1, max=15)  # (batch, T_phon)

    outputs = []
    for b in range(batch_size):
        frames = []
        for t in range(T_phon):
            n = int_durs[b, t].item()
            frames.append(x[b, t].unsqueeze(0).expand(n, -1))
        outputs.append(torch.cat(frames, dim=0))  # (T_frames_b, dim)

    # Pad to same length
    max_len = max(o.shape[0] for o in outputs)
    padded = torch.zeros(batch_size, max_len, dim, device=x.device, dtype=x.dtype)
    for b, o in enumerate(outputs):
        padded[b, :o.shape[0]] = o
    return padded


# ---------------------------------------------------------------------------
# 6. Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Frame-level features + style → mel.
    GRU-based for fast CPU training. Style vector is concatenated to each frame.
    """

    def __init__(self, dim: int = 256, n_heads: int = 4,
                 ffn_dim: int = 1024, n_layers: int = 4,
                 dropout: float = 0.1, n_mels: int = N_MELS):
        super().__init__()
        # Input: frame features (dim) + style (dim) concatenated
        self.gru = nn.GRU(dim * 2, dim, num_layers=2, batch_first=True,
                          dropout=dropout)
        self.proj = nn.Linear(dim, n_mels)
        self.dim = dim

    def forward(self, frame_features: torch.Tensor,
                style_vector: torch.Tensor) -> torch.Tensor:
        """
        frame_features: (batch, T_frames, dim)
        style_vector: (batch, dim)
        Returns: (batch, T_frames, n_mels)
        """
        T = frame_features.shape[1]
        # Broadcast style to every frame
        style_expanded = style_vector.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([frame_features, style_expanded], dim=-1)  # (batch, T, dim*2)
        out, _ = self.gru(x)                                      # (batch, T, dim)
        mel = self.proj(out)                                      # (batch, T, n_mels)
        return mel


# ---------------------------------------------------------------------------
# 7. iSTFTNet Vocoder
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """3 dilated convolutions with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size,
                              dilation=d, padding=pad),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size,
                              dilation=1, padding=kernel_size // 2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class iSTFTNetVocoder(nn.Module):
    """
    Mel spectrogram → waveform.
    4 upsampling stages (×8×8×4×2 = ×512) + residual dilated convs.
    Final conv → 1 channel (waveform directly, no iSTFT needed).
    Output: float32 waveform at SAMPLE_RATE Hz.
    """

    UPSAMPLE_RATES    = (8, 8, 4, 2)
    UPSAMPLE_CHANNELS = (512, 256, 128, 64)
    RESBLOCK_KERNEL   = 3
    RESBLOCK_DILATIONS = (1, 3, 5)

    def __init__(self, n_mels: int = N_MELS,
                 n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH):
        super().__init__()
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.win_length  = win_length
        self.n_freqs     = n_fft // 2 + 1

        self.input_conv = nn.Conv1d(n_mels, self.UPSAMPLE_CHANNELS[0],
                                    kernel_size=7, padding=3)

        self.ups        = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        in_ch = self.UPSAMPLE_CHANNELS[0]
        for rate, out_ch in zip(self.UPSAMPLE_RATES, self.UPSAMPLE_CHANNELS):
            self.ups.append(
                nn.ConvTranspose1d(in_ch, out_ch,
                                   kernel_size=rate * 2, stride=rate,
                                   padding=rate // 2)
            )
            self.res_blocks.append(
                ResidualBlock(out_ch, self.RESBLOCK_KERNEL, self.RESBLOCK_DILATIONS)
            )
            in_ch = out_ch

        # Output directly as waveform — no iSTFT, avoids double-upsampling bug
        self.final_conv = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (batch, T_frames, n_mels)
        Returns: (batch, T_samples) waveform
        """
        x = mel.transpose(1, 2)   # (batch, n_mels, T)
        x = self.input_conv(x)

        for up, res in zip(self.ups, self.res_blocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = res(x)

        x = F.leaky_relu(x, 0.1)
        x = self.final_conv(x)    # (batch, 1, T_samples)
        x = torch.tanh(x)
        return x.squeeze(1)       # (batch, T_samples)


# ---------------------------------------------------------------------------
# 8. JanusTTSv2 — main class
# ---------------------------------------------------------------------------
class JanusTTSv2:
    """
    Janus Neural TTS Engine v2.
    Pipeline: text → phonemes → TextEncoder → DurationPredictor →
              LengthRegulator → Decoder → iSTFTNetVocoder → PCM bytes
    """

    SAMPLE_RATE = SAMPLE_RATE
    N_MELS = N_MELS

    def __init__(self, weights_path: str = "janus_tts_v2_weights.pt"):
        self.phonemizer = Phonemizer()
        self.text_encoder = TextEncoder()
        self.style_encoder = StyleEncoder()
        self.duration_predictor = DurationPredictor()
        self.decoder = Decoder()
        self.vocoder = iSTFTNetVocoder()

        # Default style: random unit-sphere vector (256-dim, neutral voice)
        style = torch.randn(1, 256)
        self.style_vector = F.normalize(style, p=2, dim=-1)

        if pathlib.Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            print(f"[JanusTTSv2] No weights at '{weights_path}'. Using random init.")

        self._set_eval()

    def _set_eval(self):
        self.text_encoder.eval()
        self.style_encoder.eval()
        self.duration_predictor.eval()
        self.decoder.eval()
        self.vocoder.eval()

    def _set_train(self):
        self.text_encoder.train()
        self.duration_predictor.train()
        self.decoder.train()
        # StyleEncoder stays frozen during fine-tune
        self.style_encoder.eval()
        self.vocoder.eval()

    def _load_weights(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "text_encoder" in ckpt:
            self.text_encoder.load_state_dict(ckpt["text_encoder"])
        if "style_encoder" in ckpt:
            self.style_encoder.load_state_dict(ckpt["style_encoder"])
        if "duration_predictor" in ckpt:
            self.duration_predictor.load_state_dict(ckpt["duration_predictor"])
        if "decoder" in ckpt:
            self.decoder.load_state_dict(ckpt["decoder"])
        if "vocoder" in ckpt:
            self.vocoder.load_state_dict(ckpt["vocoder"])
        if "style_vector" in ckpt:
            self.style_vector = ckpt["style_vector"]
        print(f"[JanusTTSv2] Loaded weights from '{path}'.")

    def save_weights(self, path: str = "janus_tts_v2_weights.pt"):
        torch.save({
            "text_encoder": self.text_encoder.state_dict(),
            "style_encoder": self.style_encoder.state_dict(),
            "duration_predictor": self.duration_predictor.state_dict(),
            "decoder": self.decoder.state_dict(),
            "vocoder": self.vocoder.state_dict(),
            "style_vector": self.style_vector,
        }, path)
        print(f"[JanusTTSv2] Saved weights to '{path}'.")

    @torch.no_grad()
    def synthesize(self, text: str, speed: float = 1.0) -> bytes:
        """
        Synthesize text to raw PCM int16 bytes at SAMPLE_RATE Hz.
        Uses fixed duration (6 frames/phoneme) until duration predictor is
        properly trained on real speech data.
        """
        phonemes = self.phonemizer.text_to_phonemes(text)
        ids = self.phonemizer.phonemes_to_ids(phonemes)
        ids_tensor = torch.tensor([ids], dtype=torch.long)

        encoded = self.text_encoder(ids_tensor)              # (1, T, 256)

        # Fixed duration: 6 frames per phoneme ≈ 64ms at 24kHz/256hop
        # This gives natural-sounding speech without needing a trained predictor
        T = encoded.shape[1]
        durations = torch.full((1, T), 6.0)

        frame_feats = length_regulate(encoded, durations, speed=speed)
        mel = self.decoder(frame_feats, self.style_vector)   # (1, T_frames, 80)
        waveform = self.vocoder(mel)                         # (1, T_samples)
        wav_np = waveform[0].detach().cpu().numpy()

        wav_np = np.clip(wav_np, -1.0, 1.0)
        pcm16 = (wav_np * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def load_style_from_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load a WAV file, compute mel spectrogram, encode with StyleEncoder.
        Returns 512-dim style vector (1, 512).
        """
        import scipy.io.wavfile as wavfile

        sr, audio = wavfile.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        # Resample to SAMPLE_RATE if needed (linear interpolation)
        if sr != SAMPLE_RATE:
            n_target = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_target),
                np.arange(len(audio)), audio
            ).astype(np.float32)

        mel_np = _audio_to_mel_np(audio)                     # (T, 80)
        mel_t = torch.from_numpy(mel_np).unsqueeze(0)        # (1, T, 80)

        with torch.no_grad():
            style = self.style_encoder(mel_t)                # (1, 512)
        return style

    def set_style(self, style_vector: torch.Tensor):
        """Set the active style vector. Accepts (512,) or (1, 512)."""
        if style_vector.dim() == 1:
            style_vector = style_vector.unsqueeze(0)
        self.style_vector = F.normalize(style_vector, p=2, dim=-1)

    def train_on_sample(self, text: str, audio_path: str,
                        steps: int = 100, lr: float = 1e-4):
        """
        Fine-tune on a single (text, audio) pair.
        Updates: TextEncoder, DurationPredictor, Decoder.
        StyleEncoder is frozen — it just encodes the reference.
        """
        import scipy.io.wavfile as wavfile

        # Load and preprocess audio
        sr, audio = wavfile.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if sr != SAMPLE_RATE:
            n_target = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_target),
                np.arange(len(audio)), audio
            ).astype(np.float32)

        # Compute target mel and style
        target_mel_np = _audio_to_mel_np(audio)              # (T, 80)
        target_mel = torch.from_numpy(target_mel_np).unsqueeze(0)  # (1, T, 80)

        with torch.no_grad():
            style = self.style_encoder(target_mel)           # (1, 512)
        self.set_style(style)

        # Phonemize
        phonemes = self.phonemizer.text_to_phonemes(text)
        ids = self.phonemizer.phonemes_to_ids(phonemes)
        ids_tensor = torch.tensor([ids], dtype=torch.long)

        # Trainable params: TextEncoder + DurationPredictor + Decoder
        params = (
            list(self.text_encoder.parameters()) +
            list(self.duration_predictor.parameters()) +
            list(self.decoder.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        self._set_train()
        for step in range(steps):
            optimizer.zero_grad()
            encoded = self.text_encoder(ids_tensor)
            durations = self.duration_predictor(encoded)
            frame_feats = length_regulate(encoded, durations)
            mel_pred = self.decoder(frame_feats, self.style_vector)

            t = min(mel_pred.shape[1], target_mel.shape[1])
            mel_loss = F.mse_loss(mel_pred[:, :t, :], target_mel[:, :t, :])
            dur_loss = ((durations - 5.0) ** 2).mean() * 0.01
            loss = mel_loss + dur_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if (step + 1) % 20 == 0:
                print(f"  [fine-tune] step {step+1}/{steps} "
                      f"mel={mel_loss.item():.4f} dur={dur_loss.item():.4f}")

        self._set_eval()
        print(f"[JanusTTSv2] Fine-tuned on '{audio_path}'.")


# ---------------------------------------------------------------------------
# 9. JanusTTSv2Trainer
# ---------------------------------------------------------------------------
class JanusTTSv2Trainer:
    """
    Trains TextEncoder + DurationPredictor + Decoder on synthetic (phoneme, mel) pairs.
    StyleEncoder initialized with random style (fine-tuned later from real audio).
    Loss: MSE(predicted_mel, target_mel) + duration_loss
    """

    _SAMPLE_TEXTS = [
        "hello how are you today",
        "i am janus your voice assistant",
        "how can i help you",
        "the quick brown fox jumps over the lazy dog",
        "please tell me what you need",
        "i will do my best to help",
        "good morning have a great day",
        "thank you for using janus",
        "i am here to assist you",
        "what would you like to know",
        "let me think about that",
        "i understand what you mean",
        "that is a great question",
        "i can help you with that",
        "please wait a moment",
        "the answer is very simple",
        "i will find that for you",
        "you are welcome",
        "have a wonderful day",
        "see you next time",
    ]

    def __init__(self):
        self.phonemizer = Phonemizer()
        self.text_encoder = TextEncoder()
        self.style_encoder = StyleEncoder()
        self.duration_predictor = DurationPredictor()
        self.decoder = Decoder()
        self.vocoder = iSTFTNetVocoder()

        # Fixed random style vector for training (256-dim to match smaller model)
        style = torch.randn(1, 256)
        self.style_vector = F.normalize(style, p=2, dim=-1)

    def _make_reference_mel(self, n_frames: int = 100) -> np.ndarray:
        """Generate a smooth synthetic mel target (not pure noise)."""
        mel = np.random.randn(n_frames, N_MELS).astype(np.float32) * 0.5 - 3.0
        for i in range(1, n_frames):
            mel[i] = 0.85 * mel[i - 1] + 0.15 * mel[i]
        return mel

    def _generate_training_data(self, n: int = 2000):
        """Return list of (ids_tensor, mel_tensor, dur_tensor) pairs."""
        data = []
        texts = self._SAMPLE_TEXTS
        for i in range(n):
            text = texts[i % len(texts)]
            phonemes = self.phonemizer.text_to_phonemes(text)
            ids = self.phonemizer.phonemes_to_ids(phonemes)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            T_phon = len(ids)
            # Target durations: 4-8 frames per phoneme (realistic speech rate)
            dur_targets = torch.randint(4, 9, (T_phon,)).float()
            n_frames = int(dur_targets.sum().item())
            mel_np = self._make_reference_mel(n_frames)
            mel_tensor = torch.from_numpy(mel_np)
            data.append((ids_tensor, mel_tensor, dur_targets))
        return data

    def train(self, epochs: int = 10, lr: float = 1e-3, n_samples: int = 2000,
              batch_size: int = 8):
        """Train TextEncoder + DurationPredictor + Decoder on synthetic data."""
        print(f"[JanusTTSv2Trainer] Generating {n_samples} training samples...")
        data = self._generate_training_data(n=n_samples)

        params = (
            list(self.text_encoder.parameters()) +
            list(self.duration_predictor.parameters()) +
            list(self.decoder.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        self.text_encoder.train()
        self.duration_predictor.train()
        self.decoder.train()
        self.style_encoder.eval()

        for epoch in range(epochs):
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), batch_size):
                batch = [data[j] for j in indices[i:i + batch_size]]
                max_phon = max(ids.shape[0] for ids, _, _ in batch)
                max_mel  = max(mel.shape[0] for _, mel, _ in batch)

                ids_batch = torch.zeros(len(batch), max_phon, dtype=torch.long)
                mel_batch = torch.zeros(len(batch), max_mel, N_MELS)
                dur_batch = torch.zeros(len(batch), max_phon)
                for j, (ids, mel, dur) in enumerate(batch):
                    ids_batch[j, :ids.shape[0]] = ids
                    mel_batch[j, :mel.shape[0]] = mel
                    dur_batch[j, :dur.shape[0]] = dur

                pad_mask = (ids_batch == 0)

                optimizer.zero_grad()

                encoded   = self.text_encoder(ids_batch, src_key_padding_mask=pad_mask)
                durations = self.duration_predictor(encoded)

                style = self.style_vector.expand(len(batch), -1)

                frame_feats = length_regulate(encoded, durations)
                mel_pred    = self.decoder(frame_feats, style)

                t = min(mel_pred.shape[1], mel_batch.shape[1])
                mel_loss = F.mse_loss(mel_pred[:, :t, :], mel_batch[:, :t, :])
                # Supervise durations directly — teach realistic phoneme lengths
                dur_loss = F.mse_loss(durations, dur_batch) * 0.1
                loss = mel_loss + dur_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        self.text_encoder.eval()
        self.duration_predictor.eval()
        self.decoder.eval()
        print("[JanusTTSv2Trainer] Training complete.")

    def save(self, path: str = "janus_tts_v2_weights.pt"):
        torch.save({
            "text_encoder": self.text_encoder.state_dict(),
            "style_encoder": self.style_encoder.state_dict(),
            "duration_predictor": self.duration_predictor.state_dict(),
            "decoder": self.decoder.state_dict(),
            "vocoder": self.vocoder.state_dict(),
            "style_vector": self.style_vector,
        }, path)
        print(f"[JanusTTSv2Trainer] Saved weights to '{path}'.")


# ---------------------------------------------------------------------------
# 10. __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    weights_path = "janus_tts_v2_weights.pt"
    output_path = "janus_v2_test.wav"

    # Train if no weights exist
    if not pathlib.Path(weights_path).exists():
        print("[Janus TTS v2] No weights found. Training for 5 epochs...")
        trainer = JanusTTSv2Trainer()
        trainer.train(epochs=5, lr=1e-3, n_samples=500, batch_size=16)
        trainer.save(weights_path)
    else:
        print(f"[Janus TTS v2] Found existing weights at '{weights_path}'.")

    # Verify "janus" phonemization
    ph = Phonemizer()
    janus_phones = ph.text_to_phonemes("janus")
    print(f"[Janus TTS v2] Phonemes for 'janus': {janus_phones}")
    assert "Y" in janus_phones, "ERROR: 'janus' must start with Y (YAH-nus)"
    assert "JH" not in janus_phones, "ERROR: 'janus' must NOT use JH"

    # Initialize TTS
    print("[Janus TTS v2] Initializing TTS engine...")
    tts = JanusTTSv2(weights_path=weights_path)

    test_text = "Hello, I am Janus. How can I help you today?"
    print(f"[Janus TTS v2] Synthesizing: \"{test_text}\"")

    t0 = time.time()
    pcm_bytes = tts.synthesize(test_text, speed=1.0)
    elapsed = time.time() - t0

    # Write WAV
    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    duration_sec = len(pcm_array) / SAMPLE_RATE
    _save_wav(output_path, pcm_array, sr=SAMPLE_RATE)

    print(f"[Janus TTS v2] Audio saved to: {pathlib.Path(output_path).resolve()}")
    print(f"[Janus TTS v2] Audio duration: {duration_sec:.2f}s  "
          f"(synthesized in {elapsed:.2f}s)")
