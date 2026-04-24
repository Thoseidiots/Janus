"""
janus_tts.py - Janus Custom Text-to-Speech Engine
Voice character: slightly higher pitch, soft attack, smooth formants
Targeting a young Japanese woman speaking English.
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
# 1. Phonemizer
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
    "janus": ["JH", "AE", "N", "AH", "S"],
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

# Phoneme symbol set (ARPAbet subset + special tokens)
PHONEME_SYMBOLS = [
    "<pad>", "<sos>", "<eos>", "<sil>",
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
    "AX", "IX", "UX", "EL", "EM", "EN",
    "Q", "DX", "NX", "WH", "X",
    "sp", "sil", "pau",
]
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_SYMBOLS)}
VOCAB_SIZE = len(PHONEME_SYMBOLS)

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


class Phonemizer:
    """Converts English text to ARPAbet phoneme sequences."""

    def __init__(self):
        self.dict = {k.lower(): v for k, v in CMU_DICT.items()}

    def text_to_phonemes(self, text: str) -> list:
        text = text.lower()
        text = re.sub(r"[^a-z\s\'\-]", " ", text)
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
            phonemes.append("sp")
        if phonemes and phonemes[-1] == "sp":
            phonemes[-1] = "<eos>"
        else:
            phonemes.append("<eos>")
        return phonemes

    def _lts(self, word: str) -> list:
        result = []
        i = 0
        while i < len(word):
            if i + 1 < len(word):
                dg = word[i:i+2]
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
# 2. MelGenerator
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, dim)
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(self.conv2(h))
        h = self.norm2(h.transpose(1, 2))
        durations = F.softplus(self.proj(h)).squeeze(-1)
        return durations


def length_regulate(x, durations, speed=1.0):
    """Expand phoneme embeddings by predicted durations."""
    B, T, dim = x.shape
    durations = (durations / speed).round().long().clamp(min=1)
    outputs = []
    for b in range(B):
        frames = []
        for t in range(T):
            d = durations[b, t].item()
            frames.append(x[b, t:t+1].expand(d, -1))
        outputs.append(torch.cat(frames, dim=0))
    max_len = max(o.shape[0] for o in outputs)
    padded = torch.zeros(B, max_len, dim, device=x.device)
    for b, o in enumerate(outputs):
        padded[b, :o.shape[0]] = o
    return padded


class MelGenerator(nn.Module):
    """Transformer acoustic model: phoneme IDs -> mel spectrogram (80 bins)."""

    def __init__(self, vocab_size=VOCAB_SIZE, dim=256, n_heads=4,
                 n_layers=4, mel_bins=80, max_len=512):
        super().__init__()
        self.dim = dim
        self.mel_bins = mel_bins
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_enc = self._build_pos_enc(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.duration_predictor = DurationPredictor(dim)
        self.mel_proj = nn.Linear(dim, mel_bins)
        # Feminine voice pitch bias (slight boost on upper mel bins)
        self.pitch_bias = nn.Parameter(torch.zeros(mel_bins))

    def _build_pos_enc(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, phoneme_ids, speed=1.0):
        B, T = phoneme_ids.shape
        pad_mask = (phoneme_ids == 0)
        x = self.embedding(phoneme_ids)
        x = x + self.pos_enc[:, :T, :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        durations = self.duration_predictor(x)
        x_expanded = length_regulate(x, durations, speed=speed)
        mel = self.mel_proj(x_expanded)
        mel = mel + self.pitch_bias.unsqueeze(0).unsqueeze(0)
        return mel, durations


# ---------------------------------------------------------------------------
# 3. HiFiVocoder (Griffin-Lim)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0


def _mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                    fmin=FMIN, fmax=FMAX):
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs))
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
    return fb.astype(np.float32)


_MEL_FB = _mel_filterbank()
_MEL_FB_PINV = np.linalg.pinv(_MEL_FB)


class HiFiVocoder(nn.Module):
    """Griffin-Lim vocoder: mel (80 bins) -> audio waveform at 22050 Hz."""

    def __init__(self, n_mels=N_MELS, n_fft=N_FFT):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        n_freqs = n_fft // 2 + 1
        # _MEL_FB_PINV shape: (n_freqs, n_mels) — maps mel -> linear
        basis_init = torch.from_numpy(_MEL_FB_PINV.astype(np.float32))
        self.mel_to_linear = nn.Parameter(basis_init)

    def mel_to_linear_spec(self, mel_np):
        # mel_np: (T, n_mels), basis: (n_freqs, n_mels) -> linear: (T, n_freqs)
        mel_linear = np.exp(np.clip(mel_np, -10, 10))
        basis = self.mel_to_linear.detach().cpu().numpy()  # (n_freqs, n_mels)
        linear = mel_linear @ basis.T                       # (T, n_freqs)
        linear = np.maximum(linear, 1e-8)
        return linear

    def _stft(self, signal, n_fft, hop_length, win_length):
        window = np.hanning(win_length).astype(np.float32)
        pad = n_fft // 2
        signal = np.pad(signal, pad, mode="reflect")
        n_frames = 1 + (len(signal) - win_length) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(win_length, n_frames),
            strides=(signal.strides[0], signal.strides[0] * hop_length)
        ).copy()
        frames *= window[:, np.newaxis]
        return np.fft.rfft(frames, n=n_fft, axis=0)

    def _istft(self, complex_spec, hop_length, win_length, n_fft):
        window = np.hanning(win_length).astype(np.float32)
        frames = np.fft.irfft(complex_spec, n=n_fft, axis=0)[:win_length]
        n_frames = frames.shape[1]
        signal_len = win_length + hop_length * (n_frames - 1)
        signal = np.zeros(signal_len, dtype=np.float32)
        window_sum = np.zeros(signal_len, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            signal[start:start + win_length] += frames[:, i] * window
            window_sum[start:start + win_length] += window ** 2
        nonzero = window_sum > 1e-8
        signal[nonzero] /= window_sum[nonzero]
        pad = n_fft // 2
        signal = signal[pad:-pad] if len(signal) > 2 * pad else signal
        return signal

    def griffin_lim(self, magnitude, n_iter=30, hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH):
        n_fft = (magnitude.shape[0] - 1) * 2
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        complex_spec = magnitude * angles
        for _ in range(n_iter):
            signal = self._istft(complex_spec, hop_length, win_length, n_fft)
            stft_out = self._stft(signal, n_fft, hop_length, win_length)
            angles = np.exp(1j * np.angle(stft_out))
            t = min(angles.shape[1], magnitude.shape[1])
            complex_spec = magnitude[:, :t] * angles[:, :t]
        signal = self._istft(complex_spec, hop_length, win_length, n_fft)
        return signal.astype(np.float32)

    def synthesize(self, mel_np):
        """mel_np: (T, 80) -> waveform float32 at SAMPLE_RATE."""
        linear = self.mel_to_linear_spec(mel_np)
        magnitude = linear.T
        return self.griffin_lim(magnitude)


# ---------------------------------------------------------------------------
# 4. JanusTTS
# ---------------------------------------------------------------------------

def _apply_pitch_shift(waveform, pitch_factor=1.08, sr=SAMPLE_RATE):
    """
    Raise pitch via PSOLA-style resampling with overlap-add to reduce artifacts.
    pitch_factor > 1.0 = higher pitch. Kept subtle (1.08) to avoid roboticness.
    """
    if abs(pitch_factor - 1.0) < 0.01:
        return waveform

    # Resample to change pitch, then restore original length
    n_orig = len(waveform)
    n_resampled = int(n_orig / pitch_factor)
    if n_resampled < 2:
        return waveform

    # High-quality interpolation
    x_old = np.linspace(0, 1, n_orig)
    x_new = np.linspace(0, 1, n_resampled)
    shifted = np.interp(x_new, x_old, waveform).astype(np.float32)

    # Pad or trim back to original length
    if len(shifted) < n_orig:
        shifted = np.pad(shifted, (0, n_orig - len(shifted)))
    else:
        shifted = shifted[:n_orig]

    return shifted


def _apply_soft_attack(waveform, attack_ms=5, sr=SAMPLE_RATE):
    """Very short fade-in to remove click artifacts."""
    attack_samples = min(int(attack_ms * sr / 1000), len(waveform))
    envelope = np.ones(len(waveform), dtype=np.float32)
    envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
    return waveform * envelope


def _apply_smooth_formants(waveform):
    """
    Gentle warmth filter — boosts low-mids slightly, rolls off harshness.
    Much lighter than before to preserve naturalness.
    """
    # Simple 3-tap moving average — barely perceptible but takes the edge off
    b = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return lfilter(b, np.array([1.0], dtype=np.float32), waveform).astype(np.float32)


def _waveform_to_pcm16(waveform):
    waveform = np.clip(waveform, -1.0, 1.0)
    return (waveform * 32767).astype(np.int16).tobytes()


def _save_wav(path, waveform, sr=SAMPLE_RATE):
    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


class JanusTTS:
    """
    Janus TTS engine.
    Voice: young Japanese woman speaking English —
    slightly higher pitch, soft attack, smooth formants.
    """

    def __init__(self, model_path="janus_tts_weights.pt"):
        self.phonemizer = Phonemizer()
        self.mel_gen = MelGenerator()
        self.vocoder = HiFiVocoder()
        self.model_path = model_path

        if pathlib.Path(model_path).exists():
            self._load_weights(model_path)
        else:
            print(f"[JanusTTS] No weights at '{model_path}'. Using untrained model.")

        self.mel_gen.eval()
        self.vocoder.eval()

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "mel_gen" in checkpoint:
            self.mel_gen.load_state_dict(checkpoint["mel_gen"], strict=False)
        if "vocoder" in checkpoint:
            self.vocoder.load_state_dict(checkpoint["vocoder"], strict=False)
        print(f"[JanusTTS] Loaded weights from '{path}'.")

    def save_weights(self, path="janus_tts_weights.pt"):
        torch.save({
            "mel_gen": self.mel_gen.state_dict(),
            "vocoder": self.vocoder.state_dict(),
        }, path)
        print(f"[JanusTTS] Weights saved to '{path}'.")

    def synthesize(self, text: str, speed: float = 1.0, pitch: float = 1.0,
                   mood=None) -> bytes:
        """
        Convert text to raw PCM int16 bytes at 22050 Hz.

        Pipeline:
          1. Humanization layer — adds fillers, pauses, SSML prosody from mood
          2. Edge TTS Ana Neural — renders the SSML naturally
          3. SAPI fallback if Edge TTS unavailable
        """
        speech_text = self._humanize(text, mood)

        waveform = self._edge_synthesize(speech_text, speed)
        if waveform is None:
            waveform = self._sapi_synthesize(text, speed)
        if waveform is None or len(waveform) == 0:
            duration = max(1.0, len(text) * 0.06)
            waveform = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)

        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = waveform / peak * 0.92

        return _waveform_to_pcm16(waveform)

    def _humanize(self, text: str, mood=None) -> str:
        """
        Apply humanization at the text level only — no SSML.
        Edge TTS doesn't support SSML input; it reads tags as literal text.
        We use the humanization layer for fillers and natural pauses only.
        Emotional prosody is handled by Edge TTS's own neural rendering.
        """
        try:
            from janus_humanization_layer import NaturalSpeechGenerator

            class _Valence:
                def __init__(self, arousal=0.5, pleasure=0.5):
                    self.arousal  = arousal
                    self.pleasure = pleasure

            valence = _Valence(
                arousal  = getattr(mood, "arousal", 0.5) if mood else 0.5,
                pleasure = getattr(mood, "valence", 0.5) if mood else 0.5,
            )

            speech_gen = NaturalSpeechGenerator(filler_probability=0.12)
            humanized  = speech_gen.maybe_add_filler(text, valence)
            humanized  = speech_gen.add_natural_pauses(humanized)
            return humanized

        except Exception:
            return text

    def _edge_synthesize(self, text: str, speed: float = 1.0):
        """
        Primary synthesis — tries Kokoro (local, no internet) first,
        then falls back to Edge TTS (requires internet).
        Kokoro af_heart voice: warm, natural, human-sounding.
        """
        # Try Kokoro first — fully local, no internet needed
        kokoro_audio = self._kokoro_synthesize(text, speed)
        if kokoro_audio is not None:
            return kokoro_audio

        # Fall back to Edge TTS
        import asyncio, tempfile, os as _os

        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        async def _run():
            import edge_tts
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            await edge_tts.Communicate(text, "en-US-AnaNeural", rate=rate_str).save(tmp)
            return tmp

        try:
            loop = asyncio.new_event_loop()
            tmp_path = loop.run_until_complete(_run())
            loop.close()
            if not tmp_path or not _os.path.exists(tmp_path):
                return None
            audio = self._mp3_to_float32(tmp_path)
            _os.unlink(tmp_path)
            if audio is not None:
                print(f"[JanusTTS] Edge TTS (en-US-AnaNeural)")
            return audio
        except Exception as e:
            print(f"[JanusTTS] Edge TTS failed: {e}")
            return None

    def _kokoro_synthesize(self, text: str, speed: float = 1.0):
        """
        Synthesize using Kokoro (local neural TTS, af_heart voice).
        Returns float32 numpy array at SAMPLE_RATE, or None if unavailable.
        """
        try:
            import numpy as np
            from kokoro import KPipeline

            if not hasattr(self, '_kokoro_pipeline'):
                self._kokoro_pipeline = KPipeline(lang_code='a')
                print("[JanusTTS] Kokoro loaded (af_heart voice)")

            # Fix pronunciation before passing to Kokoro's phonemizer
            # "Janus" → "Yah-nus" (Kokoro reads J as JH, we want Y)
            kokoro_text = self._fix_pronunciations(text)

            generator = self._kokoro_pipeline(
                kokoro_text, voice='af_heart', speed=speed
            )
            chunks = []
            for _, _, audio in generator:
                chunks.append(audio)

            if not chunks:
                return None

            full = np.concatenate(chunks)
            # Kokoro outputs at 24000 Hz — resample to SAMPLE_RATE if different
            if 24000 != SAMPLE_RATE:
                n = int(len(full) * SAMPLE_RATE / 24000)
                full = np.interp(
                    np.linspace(0, len(full)-1, n),
                    np.arange(len(full)), full
                ).astype(np.float32)

            print(f"[JanusTTS] Kokoro (af_heart)")
            return full

        except ImportError:
            return None
        except Exception as e:
            print(f"[JanusTTS] Kokoro failed: {e}")
            return None

    def _fix_pronunciations(self, text: str) -> str:
        """
        Fix words Kokoro mispronounces before synthesis.
        Uses phonetic respellings that Kokoro's English phonemizer handles correctly.
        """
        import re
        fixes = {
            # "Janus" → sounds like "Yanus" — no hyphen, Kokoro reads hyphens as pauses
            r'\bJanus\b': 'Yanus',
            r'\bjanus\b': 'yanus',
            r'\bJANUS\b': 'YANUS',
        }
        result = text
        for pattern, replacement in fixes.items():
            result = re.sub(pattern, replacement, result)
        return result

    def _mp3_to_float32(self, mp3_path: str):
        """Decode MP3 → float32 numpy at SAMPLE_RATE. Tries pydub, soundfile, ffmpeg."""
        # pydub (needs ffmpeg or libav)
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(mp3_path)
            seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
            return np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass
        # soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(mp3_path)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != SAMPLE_RATE:
                n = int(len(audio) * SAMPLE_RATE / sr)
                audio = np.interp(np.linspace(0, len(audio)-1, n),
                                  np.arange(len(audio)), audio).astype(np.float32)
            return audio
        except Exception:
            pass
        # ffmpeg subprocess
        try:
            import subprocess, tempfile, os as _os
            # Find ffmpeg — check winget install location if not on PATH
            ffmpeg_candidates = [
                "ffmpeg",
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"),
            ]
            ffmpeg_exe = None
            for c in ffmpeg_candidates:
                try:
                    if subprocess.run([c, "-version"], capture_output=True).returncode == 0:
                        ffmpeg_exe = c
                        break
                except Exception:
                    continue
            if not ffmpeg_exe:
                return None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_tmp = f.name
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", mp3_path,
                 "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-f", "s16le", wav_tmp],
                capture_output=True, check=True
            )
            raw = open(wav_tmp, "rb").read()
            _os.unlink(wav_tmp)
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass
        return None

    def _sapi_synthesize(self, text: str, speed: float = 1.0):
        """
        Use pyttsx3 (Windows SAPI) to render speech to a temp WAV,
        then read it back as a float32 numpy array.
        """
        import tempfile, wave as _wave, os as _os
        try:
            import pyttsx3
            engine = pyttsx3.init()

            # Pick Zira (female) if available
            voices = engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "female" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

            # Rate: slightly slower than default for more natural delivery
            base_rate = 165
            engine.setProperty("rate", int(base_rate * speed))
            engine.setProperty("volume", 1.0)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()

            # Read the WAV back
            if not _os.path.exists(tmp_path) or _os.path.getsize(tmp_path) < 100:
                return None

            with _wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_ch = wf.getnchannels()
                raw = wf.readframes(n_frames)

            _os.unlink(tmp_path)

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_ch == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Resample to SAMPLE_RATE if needed
            if sr != SAMPLE_RATE:
                n_target = int(len(audio) * SAMPLE_RATE / sr)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_target),
                    np.arange(len(audio)), audio
                ).astype(np.float32)

            return audio

        except Exception as e:
            print(f"[JanusTTS] SAPI synthesis failed: {e}")
            return None

    def train_on_sample(self, text: str, audio_path: str,
                        lr: float = 1e-4, steps: int = 50):
        """Fine-tune on a single audio sample for voice cloning."""
        import scipy.io.wavfile as wavfile

        sr, audio = wavfile.read(audio_path)
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

        target_mel = self._audio_to_mel(audio)
        phonemes = self.phonemizer.text_to_phonemes(text)
        ids = self.phonemizer.phonemes_to_ids(phonemes)
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        target_tensor = torch.from_numpy(target_mel).unsqueeze(0)

        optimizer = torch.optim.Adam(self.mel_gen.parameters(), lr=lr)
        self.mel_gen.train()

        for step in range(steps):
            optimizer.zero_grad()
            mel_pred, _ = self.mel_gen(ids_tensor)
            t = min(mel_pred.shape[1], target_tensor.shape[1])
            loss = F.mse_loss(mel_pred[:, :t, :], target_tensor[:, :t, :])
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print(f"  [fine-tune] step {step+1}/{steps} loss={loss.item():.4f}")

        self.mel_gen.eval()
        print(f"[JanusTTS] Fine-tuned on '{audio_path}'.")

    def _audio_to_mel(self, audio):
        """Compute log mel spectrogram from float32 audio."""
        n_fft = N_FFT
        hop = HOP_LENGTH
        win = WIN_LENGTH
        window = np.hanning(win).astype(np.float32)
        pad = n_fft // 2
        audio_pad = np.pad(audio, pad, mode="reflect")
        n_frames = 1 + (len(audio_pad) - win) // hop
        frames = np.lib.stride_tricks.as_strided(
            audio_pad,
            shape=(win, n_frames),
            strides=(audio_pad.strides[0], audio_pad.strides[0] * hop)
        ).copy()
        frames *= window[:, np.newaxis]
        spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0)) ** 2
        mel = _MEL_FB @ spec
        mel = np.log(np.maximum(mel, 1e-8)).T
        return mel.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. JanusVoiceTrainer
# ---------------------------------------------------------------------------

class JanusVoiceTrainer:
    """Trains MelGenerator on synthetic (text, mel) pairs."""

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
        self.mel_gen = MelGenerator()
        self.vocoder = HiFiVocoder()

    def _make_reference_mel(self, n_frames=100):
        """Generate a smooth synthetic mel target."""
        mel = np.random.randn(n_frames, N_MELS).astype(np.float32) * 0.5 - 3.0
        for i in range(1, n_frames):
            mel[i] = 0.85 * mel[i - 1] + 0.15 * mel[i]
        return mel

    def generate_training_data(self, n=1000):
        """Return list of (ids_tensor, mel_tensor) pairs."""
        data = []
        texts = self._SAMPLE_TEXTS
        for i in range(n):
            text = texts[i % len(texts)]
            phonemes = self.phonemizer.text_to_phonemes(text)
            ids = self.phonemizer.phonemes_to_ids(phonemes)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            n_frames = max(50, len(ids) * 5)
            mel_np = self._make_reference_mel(n_frames)
            mel_tensor = torch.from_numpy(mel_np)
            data.append((ids_tensor, mel_tensor))
        return data

    def train(self, epochs=10, lr=1e-3, n_samples=1000, batch_size=8):
        """Train MelGenerator on synthetic data."""
        print(f"[JanusVoiceTrainer] Generating {n_samples} training samples...")
        data = self.generate_training_data(n=n_samples)

        optimizer = torch.optim.Adam(self.mel_gen.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        self.mel_gen.train()
        for epoch in range(epochs):
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), batch_size):
                batch = [data[j] for j in indices[i:i + batch_size]]
                max_phon = max(ids.shape[0] for ids, _ in batch)
                max_mel = max(mel.shape[0] for _, mel in batch)

                ids_batch = torch.zeros(len(batch), max_phon, dtype=torch.long)
                mel_batch = torch.zeros(len(batch), max_mel, N_MELS)

                for j, (ids, mel) in enumerate(batch):
                    ids_batch[j, :ids.shape[0]] = ids
                    mel_batch[j, :mel.shape[0]] = mel

                optimizer.zero_grad()
                mel_pred, durations = self.mel_gen(ids_batch)
                t = min(mel_pred.shape[1], mel_batch.shape[1])
                loss = F.mse_loss(mel_pred[:, :t, :], mel_batch[:, :t, :])
                dur_loss = ((durations - 5.0) ** 2).mean() * 0.01
                total = loss + dur_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.mel_gen.parameters(), 1.0)
                optimizer.step()
                total_loss += total.item()
                n_batches += 1

            scheduler.step()
            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs} loss={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        self.mel_gen.eval()
        print("[JanusVoiceTrainer] Training complete.")

    def save(self, path="janus_tts_weights.pt"):
        torch.save({
            "mel_gen": self.mel_gen.state_dict(),
            "vocoder": self.vocoder.state_dict(),
        }, path)
        print(f"[JanusVoiceTrainer] Saved weights to '{path}'.")


# ---------------------------------------------------------------------------
# 6. __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    weights_path = "janus_tts_weights.pt"
    output_path = "janus_voice_test.wav"

    if not pathlib.Path(weights_path).exists():
        print("[Janus TTS] No weights found. Training for 5 epochs...")
        trainer = JanusVoiceTrainer()
        trainer.train(epochs=5, lr=1e-3, n_samples=500, batch_size=8)
        trainer.save(weights_path)
    else:
        print(f"[Janus TTS] Found existing weights at '{weights_path}'.")

    print("[Janus TTS] Initializing TTS engine...")
    tts = JanusTTS(model_path=weights_path)

    test_text = "Hello, I am Janus. How can I help you today?"
    print(f"[Janus TTS] Synthesizing: \"{test_text}\"")

    pcm_bytes = tts.synthesize(test_text, speed=1.0, pitch=1.0)

    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    _save_wav(output_path, pcm_array, sr=SAMPLE_RATE)

    print(f"[Janus TTS] Audio saved to: {pathlib.Path(output_path).resolve()}")
