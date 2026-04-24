"""
janus_rvc.py
============
Janus Voice Cloning — Retrieval-based Voice Conversion (RVC-style).

Clones Ana Neural's voice from the 280 WAV samples in voice_training_data/.

Pipeline:
  Input audio → F0 extraction → Content features → Voice conversion → Output

Usage:
    # Train on Ana's voice samples
    python janus_rvc.py --train

    # Convert any audio to Ana's voice
    python janus_rvc.py --convert input.wav output.wav

    # Test with synthesized text
    python janus_rvc.py --test "Hello, I am Janus."
"""

import argparse
import math
import pathlib
import wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE   = 24000
HOP_LENGTH    = 256
WIN_LENGTH    = 1024
N_FFT         = 1024
N_MELS        = 80
VOICE_DATA    = pathlib.Path("voice_training_data")
WEIGHTS_PATH  = pathlib.Path("janus_rvc_weights.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load WAV file as float32 numpy array, resampled to target_sr."""
    import scipy.io.wavfile as wf
    sr, audio = wf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        n = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, len(audio)-1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def save_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram. Returns (T, N_MELS)."""
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    pad = N_FFT // 2
    audio_pad = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (len(audio_pad) - WIN_LENGTH) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio_pad,
        shape=(WIN_LENGTH, n_frames),
        strides=(audio_pad.strides[0], audio_pad.strides[0] * HOP_LENGTH)
    ).copy()
    frames *= window[:, np.newaxis]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=0)) ** 2

    # Mel filterbank
    n_freqs = N_FFT // 2 + 1
    mel_min = 2595 * math.log10(1 + 0 / 700)
    mel_max = 2595 * math.log10(1 + 8000 / 700)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        for k in range(bins[m-1], bins[m]):
            if bins[m] != bins[m-1]:
                fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
        for k in range(bins[m], bins[m+1]):
            if bins[m+1] != bins[m]:
                fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])

    mel = fb @ spec
    return np.log(np.maximum(mel, 1e-8)).T.astype(np.float32)


def extract_f0(audio: np.ndarray, sr: int = SAMPLE_RATE,
               hop: int = HOP_LENGTH) -> np.ndarray:
    """
    Extract fundamental frequency (pitch) using autocorrelation.
    Returns F0 contour in Hz, shape (T,). 0 = unvoiced.
    """
    frame_len = 1024
    n_frames  = 1 + (len(audio) - frame_len) // hop
    f0 = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        frame = audio[i * hop: i * hop + frame_len]
        if len(frame) < frame_len:
            break
        # Autocorrelation
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        # Find first peak after minimum lag (f0 range: 80-800 Hz)
        min_lag = int(sr / 800)
        max_lag = int(sr / 80)
        if max_lag >= len(corr):
            continue
        peak = np.argmax(corr[min_lag:max_lag]) + min_lag
        if corr[peak] > 0.3 * corr[0] and corr[0] > 0:
            f0[i] = sr / peak

    return f0


# ─────────────────────────────────────────────────────────────────────────────
# Voice Encoder — learns Ana's voice embedding from mel spectrograms
# ─────────────────────────────────────────────────────────────────────────────

class VoiceEncoder(nn.Module):
    """
    Encodes a mel spectrogram into a speaker embedding.
    Trained on Ana's voice samples to capture her timbre.
    """
    def __init__(self, n_mels: int = N_MELS, embed_dim: int = 256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2),    nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=2),    nn.ReLU(),
        )
        self.gru  = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, n_mels) → embedding: (B, embed_dim)"""
        x = mel.transpose(1, 2)          # (B, n_mels, T)
        x = self.convs(x).transpose(1, 2)  # (B, T, 256)
        _, h = self.gru(x)               # h: (2, B, 256)
        h = torch.cat([h[0], h[1]], dim=-1)  # (B, 512)
        return F.normalize(self.proj(h), p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Voice Converter — converts input mel to target voice
# ─────────────────────────────────────────────────────────────────────────────

class VoiceConverter(nn.Module):
    """
    Converts input mel spectrogram to target voice.
    Conditions on: speaker embedding + F0 contour.
    """
    def __init__(self, n_mels: int = N_MELS, embed_dim: int = 256):
        super().__init__()
        # Input: mel (n_mels) + F0 (1) + speaker embed (embed_dim)
        in_dim = n_mels + 1 + embed_dim
        self.gru = nn.GRU(in_dim, 512, num_layers=2, batch_first=True,
                          dropout=0.1, bidirectional=True)
        self.proj = nn.Linear(1024, n_mels)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor,
                speaker_embed: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, T, n_mels)
        f0:  (B, T, 1)  — normalized F0
        speaker_embed: (B, embed_dim)
        Returns: converted mel (B, T, n_mels)
        """
        T = mel.shape[1]
        spk = speaker_embed.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)
        x = torch.cat([mel, f0, spk], dim=-1)               # (B, T, in_dim)
        out, _ = self.gru(x)                                 # (B, T, 1024)
        return self.proj(out)                                # (B, T, n_mels)


# ─────────────────────────────────────────────────────────────────────────────
# Vocoder — mel → waveform (reuses janus_tts_v2 vocoder)
# ─────────────────────────────────────────────────────────────────────────────

class RVCVocoder(nn.Module):
    """Lightweight HiFi-GAN style vocoder: mel → waveform."""

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.input_conv = nn.Conv1d(n_mels, 256, 7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64,  16, stride=8, padding=4),
            nn.ConvTranspose1d(64,  32,  8,  stride=4, padding=2),
            nn.ConvTranspose1d(32,  16,  4,  stride=2, padding=1),
        ])
        self.res = nn.ModuleList([
            nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.LeakyReLU(0.1),
                          nn.Conv1d(128, 128, 3, padding=1)),
            nn.Sequential(nn.Conv1d(64,  64,  3, padding=1), nn.LeakyReLU(0.1),
                          nn.Conv1d(64,  64,  3, padding=1)),
            nn.Sequential(nn.Conv1d(32,  32,  3, padding=1), nn.LeakyReLU(0.1),
                          nn.Conv1d(32,  32,  3, padding=1)),
            nn.Sequential(nn.Conv1d(16,  16,  3, padding=1), nn.LeakyReLU(0.1),
                          nn.Conv1d(16,  16,  3, padding=1)),
        ])
        self.final = nn.Conv1d(16, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, n_mels) → waveform: (B, T_samples)"""
        x = mel.transpose(1, 2)
        x = self.input_conv(x)
        for up, res in zip(self.ups, self.res):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = x + res(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.final(x))
        return x.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# JanusRVC — main class
# ─────────────────────────────────────────────────────────────────────────────

class JanusRVC:
    """
    Voice cloning system trained on Ana Neural's voice.
    Converts any input audio to sound like Ana.
    """

    def __init__(self, weights_path: str = str(WEIGHTS_PATH)):
        self.encoder   = VoiceEncoder()
        self.converter = VoiceConverter()
        self.vocoder   = RVCVocoder()
        self.target_embed = None  # Ana's voice embedding

        if pathlib.Path(weights_path).exists():
            self._load(weights_path)
        else:
            print(f"[RVC] No weights at '{weights_path}'. Train first.")

    def _load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.converter.load_state_dict(ckpt["converter"])
        self.vocoder.load_state_dict(ckpt["vocoder"])
        self.target_embed = ckpt.get("target_embed")
        print(f"[RVC] Loaded weights from '{path}'.")

    def save(self, path: str = str(WEIGHTS_PATH)):
        torch.save({
            "encoder":      self.encoder.state_dict(),
            "converter":    self.converter.state_dict(),
            "vocoder":      self.vocoder.state_dict(),
            "target_embed": self.target_embed,
        }, path)
        print(f"[RVC] Saved to '{path}'.")

    def train(self, voice_data_dir: str = str(VOICE_DATA),
              epochs: int = 10, lr: float = 1e-3, batch_size: int = 4):
        """Train on Ana Neural's voice samples."""
        wav_files = sorted(pathlib.Path(voice_data_dir).glob("*.wav"))
        if not wav_files:
            print(f"[RVC] No WAV files found in {voice_data_dir}")
            return

        print(f"[RVC] Training on {len(wav_files)} samples...")

        # Load all mels
        mels = []
        for wf in wav_files:
            try:
                audio = load_wav(str(wf))
                mel   = compute_mel(audio)
                if len(mel) > 10:
                    mels.append(mel)
            except Exception as e:
                print(f"  Skip {wf.name}: {e}")

        print(f"[RVC] Loaded {len(mels)} valid samples")

        params = (list(self.encoder.parameters()) +
                  list(self.converter.parameters()) +
                  list(self.vocoder.parameters()))
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        self.encoder.train()
        self.converter.train()
        self.vocoder.train()

        for epoch in range(epochs):
            import random
            random.shuffle(mels)
            total_loss = 0.0
            n_batches  = 0

            for i in range(0, len(mels), batch_size):
                batch_mels = mels[i:i + batch_size]
                # Pad to same length
                max_t = max(m.shape[0] for m in batch_mels)
                padded = np.zeros((len(batch_mels), max_t, N_MELS), dtype=np.float32)
                for j, m in enumerate(batch_mels):
                    padded[j, :m.shape[0]] = m

                mel_t = torch.from_numpy(padded)  # (B, T, 80)

                # Extract F0 for each sample
                f0_batch = np.zeros((len(batch_mels), max_t, 1), dtype=np.float32)
                for j, wf in enumerate(wav_files[i:i + batch_size]):
                    try:
                        audio = load_wav(str(wf))
                        f0    = extract_f0(audio)
                        t     = min(len(f0), max_t)
                        # Normalize F0 to [0, 1]
                        f0_norm = f0[:t] / 800.0
                        f0_batch[j, :t, 0] = f0_norm
                    except Exception:
                        pass
                f0_t = torch.from_numpy(f0_batch)

                opt.zero_grad()

                # Encode speaker
                embed = self.encoder(mel_t)  # (B, 256)

                # Convert (self-reconstruction: same voice in, same voice out)
                mel_pred = self.converter(mel_t, f0_t, embed)  # (B, T, 80)

                # Reconstruction loss
                recon_loss = F.mse_loss(mel_pred, mel_t)

                # Vocoder loss — reconstruct waveform from predicted mel
                wav_pred = self.vocoder(mel_pred)  # (B, T_samples)

                # Load target waveforms for vocoder supervision
                wav_targets = []
                for wf in wav_files[i:i + batch_size]:
                    try:
                        audio = load_wav(str(wf))
                        wav_targets.append(torch.from_numpy(audio))
                    except Exception:
                        wav_targets.append(torch.zeros(1))

                max_wav = max(w.shape[0] for w in wav_targets)
                wav_t = torch.zeros(len(wav_targets), max_wav)
                for j, w in enumerate(wav_targets):
                    wav_t[j, :w.shape[0]] = w

                tw = min(wav_pred.shape[1], wav_t.shape[1])
                voc_loss = F.l1_loss(wav_pred[:, :tw], wav_t[:, :tw])

                loss = recon_loss + voc_loss * 0.1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                total_loss += loss.item()
                n_batches  += 1

            sched.step()
            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.6f}")

        # Compute Ana's average voice embedding
        self.encoder.eval()
        embeds = []
        with torch.no_grad():
            for mel in mels[:50]:  # use first 50 for embedding
                mel_t = torch.from_numpy(mel).unsqueeze(0)
                embeds.append(self.encoder(mel_t))
        self.target_embed = torch.stack(embeds).mean(0)
        print(f"[RVC] Ana's voice embedding computed.")

        self.encoder.eval()
        self.converter.eval()
        self.vocoder.eval()

    def convert(self, input_wav: str, output_wav: str):
        """Convert input audio to Ana's voice."""
        if self.target_embed is None:
            print("[RVC] No target embedding. Train first.")
            return

        audio = load_wav(input_wav)
        mel   = compute_mel(audio)
        f0    = extract_f0(audio)

        mel_t = torch.from_numpy(mel).unsqueeze(0)
        t     = min(len(f0), mel.shape[0])
        f0_norm = np.zeros((1, mel.shape[0], 1), dtype=np.float32)
        f0_norm[0, :t, 0] = f0[:t] / 800.0
        f0_t = torch.from_numpy(f0_norm)

        with torch.no_grad():
            mel_conv = self.converter(mel_t, f0_t, self.target_embed)
            wav_out  = self.vocoder(mel_conv)

        audio_out = wav_out[0].numpy()
        peak = np.max(np.abs(audio_out))
        if peak > 1e-6:
            audio_out = audio_out / peak * 0.92
        save_wav(output_wav, audio_out)
        print(f"[RVC] Converted → {output_wav}")

    def test(self, text: str, output_wav: str = "rvc_test.wav"):
        """Synthesize text with Ana's voice via Kokoro → convert."""
        import tempfile, os
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            # Synthesize with Kokoro
            from kokoro import KPipeline
            import soundfile as sf
            pipeline = KPipeline(lang_code="a")
            chunks = []
            for _, _, audio in pipeline(text, voice="af_heart"):
                chunks.append(audio)
            if chunks:
                audio = np.concatenate(chunks)
                sf.write(tmp, audio, 24000)
                self.convert(tmp, output_wav)
        except Exception as e:
            print(f"[RVC] Test failed: {e}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janus RVC Voice Cloning")
    parser.add_argument("--train",   action="store_true", help="Train on Ana's voice samples")
    parser.add_argument("--convert", nargs=2, metavar=("INPUT", "OUTPUT"), help="Convert audio")
    parser.add_argument("--test",    type=str, help="Test with text")
    parser.add_argument("--epochs",  type=int, default=10)
    args = parser.parse_args()

    rvc = JanusRVC()

    if args.train:
        rvc.train(epochs=args.epochs)
        rvc.save()

    elif args.convert:
        rvc.convert(args.convert[0], args.convert[1])

    elif args.test:
        rvc.test(args.test)
        import subprocess
        subprocess.Popen(["powershell", "-Command",
            "(New-Object Media.SoundPlayer 'rvc_test.wav').PlaySync()"])

    else:
        parser.print_help()
