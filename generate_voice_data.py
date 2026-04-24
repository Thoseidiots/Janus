"""
generate_voice_data.py
======================
Generate training audio from Ana Neural (Edge TTS) to fine-tune janus_tts_v2.

Produces ~50 WAV files covering:
- Common phrases Janus will actually say
- Varied sentence lengths and structures
- Emotional range (questions, statements, exclamations)

Then fine-tunes janus_tts_v2 on all of them.
"""

import asyncio
import os
import subprocess
import wave
import numpy as np
import pathlib
import time

FFMPEG = None
for candidate in [
    "ffmpeg",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"),
]:
    try:
        if subprocess.run([candidate, "-version"], capture_output=True).returncode == 0:
            FFMPEG = candidate
            break
    except Exception:
        continue

SAMPLE_RATE = 24000
VOICE = "en-US-AnaNeural"
OUTPUT_DIR = pathlib.Path("voice_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Training sentences — things Janus will actually say
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = [
    # Identity
    "Hello, I am Janus.",
    "My name is Janus.",
    "I am Janus, your assistant.",
    "Hi there, I am Janus.",
    "Hello, how can I help you today?",

    # Responses
    "I understand.",
    "Got it.",
    "Of course.",
    "Sure, I can help with that.",
    "Let me think about that.",
    "That is a great question.",
    "I will do my best.",
    "I am not sure about that.",
    "I do not know, but I can find out.",
    "You are welcome.",
    "Thank you for asking.",

    # Questions
    "What would you like to know?",
    "How can I assist you?",
    "Is there anything else you need?",
    "What are you working on?",
    "Can you tell me more?",
    "What do you mean by that?",

    # Statements
    "The answer is simple.",
    "This is interesting.",
    "I found something useful.",
    "Let me explain.",
    "Here is what I know.",
    "That makes sense.",
    "I agree with you.",
    "I see what you mean.",

    # Technical
    "I am running the code now.",
    "The training is complete.",
    "I saved the file.",
    "The process finished successfully.",
    "There was an error. Let me fix it.",
    "I will check the logs.",

    # Emotional range
    "That is wonderful news!",
    "I am happy to help.",
    "I am sorry to hear that.",
    "Do not worry, I will figure it out.",
    "This is going to work.",
    "I am a little tired, but I am here.",

    # Longer sentences
    "I have been thinking about this problem and I believe I have a solution.",
    "Let me walk you through what I found step by step.",
    "The best approach here would be to start with the simplest version and build from there.",
    "I want to make sure I understand what you are asking before I respond.",
    "Based on what you told me, I think the answer is somewhere in the middle.",

    # Janus name (correct pronunciation)
    "I am Janus and I am here to help.",
    "You can call me Janus.",
    "Janus is my name.",
]


async def synth_sentence(text: str, index: int) -> pathlib.Path:
    """Synthesize one sentence with Edge TTS, save as WAV."""
    import edge_tts
    import tempfile

    mp3_tmp = tempfile.mktemp(suffix=".mp3")
    wav_out = OUTPUT_DIR / f"janus_{index:03d}.wav"

    try:
        communicate = edge_tts.Communicate(text, VOICE, rate="-5%")
        await communicate.save(mp3_tmp)

        if FFMPEG:
            subprocess.run(
                [FFMPEG, "-y", "-i", mp3_tmp,
                 "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_out)],
                capture_output=True, check=True
            )
        else:
            # pydub fallback
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(mp3_tmp)
            seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
            seg.export(str(wav_out), format="wav")

        return wav_out
    finally:
        if os.path.exists(mp3_tmp):
            os.unlink(mp3_tmp)


async def generate_all():
    print(f"Generating {len(SENTENCES)} training samples with {VOICE}...")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print()

    wavs = []
    for i, sentence in enumerate(SENTENCES):
        try:
            wav = await synth_sentence(sentence, i)
            size_kb = wav.stat().st_size // 1024
            print(f"  [{i+1:2d}/{len(SENTENCES)}] {sentence[:50]:<50} → {size_kb}KB")
            wavs.append((sentence, wav))
        except Exception as e:
            print(f"  [{i+1:2d}/{len(SENTENCES)}] FAILED: {e}")

    print(f"\nGenerated {len(wavs)} WAV files in {OUTPUT_DIR}/")
    return wavs


def finetune(wavs):
    """Fine-tune janus_tts_v2 on the generated audio."""
    print("\nFine-tuning janus_tts_v2 on generated audio...")

    from janus_tts_v2 import JanusTTSv2

    weights = "janus_tts_v2_weights.pt"
    if not pathlib.Path(weights).exists():
        print(f"ERROR: {weights} not found. Run python janus_tts_v2.py first.")
        return

    tts = JanusTTSv2(weights)

    total = len(wavs)
    for i, (text, wav_path) in enumerate(wavs):
        print(f"\n[{i+1}/{total}] Fine-tuning on: {text[:60]}")
        try:
            tts.train_on_sample(
                text=text,
                audio_path=str(wav_path),
                steps=30,       # 30 steps per sample × 50 samples = 1500 total
                lr=5e-5         # small LR to avoid forgetting
            )
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save updated weights
    tts.save_weights(weights)
    print(f"\nFine-tuning complete. Weights saved to {weights}")

    # Test synthesis
    print("\nTesting synthesis...")
    pcm = tts.synthesize("Hello, I am Janus. How can I help you today?")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    duration = len(audio) / tts.SAMPLE_RATE

    out_path = "janus_v2_finetuned_test.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.SAMPLE_RATE)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    print(f"Test audio saved: {out_path} ({duration:.1f}s)")
    return out_path


if __name__ == "__main__":
    # Step 1: Generate training audio
    wavs = asyncio.run(generate_all())

    if not wavs:
        print("No audio generated. Check edge-tts and ffmpeg.")
        exit(1)

    # Step 2: Fine-tune
    finetune(wavs)

    print("\nDone! Run: Start-Process janus_v2_finetuned_test.wav")
