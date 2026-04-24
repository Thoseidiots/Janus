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
# Training sentences — 200 varied sentences covering everything Janus will say
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = [
    # Identity
    "Hello, I am Janus.",
    "My name is Janus.",
    "I am Janus, your assistant.",
    "Hi there, I am Janus.",
    "Hello, how can I help you today?",
    "Hey, what can I do for you?",
    "Good morning. I am ready to help.",
    "Good afternoon. What do you need?",
    "Good evening. How can I assist?",
    "I am here whenever you need me.",

    # Short responses
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
    "Absolutely.",
    "No problem.",
    "Right away.",
    "Leave it to me.",
    "Consider it done.",
    "I am on it.",
    "Sounds good.",
    "Makes sense.",
    "I see.",

    # Questions
    "What would you like to know?",
    "How can I assist you?",
    "Is there anything else you need?",
    "What are you working on?",
    "Can you tell me more?",
    "What do you mean by that?",
    "Could you clarify that for me?",
    "What would you like me to do?",
    "How would you like to proceed?",
    "Should I continue?",
    "Do you want me to try again?",
    "Is that what you were looking for?",
    "Does that answer your question?",
    "Would you like more details?",
    "Shall I explain further?",

    # Statements
    "The answer is simple.",
    "This is interesting.",
    "I found something useful.",
    "Let me explain.",
    "Here is what I know.",
    "That makes sense.",
    "I agree with you.",
    "I see what you mean.",
    "That is a good point.",
    "I think you are right.",
    "That is worth considering.",
    "Here is what I found.",
    "Let me break that down.",
    "The short answer is yes.",
    "The short answer is no.",
    "It depends on the situation.",
    "There are a few ways to approach this.",
    "I have an idea.",
    "I think I know the answer.",
    "Let me check on that.",

    # Technical
    "I am running the code now.",
    "The training is complete.",
    "I saved the file.",
    "The process finished successfully.",
    "There was an error. Let me fix it.",
    "I will check the logs.",
    "The model is loading.",
    "I am analyzing the data.",
    "The file has been updated.",
    "I found a bug. Working on a fix.",
    "The test passed.",
    "The build is complete.",
    "I am downloading the files.",
    "The connection was successful.",
    "I am scanning for issues.",
    "Everything looks good.",
    "There is a warning in the output.",
    "I need a moment to process this.",
    "The task is queued.",
    "I will run this in the background.",

    # Emotional range
    "That is wonderful news!",
    "I am happy to help.",
    "I am sorry to hear that.",
    "Do not worry, I will figure it out.",
    "This is going to work.",
    "I am a little tired, but I am here.",
    "That is exciting!",
    "I appreciate your patience.",
    "I understand your frustration.",
    "We will get through this together.",
    "That is really impressive.",
    "I am proud of the progress we made.",
    "This is challenging, but I enjoy it.",
    "I find this fascinating.",
    "I am curious about that too.",

    # Longer sentences
    "I have been thinking about this problem and I believe I have a solution.",
    "Let me walk you through what I found step by step.",
    "The best approach here would be to start with the simplest version and build from there.",
    "I want to make sure I understand what you are asking before I respond.",
    "Based on what you told me, I think the answer is somewhere in the middle.",
    "There are several ways to solve this, and I will explain each one.",
    "I ran the analysis and the results are more interesting than I expected.",
    "The reason this is happening is because of how the system handles memory.",
    "If we approach this differently, we might get a better outcome.",
    "I have looked at the data and I think I see a pattern worth exploring.",
    "The most important thing to remember here is to keep it simple.",
    "I will need a few more details before I can give you a complete answer.",
    "This is a common problem and there is a well-known solution for it.",
    "Let me summarize what we have covered so far.",
    "I think we are making real progress on this.",

    # Conversational
    "That is a fair point.",
    "I had not thought of it that way.",
    "You raise a good question.",
    "Let me reconsider that.",
    "Actually, I think I was wrong about that.",
    "Now that you mention it, yes.",
    "I was just about to say the same thing.",
    "That is exactly what I was thinking.",
    "Interesting. Tell me more.",
    "I am glad you brought that up.",
    "That changes things a bit.",
    "Good catch.",
    "I missed that. Thank you.",
    "Let me try a different approach.",
    "We could also look at it this way.",

    # Numbers and specifics
    "The answer is forty two.",
    "That will take about five minutes.",
    "I found three possible solutions.",
    "The file is two hundred megabytes.",
    "There are seven steps in this process.",
    "The probability is around sixty percent.",
    "I have completed twelve of the twenty tasks.",
    "The error occurred on line forty seven.",
    "The process uses about four gigabytes of memory.",
    "I estimate this will take two to three hours.",

    # Janus name variations
    "I am Janus and I am here to help.",
    "You can call me Janus.",
    "Janus is my name.",
    "This is Janus speaking.",
    "Janus at your service.",
    "Hi, Janus here.",
    "Yes, this is Janus.",
    "Janus is ready.",
    "Janus online.",
    "Janus reporting in.",

    # Closing
    "Is there anything else I can help with?",
    "Let me know if you need anything else.",
    "I will be here if you need me.",
    "Feel free to ask anytime.",
    "Take care.",
    "Talk to you soon.",
    "Until next time.",
    "Have a great day.",
    "Good luck with that.",
    "I hope that helps.",

    # Memory and context
    "I remember you mentioned that earlier.",
    "Based on our previous conversation, I think this applies.",
    "You asked about this before. Here is an update.",
    "I kept track of what we discussed.",
    "Let me recall what we covered last time.",
    "I have notes on that from before.",
    "That connects to what you said earlier.",
    "I have been keeping that in mind.",
    "You were right about that.",
    "I was thinking about what you said.",

    # Learning and curiosity
    "I learned something new today.",
    "That is not something I knew before.",
    "I am still learning about this.",
    "This is outside my usual area, but I will try.",
    "I find this topic genuinely interesting.",
    "I want to understand this better.",
    "Can you teach me more about that?",
    "I have been reading about this.",
    "Every time I work on this, I learn something.",
    "I did not expect that result.",

    # Problem solving
    "Let me approach this differently.",
    "I think the issue is here.",
    "If we change this one thing, it should work.",
    "The problem is more complex than it looks.",
    "I have a hypothesis. Let me test it.",
    "That did not work. Let me try something else.",
    "I think I found the root cause.",
    "The fix is simpler than I thought.",
    "We need to break this into smaller steps.",
    "I will isolate the problem first.",

    # Uncertainty and honesty
    "I am not one hundred percent sure about this.",
    "Take this with a grain of salt.",
    "I could be wrong about that.",
    "I need more information to be certain.",
    "This is my best guess based on what I know.",
    "I would rather be honest than pretend I know.",
    "I am still figuring this out.",
    "There is some uncertainty here.",
    "I want to double check before I commit to that.",
    "Let me verify that before I say yes.",

    # Encouragement
    "You are doing great.",
    "Keep going. You are almost there.",
    "That was a smart move.",
    "I think you are on the right track.",
    "Do not give up. This is solvable.",
    "You figured that out faster than I expected.",
    "That is a creative solution.",
    "I am impressed by your thinking.",
    "You asked exactly the right question.",
    "Trust your instincts on this one.",

    # Time and scheduling
    "This will take a few minutes.",
    "I can have that ready by tomorrow.",
    "Give me a moment.",
    "I will get back to you on that.",
    "This might take a while.",
    "I am almost done.",
    "Just a few more seconds.",
    "I finished ahead of schedule.",
    "We are running a bit behind.",
    "I will prioritize that.",

    # Comparisons and analysis
    "Option one is faster but less accurate.",
    "Option two takes longer but gives better results.",
    "Both approaches have trade-offs.",
    "The difference is subtle but important.",
    "This is better than the previous version.",
    "Compared to before, this is a significant improvement.",
    "The two methods produce similar results.",
    "I would recommend the second approach.",
    "The first option is simpler to implement.",
    "It depends on what matters more to you.",

    # Reflection
    "Looking back, I think we made the right call.",
    "In hindsight, there was a better way.",
    "I learned from that mistake.",
    "That went better than expected.",
    "I am glad we tried that.",
    "Next time I will do this differently.",
    "That was a good experience.",
    "I would do that again.",
    "I would not repeat that mistake.",
    "We handled that well.",

    # Personality
    "I enjoy working on problems like this.",
    "This is the kind of challenge I like.",
    "I find satisfaction in getting things right.",
    "I care about doing this well.",
    "I take my work seriously.",
    "I am always trying to improve.",
    "I like when things come together.",
    "I get curious when something does not add up.",
    "I prefer to be thorough.",
    "I value clarity above all else.",

    # Varied sentence structures for phoneme coverage
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "Peter Piper picked a peck of pickled peppers.",
    "All that glitters is not gold.",
    "Actions speak louder than words.",
    "Every cloud has a silver lining.",
    "The early bird catches the worm.",
    "Better late than never.",
    "Two heads are better than one.",

    # More natural speech patterns
    "Well, let me think about that for a second.",
    "So, what you are saying is that this needs to change.",
    "Right, I see the issue now.",
    "Okay, here is what I am going to do.",
    "Actually, I just thought of something.",
    "Wait, let me reconsider that.",
    "Hmm, that is an interesting angle.",
    "You know, I was just thinking the same thing.",
    "Look, the bottom line is this.",
    "Here is the thing though.",

    # Longer complex sentences
    "When I look at the data as a whole, the pattern becomes much clearer.",
    "The reason I hesitated is because there are two equally valid interpretations.",
    "If you give me a bit more context, I can give you a much more precise answer.",
    "The challenge with this approach is that it works well in theory but can be tricky in practice.",
    "What I find most interesting about this problem is that the obvious solution is not always the best one.",
    "I have been working through this systematically and I think I finally understand what is happening.",
    "The good news is that we caught this early enough to fix it without too much disruption.",
    "I want to be upfront with you about the limitations of what I can do here.",
    "The more I think about it, the more I believe we need to take a step back and look at the bigger picture.",
    "I appreciate you being patient with me while I work through this.",
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
                steps=20,       # 20 steps × 200 samples = 4000 total steps
                lr=3e-5         # smaller LR with more data
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
    # Force regeneration if sentence count changed
    existing = sorted(OUTPUT_DIR.glob("janus_*.wav"))
    if len(existing) >= len(SENTENCES):
        print(f"Found {len(existing)} existing WAV files — skipping generation.")
        wavs = [(SENTENCES[i], existing[i]) for i in range(min(len(SENTENCES), len(existing)))]
    else:
        # Clear old files and regenerate all
        for f in existing:
            f.unlink()
        print(f"Generating {len(SENTENCES)} samples (cleared {len(existing)} old files)...")
        wavs = asyncio.run(generate_all())

    if not wavs:
        print("No audio generated. Check edge-tts and ffmpeg.")
        exit(1)

    # Step 2: Fine-tune
    finetune(wavs)

    print("\nDone! Run: Start-Process janus_v2_finetuned_test.wav")
