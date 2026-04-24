# Voice App Analysis — What We Have & What We Need to Build

## Apps Installed

### 1. Voicemod V3
- **Type**: Real-time voice changer (virtual audio device)
- **How it works**: Routes microphone audio through DSP effects, outputs to virtual mic
- **API**: WebSocket on localhost:59129 (requires paid developer key for programmatic control)
- **Audio routing**: Device index 5/11/12 (Line Out) → processes → Device 1/7/14 (Virtual Mic)
- **Voices**: Preset effects (anime, robot, female pitch, etc.)
- **Strength**: Real-time, low latency, many presets
- **Weakness**: No programmatic voice selection without paid key, not neural

### 2. HitPaw VoicePea
- **Type**: AI voice changer (ONNX neural models)
- **How it works**: ONNX runtime + DirectML GPU acceleration
- **Models**: HPvoice64.dat (66MB) + HPvoice.dat (50MB) — neural voice conversion
- **Strength**: Neural quality, likely better than Voicemod for voice cloning
- **Weakness**: Closed source, no Python API visible
- **Potential**: The .dat files are ONNX models — we may be able to load them directly

### 3. Voxal Voice Changer (NCH Software)
- **Type**: Real-time DSP voice changer
- **How it works**: Traditional DSP (pitch shift, formant shift, reverb)
- **Strength**: Simple, lightweight
- **Weakness**: Not neural, sounds robotic

---

## What We Currently Have (Built)

| Component | Status | Quality |
|-----------|--------|---------|
| STT (faster-whisper) | ✅ Working | Good — base model |
| TTS acoustic model | ⚠️ Noisy | Needs more training |
| TTS vocoder | ⚠️ Noisy | Needs 10k+ steps |
| Voice (Kokoro af_heart) | ✅ Working | Human-sounding |
| Voice (Edge TTS Ana) | ✅ Working | Most human |
| Wake word detection | ✅ Working | Via Whisper |
| Brain (JanusGPT) | ⚠️ Weak | Needs Avus training |
| Voice loop | ✅ Wired | Ready to test |

---

## What We're Missing & Plan to Build

### Priority 1: Better STT
**Current**: faster-whisper base (good but not great)
**Target**: faster-whisper large-v3 or distil-whisper (much better accuracy)
**Plan**: Just upgrade the model size — same code, better model
**Effort**: 1 line change

### Priority 2: Voice Conversion (RVC-style)
**Current**: Kokoro af_heart (good but not Janus's own voice)
**Target**: Custom voice that sounds like a specific person
**What to build**: RVC (Retrieval-based Voice Conversion)
  - Encode input voice → extract pitch + timbre features
  - Replace timbre with target voice embedding
  - Decode back to audio
**Key insight from HitPaw**: Their HPvoice.dat is likely an RVC-style ONNX model
**Plan**: 
  1. Try to load HPvoice.dat as ONNX and understand its input/output
  2. Build our own RVC pipeline using the same approach
  3. Train on Ana Neural samples (already have 280 WAVs)

### Priority 3: Real-time Voice Pipeline
**Current**: Record → transcribe → generate → synthesize → play (sequential, ~3-5s latency)
**Target**: Streaming pipeline with <1s latency
**Plan**:
  - Stream STT: process audio in 0.5s chunks, emit partial transcripts
  - Stream TTS: start speaking first sentence while generating the rest
  - Overlap: while Janus speaks, pre-buffer next response

### Priority 4: Stronger Brain
**Current**: JanusGPT (small, undertrained)
**Target**: Avus 1B after proper training
**Plan**: Next Kaggle run with fixed training data

---

## Immediate Next Steps

1. **Test `python janus_voice.py`** — confirm the full loop works
2. **Try loading HPvoice.dat as ONNX** — understand HitPaw's voice model
3. **Upgrade STT to large-v3** — better transcription accuracy
4. **Build RVC pipeline** — voice conversion from Ana → custom Janus voice
