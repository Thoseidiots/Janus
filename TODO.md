# TODO

Ordered roughly by priority. Items marked `[~]` are partially done.

---

## Training

- [ ] Collect real training data — business strategy, code, reasoning, conversation
- [ ] Build a data pipeline that feeds all dataset generators into one unified token stream
- [ ] Run first real training pass on Avus-1B (needs cloud GPU — Kaggle/Lightning AI)
- [ ] Validate HBM training loop produces meaningful bind/unbind similarity scores
- [ ] Train specialist models per curriculum, then merge with DARE
- [ ] Set up checkpoint auto-save and resume on Kaggle between sessions

---

## Avus / Architecture

- [~] Avus transformer scales to 70B — configs exist, untrained
- [ ] KV cache validation at inference time (currently untested end-to-end)
- [ ] Flash attention integration for memory-efficient training at 7B+
- [ ] Gradient checkpointing for large model training on limited VRAM
- [ ] Quantization (int8/int4) for inference on CPU/low-VRAM hardware
- [ ] Context length extension beyond 4096 (RoPE scaling / YaRN)

---

## Holographic Brain Memory

- [~] HBM package built (core, real_valued, spawning, visualization)
- [ ] Prove retrieval similarity holds at scale (>100 stored items)
- [ ] Connect HBM to Avus as an external memory layer (read/write during generation)
- [ ] Persistent HBM state — save/load memory vector to disk between sessions
- [ ] Memory consolidation — compress old memories without losing them
- [ ] Episodic memory indexing — search HBM by semantic similarity

---

## Video Comprehension

- [~] Frame capture, motion detection, subtitle scanning built
- [~] AvusBrain bridge connected to video comprehension
- [ ] Real OCR integration (pytesseract or easyocr) to replace heuristic subtitle detection
- [ ] Audio transcription from video (Whisper.cpp, local)
- [ ] Janus autonomously decides what to watch based on current goals
- [ ] Learning loop — video observations update HBM automatically

---

## Speech

- [~] Source-filter synthesis engine built (speech_synthesis.py)
- [~] Voice I/O with wake word detection (voice_io_enhanced.py)
- [ ] Piper TTS model download and integration (free, local, high quality)
- [ ] Whisper.cpp model download and integration for STT
- [ ] Connect speech_synthesis.py to voice_io_enhanced.py as the TTS backend
- [ ] Prosody driven by Avus output (emotion tag -> SpeechContext)
- [ ] Speaker identity — consistent voice across sessions

---

## Autonomy & Planning

- [ ] Long-horizon goal planner — break multi-step goals into subtasks
- [ ] Self-directed learning — Janus identifies knowledge gaps and seeks to fill them
- [ ] Inter-session continuity — Janus resumes where it left off on restart
- [ ] Sandboxed code execution — run generated code safely
- [ ] Critic/verifier — check outputs before acting on them
- [ ] Error recovery — when a tool fails, reason about why and retry differently

---

## Memory (Long-term)

- [ ] Episodic memory store — searchable log of past events
- [ ] Working memory scratchpad — readable/writable mid-generation
- [ ] Semantic memory — distilled facts extracted from experience
- [ ] Memory consolidation during idle time (sleep-like offline processing)

---

## Multimodal

- [ ] Image encoder — feed visual context into Avus alongside text
- [ ] Screen understanding beyond pixel heuristics (UI element detection)
- [ ] Audio input pipeline — microphone -> tokens without external API

---

## Infrastructure

- [ ] Proper logging system across all modules (replace scattered print statements)
- [ ] Config file for system-wide settings (device, paths, model size)
- [ ] Health check script — verify all components load correctly
- [ ] Requirements.txt that actually reflects current dependencies
- [ ] Remove or archive dead code (archive/ folder, orphan files)

---

## Finance / CEO Agent

- [ ] Audit autonomous_finance.py — determine what's functional vs stub
- [ ] Connect finance agent to real data sources (local scraping, no API keys)
- [ ] Goal-driven financial planning loop
- [ ] CEO agent decision loop with memory of past decisions and outcomes

---

## Known Issues

- [ ] `updated_cognitive_loop.py` is in archive — needs to be restored or replaced
- [ ] `avus_tokenizer .py` (with space in filename) is a duplicate — delete it
- [ ] `config_avus_1b .json` (with space in filename) — rename to clean version
- [ ] `llm_integration.py` previously used Claude/OpenAI — now uses Avus, verify all callers updated
- [ ] `consciousness.py` has dead Gemini key reference — already cleaned, verify
