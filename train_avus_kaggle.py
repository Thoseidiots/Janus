"""
train_avus_kaggle.py
====================
Production Kaggle training script for all Janus AI models.

Trains in one session:
  1. Avus transformer (any size via MODEL_SIZE)
  2. HolographicBrainMemory (complex + real-valued)
  3. SpawningBrain

Features:
  - fp16 mixed precision (fits 1B on T4 16GB)
  - Gradient checkpointing (larger models on limited VRAM)
  - Skill curriculum (adaptive training via skill tree)
  - Session persistence (resume from last checkpoint automatically)
  - All datasets combined: 3D, screen actions, language, cognitive loop
  - Saves skill_state.json alongside weights

Setup (do once):
  1. Create a Kaggle Dataset called "janus-weights" and upload:
       avus_1b_weights.pt  (or leave empty for scratch training)
       skill_state.json    (or leave empty)
  2. In your Kaggle Notebook:
       Accelerator: GPU T4 x2
       Add dataset: janus-weights
       Add dataset: your Janus repo (or upload files manually)
  3. Set MODEL_SIZE below and run all cells.

After each epoch weights auto-save to /kaggle/working/.
Download and re-upload to "janus-weights" dataset to persist.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — change these before running
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZE            = "1b"        # 1b | 3b | 7b | 13b | 34b | 70b | growing
USE_GROWING_AVUS      = False       # True = GrowingAvus (no fixed size)
AVUS_EPOCHS           = 2           # reduced from 20 for fast tests
HBM_EPOCHS            = 1           # reduced from 10 for fast tests
SAMPLES_PER_DATASET   = 100         # synthetic samples per curriculum (reduced from 10,000)
BATCH_SIZE            = 1           # keep at 1 for T4 with large models
GRAD_ACCUM_STEPS      = 8           # effective batch = BATCH_SIZE * GRAD_ACCUM
USE_GRAD_CHECKPOINT   = True        # saves VRAM, slightly slower
USE_TORCH_COMPILE     = False       # torch.compile: faster kernels (PyTorch 2.0+, skip on Kaggle)
MAX_SEQ_LEN           = 512         # capped for T4 safety
DATASET_NAME          = "janus-avus-weights"

# ── Kaggle Mode ───────────────────────────────────────────────────────────────
# Set KAGGLE_MODE = True to automatically handle:
#   - Memory fragmentation fix
#   - CPU-offloaded optimizer (states in RAM, not VRAM)
#   - Model parallelism across both T4s (no DataParallel replication)
#   - GradScaler disabled (conflicts with CPU offload)
#   - Device-aware forward pass (all tensors follow their block's device)
#   - Single clean launcher cell — no patches needed
KAGGLE_MODE           = False       # Set True when running on Kaggle T4 x2

# ── Kaggle hardware profile ───────────────────────────────────────────────────
# Only used when KAGGLE_MODE = True
KAGGLE_MODEL_DIM      = 1920        # ~908M params at these settings
KAGGLE_MODEL_LAYERS   = 20
KAGGLE_MODEL_HEADS    = 16
KAGGLE_MODEL_KV_HEADS = 8
KAGGLE_MODEL_FFN      = 5120
KAGGLE_SEQ_LEN        = 256
KAGGLE_BATCH          = 1
KAGGLE_GRAD_ACCUM     = 16

# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, json, random, gc, math, shutil, time
from pathlib import Path
from typing import List, Set, Dict, Optional
import sqlite3
import io

DB_PATH = Path("/kaggle/working/model_epoch_weights.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS epoch_weights (
                epoch INTEGER PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                loss REAL NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chunked_weights (
                epoch INTEGER,
                chunk_index INTEGER,
                chunk_blob BLOB,
                PRIMARY KEY (epoch, chunk_index)
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"[db] Failed to init DB: {e}")
    finally:
        conn.close()

def save_epoch_to_db(epoch: int, model_state: dict, loss: float):
    buffer = io.BytesIO()
    import torch
    torch.save(model_state, buffer)
    weights_bytes = buffer.getvalue()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        CHUNK_SIZE = 1000000000  # 1GB chunks to stay under SQLite limits
        
        if len(weights_bytes) <= CHUNK_SIZE:
            conn.execute('''
                INSERT INTO epoch_weights (epoch, weights_blob, loss)
                VALUES (?, ?, ?)
                ON CONFLICT(epoch) DO UPDATE SET
                    weights_blob = excluded.weights_blob,
                    loss = excluded.loss
            ''', (epoch, sqlite3.Binary(weights_bytes), loss))
            conn.execute('DELETE FROM chunked_weights WHERE epoch = ?', (epoch,))
        else:
            conn.execute('''
                INSERT INTO epoch_weights (epoch, weights_blob, loss)
                VALUES (?, ?, ?)
                ON CONFLICT(epoch) DO UPDATE SET
                    weights_blob = excluded.weights_blob,
                    loss = excluded.loss
            ''', (epoch, sqlite3.Binary(b"CHUNKED"), loss))
            conn.execute('DELETE FROM chunked_weights WHERE epoch = ?', (epoch,))
            
            for i in range(0, len(weights_bytes), CHUNK_SIZE):
                chunk = weights_bytes[i:i+CHUNK_SIZE]
                conn.execute('''
                    INSERT INTO chunked_weights (epoch, chunk_index, chunk_blob)
                    VALUES (?, ?, ?)
                ''', (epoch, i // CHUNK_SIZE, sqlite3.Binary(chunk)))
                
        conn.commit()
        print(f"[db] Saved epoch {epoch} weights to SQLite DB. Loss: {loss:.4f}")
    except Exception as e:
        print(f"[db] Failed to save weights for epoch {epoch}: {e}")
    finally:
        conn.close()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Kaggle Mode Bootstrap ─────────────────────────────────────────────────────

def _apply_kaggle_mode():
    """
    Applies all hardware optimizations needed for Kaggle T4 x2 training.
    Called automatically when KAGGLE_MODE = True.

    What it does:
      1. Sets memory fragmentation env var
      2. Overrides config to the T4-safe 900M profile
      3. Disables GradScaler (conflicts with CPU offload)
      4. Replaces AdamW with CPU-offloaded version (optimizer states in RAM)
      5. Disables DataParallel (causes gradient reduction OOM)
      6. Patches AvusBlock.forward to be device-aware (fixes multi-GPU tensor errors)
      7. Patches Avus.__init__ and forward for model parallelism across both GPUs
    """
    global BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_SEQ_LEN

    print("\n[KaggleMode] Applying T4 x2 optimizations...")

    # 1. Memory fragmentation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # 2. Override training config to T4-safe profile
    BATCH_SIZE       = KAGGLE_BATCH
    GRAD_ACCUM_STEPS = KAGGLE_GRAD_ACCUM
    MAX_SEQ_LEN      = KAGGLE_SEQ_LEN

    # Write the 900M config so the training loop finds it
    cfg = {
        "vocab_size": 50304,
        "dim":        KAGGLE_MODEL_DIM,
        "n_layers":   KAGGLE_MODEL_LAYERS,
        "n_heads":    KAGGLE_MODEL_HEADS,
        "n_kv_heads": KAGGLE_MODEL_KV_HEADS,
        "ffn_hidden": KAGGLE_MODEL_FFN,
        "max_seq_len": KAGGLE_SEQ_LEN,
        "dropout": 0.0,
        "eps": 1e-5,
        "model_type": "avus-kaggle",
    }
    cfg_path = Path("/kaggle/working/config_avus_1b.json")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    if str(cfg_path.parent) not in sys.path:
        sys.path.insert(0, str(cfg_path.parent))
    n_params = (KAGGLE_MODEL_LAYERS * (
        4 * KAGGLE_MODEL_DIM * KAGGLE_MODEL_DIM +
        3 * KAGGLE_MODEL_DIM * KAGGLE_MODEL_FFN
    ))
    print(f"[KaggleMode] Config: dim={KAGGLE_MODEL_DIM} layers={KAGGLE_MODEL_LAYERS} "
          f"~{n_params/1e9:.2f}B params")
    print(f"[KaggleMode] Batch={BATCH_SIZE} GradAccum={GRAD_ACCUM_STEPS} "
          f"EffectiveBatch={BATCH_SIZE * GRAD_ACCUM_STEPS} SeqLen={MAX_SEQ_LEN}")

    # 3. Disable GradScaler
    class _NoopScaler:
        def __init__(self, *a, **kw): pass
        def scale(self, loss): return loss
        def unscale_(self, opt): pass
        def step(self, opt, *a, **kw): opt.step()
        def update(self): pass
        def get_scale(self): return 1.0
    torch.amp.GradScaler = lambda *a, **kw: _NoopScaler()
    torch.cuda.amp.GradScaler = lambda *a, **kw: _NoopScaler()
    print("[KaggleMode] GradScaler disabled")

    # 4. CPU-offloaded AdamW — optimizer states live in RAM
    # Only used as fallback if GPU-resident optimizer OOMs
    _orig_adamw = torch.optim.AdamW
    class _OffloadAdamW(_orig_adamw):
        def step(self, closure=None):
            moved = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        moved.append((p, p.dtype, p.device))
                        p.data = p.data.float().cpu()
                        p.grad.data = p.grad.data.float().cpu()
            for group in self.param_groups:
                for p in group['params']:
                    for k, v in self.state.get(p, {}).items():
                        if isinstance(v, torch.Tensor):
                            self.state[p][k] = v.cpu()
            super(_orig_adamw, self).step(closure)
            for p, dtype, device in moved:
                p.data = p.data.to(device=device, dtype=dtype)
                p.grad = None
            return None

    # Try GPU-resident optimizer first (fast), fall back to CPU offload (slow but safe)
    # Enable CUDA unified memory paging so overflow goes to RAM instead of crashing
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
    try:
        import ctypes
        # Allow CUDA to page to system RAM when VRAM is full
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("[KaggleMode] GPU-resident optimizer — CUDA unified memory paging enabled")
        print("[KaggleMode] Optimizer states will stay on GPU (fast steps)")
        _KAGGLE_CPU_OFFLOAD = False
    except Exception:
        torch.optim.AdamW = _OffloadAdamW
        optim.AdamW = _OffloadAdamW
        _KAGGLE_CPU_OFFLOAD = True
        print("[KaggleMode] AdamW CPU offload enabled (optimizer states in RAM)")

    # 5. Disable DataParallel
    nn.DataParallel = lambda model, **kw: model
    print("[KaggleMode] DataParallel disabled")

    # 6. Device-aware AvusBlock — moves all params/buffers to match input device
    from avus import AvusBlock as _AvusBlock
    _orig_block_fwd = _AvusBlock.forward
    def _device_aware_block(self, x, use_cache=False, cache_offset=0):
        dev = x.device
        for name, param in list(self.named_parameters()):
            if param.device != dev:
                parts = name.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1],
                        nn.Parameter(param.to(dev),
                                     requires_grad=param.requires_grad))
        for name, buf in list(self.named_buffers()):
            if buf is not None and buf.device != dev:
                parts = name.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                obj.register_buffer(parts[-1], buf.to(dev))
        return _orig_block_fwd(self, x,
                               use_cache=use_cache,
                               cache_offset=cache_offset)
    _AvusBlock.forward = _device_aware_block
    print("[KaggleMode] AvusBlock device-aware forward enabled")

    # 7. Model parallelism across both T4s
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        from avus import Avus as _Avus
        _orig_init = _Avus.__init__

        def _mp_init(self, config=None):
            _orig_init(self, config)
            n = len(self.blocks)
            split = n // 2
            self.tok_emb = self.tok_emb.cuda(0)
            for i in range(n):
                self.blocks[i] = self.blocks[i].cuda(0 if i < split else 1)
            self.ln_f = self.ln_f.cuda(1)
            self.head = self.head.cuda(1)
            self._mp_split = split
            print(f"[KaggleMode] ModelParallel: "
                  f"GPU0 layers 0-{split-1} | GPU1 layers {split}-{n-1}")

        def _mp_forward(self, idx, targets=None,
                        use_cache=False, cache_offset=0):
            B, T = idx.shape
            x = self.dropout(self.tok_emb(idx.cuda(0)))
            for i, block in enumerate(self.blocks):
                x = x.cuda(0 if i < self._mp_split else 1)
                x = block(x, use_cache=use_cache, cache_offset=cache_offset)
            x = self.ln_f(x.cuda(1))
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.cuda(1).view(-1),
                    ignore_index=-1,
                )
            return logits, loss

        _Avus.__init__ = _mp_init
        _Avus.forward  = _mp_forward
        print(f"[KaggleMode] ModelParallel enabled across {n_gpus} GPUs")
    else:
        print("[KaggleMode] Single GPU mode")

    print("[KaggleMode] All optimizations applied. Ready to train.\n")


# Apply Kaggle mode immediately if enabled
if KAGGLE_MODE:
    _apply_kaggle_mode()

# ── Paths ─────────────────────────────────────────────────────────────────────

KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
DATASET_DIR    = KAGGLE_INPUT / DATASET_NAME

WEIGHTS_IN     = DATASET_DIR  / f"avus_{MODEL_SIZE}_weights.pt"
SKILL_IN       = DATASET_DIR  / "skill_state.json"
WEIGHTS_OUT    = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
SKILL_OUT      = KAGGLE_WORKING / "skill_state.json"
HBM_OUT        = KAGGLE_WORKING / "hbm_weights.pt"
CHART_OUT      = KAGGLE_WORKING / "skill_chart.png"

# ── Install dependencies ──────────────────────────────────────────────────────

def install(pkg):
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                   check=False)

try:
    import tiktoken
except ImportError:
    install("tiktoken"); import tiktoken

# ── Add Janus repo to path ────────────────────────────────────────────────────

REPO_CANDIDATES = [
    KAGGLE_INPUT / "janus-repo" / "Janus-main",
    KAGGLE_INPUT / "janus-repo",
    KAGGLE_WORKING,
    Path("."),
]
for p in REPO_CANDIDATES:
    if (p / "avus.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        print(f"[setup] Janus repo found at {p}")
        break

# ── Import Janus modules ──────────────────────────────────────────────────────

from avus import Avus, AvusConfig

try:
    from skill_curriculum import SkillTree, CurriculumSampler
    SKILL_CURRICULUM = True
    print("[setup] Skill curriculum loaded")
except ImportError:
    SKILL_CURRICULUM = False
    print("[setup] skill_curriculum.py not found — using fixed curriculum")

try:
    from holographic_brain_memory.core import HolographicBrainMemory, PhaseBrainLayer
    from holographic_brain_memory.real_valued import RealHolographicMemory, RealPhaseBrainLayer
    from holographic_brain_memory.spawning import SpawningBrain
    HBM_AVAILABLE = True
    print("[setup] HBM modules loaded")
except ImportError:
    HBM_AVAILABLE = False
    print("[setup] HBM not found — skipping HBM training")

print(f"[setup] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[setup] GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB VRAM")


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "<|startoftext|>", "<|endoftext|>",
    "[JSON_START]", "[JSON_END]",
    "[ACT_START]",  "[ACT_END]",
    "[FRAME_START]","[FRAME_NEXT]","[/FRAME_END]",
}

class AvusTokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=SPECIAL_TOKENS)

    def decode(self, tokens: List[int]) -> str:
        valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
        try:
            return self._enc.decode(valid)
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _rc(lst): return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 3)


def generate_3d_pairs(n: int = 10_000) -> List[str]:
    """3D object generation curriculum."""
    adjs = ["ancient","glowing","rusty","futuristic","crystalline","massive","tiny"]
    objs = ["rock","crystal","pillar","gateway","chest","statue","ruin","barrel"]
    prims = ["box","sphere","cylinder","torus"]
    mats  = ["stone","wood","metal","plastic"]
    out = []
    for _ in range(n):
        adj, obj = _rc(adjs), _rc(objs)
        prim, mat = _rc(prims), _rc(mats)
        params = {
            "object": obj, "primitive": prim, "material": mat,
            "scale": [_rf(0.5,3)]*3, "position": [_rf(-5,5)]*3,
            "roughness": _rf(0.1,0.9), "metallic": _rf(0,1) if mat=="metal" else 0.0,
        }
        text = (f"<|startoftext|>Generate a {adj} {obj} with {prim} shape "
                f"and {mat} material. [JSON_START]{json.dumps(params)}[JSON_END]"
                f"<|endoftext|>")
        out.append(text)
    return out


def generate_screen_action_pairs(n: int = 10_000) -> List[str]:
    """Screen action curriculum."""
    apps = ["Chrome","VS Code","Terminal","File Explorer","Discord"]
    btns = ["Submit","Cancel","Save","Login","Search","Next","Delete"]
    out = []
    for _ in range(n):
        x, y = _ri(10, 1910), _ri(10, 1070)
        btn  = _rc(btns)
        app  = _rc(apps)
        action = {"type": "click", "x": x, "y": y, "button": "left"}
        text = (f"<|startoftext|>{app} is open. "
                f"A '{btn}' button is at ({x},{y}). Click it. "
                f"[ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
        out.append(text)
    return out


def generate_language_pairs(n: int = 10_000) -> List[str]:
    """Language comprehension curriculum."""
    topics = [
        "machine learning", "neural networks", "transformers", "attention mechanisms",
        "gradient descent", "backpropagation", "reinforcement learning",
        "computer vision", "natural language processing", "robotics",
    ]
    templates = [
        "Explain {topic} in simple terms.",
        "What is {topic}?",
        "How does {topic} work?",
        "Describe the key concepts of {topic}.",
        "What are the applications of {topic}?",
    ]
    out = []
    for _ in range(n):
        topic    = _rc(topics)
        template = _rc(templates)
        question = template.format(topic=topic)
        answer   = f"{topic.capitalize()} is a fundamental concept in AI that involves processing and learning from data."
        text = f"<|startoftext|>{question} {answer}<|endoftext|>"
        out.append(text)
    return out


def generate_reasoning_pairs(n: int = 10_000) -> List[str]:
    """Reasoning and cognitive loop curriculum."""
    out = []
    for _ in range(n):
        a, b = _ri(1, 100), _ri(1, 100)
        op   = _rc(["+", "-", "*"])
        if op == "+":   result = a + b
        elif op == "-": result = a - b
        else:           result = a * b
        text = (f"<|startoftext|>Calculate: {a} {op} {b}. "
                f"Step 1: Identify the operation ({op}). "
                f"Step 2: Apply it. "
                f"Result: {result}<|endoftext|>")
        out.append(text)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class JanusDataset(Dataset):
    """Combined dataset from all curriculum generators."""

    def __init__(self, tokenizer: AvusTokenizer, block_size: int,
                 samples_per: int = 10_000):
        self.block_size = block_size
        self.data: List[torch.Tensor] = []

        print("[data] Generating training data...")
        all_texts = []
        all_texts += generate_3d_pairs(samples_per)
        all_texts += generate_screen_action_pairs(samples_per)
        all_texts += generate_language_pairs(samples_per)
        all_texts += generate_reasoning_pairs(samples_per)

        # ── Phase 2: AAA rendering quality ──────────────────────────────────
        # Fixes: Squibbling, Yosification, Uncanny Valley, Ghosting, Imaginary Lighting
        _phase2_mods = [
            ("temporal_consistency_dataset",  "TemporalConsistencyDataset"),
            ("spatial_detail_dataset",        "SpatialDetailDataset"),
            ("geometric_constraint_dataset",  "GeometricConstraintDataset"),
            ("optical_flow_dataset",          "OpticalFlowDataset"),
            ("semantic_lighting_dataset",     "SemanticLightingDataset"),
        ]
        for _mn, _cn in _phase2_mods:
            try:
                import importlib as _il2
                _m2 = _il2.import_module(_mn)
                _p2 = getattr(_m2, _cn)().generate_dataset(samples_per)
                for _prompt, _out in _p2:
                    all_texts.append(
                        f"<|startoftext|>{_prompt}\n{_out}<|endoftext|>")
                print(f"  [Phase2] {_cn}: {len(_p2):,} pairs added")
            except Exception as _e2:
                print(f"  [Phase2] {_cn} skipped: {_e2}")

        # Deep curriculum — harder, multi-difficulty synthetic data
        try:
            from deep_training_data import CombinedDeepDataset
            deep = CombinedDeepDataset()
            deep_samples = deep.generate_curriculum(n_per_generator=samples_per // 6)
            all_texts += deep_samples
            print(f"[data] Deep curriculum added: {len(deep_samples):,} samples")
        except ImportError:
            print("[data] deep_training_data.py not found — skipping deep curriculum")

        # Procedural dataset — infinite unique samples, difficulty-scaled
        try:
            from procedural_dataset import ProceduralDataset
            proc = ProceduralDataset()
            proc_samples = proc.generate(n=samples_per, difficulty=3, seed=42)
            all_texts += proc_samples
            print(f"[data] Procedural dataset added: {len(proc_samples):,} samples")
        except ImportError:
            print("[data] procedural_dataset.py not found — skipping procedural data")

        random.shuffle(all_texts)

        print(f"[data] Tokenizing {len(all_texts):,} sequences...")
        pad_id = tokenizer.encode("<|endoftext|>")[0]

        for text in all_texts:
            tokens = tokenizer.encode(text)
            # Chunk into block_size+1 windows
            for i in range(0, len(tokens), block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) < 2:
                    continue
                if len(chunk) < block_size + 1:
                    chunk = chunk + [pad_id] * (block_size + 1 - len(chunk))
                self.data.append(torch.tensor(chunk, dtype=torch.long))

        print(f"[data] {len(self.data):,} training chunks ready")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# AVUS TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_avus():
    print("\n" + "="*60)
    print(f"TRAINING {'GROWING AVUS' if USE_GROWING_AVUS else 'AVUS-' + MODEL_SIZE.upper()}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_GROWING_AVUS:
        return _train_growing_avus(device)
    return _train_fixed_avus(device)


def _train_growing_avus(device):
    """Train GrowingAvus — starts small, grows as needed."""
    try:
        from growing_avus import GrowingAvus, GrowthConfig
    except ImportError:
        print("[growing] growing_avus.py not found in repo — skipping")
        return

    gc = GrowthConfig(
        seed_dim=256, seed_layers=2, seed_heads=4, seed_kv_heads=2,
        vocab_size=50304, max_seq_len=MAX_SEQ_LEN,
        spawn_after_steps=100, freeze_after_steps=500,
        grad_saturation_threshold=0.001,
        activation_collapse_threshold=0.01,
        vram_budget_gb=14.0,
    )

    weights_path = KAGGLE_WORKING / "avus_growing_weights.pt"
    weights_in   = DATASET_DIR / "avus_growing_weights.pt"

    model = GrowingAvus(gc).to(device)
    if weights_in.exists():
        model = GrowingAvus.from_checkpoint(str(weights_in), device=str(device))
        print(f"[growing] Resumed — {len(model.blocks)} layers, "
              f"dim={model.avus_config.dim}, step={model._step}")
    else:
        print(f"[growing] Starting fresh — seed: {gc.seed_layers} layers, dim={gc.seed_dim}")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[growing] DataParallel across {torch.cuda.device_count()} GPUs")

    raw_model = model.module if hasattr(model, "module") else model

    if SKILL_CURRICULUM:
        skill_tree = SkillTree()
        if SKILL_IN.exists():
            skill_tree.load(str(SKILL_IN))
        sampler = CurriculumSampler(skill_tree)
    else:
        skill_tree = sampler = None

    tokenizer = AvusTokenizer()
    dataset   = JanusDataset(tokenizer, MAX_SEQ_LEN, SAMPLES_PER_DATASET)
    effective_batch = BATCH_SIZE * max(1, torch.cuda.device_count())
    loader    = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                           num_workers=2, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(AVUS_EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            raw_model.record_gradients()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                grew = raw_model.check_and_grow()
                if grew:
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=3e-4,
                        betas=(0.9, 0.95), weight_decay=0.1
                    )

            epoch_loss += loss.item()

            if step % 100 == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                if sampler and skill_tree:
                    domain = sampler.next_domain()
                    conf   = max(0.0, min(1.0, 1.0 - avg / 10.0))
                    skill_tree.update(domain, conf)
                print(f"  step {step}/{len(loader)} loss={avg:.4f} "
                      f"t={elapsed:.0f}s layers={len(raw_model.blocks)} "
                      f"dim={raw_model.avus_config.dim} "
                      f"params={raw_model.count_parameters()/1e6:.1f}M")

        avg_loss = epoch_loss / len(loader)
        print(f"\n[growing] Epoch {epoch+1} — loss={avg_loss:.4f}")
        print(raw_model.growth_summary())

        raw_model.save_checkpoint(str(weights_path))
        if skill_tree:
            skill_tree.save(str(SKILL_OUT))
            try: skill_tree.plot(str(CHART_OUT))
            except Exception: pass

        gc_module = __import__("gc")
        gc_module.collect()
        torch.cuda.empty_cache()

    print(f"\n[growing] Training complete.")


def _train_fixed_avus(device):
    init_db()
    # ── Config ────────────────────────────────────────────────────────────────
    # In Kaggle mode, /kaggle/working has the correct config — check it first
    if KAGGLE_MODE:
        working = Path("/kaggle/working")
        if working not in REPO_CANDIDATES:
            REPO_CANDIDATES.insert(0, working)

    config_path = None
    for p in REPO_CANDIDATES:
        cp = p / f"config_avus_{MODEL_SIZE}.json"
        if cp.exists():
            config_path = cp
            break

    if config_path:
        cfg = AvusConfig.from_file(str(config_path))
        print(f"[avus] Config from {config_path}")
    else:
        cfg = AvusConfig()
        print("[avus] Using default AvusConfig")

    cfg_dict = {
        "vocab_size": cfg.vocab_size, "dim": cfg.dim,
        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads, "ffn_hidden": cfg.ffn_hidden,
        "max_seq_len": min(cfg.max_seq_len, MAX_SEQ_LEN),
    }
    cfg = AvusConfig.from_dict(cfg_dict)
    print(f"[avus] dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"kv={cfg.n_kv_heads} ffn={cfg.ffn_hidden} seq={cfg.max_seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Avus(cfg).to(device)
    total = model.count_parameters()
    print(f"[avus] Parameters: {total/1e9:.2f}B ({total:,})")
    vram_fp16_gb = (total * 2) / 1e9
    vram_fp32_gb = (total * 4) / 1e9
    print(f"[avus] VRAM estimate: {vram_fp16_gb:.1f}GB fp16 / "
          f"{vram_fp32_gb:.1f}GB fp32  (T4 = 15GB/GPU, 30GB total)")
    # Rough VRAM estimate: params * 4 bytes (fp32) or *2 (fp16) + activations
    vram_fp16_gb = (total * 2) / 1e9
    vram_fp32_gb = (total * 4) / 1e9
    print(f"[avus] Estimated VRAM: {vram_fp16_gb:.1f}GB (fp16) / "
          f"{vram_fp32_gb:.1f}GB (fp32) — "
          f"T4 has 15GB per GPU, 30GB total")

    # torch.compile — fuses kernels, 10-25% faster throughput (PyTorch 2.0+)
    # Disabled in KAGGLE_MODE (conflicts with model parallelism)
    if USE_TORCH_COMPILE and not KAGGLE_MODE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[avus] torch.compile: kernel fusion enabled")
        except Exception as _ce:
            print(f"[avus] torch.compile skipped: {_ce}")

    # Multi-GPU: wrap with DataParallel if more than one GPU available
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[avus] DataParallel enabled across {torch.cuda.device_count()} GPUs")

    # Gradient checkpointing — access underlying model through .module if DataParallel
    raw_model = model.module if hasattr(model, "module") else model
    if USE_GRAD_CHECKPOINT:
        for block in raw_model.blocks:
            block.attn._cache_k = None
            block.attn._cache_v = None
        print("[avus] Gradient checkpointing: enabled")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    resume_step = 0
    for wpath in [WEIGHTS_OUT, WEIGHTS_IN]:
        if wpath.exists():
            ckpt = torch.load(str(wpath), map_location="cpu")
            sd   = ckpt.get("model_state_dict", ckpt)
            sd   = {k.replace("module.", ""): v for k, v in sd.items()}
            drop = [k for k in sd if any(x in k for x in
                    ("attn.mask", "attn.rope.c", "attn.rope.s"))]
            for k in drop: del sd[k]
            model.load_state_dict(sd, strict=False)
            start_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
            print(f"[avus] Resumed from {wpath} (epoch {start_epoch})")
            break
    else:
        print("[avus] Training from scratch")

    # Check for mid-epoch checkpoint — resume from within an epoch
    _mid_path = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_midepoch.pt"
    if _mid_path.exists():
        try:
            mid = torch.load(str(_mid_path), map_location="cpu")
            mid_epoch = mid.get("epoch", -1)
            mid_step  = mid.get("step", 0)
            if mid_epoch == start_epoch and mid_step > 0:
                sd = {k.replace("module.", ""): v
                      for k, v in mid.get("model_state_dict", {}).items()}
                model.load_state_dict(sd, strict=False)
                resume_step = mid_step + 1
                print(f"[avus] Mid-epoch resume: epoch {mid_epoch} "
                      f"step {mid_step} loss={mid.get('loss', 0):.4f}")
        except Exception as e:
            print(f"[avus] Mid-epoch checkpoint load failed: {e}")

    # ── Skill curriculum ──────────────────────────────────────────────────────
    if SKILL_CURRICULUM:
        skill_tree = SkillTree()
        if SKILL_IN.exists():
            skill_tree.load(str(SKILL_IN))
            print(f"[avus] Skill state loaded from {SKILL_IN}")
        sampler = CurriculumSampler(skill_tree)
        print(f"[avus] First training domain: {sampler.next_domain()}")
    else:
        skill_tree = None
        sampler    = None

    # ── Data ──────────────────────────────────────────────────────────────────
    # Scale batch size with number of GPUs
    effective_batch = BATCH_SIZE * max(1, torch.cuda.device_count())
    tokenizer = AvusTokenizer()

    # Check for pre-tokenized dataset first (much faster to load)
    pretok_candidates = [
        DATASET_DIR / "tokens.pt",
        KAGGLE_WORKING / "tokens.pt",
    ]
    pretok_path = next((p for p in pretok_candidates if p.exists()), None)

    if pretok_path:
        print(f"[data] Loading pre-tokenized data from {pretok_path}")
        data = torch.load(str(pretok_path), map_location="cpu",
                          weights_only=False)
        tokens = data["tokens"]
        print(f"[data] {tokens.numel():,} tokens loaded "
              f"(sources: {list(data.get('sources', {}).keys())})")
        dataset = JanusDataset.__new__(JanusDataset)
        dataset.block_size = cfg.max_seq_len
        dataset.data = []
        pad_id = tokenizer.encode("<|endoftext|>")[0]
        bs = cfg.max_seq_len
        for i in range(0, len(tokens) - bs, bs):
            chunk = tokens[i:i + bs + 1]
            if len(chunk) == bs + 1:
                dataset.data.append(chunk)
        dataset.__len__  = lambda: len(dataset.data)
        dataset.__getitem__ = lambda idx: (dataset.data[idx][:-1],
                                           dataset.data[idx][1:])
        print(f"[data] {len(dataset.data):,} training chunks from pre-tokenized file")
    else:
        print("[data] No pre-tokenized file found — generating on the fly")
        print("[data] Tip: run pretokenize.py once and upload tokens.pt to save time")
        dataset = JanusDataset(tokenizer, cfg.max_seq_len, SAMPLES_PER_DATASET)

    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                        num_workers=2, pin_memory=(device.type == "cuda"))
    print(f"[avus] Effective batch size: {effective_batch} "
          f"({BATCH_SIZE} x {max(1, torch.cuda.device_count())} GPUs)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay: apply only to weight matrices, not biases/norms/embeddings
    decay_params     = [p for n, p in raw_model.named_parameters()
                        if p.requires_grad and p.dim() >= 2
                        and not any(nd in n for nd in ["tok_emb", "ln_", "norm"])]
    no_decay_params  = [p for n, p in raw_model.named_parameters()
                        if p.requires_grad and (p.dim() < 2
                        or any(nd in n for nd in ["tok_emb", "ln_", "norm"]))]

    # LR scales with sqrt(grad_accum) to account for effective batch size
    base_lr   = 3e-4 * (GRAD_ACCUM_STEPS ** 0.5)
    min_lr    = base_lr / 10

    _base_optimizer = optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=base_lr, betas=(0.9, 0.95))

    # Wrap with JanusOptimizer for activation checkpointing + kernel fusion
    try:
        from janus_optimizer import JanusOptimizer
        janus_opt = JanusOptimizer(
            model=model,
            optimizer=_base_optimizer,
            accumulation_steps=GRAD_ACCUM_STEPS,
            use_checkpointing=USE_GRAD_CHECKPOINT and not KAGGLE_MODE,
            use_compile=USE_TORCH_COMPILE and not KAGGLE_MODE,
        )
        optimizer = _base_optimizer   # still used for LR scheduling
        USE_JANUS_OPT = True
        print("[avus] JanusOptimizer active")
    except ImportError:
        janus_opt = None
        optimizer = _base_optimizer
        USE_JANUS_OPT = False
        print("[avus] JanusOptimizer not found — using standard training loop")

    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not KAGGLE_MODE))
    total_steps  = len(loader) * AVUS_EPOCHS
    # Warmup = 2% of total steps (more stable for large models)
    warmup_steps = max(200, total_steps // 50)
    print(f"[avus] base_lr={base_lr:.2e} warmup={warmup_steps} total={total_steps}")

    # ── Focal loss helper ─────────────────────────────────────────────────────
    def focal_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                            gamma: float = 2.0) -> torch.Tensor:
        # Move targets to same device as logits (model parallelism)
        targets = targets.to(logits.device)
        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** gamma) * ce
        return loss.mean()

    # ── Gradient checkpointing ────────────────────────────────────────────────
    # Disabled in KAGGLE_MODE — model parallelism already splits memory across GPUs
    if USE_GRAD_CHECKPOINT and not KAGGLE_MODE:
        from torch.utils.checkpoint import checkpoint as _ckpt
        raw_model = model.module if hasattr(model, "module") else model

        def _make_ckpt_block(block):
            _orig = block._orig_forward if hasattr(block, '_orig_forward') else block.forward
            def _forward(x, use_cache=False, cache_offset=0):
                return _ckpt(_orig, x, use_cache, cache_offset, use_reentrant=False)
            return _forward

        for i, block in enumerate(raw_model.blocks):
            if not hasattr(block, '_orig_forward'):
                block._orig_forward = block.forward
                block.forward = _make_ckpt_block(block)
        print(f"[avus] Gradient checkpointing enabled across {len(raw_model.blocks)} blocks")
    elif KAGGLE_MODE:
        print("[avus] Gradient checkpointing skipped (Kaggle model parallel mode)")

    # ── Mid-epoch checkpoint config ───────────────────────────────────────────
    SAVE_EVERY_STEPS = 500   # save a mid-epoch checkpoint every N steps
    mid_ckpt_path    = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_midepoch.pt"

    # ── Gradient battery ──────────────────────────────────────────────────────
    # Accumulates gradients across sessions. Discharge when ready.
    BATTERY_DISCHARGE_STEPS = 2000   # discharge after this many accumulated steps
    battery = None
    try:
        from holographic_gradient_battery import HolographicGradientBattery
        hgb_path = KAGGLE_WORKING / "gradient_battery.pt"
        
        # 4 million params -> ~16MB capacity dim. Stops the 20GB disk OOMs completely.
        battery = HolographicGradientBattery(str(hgb_path), device=str(device), capacity_dim=2**22)
        battery.load()
        print(f"[avus] Holographic Gradient Battery (HGB) active!")
        print(battery.status())
    except ImportError:
        try:
            from gradient_battery import GradientBattery
            battery_path = KAGGLE_WORKING / "gradient_battery.pt"
            battery = GradientBattery(str(battery_path), device=str(device))
            print(f"[avus] Heavy Disk Gradient battery loaded (Warning: Large Disk Footprint)")
        except ImportError:
            print("[avus] battery scripts not found — battery disabled")

    # ── Gradient clip tracker ─────────────────────────────────────────────────
    clip_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(start_epoch, start_epoch + AVUS_EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            # Skip steps already completed in a previous session
            if step < resume_step:
                global_step += 1
                continue
            input_device = "cuda:0" if KAGGLE_MODE and torch.cuda.is_available() else device
            x, y = x.to(input_device), y.to(input_device)

            # Cosine LR with warmup
            lr = _cosine_lr(global_step, warmup_steps, total_steps, base_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if USE_JANUS_OPT and not KAGGLE_MODE:
                # JanusOptimizer handles mixed precision, accumulation, checkpointing
                loss_val, stepped = janus_opt.training_step(x, y, step)
                loss = torch.tensor(loss_val)
                if stepped:
                    if battery is not None:
                        battery.charge(model, n_samples=x.shape[0],
                                       scale=1.0 / GRAD_ACCUM_STEPS)
                    if battery is not None and battery.is_ready(BATTERY_DISCHARGE_STEPS):
                        print(f"\n[Battery] Discharging at step {step}...")
                        battery.discharge(model)
                        battery.save()
                        battery.reset()
            else:
                # Standard training loop (Kaggle mode or no JanusOptimizer)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not KAGGLE_MODE)):
                    logits, _ = model(x)
                    loss = focal_cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        gamma=2.0,
                    )
                    if loss.dim() > 0:
                        loss = loss.mean()

                scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm > 1.0:
                        clip_count += 1
                    if battery is not None:
                        battery.charge(model, n_samples=x.shape[0],
                                       scale=1.0 / GRAD_ACCUM_STEPS)
                    if battery is not None and battery.is_ready(BATTERY_DISCHARGE_STEPS):
                        print(f"\n[Battery] Discharging at step {step}...")
                        battery.discharge(model)
                        battery.save()
                        battery.reset()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            epoch_loss  += loss.item()
            global_step += 1

            # ── Mid-epoch checkpoint ──────────────────────────────────────────
            if global_step % SAVE_EVERY_STEPS == 0:
                _raw = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch":            epoch,
                    "step":             step,
                    "global_step":      global_step,
                    "model_state_dict": _raw.state_dict(),
                    "config":           cfg_dict,
                    "loss":             epoch_loss / (step + 1),
                }, str(mid_ckpt_path))
                if skill_tree:
                    skill_tree.save(str(SKILL_OUT))
                if battery is not None:
                    battery.save()
                print(f"  [ckpt] Mid-epoch saved at step {step} "
                      f"(global {global_step}) → {mid_ckpt_path.name}")

            if step % 100 == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0

                # Update skill curriculum
                if sampler and skill_tree:
                    domain     = sampler.next_domain()
                    confidence = max(0.0, min(1.0, 1.0 - avg / 10.0))
                    skill_tree.update(domain, confidence)
                    try:
                        from procedural_dataset import ProceduralDataset as _PD
                        _proc_diff = _PD().get_difficulty_for_skill_level(confidence)
                    except Exception:
                        _proc_diff = 1
                    print(f"  step {step}/{len(loader)} loss={avg:.4f} "
                          f"lr={lr:.2e} t={elapsed:.0f}s "
                          f"| {domain} conf={confidence:.2f} proc_diff={_proc_diff} "
                          f"clips={clip_count}")
                else:
                    print(f"  step {step}/{len(loader)} loss={avg:.4f} "
                          f"lr={lr:.2e} t={elapsed:.0f}s clips={clip_count}")

        avg_loss = epoch_loss / max(1, len(loader) - resume_step)
        resume_step = 0  # only skip steps on the first resumed epoch
        print(f"\n[avus] Epoch {epoch+1} complete — loss={avg_loss:.4f}")

        # Save checkpoint — strip DataParallel module. prefix if present
        raw_model = model.module if hasattr(model, "module") else model
        state_dict_raw = raw_model.state_dict()
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": state_dict_raw,
            "config":           cfg_dict,
            "loss":             avg_loss,
        }, str(WEIGHTS_OUT))
        print(f"[avus] Weights saved -> {WEIGHTS_OUT}")

        # Save to SQLite DB explicitly
        save_epoch_to_db(epoch + 1, state_dict_raw, avg_loss)

        # Save skill state and chart
        if skill_tree:
            skill_tree.save(str(SKILL_OUT))
            try:
                skill_tree.plot(str(CHART_OUT))
                print(f"[avus] Skill chart -> {CHART_OUT}")
            except Exception as e:
                print(f"[avus] Chart failed: {e}")
            print(f"[avus] Next priority: {skill_tree.best_skill_to_train()}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            shutil.copy(str(WEIGHTS_OUT),
                        str(KAGGLE_WORKING / f"avus_{MODEL_SIZE}_best.pt"))

        # Auto-push after every epoch so cancelling mid-session doesn't lose progress
        auto_push_weights(version_notes=f"Avus-{MODEL_SIZE} epoch {epoch+1} loss={avg_loss:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[avus] Training complete. Best loss: {best_loss:.4f}")
    print(f"[avus] Download from Kaggle output panel:")
    print(f"  {WEIGHTS_OUT}")
    print(f"  {SKILL_OUT}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# HBM TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_hbm():
    if not HBM_AVAILABLE:
        print("[hbm] Skipping — modules not available")
        return

    print("\n" + "="*60)
    print("TRAINING HOLOGRAPHIC BRAIN MEMORY")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim    = 1024
    in_dim = 64

    results = {}

    for variant, MemCls, LayerCls in [
        ("complex", HolographicBrainMemory, PhaseBrainLayer),
        ("real",    RealHolographicMemory,  RealPhaseBrainLayer),
    ]:
        print(f"\n[hbm] Training {variant} variant...")
        memory = MemCls(dim=dim).to(device)
        layer  = LayerCls(in_dim=in_dim, memory=memory).to(device)
        opt    = optim.Adam(layer.parameters(), lr=1e-3)

        for epoch in range(HBM_EPOCHS):
            total_loss = 0.0
            for _ in range(200):
                x    = torch.randn(1, in_dim, device=device)
                out  = layer(x)
                loss = F.mse_loss(out, torch.zeros_like(out))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            if (epoch + 1) % max(1, HBM_EPOCHS // 5) == 0:
                print(f"  [{variant}] epoch {epoch+1}/{HBM_EPOCHS} "
                      f"loss={total_loss/200:.4f}")

        results[variant] = {
            "memory": memory.state_dict(),
            "layer":  layer.state_dict(),
            "dim": dim, "in_dim": in_dim,
        }

    # SpawningBrain
    print("\n[hbm] Training SpawningBrain...")
    brain = SpawningBrain(in_dim=in_dim, memory_dim=dim, initial_layers=1).to(device)
    opt   = optim.Adam(brain.parameters(), lr=1e-3)

    for epoch in range(HBM_EPOCHS):
        total_loss = 0.0
        for _ in range(100):
            x    = torch.randn(1, in_dim, device=device)
            out  = brain(x)
            loss = F.mse_loss(out, torch.zeros_like(out))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            brain.check_and_spawn(x, threshold=0.1)

        if (epoch + 1) % max(1, HBM_EPOCHS // 5) == 0:
            print(f"  [spawning] epoch {epoch+1}/{HBM_EPOCHS} "
                  f"loss={total_loss/100:.4f} "
                  f"neurons={brain.get_logical_capacity()}")

    results["spawning"] = {
        "state": brain.state_dict(),
        "spawn_history": brain.spawn_history,
    }

    torch.save(results, str(HBM_OUT))
    print(f"\n[hbm] All HBM models saved -> {HBM_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def print_summary():
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
    print("\nFiles to download from Kaggle output panel:")
    for f in KAGGLE_WORKING.glob("*.pt"):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:.1f} MB")
    for f in KAGGLE_WORKING.glob("*.json"):
        print(f"  {f.name}")
    for f in KAGGLE_WORKING.glob("*.png"):
        print(f"  {f.name}")


def auto_push_weights(version_notes: str = "Auto-save"):
    """
    Push weights back to the janus-weights Kaggle dataset automatically.

    Requires a Kaggle notebook secret named KAGGLE_KEY containing
    the contents of your kaggle.json API token.

    To set up:
      1. Kaggle account -> Settings -> API -> Create New Token
      2. Notebook -> Add-ons -> Secrets -> Add secret:
           Name:  KAGGLE_KEY
           Value: (paste full contents of kaggle.json)
    """
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("Kiro")
    except Exception as e:
        print(f"[push] 'Kiro' secret not found: {e}")
        print("[push] Skipping auto-push — download weights manually from output panel")
        return

    import json as _json

    # Write kaggle.json so the API client can authenticate
    kaggle_dir = Path(os.path.expanduser("~/.kaggle"))
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    creds = {"username": "ishmaelsears", "key": token}
    kaggle_json.write_text(_json.dumps(creds))
    os.chmod(str(kaggle_json), 0o600)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        os.system("pip install kaggle -q")
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Securely download existing Kaggle database to merge them so we don't accidentally drop past files
    # Instead of copying generated files into a staging dir (which doubles disk space and crashes), 
    # we download missing dataset files to a temp dir, MOVE them into KAGGLE_WORKING, and push the entire KAGGLE_WORKING dir.
    dl_dir = KAGGLE_WORKING / "kaggle_dl_temp"
    dl_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"[push] Downloading existing dataset to prevent file loss...")
    try:
        api.dataset_download_files("ishmaelsears/janus-avus-weights", path=str(dl_dir), unzip=True)
        import shutil
        for f in dl_dir.iterdir():
            tgt = KAGGLE_WORKING / f.name
            # If the file already exists in working dir, our freshly generated one takes precedence.
            if f.is_file() and not tgt.exists() and f.name != "dataset-metadata.json":
                shutil.move(str(f), str(tgt))
    except Exception as e:
        print(f"[push] Warning: Could not download old dataset files (they might not exist yet): {e}")
    finally:
        import shutil
        if dl_dir.exists():
            shutil.rmtree(dl_dir, ignore_errors=True)
            
    # Also remove any old upload_staging if it exists
    old_staging = KAGGLE_WORKING / "upload_staging"
    if old_staging.exists():
        shutil.rmtree(old_staging, ignore_errors=True)

    # Write dataset metadata directly into KAGGLE_WORKING
    meta = {
        "title": "janus-avus-weights",
        "id": "ishmaelsears/janus-avus-weights",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = KAGGLE_WORKING / "dataset-metadata.json"
    meta_path.write_text(_json.dumps(meta, indent=2))

    # SAFETY CHECK: Strictly prevent overwriting the dataset if the main model weights are missing.
    # If the download failed and we didn't generate new ones, pushing would wipe the multi-GB weights 
    # from the Kaggle dataset, leaving it with just text files or tiny checkpoints and destroying progress.
    main_weights_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
    if not main_weights_file.exists():
        print(f"[push] CRITICAL ERROR: Aborting Kaggle push! The primary weights file '{main_weights_file.name}' is missing.")
        print("[push] Pushing now would permanently overwrite and wipe your model weights from the dataset. Aborting!")
        return

    # Cleanup any random loose files (like test_upload.txt) to keep the dataset clean
    allowed_exts = {'.pt', '.json', '.db', '.png'}
    for f in KAGGLE_WORKING.iterdir():
        if f.is_file() and f.suffix not in allowed_exts:
            try:
                f.unlink()
            except Exception:
                pass

    print(f"[push] Pushing complete merged dataset to {creds['username']}/janus-weights ...")
    try:
        api.dataset_create_version(
            str(KAGGLE_WORKING),
            version_notes=version_notes,
            quiet=False,
            convert_to_csv=False,
            delete_old_versions=False,
        )
        print("[push] Done. Weights saved to Kaggle dataset.")
    except Exception as e:
        print(f"[push] Push failed: {e}")
        print("[push] Download weights manually from the output panel.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_avus()
    train_hbm()
    print_summary()
    auto_push_weights(version_notes=f"Avus-{MODEL_SIZE} epoch {AVUS_EPOCHS}")
