"""
train_avus_lightning.py
=======================
Self-contained Avus 3D curriculum training script for Lightning AI.

What this does:
  1. Clones the Janus repo (or uses existing /teamspace/studios/this_studio/Janus)
  2. Loads/creates config_avus_1b.json  (vocab=50304, dim=768, 12 layers, max_seq_len=512)
  3. Generates 10,000 (prompt, JSON) pairs via Grade3DGeneration
  4. Trains Avus for 20 epochs with dynamic LR, batch_size=1, mixed precision
  5. Saves avus_1b_weights.pt after every epoch

Lightning AI notes:
  - Run in a Studio with at least a T4 (16 GB VRAM). An A10G is ideal.
  - All files land in /teamspace/studios/this_studio/Janus/
  - avus_1b_weights.pt persists between sessions automatically (it's in the Studio volume).
  - No Google Drive needed.

Usage:
  python train_avus_lightning.py
"""

import os
import sys
import json
import random
import importlib
import gc
from typing import List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────
REPO_URL   = "https://github.com/Thoseidiots/Janus.git"
WORK_DIR   = "/teamspace/studios/this_studio/Janus"
WEIGHTS    = os.path.join(WORK_DIR, "avus_1b_weights.pt")
CONFIG     = os.path.join(WORK_DIR, "config_avus_1b.json")

# ─────────────────────────────────────────────
# 1.  Clone repo if needed & set working dir
# ─────────────────────────────────────────────
if not os.path.exists(WORK_DIR):
    print(f"Cloning Janus repo into {WORK_DIR} ...")
    os.system(f"git clone {REPO_URL} {WORK_DIR}")
else:
    print(f"Repo already present at {WORK_DIR}")

os.chdir(WORK_DIR)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print(f"Working directory: {os.getcwd()}")

# ─────────────────────────────────────────────
# 2.  Install tiktoken if missing
# ─────────────────────────────────────────────
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken ...")
    os.system("pip install tiktoken -q")
    import tiktoken

# ─────────────────────────────────────────────
# 3.  Write / verify config_avus_1b.json
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "vocab_size": 50304,
    "dim":        768,
    "n_layers":   12,
    "n_heads":    12,
    "n_kv_heads": 4,
    "max_seq_len": 512
}

if not os.path.exists(CONFIG):
    with open(CONFIG, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Created {CONFIG}")
else:
    with open(CONFIG) as f:
        saved = json.load(f)
    # Force max_seq_len to 512 to avoid OOM
    if saved.get("max_seq_len", 0) > 512:
        saved["max_seq_len"] = 512
        with open(CONFIG, "w") as f:
            json.dump(saved, f, indent=2)
        print("Clamped max_seq_len to 512 to avoid OOM.")
    print(f"Config: {saved}")

# ─────────────────────────────────────────────
# 4.  Reload model.py  (gets C and Avus)
# ─────────────────────────────────────────────
if "model" in sys.modules:
    del sys.modules["model"]
import model as _m
importlib.reload(_m)
from model import C, Avus

print(f"Avus loaded.  Config fields: {list(DEFAULT_CONFIG.keys())}")

# ─────────────────────────────────────────────
# 5.  AvusTokenizer  (inlined – no reload needed)
# ─────────────────────────────────────────────
class AvusTokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, allowed_special: Set[str] = None) -> List[int]:
        special = allowed_special or set()
        special.update({"<|startoftext|>", "<|endoftext|>", "[JSON_START]", "[JSON_END]"})
        return self._enc.encode(text, allowed_special=special)

    def decode(self, tokens: List[int]) -> str:
        valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
        try:
            return self._enc.decode(valid)
        except Exception:
            return ""

# ─────────────────────────────────────────────
# 6.  Grade3DGeneration  (curriculum dataset)
# ─────────────────────────────────────────────
class Grade3DGeneration:
    """
    Generates (natural-language prompt, JSON parameter string) pairs
    for training Avus to output structured 3D generation parameters.
    """

    ADJECTIVES = [
        "ancient", "mystical", "glowing", "rusty", "futuristic",
        "organic", "sharp", "smooth", "rugged", "floating",
        "massive", "tiny", "vibrant", "monochromatic", "crystalline"
    ]
    OBJECTS = [
        "rock", "tree", "crystal", "pillar", "gateway",
        "chest", "barrel", "lantern", "statue", "ruin"
    ]
    PRIMITIVES  = ["box", "sphere", "cylinder", "torus"]
    MATERIALS   = ["stone", "wood", "metal", "plastic"]

    def __init__(self):
        self._tasks = [
            self._simple_object,
            self._terrain_feature,
            self._composite_object,
        ]

    # ── helpers ──────────────────────────────
    def _rc(self, lst):  return random.choice(lst)
    def _ri(self, a, b): return random.randint(a, b)
    def _rf(self, a, b): return random.uniform(a, b)

    def _geo(self, p):
        if p == "box":
            return {"w": self._rf(.5,10), "h": self._rf(.5,10), "d": self._rf(.5,10),
                    "bevel": self._rf(0,.2) if random.random()<.3 else 0,
                    "subdiv": self._ri(0,3) if random.random()<.5 else 0}
        if p == "sphere":
            return {"radius": self._rf(.5,5), "segments": self._ri(8,64), "rings": self._ri(8,64)}
        if p == "cylinder":
            return {"radius": self._rf(.2,3), "height": self._rf(1,10),
                    "segments": self._ri(8,64), "cap_segments": self._ri(1,5)}
        if p == "torus":
            return {"outer_radius": self._rf(1,5), "inner_radius": self._rf(.1,.9),
                    "radial_segments": self._ri(8,64), "tubular_segments": self._ri(8,64)}
        if p == "terrain":
            return {"grid_size": [self._ri(16,64), self._ri(16,64)],
                    "height_scale": self._rf(.1,2), "scale": self._rf(.05,.5),
                    "octaves": self._ri(3,8), "iso_level": self._rf(-.5,.5)}
        if p == "sdf_composite":
            sdfs = []
            for _ in range(self._ri(2,4)):
                t = self._rc(["sphere","box"])
                sp = {"center": [self._rf(-1,1),self._rf(-1,1),self._rf(-1,1)]}
                if t=="sphere": sp["radius"] = self._rf(.1,.8)
                else:           sp["size"]   = [self._rf(.2,1.5)]*3
                sdfs.append({"type":t,"params":sp,"op":self._rc(["union","subtract","intersect"])})
            return {"sdfs":sdfs,"grid_resolution":self._ri(16,64),"threshold":self._rf(-.1,.1)}

    def _mat(self, m):
        res = [self._ri(128,1024), self._ri(128,1024)]
        if m=="stone":
            return {"resolution":res,"roughness_scale":self._rf(.5,.9),
                    "metallic_scale":0.0,"normal_strength":self._rf(.5,1.5),
                    "color_variation":self._rf(.05,.2)}
        if m=="wood":
            return {"resolution":res,"roughness_scale":self._rf(.6,.8),
                    "metallic_scale":0.0,"normal_strength":self._rf(.6,1.2),
                    "grain_frequency":self._rf(.01,.1)}
        if m=="metal":
            return {"resolution":res,"roughness_scale":self._rf(.1,.5),
                    "metallic_scale":self._rf(.8,1),"normal_strength":self._rf(.3,.8),
                    "polish":self._rf(.1,.5)}
        if m=="plastic":
            return {"resolution":res,"roughness_scale":self._rf(.3,.7),
                    "metallic_scale":0.0,"normal_strength":self._rf(.1,.5),
                    "color":[self._ri(0,255),self._ri(0,255),self._ri(0,255)]}

    def _make(self, name, ptype, mtype):
        return {
            "object_name": name,
            "position":    [self._rf(-10,10), self._rf(0,10), self._rf(-10,10)],
            "rotation":    [self._rf(0,360)]*3,
            "scale":       [self._rf(.5,2)]*3,
            "geometry":    {"primitive_type": ptype, "geometry_params": self._geo(ptype)},
            "material":    {"material_type":  mtype, "material_params": self._mat(mtype)},
        }

    # ── task generators ───────────────────────
    def _simple_object(self):
        name  = self._rc(self.OBJECTS)
        adj   = self._rc(self.ADJECTIVES)
        ptype = self._rc(self.PRIMITIVES)
        mtype = self._rc(self.MATERIALS)
        prompt = f"Generate a {adj} {name} with a {ptype}-like shape and a {mtype} material."
        return prompt, self._make(name, ptype, mtype)

    def _terrain_feature(self):
        name  = self._rc(["hill","mountain","crater","plateau"])
        adj   = self._rc(self.ADJECTIVES)
        mtype = self._rc(["stone","plastic"])
        prompt = f"Create a {adj} {name} feature in a landscape, made of {mtype}."
        return prompt, self._make(name, "terrain", mtype)

    def _composite_object(self):
        name  = self._rc(["complex statue","abstract sculpture","alien structure"])
        adj   = self._rc(self.ADJECTIVES)
        mtype = self._rc(self.MATERIALS)
        prompt = f"Design a {adj} {name} using various combined shapes and a {mtype} finish."
        return prompt, self._make(name, "sdf_composite", mtype)

    def generate_dataset(self, samples=10000, seed=42):
        random.seed(seed)
        out = []
        for _ in range(samples):
            fn = self._rc(self._tasks)
            prompt, params = fn()
            out.append((prompt, json.dumps(params, indent=2)))
        return out


# ─────────────────────────────────────────────
# 7.  StructuredTextDataset
# ─────────────────────────────────────────────
class StructuredTextDataset(Dataset):
    def __init__(self, pairs, tokenizer: AvusTokenizer, block_size: int):
        self.block_size = block_size
        self.data: List[List[int]] = []
        special = {"<|startoftext|>","<|endoftext|>","[JSON_START]","[JSON_END]"}
        pad_id  = tokenizer.encode("<|endoftext|>", allowed_special=special)[0]

        for prompt, js in pairs:
            text   = f"<|startoftext|>{prompt} [JSON_START] {js} [JSON_END]<|endoftext|>"
            tokens = tokenizer.encode(text, allowed_special=special)

            idx = 0
            while idx < len(tokens):
                chunk = tokens[idx : idx + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.data.append(chunk)
                elif len(chunk) > 1:
                    chunk = chunk + [pad_id] * (block_size + 1 - len(chunk))
                    self.data.append(chunk)
                idx += block_size

        if not self.data:
            raise ValueError("Dataset is empty – check block_size vs token lengths.")

        print(f"Dataset: {len(self.data)} sequences of length {block_size}")

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


# ─────────────────────────────────────────────
# 8.  Training loop
# ─────────────────────────────────────────────
def train(epochs: int = 20):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB")

    # ── config & model ──
    with open(CONFIG) as f:
        cfg = json.load(f)
    config = C(**cfg)
    print(f"Config: {cfg}")

    model = Avus(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ── load existing weights ──
    if os.path.exists(WEIGHTS):
        sd = torch.load(WEIGHTS, map_location=device)
        # Remove keys that break when max_seq_len changes
        drop = [k for k in sd if any(x in k for x in ("attn.mask","attn.rope.c","attn.rope.s"))]
        for k in drop: del sd[k]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights from {WEIGHTS}  (dropped {len(drop)} shape-incompatible keys)")
    else:
        print("No existing weights – training from scratch.")

    # ── tokenizer & dataset ──
    tokenizer    = AvusTokenizer()
    curriculum   = Grade3DGeneration()
    pairs        = curriculum.generate_dataset(samples=10_000)
    dataset      = StructuredTextDataset(pairs, tokenizer, config.max_seq_len)
    num_workers  = min(os.cpu_count() or 1, 4)
    dataloader   = DataLoader(dataset, batch_size=1, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # ── optimizer & scaler ──
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    best_loss      = float("inf")
    no_improve     = 0
    patience       = 5
    current_lr     = 5e-6

    # ── epoch loop ──
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:>3}/{epochs}  avg_loss={avg:.4f}  lr={current_lr:.1e}")

        # ── dynamic LR ──
        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and current_lr > 1e-6:
            current_lr = 1e-6
            for g in optimizer.param_groups:
                g["lr"] = current_lr
            print(f"  → LR reduced to {current_lr:.1e}")
            no_improve = 0

        # ── save every epoch ──
        torch.save(model.state_dict(), WEIGHTS)

    print(f"\nTraining complete.  Weights saved to {WEIGHTS}")
    return model


# ─────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train(epochs=20)
