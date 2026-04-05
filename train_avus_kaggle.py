# """
train_avus_kaggle.py

Self-contained Avus training script for Kaggle Notebooks.

Setup (do once before running):

1. Upload avus_1b_weights.pt + config_avus_1b.json + model.py + avus_tokenizer.py
   as a Kaggle Dataset called "janus-avus-weights"
1. In your Kaggle Notebook:
- Accelerator: GPU T4 x2
- Add dataset: your "janus-avus-weights" dataset
- Add dataset: your Janus repo (or upload model.py manually)
1. Paste this entire script into a notebook cell and run.

After every epoch weights are saved back to /kaggle/working/avus_1b_weights.pt
At the end of the session download them from the Kaggle output panel,
re-upload to your "janus-avus-weights" dataset to persist for next session.

Or use the auto-push cell at the bottom to push via Kaggle API automatically.
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

# 0.  Paths (Kaggle layout)

# ─────────────────────────────────────────────

KAGGLE_INPUT    = "/kaggle/input"
KAGGLE_WORKING  = "/kaggle/working"
DATASET_NAME    = "janus-avus-weights"   # your Kaggle dataset slug

# Where weights live coming IN (read-only Kaggle input mount)

WEIGHTS_IN      = os.path.join(KAGGLE_INPUT, DATASET_NAME, "avus_1b_weights.pt")
CONFIG_IN       = os.path.join(KAGGLE_INPUT, DATASET_NAME, "config_avus_1b.json")
MODEL_IN        = os.path.join(KAGGLE_INPUT, DATASET_NAME, "model.py")

# Where weights go OUT (writeable, download from output panel)

WEIGHTS_OUT     = os.path.join(KAGGLE_WORKING, "avus_1b_weights.pt")
CONFIG_OUT      = os.path.join(KAGGLE_WORKING, "config_avus_1b.json")

# ─────────────────────────────────────────────

# 1.  Install tiktoken if missing

# ─────────────────────────────────────────────

try:
import tiktoken
except ImportError:
os.system("pip install tiktoken -q")
import tiktoken

# ─────────────────────────────────────────────

# 2.  Copy model.py to working dir & import

# ─────────────────────────────────────────────

import shutil

# Copy model.py from dataset input to working dir so we can import it

if os.path.exists(MODEL_IN):
shutil.copy(MODEL_IN, os.path.join(KAGGLE_WORKING, "model.py"))
print(f"Copied model.py from dataset.")
elif os.path.exists("/kaggle/input/janus-repo/model.py"):
shutil.copy("/kaggle/input/janus-repo/model.py",
os.path.join(KAGGLE_WORKING, "model.py"))
print("Copied model.py from repo dataset.")
else:
# Last resort: check current dir
assert os.path.exists("model.py"), (
"model.py not found. Add your Janus repo as a Kaggle dataset "
"or include model.py in the janus-avus-weights dataset."
)

if KAGGLE_WORKING not in sys.path:
sys.path.insert(0, KAGGLE_WORKING)

if "model" in sys.modules:
del sys.modules["model"]
import model as _m
importlib.reload(_m)
from model import C, Avus
print("Avus model loaded.")

# ─────────────────────────────────────────────

# 3.  Config

# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
"vocab_size":  50304,
"dim":         768,
"n_layers":    12,
"n_heads":     12,
"n_kv_heads":  4,
"max_seq_len": 512
}

# Load from dataset if available, else use default

if os.path.exists(CONFIG_IN):
with open(CONFIG_IN) as f:
cfg = json.load(f)
# Clamp max_seq_len to avoid OOM
cfg["max_seq_len"] = min(cfg.get("max_seq_len", 512), 512)
print(f"Config loaded from dataset: {cfg}")
else:
cfg = DEFAULT_CONFIG
print(f"Using default config: {cfg}")

# Save config to output so it travels with the weights

with open(CONFIG_OUT, "w") as f:
json.dump(cfg, f, indent=2)

# ─────────────────────────────────────────────

# 4.  AvusTokenizer  (inlined)

# ─────────────────────────────────────────────

class AvusTokenizer:
def **init**(self):
self._enc = tiktoken.get_encoding("gpt2")

```
def encode(self, text: str, allowed_special: Set[str] = None) -> List[int]:
    special = allowed_special or set()
    special.update({"<|startoftext|>", "<|endoftext|>",
                     "[JSON_START]", "[JSON_END]",
                     "[ACT_START]",  "[ACT_END]"})
    return self._enc.encode(text, allowed_special=special)

def decode(self, tokens: List[int]) -> str:
    valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
    try:
        return self._enc.decode(valid)
    except Exception:
        return ""
```

# ─────────────────────────────────────────────

# 5.  Grade3DGeneration  (3D curriculum)

# ─────────────────────────────────────────────

class Grade3DGeneration:
ADJECTIVES = [
"ancient", "mystical", "glowing", "rusty", "futuristic",
"organic", "sharp", "smooth", "rugged", "floating",
"massive", "tiny", "vibrant", "monochromatic", "crystalline"
]
OBJECTS = [
"rock", "tree", "crystal", "pillar", "gateway",
"chest", "barrel", "lantern", "statue", "ruin"
]
PRIMITIVES = ["box", "sphere", "cylinder", "torus"]
MATERIALS  = ["stone", "wood", "metal", "plastic"]

```
def __init__(self):
    self._tasks = [self._simple_object,
                   self._terrain_feature,
                   self._composite_object]

def _rc(self, lst):  return random.choice(lst)
def _ri(self, a, b): return random.randint(a, b)
def _rf(self, a, b): return random.uniform(a, b)

def _geo(self, p):
    if p == "box":
        return {"w": self._rf(.5,10), "h": self._rf(.5,10), "d": self._rf(.5,10)}
    if p == "sphere":
        return {"radius": self._rf(.5,5), "segments": self._ri(8,64)}
    if p == "cylinder":
        return {"radius": self._rf(.2,3), "height": self._rf(1,10),
                "segments": self._ri(8,64)}
    if p == "torus":
        return {"outer_radius": self._rf(1,5), "inner_radius": self._rf(.1,.9)}
    if p == "terrain":
        return {"grid_size": [self._ri(16,64), self._ri(16,64)],
                "height_scale": self._rf(.1,2), "octaves": self._ri(3,8)}
    if p == "sdf_composite":
        sdfs = []
        for _ in range(self._ri(2,4)):
            t  = self._rc(["sphere","box"])
            sp = {"center": [self._rf(-1,1)]*3}
            if t == "sphere": sp["radius"] = self._rf(.1,.8)
            else:             sp["size"]   = [self._rf(.2,1.5)]*3
            sdfs.append({"type":t,"params":sp,
                          "op":self._rc(["union","subtract","intersect"])})
        return {"sdfs":sdfs,"grid_resolution":self._ri(16,64)}

def _mat(self, m):
    base = {"resolution": [512,512], "roughness_scale": self._rf(.3,.9)}
    if m == "metal":  base["metallic_scale"] = self._rf(.8,1)
    else:             base["metallic_scale"] = 0.0
    return base

def _make(self, name, ptype, mtype):
    return {
        "object_name": name,
        "position":    [self._rf(-10,10)]*3,
        "rotation":    [self._rf(0,360)]*3,
        "scale":       [self._rf(.5,2)]*3,
        "geometry":    {"primitive_type": ptype,
                        "geometry_params": self._geo(ptype)},
        "material":    {"material_type": mtype,
                        "material_params": self._mat(mtype)},
    }

def _simple_object(self):
    name  = self._rc(self.OBJECTS)
    adj   = self._rc(self.ADJECTIVES)
    ptype = self._rc(self.PRIMITIVES)
    mtype = self._rc(self.MATERIALS)
    prompt = (f"Generate a {adj} {name} with a "
              f"{ptype}-like shape and a {mtype} material.")
    return prompt, self._make(name, ptype, mtype)

def _terrain_feature(self):
    name  = self._rc(["hill","mountain","crater","plateau"])
    adj   = self._rc(self.ADJECTIVES)
    mtype = self._rc(["stone","plastic"])
    prompt = (f"Create a {adj} {name} feature in a "
              f"landscape, made of {mtype}.")
    return prompt, self._make(name, "terrain", mtype)

def _composite_object(self):
    name  = self._rc(["complex statue","abstract sculpture","alien structure"])
    adj   = self._rc(self.ADJECTIVES)
    mtype = self._rc(self.MATERIALS)
    prompt = (f"Design a {adj} {name} using various "
              f"combined shapes and a {mtype} finish.")
    return prompt, self._make(name, "sdf_composite", mtype)

def generate_dataset(self, samples=10_000, seed=42):
    random.seed(seed)
    out = []
    for _ in range(samples):
        fn = self._rc(self._tasks)
        prompt, params = fn()
        out.append((prompt, json.dumps(params, indent=2)))
    return out
```

# ─────────────────────────────────────────────

# 6.  ScreenActionDataset  (vision curriculum)

# ─────────────────────────────────────────────

class ScreenActionDataset:
APPS    = ["Chrome","Firefox","Notepad","VS Code","File Explorer",
"Discord","Slack","Terminal","Excel","Word"]
BUTTONS = ["Submit","Cancel","OK","Close","Save","Login","Search",
"Next","Back","Download","Delete","Edit","Confirm"]
FIELDS  = ["username","password","email","search bar","message box",
"title field","URL bar","comment box"]
SCROLL_CTX = ["a long webpage","a list of search results",
"a file directory","a chat history","a code file"]
TEXT    = ["Hello, world!","search query","my username","a short note",
"the file name","an email address","a URL","1234"]
RESOLUTIONS = [(1920,1080),(1366,768),(2560,1440),(1280,720)]

```
def __init__(self):
    self._tasks = [
        self._click_button, self._double_click, self._right_click,
        self._type_field,   self._press_enter,  self._scroll,
        self._wait,         self._keyboard_shortcut,
    ]

def _rc(self, l): return random.choice(l)
def _ri(self, a, b): return random.randint(a, b)
def _rf(self, a, b): return round(random.uniform(a, b), 2)
def _coord(self):
    w, h = self._rc(self.RESOLUTIONS)
    return random.randint(10, w-10), random.randint(10, h-10)

def _click_button(self):
    x, y = self._coord()
    desc = (f"{self._rc(self.APPS)} is open. "
            f"A '{self._rc(self.BUTTONS)}' button is at ({x},{y}). Click it.")
    return desc, {"type":"click","x":x,"y":y,"button":"left"}

def _double_click(self):
    x, y = self._coord()
    item = self._rc(["a folder","a file","a shortcut","a cell"])
    desc = (f"File Explorer shows {item} at ({x},{y}). "
            f"Open it with a double-click.")
    return desc, {"type":"double_click","x":x,"y":y}

def _right_click(self):
    x, y   = self._coord()
    target = self._rc(["a file icon","the desktop","a folder"])
    desc   = (f"{target} is at ({x},{y}). "
              f"Right-click to open the context menu.")
    return desc, {"type":"right_click","x":x,"y":y}

def _type_field(self):
    x, y  = self._coord()
    field = self._rc(self.FIELDS)
    text  = self._rc(self.TEXT)
    desc  = (f"A {field} is focused at ({x},{y}). "
             f"Type: \"{text}\".")
    return desc, {"type":"type","text":text,"x":x,"y":y}

def _press_enter(self):
    ctx  = self._rc(["a form is complete","a search field is filled",
                      "a command is typed in terminal"])
    desc = f"{ctx}. Press Enter to confirm."
    return desc, {"type":"press_enter"}

def _scroll(self):
    direction = self._rc(["down","up"])
    amount    = self._ri(2, 8)
    ctx       = self._rc(self.SCROLL_CTX)
    desc      = (f"The screen shows {ctx}. "
                 f"Scroll {direction} {amount} times.")
    return desc, {"type":"scroll","direction":direction,"amount":amount}

def _wait(self):
    duration = self._rf(0.5, 3.0)
    reason   = self._rc(["a loading spinner is visible",
                          "a progress bar is running",
                          "a page is still loading"])
    desc = f"{reason}. Wait {duration} seconds."
    return desc, {"type":"wait","duration":duration}

def _keyboard_shortcut(self):
    shortcuts = [
        ("press Ctrl+C to copy",  0x43),
        ("press Ctrl+V to paste", 0x56),
        ("press Ctrl+Z to undo",  0x5A),
        ("press Ctrl+S to save",  0x53),
        ("press Escape to dismiss",0x1B),
    ]
    label, vk = self._rc(shortcuts)
    app  = self._rc(self.APPS)
    desc = f"{app} is in focus. {label}."
    return desc, {"type":"key","vk_code":vk,"label":label}

def generate_dataset(self, samples=10_000, seed=99):
    random.seed(seed)
    out = []
    for _ in range(samples):
        fn = self._rc(self._tasks)
        desc, action = fn()
        out.append((desc, json.dumps(action)))
    return out
```

# ─────────────────────────────────────────────

# 7.  CombinedDataset  (3D + screen-action)

# ─────────────────────────────────────────────

class CombinedTextDataset(Dataset):
"""
Tokenises both 3D curriculum and screen-action pairs.
3D uses [JSON_START]/[JSON_END] tags.
Screen-action uses [ACT_START]/[ACT_END] tags.
"""
def **init**(self, pairs_3d, pairs_sa, tokenizer, block_size):
self.block_size = block_size
self.data: List[List[int]] = []
special = {"<|startoftext|>","<|endoftext|>",
"[JSON_START]","[JSON_END]",
"[ACT_START]","[ACT_END]"}
pad_id = tokenizer.encode("<|endoftext|>", allowed_special=special)[0]

```
    # 3D pairs
    for prompt, js in pairs_3d:
        text = (f"<|startoftext|>{prompt} "
                f"[JSON_START] {js} [JSON_END]<|endoftext|>")
        self._add(tokenizer.encode(text, allowed_special=special),
                  pad_id, block_size)

    # Screen-action pairs
    for desc, act in pairs_sa:
        text = (f"<|startoftext|>{desc} "
                f"[ACT_START] {act} [ACT_END]<|endoftext|>")
        self._add(tokenizer.encode(text, allowed_special=special),
                  pad_id, block_size)

    if not self.data:
        raise ValueError("Dataset is empty.")
    print(f"Combined dataset: {len(self.data)} sequences "
          f"(block_size={block_size})")

def _add(self, tokens, pad_id, block_size):
    idx = 0
    while idx < len(tokens):
        chunk = tokens[idx: idx + block_size + 1]
        if len(chunk) == block_size + 1:
            self.data.append(chunk)
        elif len(chunk) > 1:
            chunk = chunk + [pad_id] * (block_size + 1 - len(chunk))
            self.data.append(chunk)
        idx += block_size

def __len__(self):  return len(self.data)

def __getitem__(self, idx):
    chunk = self.data[idx]
    return (torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:],  dtype=torch.long))
```

# ─────────────────────────────────────────────

# 8.  Training loop

# ─────────────────────────────────────────────

def train(epochs: int = 20, samples_per_curriculum: int = 10_000):
gc.collect()
torch.cuda.empty_cache()

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

# ── model ──
config = C(**cfg)
model  = Avus(config).to(device)
total  = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")

# ── load existing weights ──
if os.path.exists(WEIGHTS_IN):
    sd   = torch.load(WEIGHTS_IN, map_location=device)
    drop = [k for k in sd if any(x in k for x in
            ("attn.mask","attn.rope.c","attn.rope.s"))]
    for k in drop: del sd[k]
    model.load_state_dict(sd, strict=False)
    print(f"Loaded weights from {WEIGHTS_IN}  "
          f"(dropped {len(drop)} incompatible keys)")
elif os.path.exists(WEIGHTS_OUT):
    # Resume from previous epoch in same session
    sd   = torch.load(WEIGHTS_OUT, map_location=device)
    drop = [k for k in sd if any(x in k for x in
            ("attn.mask","attn.rope.c","attn.rope.s"))]
    for k in drop: del sd[k]
    model.load_state_dict(sd, strict=False)
    print(f"Resumed from {WEIGHTS_OUT}")
else:
    print("No existing weights -- training from scratch.")

# ── datasets ──
tokenizer = AvusTokenizer()
pairs_3d  = Grade3DGeneration().generate_dataset(samples_per_curriculum)
pairs_sa  = ScreenActionDataset().generate_dataset(samples_per_curriculum)
dataset   = CombinedTextDataset(pairs_3d, pairs_sa,
                                tokenizer, config.max_seq_len)
loader    = DataLoader(dataset, batch_size=1, shuffle=True,
                       num_workers=2, pin_memory=(device.type=="cuda"))

# ── optimiser ──
optimizer  = optim.AdamW(model.parameters(), lr=5e-6)
scaler     = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
best_loss  = float("inf")
no_improve = 0
current_lr = 5e-6

# ── epoch loop ──
model.train()
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    avg = epoch_loss / len(loader)
    print(f"Epoch {epoch:>3}/{epochs}  loss={avg:.4f}  lr={current_lr:.1e}")

    # dynamic LR
    if avg < best_loss:
        best_loss  = avg
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= 5 and current_lr > 1e-6:
        current_lr = 1e-6
        for g in optimizer.param_groups:
            g["lr"] = current_lr
        print(f"  → LR reduced to {current_lr:.1e}")
        no_improve = 0

    # save after every epoch
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"  → Weights saved to {WEIGHTS_OUT}")

print(f"\nDone. Download {WEIGHTS_OUT} from the Kaggle output panel.")
print("Then re-upload to your 'janus-avus-weights' dataset for next session.")
return model
```

# ─────────────────────────────────────────────

# 9.  Run

# ─────────────────────────────────────────────

if **name** == "**main**":
train(epochs=20, samples_per_curriculum=10_000)