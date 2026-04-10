"""
pretokenize.py
==============
Pre-tokenizes all Janus training datasets and saves as a flat token tensor.

Run this once locally or on Kaggle before training.
The output file is uploaded to your janus-avus-weights dataset and
loaded directly by train_avus_kaggle.py — no re-generation each session.

Usage:
    # Generate and save
    python pretokenize.py --output tokens.pt --samples 10000

    # Preview what's in an existing file
    python pretokenize.py --inspect tokens.pt

Output:
    tokens.pt — dict with keys:
        "tokens"   : torch.LongTensor of shape (N,) — flat token stream
        "n_tokens" : int
        "vocab_size": int
        "sources"  : dict of source -> token count
        "created_at": str
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def get_tokenizer():
    try:
        import tiktoken
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "tiktoken", "-q"])
        import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    SPECIAL = {
        "<|startoftext|>", "<|endoftext|>",
        "[JSON_START]", "[JSON_END]",
        "[ACT_START]",  "[ACT_END]",
    }

    def encode(text: str):
        return enc.encode(text, allowed_special=SPECIAL)

    return encode


# ── Dataset generators ────────────────────────────────────────────────────────

def collect_texts(samples_per: int = 10_000) -> dict:
    """Collect raw text strings from all generators. Returns {source: [texts]}"""
    texts = {}

    # 1. Basic synthetic generators from train_avus_kaggle
    import random

    def rc(lst): return random.choice(lst)
    def ri(a, b): return random.randint(a, b)
    def rf(a, b): return round(random.uniform(a, b), 3)

    # 3D generation
    adjs  = ["ancient","glowing","rusty","futuristic","crystalline","massive","tiny"]
    objs  = ["rock","crystal","pillar","gateway","chest","statue","ruin","barrel"]
    prims = ["box","sphere","cylinder","torus"]
    mats  = ["stone","wood","metal","plastic"]
    t3d = []
    for _ in range(samples_per):
        params = {"object": rc(objs), "primitive": rc(prims), "material": rc(mats),
                  "scale": [rf(0.5,3)]*3, "roughness": rf(0.1,0.9)}
        t3d.append(f"<|startoftext|>Generate a {rc(adjs)} {rc(objs)} with "
                   f"{rc(prims)} shape and {rc(mats)} material. "
                   f"[JSON_START]{json.dumps(params)}[JSON_END]<|endoftext|>")
    texts["3d"] = t3d
    print(f"  3d:       {len(t3d):>8,} samples")

    # Screen actions
    apps = ["Chrome","VS Code","Terminal","File Explorer","Discord"]
    btns = ["Submit","Cancel","Save","Login","Search","Next","Delete"]
    tsa = []
    for _ in range(samples_per):
        x, y = ri(10,1910), ri(10,1070)
        action = {"type":"click","x":x,"y":y,"button":"left"}
        tsa.append(f"<|startoftext|>{rc(apps)} is open. "
                   f"A '{rc(btns)}' button is at ({x},{y}). Click it. "
                   f"[ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
    texts["screen_action"] = tsa
    print(f"  screen:   {len(tsa):>8,} samples")

    # Language
    topics = ["machine learning","neural networks","transformers","attention",
              "gradient descent","backpropagation","reinforcement learning",
              "computer vision","NLP","robotics"]
    templates = ["Explain {t}.", "What is {t}?", "How does {t} work?",
                 "Describe {t}.", "What are applications of {t}?"]
    tl = []
    for _ in range(samples_per):
        t = rc(topics)
        q = rc(templates).format(t=t)
        a = f"{t.capitalize()} is a fundamental AI concept involving data processing."
        tl.append(f"<|startoftext|>{q} {a}<|endoftext|>")
    texts["language"] = tl
    print(f"  language: {len(tl):>8,} samples")

    # Reasoning
    tr = []
    for _ in range(samples_per):
        a, b = ri(1,100), ri(1,100)
        op = rc(["+","-","*"])
        res = a+b if op=="+" else a-b if op=="-" else a*b
        tr.append(f"<|startoftext|>Calculate: {a} {op} {b}. "
                  f"Step 1: operation is {op}. Step 2: apply. "
                  f"Result: {res}<|endoftext|>")
    texts["reasoning"] = tr
    print(f"  reasoning:{len(tr):>8,} samples")

    # 2. Deep training data
    try:
        from deep_training_data import CombinedDeepDataset
        deep = CombinedDeepDataset()
        td = deep.generate_curriculum(n_per_generator=samples_per // 6)
        texts["deep"] = td
        print(f"  deep:     {len(td):>8,} samples")
    except ImportError:
        print("  deep:     skipped (deep_training_data.py not found)")

    # 3. Procedural dataset
    try:
        from procedural_dataset import ProceduralDataset
        proc = ProceduralDataset()
        tp = proc.generate(n=samples_per, difficulty=3, seed=42)
        texts["procedural"] = tp
        print(f"  procedural:{len(tp):>7,} samples")
    except ImportError:
        print("  procedural: skipped (procedural_dataset.py not found)")

    return texts


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_all(texts: dict, encode) -> tuple[torch.Tensor, dict]:
    """Tokenize all texts and concatenate into a flat token stream."""
    all_tokens = []
    sources    = {}
    total      = sum(len(v) for v in texts.values())
    done       = 0

    for source, samples in texts.items():
        source_tokens = 0
        for text in samples:
            toks = encode(text)
            all_tokens.extend(toks)
            source_tokens += len(toks)
            done += 1
            if done % 5000 == 0:
                pct = done / total * 100
                print(f"  tokenizing... {done:,}/{total:,} ({pct:.0f}%)")
        sources[source] = source_tokens
        print(f"  {source}: {source_tokens:,} tokens")

    flat = torch.tensor(all_tokens, dtype=torch.long)
    return flat, sources


# ── Main ──────────────────────────────────────────────────────────────────────

def build(output: str, samples_per: int):
    print(f"\nPre-tokenizing Janus training data")
    print(f"Samples per generator: {samples_per:,}")
    print(f"Output: {output}\n")

    encode = get_tokenizer()

    print("Generating texts...")
    t0    = time.time()
    texts = collect_texts(samples_per)

    print(f"\nTokenizing {sum(len(v) for v in texts.values()):,} samples...")
    flat, sources = tokenize_all(texts, encode)

    elapsed = time.time() - t0
    size_mb = flat.element_size() * flat.numel() / 1e6

    print(f"\nSaving to {output}...")
    torch.save({
        "tokens":     flat,
        "n_tokens":   flat.numel(),
        "vocab_size": 50304,
        "sources":    sources,
        "samples_per": samples_per,
        "created_at": datetime.now().isoformat(),
    }, output)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Total tokens: {flat.numel():,}")
    print(f"  File size:    {size_mb:.1f} MB")
    print(f"  Sources:      {list(sources.keys())}")
    print(f"\nUpload {output} to your janus-avus-weights Kaggle dataset.")
    print("The training script will load it automatically.")


def inspect(path: str):
    print(f"\nInspecting {path}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Total tokens:  {data['n_tokens']:,}")
    print(f"  Vocab size:    {data['vocab_size']:,}")
    print(f"  Created:       {data.get('created_at', 'unknown')}")
    print(f"  Samples/gen:   {data.get('samples_per', 'unknown')}")
    print(f"  Sources:")
    for src, count in data.get("sources", {}).items():
        print(f"    {src:<20} {count:>12,} tokens")
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  File size:     {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize Janus training data")
    parser.add_argument("--output",  default="tokens.pt",
                        help="Output file path (default: tokens.pt)")
    parser.add_argument("--samples", type=int, default=10_000,
                        help="Samples per generator (default: 10000)")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Inspect an existing tokens.pt file")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
    else:
        build(args.output, args.samples)
