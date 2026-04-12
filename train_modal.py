"""
train_modal.py
==============
Modal training script for Avus — runs on A10G (24GB VRAM) with no session limits.

Cost: ~$1.10/hr on A10G. A full epoch (~12hrs) costs ~$13.
No CPU offload needed — everything fits in 24GB VRAM.

Setup:
    pip install modal
    python3 -m modal setup   # authenticate once

Run:
    modal run train_modal.py              # train with defaults
    modal run train_modal.py --detach     # run in background, check later

Check status:
    modal app list
    modal app logs <app-id>

The trained weights are saved to a Modal Volume and can be downloaded:
    modal volume get janus-weights avus_1b_weights.pt ./avus_1b_weights.pt
"""

import modal
import os
import sys
from pathlib import Path

# ── Modal app definition ──────────────────────────────────────────────────────

app = modal.App("janus-avus-training")

# Persistent volume for weights — survives between runs
volume = modal.Volume.from_name("janus-weights", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "tiktoken",
        "matplotlib",
        "numpy",
    )
    .add_local_python_source(
        "avus",
        "avus_inference",
        "avus_brain",
        "avus_tokenizer",
        "skill_curriculum",
        "deep_training_data",
        "procedural_dataset",
        "gradient_battery",
        "holographic_brain_memory",
    )
)

# ── Training config ───────────────────────────────────────────────────────────

MODEL_DIM      = 1920
MODEL_LAYERS   = 20
MODEL_HEADS    = 16
MODEL_KV_HEADS = 8
MODEL_FFN      = 5120
SEQ_LEN        = 512    # full 512 — A10G has 24GB, no need to reduce
BATCH_SIZE     = 1      # reduced to fit optimizer states in 24GB
GRAD_ACCUM     = 32     # effective batch = 32
EPOCHS         = 20
SAMPLES_PER    = 10_000
WEIGHTS_PATH   = "/weights/avus_1b_weights.pt"
MIDCKPT_PATH   = "/weights/avus_1b_midepoch.pt"
SKILL_PATH     = "/weights/skill_state.json"
BATTERY_PATH   = "/weights/gradient_battery.pt"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/weights": volume},
)
def train():
    import json
    import math
    import random
    import shutil
    import time
    import gc

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Files are added via add_local_python_source — importable directly
    sys.path.insert(0, "/root")

    from avus import Avus, AvusConfig
    from skill_curriculum import SkillTree, CurriculumSampler

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = AvusConfig(
        vocab_size=50304,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        n_heads=MODEL_HEADS,
        n_kv_heads=MODEL_KV_HEADS,
        ffn_hidden=MODEL_FFN,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        eps=1e-5,
    )
    print(f"Config: dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"kv={cfg.n_kv_heads} ffn={cfg.ffn_hidden} seq={cfg.max_seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Avus(cfg).to(device)
    total = model.count_parameters()
    print(f"Parameters: {total/1e9:.2f}B ({total:,})")
    print(f"VRAM estimate: {total*2/1e9:.1f}GB fp16")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    resume_step = 0

    if os.path.exists(WEIGHTS_PATH.replace('weights', 'best')):
        best_path = WEIGHTS_PATH.replace('weights', 'best')
        ckpt = torch.load(best_path, map_location=device)
        sd   = {k.replace("module.", ""): v
                for k, v in ckpt.get("model_state_dict", ckpt).items()}
        model.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
        print(f"Resumed from BEST checkpoint {best_path} "
              f"(epoch {start_epoch}, loss={ckpt.get('loss', '?'):.4f})")
    elif os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        sd   = {k.replace("module.", ""): v
                for k, v in ckpt.get("model_state_dict", ckpt).items()}
        model.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
        print(f"Resumed from {WEIGHTS_PATH} (epoch {start_epoch})")
    else:
        print("Training from scratch")

    if os.path.exists(MIDCKPT_PATH):
        mid = torch.load(MIDCKPT_PATH, map_location=device)
        if mid.get("epoch", -1) == start_epoch and mid.get("step", 0) > 0:
            sd = {k.replace("module.", ""): v
                  for k, v in mid.get("model_state_dict", {}).items()}
            model.load_state_dict(sd, strict=False)
            resume_step = mid["step"] + 1
            print(f"Mid-epoch resume: step {mid['step']} "
                  f"loss={mid.get('loss', 0):.4f}")

    # ── Skill curriculum ──────────────────────────────────────────────────────
    skill_tree = SkillTree()
    if os.path.exists(SKILL_PATH):
        skill_tree.load(SKILL_PATH)
        print("Skill state loaded")
    sampler = CurriculumSampler(skill_tree)

    # Gradient battery — disabled for Modal (continuous runs don't need cross-session accumulation)
    # Enable on Kaggle where sessions are fragmented
    battery = None
    print("[train] Gradient battery disabled (Modal continuous run)")

    # ── Data ──────────────────────────────────────────────────────────────────
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    SPECIAL = {"<|startoftext|>", "<|endoftext|>",
               "[JSON_START]", "[JSON_END]", "[ACT_START]", "[ACT_END]"}

    def encode(text):
        return enc.encode(text, allowed_special=SPECIAL)

    # Try pre-tokenized first
    pretok = "/weights/tokens.pt"
    if os.path.exists(pretok):
        print(f"Loading pre-tokenized data from {pretok}")
        data = torch.load(pretok, map_location="cpu", weights_only=False)
        tokens = data["tokens"]
        print(f"{tokens.numel():,} tokens loaded")
    else:
        print("Generating training data...")
        try:
            from deep_training_data import CombinedDeepDataset
            from procedural_dataset import ProceduralDataset
            texts = []
            texts += CombinedDeepDataset().generate_curriculum(SAMPLES_PER // 11)
            texts += ProceduralDataset().generate(SAMPLES_PER, difficulty=3, seed=42)
            random.shuffle(texts)
        except ImportError:
            # Minimal fallback
            texts = []
            for _ in range(SAMPLES_PER):
                a, b = random.randint(1,100), random.randint(1,100)
                texts.append(f"<|startoftext|>{a} + {b} = {a+b}<|endoftext|>")

        all_toks = []
        for t in texts:
            all_toks.extend(encode(t))
        tokens = torch.tensor(all_toks, dtype=torch.long)
        print(f"{tokens.numel():,} tokens generated")

    # Build dataset
    pad_id = encode("<|endoftext|>")[0]
    chunks = []
    for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN):
        chunk = tokens[i:i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)
    print(f"{len(chunks):,} training chunks")

    class _DS(Dataset):
        def __len__(self): return len(chunks)
        def __getitem__(self, i): return chunks[i][:-1], chunks[i][1:]

    loader = DataLoader(_DS(), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)

    # ── Optimizer — GPU resident, no CPU offload ───────────────────────────────
    # Enable gradient checkpointing to reduce activation memory
    # Use the AvusBlock.forward function directly — avoids recursion
    from torch.utils.checkpoint import checkpoint as _ckpt
    from avus import AvusBlock
    _orig_block_forward = AvusBlock.forward

    def _ckpt_block_forward(self, x, use_cache=False, cache_offset=0):
        return _ckpt(_orig_block_forward, self, x, use_cache, cache_offset,
                     use_reentrant=False)

    AvusBlock.forward = _ckpt_block_forward
    print(f"Gradient checkpointing enabled across {len(model.blocks)} blocks")
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2
             and not any(x in n for x in ["tok_emb", "ln_", "norm"])]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and (p.dim() < 2
                or any(x in n for x in ["tok_emb", "ln_", "norm"]))]

    base_lr = 3e-4  # Fixed LR — sqrt scaling was too aggressive, caused NaN collapse
    # Keep model in bf16 to save VRAM
    model = model.to(torch.bfloat16)

    optimizer = optim.AdamW([
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=base_lr, betas=(0.9, 0.95), foreach=False, fused=False)

    # bf16 training — no GradScaler needed
    total_steps  = len(loader) * EPOCHS
    warmup_steps = max(200, total_steps // 50)
    print(f"base_lr={base_lr:.2e} warmup={warmup_steps} total_steps={total_steps}")

    def focal_loss(logits, targets, gamma=2.0):
        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()

    def cosine_lr(step, warmup, total, max_lr, min_lr):
        if step < warmup: return max_lr * step / max(warmup, 1)
        if step >= total: return min_lr
        p = (step - warmup) / max(total - warmup, 1)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    best_loss   = float("inf")
    SAVE_EVERY  = 500
    BATTERY_DISCHARGE = 2000
    clip_count  = 0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            if step < resume_step:
                global_step += 1
                continue

            x, y = x.to(device), y.to(device)
            lr = cosine_lr(global_step, warmup_steps, total_steps, base_lr, base_lr/10)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = focal_loss(logits.view(-1, logits.size(-1)), y.view(-1))

            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if gn > 1.0:
                    clip_count += 1

                # NaN guard
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[FATAL] Loss NaN at step {step}. Best={best_loss:.4f}. Stopping.")
                    return

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss  += loss.item()
            global_step += 1

            # Mid-epoch checkpoint
            if global_step % SAVE_EVERY == 0:
                torch.save({
                    "epoch": epoch, "step": step,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "loss": epoch_loss / (step + 1),
                }, MIDCKPT_PATH)
                skill_tree.save(SKILL_PATH)
                if battery:
                    battery.save()
                volume.commit()
                print(f"  [ckpt] step {step} saved")

            if step % 100 == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                domain = sampler.next_domain()
                conf   = max(0.0, min(1.0, 1.0 - avg / 10.0))
                skill_tree.update(domain, conf)
                bat_str = f" | battery={battery._meta['steps_charged']}steps" if battery else ""
                print(f"  step {step}/{len(loader)} loss={avg:.4f} "
                      f"lr={lr:.2e} t={elapsed:.0f}s "
                      f"| {domain} conf={conf:.2f} clips={clip_count}{bat_str}")

        avg_loss = epoch_loss / max(1, len(loader) - resume_step)
        resume_step = 0
        print(f"\nEpoch {epoch+1} complete — loss={avg_loss:.4f}")

        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss":             avg_loss,
        }, WEIGHTS_PATH)

        skill_tree.save(SKILL_PATH)
        try:
            skill_tree.plot("/weights/skill_chart.png")
        except Exception:
            pass

        if avg_loss < best_loss:
            best_loss = avg_loss
            shutil.copy(WEIGHTS_PATH, "/weights/avus_1b_best.pt")

        volume.commit()
        print(f"Weights committed to volume. Best loss: {best_loss:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print("Download weights:")
    print("  modal volume get janus-weights avus_1b_weights.pt .")


@app.function(
    image=image,
    volumes={"/weights": volume},
)
def list_weights():
    """List all files in the weights volume."""
    import os
    for f in sorted(os.listdir("/weights")):
        size = os.path.getsize(f"/weights/{f}") / 1e6
        print(f"  {f:<40} {size:.1f} MB")


@app.local_entrypoint()
def main():
    print("Starting Janus Avus training on Modal A10G...")
    print("Cost: ~$1.10/hr | No session limits | Weights persist in volume")
    print()
    train.remote()
