#!/usr/bin/env python3
"""Train Avus models of various sizes"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import torch.optim as optim
import os
import json

@dataclass  
class C:
    vocab_size: int = 3000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 128

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(b) * a

class RoPE(nn.Module):
    def __init__(self, d, m=128):
        super().__init__()
        t = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        p = torch.arange(m).float()
        f = torch.outer(p, t)
        self.register_buffer("c", f.cos()[None, None, :, :])
        self.register_buffer("s", f.sin()[None, None, :, :])
    def rh(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    def forward(self, q, k, seq_len):
        c = torch.cat([self.c[:, :, :seq_len, :]] * 2, dim=-1)
        s = torch.cat([self.s[:, :, :seq_len, :]] * 2, dim=-1)
        return (q * c) + (self.rh(q) * s), (k * c) + (self.rh(k) * s)

class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv = c.n_heads, c.n_kv_heads
        self.d = c.dim // c.n_heads
        self.g = c.n_heads // c.n_kv_heads
        self.q = nn.Linear(c.dim, c.n_heads * self.d, bias=False)
        self.k = nn.Linear(c.dim, c.n_kv_heads * self.d, bias=False)
        self.v = nn.Linear(c.dim, c.n_kv_heads * self.d, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = RoPE(self.d, c.max_seq_len)
        m = torch.tril(torch.ones(c.max_seq_len, c.max_seq_len))
        self.register_buffer("mask", m.view(1, 1, c.max_seq_len, c.max_seq_len))
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = self.rope(q, k, T)
        if self.g > 1:
            k, v = k.repeat_interleave(self.g, dim=1), v.repeat_interleave(self.g, dim=1)
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d)
        s = s.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        return self.o(torch.matmul(F.softmax(s, -1), v).transpose(1, 2).contiguous().view(B, T, C))

class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.dim, 8 * c.dim, bias=False)
        self.act = SwiGLU()
        self.fc2 = nn.Linear(4 * c.dim, c.dim, bias=False)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn, self.ffn = Attn(c), FFN(c)
    def forward(self, x):
        return x + self.ffn(self.ln2(x + self.attn(self.ln1(x))))

class Avus(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.c = c or C()
        self.emb = nn.Embedding(self.c.vocab_size, self.c.dim)
        self.blocks = nn.ModuleList([Block(self.c) for _ in range(self.c.n_layers)])
        self.ln = RMSNorm(self.c.dim)
        self.head = nn.Linear(self.c.dim, self.c.vocab_size, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def train_model(name, cfg, output_dir, epochs=3, steps=10):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")
    
    model = Avus(cfg)
    params = model.count_params()
    print(f"Parameters: {params/1e6:.2f}M")
    
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    data = torch.randint(0, cfg.vocab_size, (3000,))
    
    model.train()
    history = []
    
    for epoch in range(epochs):
        total = 0
        for i in range(steps):
            idx = torch.randint(0, len(data)-33, (1,))
            x, y = data[idx:idx+32].unsqueeze(0), data[idx+1:idx+33].unsqueeze(0)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            total += loss.item()
        history.append({"epoch": epoch+1, "loss": total/steps})
        print(f"  Epoch {epoch+1}: {total/steps:.4f}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, f"avus_{name}.pt"))
    with open(os.path.join(output_dir, f"config_{name}.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(output_dir, f"history_{name}.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved!")
    return params, history[-1]["loss"]

if __name__ == "__main__":
    output_dir = "/tmp/avus_weights"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 33M model
    cfg_33m = C(vocab_size=3000, dim=512, n_layers=8, n_heads=8, n_kv_heads=4)
    p, l = train_model("33m", cfg_33m, output_dir, epochs=3, steps=10)
    results["33m"] = {"params": p, "loss": l}
    
    # 65M model
    cfg_65m = C(vocab_size=4000, dim=640, n_layers=10, n_heads=8, n_kv_heads=4)
    p, l = train_model("65m", cfg_65m, output_dir, epochs=3, steps=10)
    results["65m"] = {"params": p, "loss": l}
    
    # 100M model
    cfg_100m = C(vocab_size=5000, dim=768, n_layers=12, n_heads=12, n_kv_heads=4)
    p, l = train_model("100m", cfg_100m, output_dir, epochs=2, steps=5)
    results["100m"] = {"params": p, "loss": l}
    
    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL MODELS TRAINED!")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"{name}: {info['params']/1e6:.2f}M params, final loss: {info['loss']:.4f}")
