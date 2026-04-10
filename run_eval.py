import sys, torch, re, random
sys.path.insert(0, 'Janus-main')
from avus import Avus, AvusConfig
import tiktoken

ckpt = torch.load('avus_1b_weights.pt', map_location='cpu', weights_only=False)
cfg  = AvusConfig(dim=1920, n_layers=20, n_heads=16, n_kv_heads=8,
                  ffn_hidden=5120, max_seq_len=512, vocab_size=50304)
model = Avus(cfg)
model.load_state_dict(
    {k.replace('module.', ''): v
     for k, v in ckpt.get('model_state_dict', ckpt).items()},
    strict=False)
model.eval()
print(f"Model loaded. Epoch: {ckpt.get('epoch')}  Loss: {round(ckpt.get('loss', 0), 4)}")

enc = tiktoken.get_encoding('gpt2')

def gen(prompt, n=50, t=0.1):
    toks = enc.encode('<|startoftext|>' + prompt,
                      allowed_special={'<|startoftext|>'})
    idx  = torch.tensor([toks])
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=n, temperature=t, top_k=10)
    new = out[0][len(toks):].tolist()
    return enc.decode([x for x in new if 0 <= x < enc.max_token_value])

N = 10
scores = {}

# ── Arithmetic ────────────────────────────────────────────────────────────────
print("\n=== ARITHMETIC ===")
rng = random.Random(42)
hits = 0
for i in range(N):
    a, b = rng.randint(1, 30), rng.randint(1, 30)
    ans  = str(a + b)
    out  = gen(f"Calculate: {a} + {b}. Step 1: Identify the operation (+). "
               f"Step 2: Apply it. Result:")
    nums = re.findall(r'\d+', out)
    hit  = ans in nums
    hits += hit
    if i < 3:
        status = "OK" if hit else "MISS"
        print(f"  {a}+{b}={ans} | got: {out[:50].strip()} | {status}")
scores['arithmetic'] = hits / N * 100

# ── Reasoning ─────────────────────────────────────────────────────────────────
print("\n=== REASONING ===")
rng2  = random.Random(42)
props = ['smart', 'fast', 'kind', 'brave', 'honest']
names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
hits  = 0
for i in range(N):
    a       = rng2.choice(names)
    p1, p2  = rng2.sample(props, 2)
    out     = gen(f"Premise: All {p1} people are {p2}. "
                  f"{a} is {p1}. Is {a} {p2}? Answer:")
    hit     = 'yes' in out.lower()
    hits   += hit
    if i < 3:
        status = "OK" if hit else "MISS"
        print(f"  All {p1}->{p2}, {a} is {p1} | got: {out[:50].strip()} | {status}")
scores['reasoning'] = hits / N * 100

# ── Memory recall ─────────────────────────────────────────────────────────────
print("\n=== MEMORY RECALL ===")
rng3   = random.Random(42)
cities = ['Berlin', 'Tokyo', 'Lagos', 'Sydney', 'Oslo']
hits   = 0
for i in range(N):
    name = rng3.choice(names)
    city = rng3.choice(cities)
    out  = gen(f"Context: {name} was born in {city}. "
               f"Question: Where was {name} born? Answer:")
    hit  = city.lower() in out.lower()
    hits += hit
    if i < 3:
        status = "OK" if hit else "MISS"
        print(f"  {name} born in {city} | got: {out[:50].strip()} | {status}")
scores['memory_recall'] = hits / N * 100

# ── Self-correction ───────────────────────────────────────────────────────────
print("\n=== SELF-CORRECTION ===")
rng4 = random.Random(42)
hits = 0
for i in range(N):
    a, b    = rng4.randint(10, 50), rng4.randint(10, 50)
    correct = str(a + b)
    wrong   = str(a + b + rng4.choice([-2, -1, 1, 2]))
    out     = gen(f"Question: What is {a} + {b}? "
                  f"Initial answer: {wrong}. "
                  f"Wait, let me recheck. The correct answer is:")
    nums    = re.findall(r'\d+', out)
    hit     = correct in nums
    hits   += hit
    if i < 3:
        status = "OK" if hit else "MISS"
        print(f"  {a}+{b}={correct} (wrong={wrong}) | got: {out[:50].strip()} | {status}")
scores['self_correction'] = hits / N * 100

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== BASELINE SCORES (Epoch 1) ===")
for domain, pct in scores.items():
    bar    = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
    print(f"  {domain:<18} [{bar}] {pct:.0f}%")
overall = sum(scores.values()) / len(scores)
print(f"\n  Overall: {overall:.0f}%")
print("\nThis is the baseline. Every future epoch should score higher.")
