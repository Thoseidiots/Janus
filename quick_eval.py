"""
quick_eval.py — fast local eval using a small fixed test set.
Runs on CPU in ~30 seconds.
"""
import sys, re, torch
import torch.nn.functional as F

WEIGHTS = r"C:\Users\legac\Downloads\Janus-workspace\avus_1b_weights (10).pt"

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {WEIGHTS}...")
ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
cfg_dict = ckpt.get("config", {})
print(f"Config: {cfg_dict}")

from avus import Avus, AvusConfig
cfg   = AvusConfig.from_dict(cfg_dict)
model = Avus(cfg)
sd    = {k.replace("module.", ""): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
model.load_state_dict(sd, strict=False)
model.eval()
print(f"Params: {model.count_parameters()/1e6:.1f}M")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
import tiktoken
enc = tiktoken.get_encoding("gpt2")
def encode(t): return enc.encode(t, allowed_special={"<|startoftext|>","<|endoftext|>"})
def decode(t):
    valid = [x for x in t if 0 <= x < enc.max_token_value]
    try: return enc.decode(valid)
    except: return ""

# ── Generate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def gen(prompt, max_new=20, temp=0.1, top_k=10):
    # Wrap with training delimiters so model knows where answer ends
    toks = encode("<|startoftext|>" + prompt)
    eos_id = encode("<|endoftext|>")[0]
    idx  = torch.tensor([toks])
    model.clear_cache()
    _, _ = model(idx, use_cache=True, cache_offset=0)
    out_toks = []
    for _ in range(max_new):
        last = idx[:, -1:]
        logits, _ = model(last, use_cache=True, cache_offset=idx.shape[1]-1)
        logits = logits[:, -1, :] / temp
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        tok_id = next_tok.item()
        # Stop at EOS or start-of-text (model thinks new example is starting)
        if tok_id == eos_id or tok_id == encode("<|startoftext|>")[0]:
            break
        out_toks.append(tok_id)
        idx = torch.cat([idx, next_tok], dim=1)
    model.clear_cache()
    return decode(out_toks)

# ── Test cases ────────────────────────────────────────────────────────────────
tests = [
    # (domain, prompt, expected)
    ("arithmetic", "Q: What is 3 + 4?\nA:", "7"),
    ("arithmetic", "Q: What is 12 + 5?\nA:", "17"),
    ("arithmetic", "Q: What is 9 * 3?\nA:", "27"),
    ("arithmetic", "Q: What is 50 - 8?\nA:", "42"),
    ("arithmetic", "Q: What is 6 * 7?\nA:", "42"),
    ("reasoning",  "All kind people are fast. Alice is kind. Is Alice fast?\nAnswer:", "yes"),
    ("reasoning",  "All smart people are honest. Bob is smart. Is Bob honest?\nAnswer:", "yes"),
    ("memory",     "Alex was born in Berlin and works as an engineer. Where was Alex born?\nAnswer:", "berlin"),
    ("memory",     "Jordan lives in Tokyo. Q: Where does Jordan live?\nA:", "tokyo"),
    ("screen_action", "Chrome is open. A 'Submit' button is at (847,392). Click it. [ACT_START]", '{"type":"click"'),
]

results = {}
print("\n" + "="*60)
for domain, prompt, expected in tests:
    out = gen(prompt)
    nums = re.findall(r'\d+', out)
    if expected.isdigit():
        ok = expected in nums
    else:
        ok = expected.lower() in out.lower()
    status = "✓" if ok else "✗"
    print(f"  [{status}] {prompt[:50]:<50} → {out[:30].strip()!r}")
    results[domain] = results.get(domain, []) + [ok]

print("\n" + "="*60)
print("SCORES:")
for domain, scores in results.items():
    pct = sum(scores)/len(scores)*100
    bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
    print(f"  {domain:<20} [{bar}] {pct:.0f}%")
overall = sum(v for s in results.values() for v in s) / sum(len(s) for s in results.values()) * 100
print(f"\n  {'OVERALL':<20} {overall:.0f}%")
print("="*60)
