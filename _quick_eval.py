"""Quick eval — 5 tasks per domain, CPU-friendly."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from avus_eval import (
    load_model, load_tokenizer, generate, AvusEvaluator,
    build_arithmetic_tasks, build_logic_tasks, build_memory_tasks,
    build_code_tasks, build_planning_tasks, build_self_correction_tasks,
    save_eval_history,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading model...")
model, cfg = load_model("avus_1b_weights.pt", device)
encode, decode = load_tokenizer()
print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
print(f"Config: dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads}")

# 5 tasks per domain for speed
tasks = {
    "arithmetic":     build_arithmetic_tasks(5),
    "reasoning":      build_logic_tasks(5),
    "memory_recall":  build_memory_tasks(5),
    "code":           build_code_tasks(5),
    "planning":       build_planning_tasks(5),
    "self_reflection": build_self_correction_tasks(5),
}

evaluator = AvusEvaluator(model, encode, decode, device=device, verbose=True)
scores = evaluator.run(tasks)

save_eval_history(scores, "avus_1b_weights.pt")

print("\nSUMMARY")
print("-" * 40)
for domain, score in scores.items():
    bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
    print(f"  {domain:<20} {score:5.1f}%  {bar}")
