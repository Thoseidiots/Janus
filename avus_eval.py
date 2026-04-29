"""
avus_eval.py
============
Evaluation suite for Avus. Measures actual reasoning ability, not just loss.

Six domains, fixed eval sets, deterministic scoring.
Run after every training epoch to track real progress.

Scores feed back into the skill tree — low score = high marginal value.

Usage:
    python avus_eval.py --weights avus_1b_weights.pt
    python avus_eval.py --weights avus_1b_weights.pt --update-skills skill_state.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: str = "cpu"):
    from avus import Avus, AvusConfig
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if not cfg_dict:
        # Infer from weight shapes
        cfg_dict = {
            "vocab_size": 50304, "dim": 1920, "n_layers": 20,
            "n_heads": 16, "n_kv_heads": 8, "ffn_hidden": 5120,
            "max_seq_len": 512,
        }
    cfg = AvusConfig.from_dict(cfg_dict)
    model = Avus(cfg)
    sd = {k.replace("module.", ""): v
          for k, v in ckpt.get("model_state_dict", ckpt).items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


def load_tokenizer():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    SPECIAL = {"<|startoftext|>", "<|endoftext|>"}
    def encode(text): return enc.encode(text, allowed_special=SPECIAL)
    def decode(tokens):
        valid = [t for t in tokens if 0 <= t < enc.max_token_value]
        try: return enc.decode(valid)
        except Exception: return ""
    return encode, decode


def generate(model, encode, decode, prompt: str,
             max_tokens: int = 60, temperature: float = 0.3,
             top_k: int = 20, device: str = "cpu") -> str:
    tokens = encode("<|startoftext|>" + prompt)
    eos_id = encode("<|endoftext|>")[0]
    sot_id = encode("<|startoftext|>")[0]
    idx = torch.tensor([tokens], device=device)
    out_toks = []
    with torch.no_grad():
        # Process prompt with cache
        model.clear_cache()
        _, _ = model(idx, use_cache=True, cache_offset=0)
        for step in range(max_tokens):
            last = idx[:, -1:]
            logits, _ = model(last, use_cache=True,
                              cache_offset=idx.shape[1] - 1)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tok_id = next_tok.item()
            # Stop at EOS or new SOT (model thinks new example starting)
            if tok_id == eos_id or tok_id == sot_id:
                break
            out_toks.append(tok_id)
            idx = torch.cat([idx, next_tok], dim=1)
    model.clear_cache()
    return decode(out_toks)


# ── Eval tasks ────────────────────────────────────────────────────────────────

class EvalTask:
    """A single evaluation task with prompt, expected answer, and scorer."""
    def __init__(self, prompt: str, expected: str, domain: str,
                 difficulty: int = 1):
        self.prompt    = prompt
        self.expected  = expected
        self.domain    = domain
        self.difficulty = difficulty

    def score(self, output: str) -> float:
        """Returns 0.0 or 1.0. Override for partial credit."""
        return 1.0 if self.expected.lower() in output.lower() else 0.0


class NumericTask(EvalTask):
    """Scores based on whether the correct number appears in output."""
    def score(self, output: str) -> float:
        numbers = re.findall(r'-?\d+\.?\d*', output)
        return 1.0 if self.expected in numbers else 0.0


class PartialTask(EvalTask):
    """Partial credit: score = fraction of expected keywords found."""
    def __init__(self, prompt, expected_keywords: List[str], domain, difficulty=1):
        super().__init__(prompt, " ".join(expected_keywords), domain, difficulty)
        self.keywords = expected_keywords

    def score(self, output: str) -> float:
        out_lower = output.lower()
        found = sum(1 for kw in self.keywords if kw.lower() in out_lower)
        return found / len(self.keywords)


# ── Fixed eval sets (seeded, never changes) ───────────────────────────────────

def build_arithmetic_tasks(n: int = 20, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            ans = str(a + b)
        elif op == "-":
            a, b = rng.randint(10, 99), rng.randint(1, 9)
            ans = str(a - b)
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            ans = str(a * b)
        # Match the training format: direct Q→A
        prompt = f"Q: What is {a} {op} {b}?\nA:"
        tasks.append(NumericTask(prompt, ans, "arithmetic", difficulty=1))
    return tasks


def build_logic_tasks(n: int = 20, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    entities = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    props = ["smart", "fast", "kind", "brave", "honest"]
    tasks = []
    for _ in range(n):
        a, b = rng.sample(entities, 2)
        p1, p2 = rng.sample(props, 2)
        prompt = (f"All {p1} people are {p2}. "
                  f"{a} is {p1}. Is {a} {p2}?\nAnswer:")
        tasks.append(EvalTask(prompt, "yes", "reasoning", difficulty=2))
    return tasks


def build_memory_tasks(n: int = 20, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    names  = ["Alex", "Jordan", "Morgan", "Taylor", "Casey"]
    cities = ["Berlin", "Tokyo", "Lagos", "Sydney", "Oslo"]
    jobs   = ["engineer", "teacher", "doctor", "designer", "pilot"]
    tasks = []
    for _ in range(n):
        name = rng.choice(names)
        city = rng.choice(cities)
        job  = rng.choice(jobs)
        prompt = (f"{name} was born in {city} and works as a {job}. "
                  f"Where was {name} born?\nAnswer:")
        tasks.append(EvalTask(prompt, city.lower(), "memory_recall", difficulty=2))
    return tasks


def build_code_tasks(n: int = 15, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    tasks = []
    # Fixed set of code bug scenarios
    scenarios = [
        (
            "def add(a, b):\n    return a - b\n\nBug: The function subtracts instead of adds. Fix:",
            "return a + b",
            ["return", "a", "+", "b"]
        ),
        (
            "for i in range(1, len(data) + 1):\n    print(data[i])\n\nBug: IndexError. Fix:",
            "range(len(data))",
            ["range", "len", "data"]
        ),
        (
            "total = 0\nfor num in numbers:\n    totl += num\n\nBug: NameError. Fix:",
            "total += num",
            ["total", "+=", "num"]
        ),
        (
            "def find(lst, target):\n    for i, val in enumerate(lst):\n        if val == target:\n            return i\n    return i\n\nBug: Returns wrong value when not found. Fix:",
            "return -1",
            ["return", "-1"]
        ),
        (
            "if x > 10\n    print('big')\n\nBug: SyntaxError. Fix:",
            "if x > 10:",
            ["if", "x", ">", "10", ":"]
        ),
    ]
    for i in range(n):
        prompt, expected, keywords = scenarios[i % len(scenarios)]
        tasks.append(PartialTask(prompt, keywords, "code", difficulty=3))
    return tasks


def build_planning_tasks(n: int = 15, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    tasks = []
    goals = [
        ("deploy a web application",
         ["requirements", "environment", "test", "deploy"]),
        ("train a machine learning model",
         ["data", "model", "train", "evaluate"]),
        ("publish a research paper",
         ["write", "review", "submit", "revise"]),
        ("onboard a new employee",
         ["paperwork", "access", "training", "introduce"]),
        ("migrate a database",
         ["backup", "schema", "transfer", "verify"]),
    ]
    for i in range(n):
        goal, keywords = goals[i % len(goals)]
        prompt = f"Goal: {goal}. List the key steps in order: 1."
        tasks.append(PartialTask(prompt, keywords, "planning", difficulty=3))
    return tasks


def build_self_correction_tasks(n: int = 15, seed: int = 42) -> List[EvalTask]:
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        correct = a + b
        wrong   = correct + rng.choice([-2, -1, 1, 2])
        prompt  = (f"Question: What is {a} + {b}? "
                   f"Initial answer: {wrong}. "
                   f"Wait, let me recheck. The correct answer is:")
        tasks.append(NumericTask(prompt, str(correct),
                                 "self_reflection", difficulty=2))
    return tasks


def build_all_tasks() -> Dict[str, List[EvalTask]]:
    return {
        "arithmetic":    build_arithmetic_tasks(),
        "reasoning":     build_logic_tasks(),
        "memory_recall": build_memory_tasks(),
        "code":          build_code_tasks(),
        "planning":      build_planning_tasks(),
        "self_reflection": build_self_correction_tasks(),
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class AvusEvaluator:
    """
    Runs all eval tasks against a loaded Avus model.
    Returns per-domain scores and an overall score.
    """

    def __init__(self, model, encode, decode, device: str = "cpu",
                 verbose: bool = False):
        self.model   = model
        self.encode  = encode
        self.decode  = decode
        self.device  = device
        self.verbose = verbose

    def run(self, tasks: Optional[Dict[str, List[EvalTask]]] = None
            ) -> Dict[str, float]:
        tasks = tasks or build_all_tasks()
        results = {}

        print("\n" + "="*60)
        print("AVUS EVALUATION")
        print("="*60)

        for domain, domain_tasks in tasks.items():
            scores = []
            for task in domain_tasks:
                output = generate(
                    self.model, self.encode, self.decode,
                    task.prompt, max_tokens=60,
                    temperature=0.1,   # low temp for eval — want deterministic
                    top_k=10,
                    device=self.device,
                )
                score = task.score(output)
                scores.append(score)

                if self.verbose:
                    status = "✓" if score >= 0.5 else "✗"
                    print(f"  [{status}] {task.prompt[:60]}...")
                    print(f"       expected: {task.expected[:40]}")
                    print(f"       got:      {output[:60].strip()}")

            domain_score = sum(scores) / len(scores) * 100
            results[domain] = round(domain_score, 1)
            bar = "█" * int(domain_score / 5) + "░" * (20 - int(domain_score / 5))
            print(f"  {domain:<20} [{bar}] {domain_score:.1f}%")

        overall = sum(results.values()) / len(results)
        results["overall"] = round(overall, 1)

        print(f"\n  {'OVERALL':<20} {'─'*20}  {overall:.1f}%")
        print("="*60)

        return results

    def update_skill_tree(self, scores: Dict[str, float],
                          skill_path: str):
        """Push eval scores back into the skill tree."""
        try:
            from skill_curriculum import SkillTree
            tree = SkillTree()
            tree.load(skill_path)

            # Map eval domains to skill tree names
            domain_map = {
                "arithmetic":     "arithmetic",
                "reasoning":      "reasoning",
                "memory_recall":  "memory_recall",
                "code":           "code",
                "planning":       "planning",
                "self_reflection": "self_reflection",
            }

            for eval_domain, skill_name in domain_map.items():
                if eval_domain in scores and skill_name in tree.skills:
                    # Convert 0-100 score to 0-1 confidence
                    confidence = scores[eval_domain] / 100.0
                    tree.update(skill_name, confidence)
                    print(f"[eval] Updated {skill_name}: {confidence:.2f}")

            tree.save(skill_path)
            print(f"[eval] Skill tree updated → {skill_path}")

            try:
                tree.plot(skill_path.replace(".json", "_eval.png"))
            except Exception:
                pass

        except Exception as e:
            print(f"[eval] Skill tree update failed: {e}")


# ── History tracking ──────────────────────────────────────────────────────────

def save_eval_history(scores: Dict[str, float], weights_path: str,
                      history_path: str = "eval_history.json"):
    history = []
    if Path(history_path).exists():
        history = json.loads(Path(history_path).read_text())

    import time
    history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "weights":   weights_path,
        "scores":    scores,
    })

    Path(history_path).write_text(json.dumps(history, indent=2))
    print(f"[eval] History saved → {history_path}")


def print_history(history_path: str = "eval_history.json"):
    if not Path(history_path).exists():
        print("No eval history found.")
        return

    history = json.loads(Path(history_path).read_text())
    print(f"\nEval history ({len(history)} runs):")
    print(f"{'Date':<18} {'Overall':>8} {'Arith':>7} {'Logic':>7} "
          f"{'Memory':>8} {'Code':>6} {'Plan':>6} {'Self':>6}")
    print("-" * 70)
    for entry in history:
        s = entry["scores"]
        print(f"{entry['timestamp']:<18} "
              f"{s.get('overall', 0):>7.1f}% "
              f"{s.get('arithmetic', 0):>6.1f}% "
              f"{s.get('reasoning', 0):>6.1f}% "
              f"{s.get('memory_recall', 0):>7.1f}% "
              f"{s.get('code', 0):>5.1f}% "
              f"{s.get('planning', 0):>5.1f}% "
              f"{s.get('self_reflection', 0):>5.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Avus reasoning ability")
    parser.add_argument("--weights",       required=True,
                        help="Path to weights .pt file")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose",       action="store_true",
                        help="Show each task result")
    parser.add_argument("--update-skills", type=str, default=None,
                        metavar="SKILL_JSON",
                        help="Update skill tree with eval scores")
    parser.add_argument("--history",       action="store_true",
                        help="Print eval history")
    parser.add_argument("--save-history",  default="eval_history.json")
    args = parser.parse_args()

    if args.history:
        print_history(args.save_history)
        return

    print(f"Loading model from {args.weights}...")
    model, cfg = load_model(args.weights, args.device)
    encode, decode = load_tokenizer()
    print(f"Model: {model.count_parameters()/1e9:.2f}B params on {args.device}")

    evaluator = AvusEvaluator(model, encode, decode,
                              device=args.device, verbose=args.verbose)
    scores = evaluator.run()

    save_eval_history(scores, args.weights, args.save_history)

    if args.update_skills:
        evaluator.update_skill_tree(scores, args.update_skills)

    print(f"\nRun with --history to see progress over time.")


if __name__ == "__main__":
    main()
