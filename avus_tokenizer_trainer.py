"""
avus_tokenizer_trainer.py
=========================
Trains a custom BPE tokenizer on Janus training data.

Why: GPT-2's vocab was trained on web text. Janus data is dominated by
screen action JSON, Python code, and structured prompts. A custom tokenizer
encodes these patterns in 40-60% fewer tokens — same information, less compute.

Usage:
    python avus_tokenizer_trainer.py                  # train from generators
    python avus_tokenizer_trainer.py --from-jsonl path/to/data.jsonl
    python avus_tokenizer_trainer.py --vocab-size 32000

Output:
    avus_tokenizer/vocab.json
    avus_tokenizer/merges.txt
    avus_tokenizer/tokenizer.json   (HuggingFace-compatible)
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Iterator, List

# ── Special tokens (must survive any tokenizer) ───────────────────────────────
SPECIAL_TOKENS = [
    "<|startoftext|>",
    "<|endoftext|>",
    "<|pad|>",
    "[JSON_START]", "[JSON_END]",
    "[ACT_START]",  "[ACT_END]",
    "[FRAME_START]","[FRAME_NEXT]","[/FRAME_END]",
    "[THINK]",      "[/THINK]",
    "[PLAN]",       "[/PLAN]",
    "[TOOL]",       "[/TOOL]",
]

VOCAB_SIZE_DEFAULT = 32_000   # smaller than GPT-2's 50k but richer merges


def _iter_training_corpus(from_jsonl: str = None,
                          samples_per: int = 5_000) -> Iterator[str]:
    """Yield training strings from generators or a JSONL file."""
    if from_jsonl:
        print(f"[tokenizer] Loading corpus from {from_jsonl}")
        with open(from_jsonl, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj.get("text", "")
        return

    # Import generators from train_avus_kaggle
    sys.path.insert(0, str(Path(__file__).parent))
    from train_avus_kaggle import (
        generate_screen_action_pairs,
        generate_3d_pairs,
        generate_language_pairs,
        generate_reasoning_pairs,
    )

    print(f"[tokenizer] Generating corpus ({samples_per} samples per domain)...")
    for text in generate_screen_action_pairs(samples_per):
        yield text
    for text in generate_3d_pairs(samples_per):
        yield text
    for text in generate_language_pairs(samples_per):
        yield text
    for text in generate_reasoning_pairs(samples_per):
        yield text

    # Also include screen action JSONL if it exists
    jsonl_path = Path("training_data/screen_actions.jsonl")
    if jsonl_path.exists():
        print(f"[tokenizer] Adding {jsonl_path}")
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj.get("text", "")


def train_tokenizer(vocab_size: int = VOCAB_SIZE_DEFAULT,
                    from_jsonl: str = None,
                    output_dir: str = "avus_tokenizer",
                    samples_per: int = 5_000):
    """Train a BPE tokenizer on Janus data and save to output_dir."""
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from tokenizers.normalizers import NFC
    except ImportError:
        print("[tokenizer] Installing HuggingFace tokenizers...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "tokenizers", "-q"])
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from tokenizers.normalizers import NFC

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build tokenizer ───────────────────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tokenizer.normalizer = NFC()
    # ByteLevel pre-tokenizer: handles any unicode, no OOV
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # ── Write corpus to temp file (tokenizers library needs file iterator) ────
    corpus_path = Path(output_dir) / "_corpus.txt"
    print(f"[tokenizer] Writing corpus to {corpus_path}...")
    count = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in _iter_training_corpus(from_jsonl, samples_per):
            f.write(text.replace("\n", " ") + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  {count:,} examples written...")
    print(f"[tokenizer] Corpus: {count:,} examples")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"[tokenizer] Training BPE tokenizer (vocab_size={vocab_size:,})...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    # ── Save ──────────────────────────────────────────────────────────────────
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))
    print(f"[tokenizer] Saved to {output_dir}/tokenizer.json")

    # Clean up corpus file
    corpus_path.unlink()

    # ── Benchmark: compare token counts vs GPT-2 ─────────────────────────────
    _benchmark(tokenizer)

    return tokenizer


def _benchmark(tokenizer):
    """Show token count comparison vs GPT-2 on representative Janus examples."""
    try:
        import tiktoken
        gpt2 = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("[tokenizer] tiktoken not available — skipping benchmark")
        return

    examples = [
        '<|startoftext|>Chrome is open. A \'Submit\' button is at (847,392). Click it. [ACT_START]{"type":"click","x":847,"y":392,"button":"left"}[ACT_END]<|endoftext|>',
        '<|startoftext|>Generate a glowing crystal with sphere shape and metal material. [JSON_START]{"object":"crystal","primitive":"sphere","material":"metal","scale":[1.2,1.2,1.2],"position":[0.5,-1.2,3.1],"roughness":0.3,"metallic":0.9}[JSON_END]<|endoftext|>',
        '<|startoftext|>Q: What is 47 + 83?\nA: 130<|endoftext|>',
        '<|startoftext|>What is machine learning?\nMachine learning is a method where computers learn patterns from data without being explicitly programmed.<|endoftext|>',
    ]

    print("\n[tokenizer] Token count comparison (GPT-2 vs Avus BPE):")
    print(f"  {'Example':<55} {'GPT-2':>6} {'Avus':>6} {'Savings':>8}")
    print("  " + "-" * 80)
    total_gpt2 = total_avus = 0
    for ex in examples:
        n_gpt2 = len(gpt2.encode(ex, allowed_special="all"))
        n_avus = len(tokenizer.encode(ex).ids)
        savings = (1 - n_avus / n_gpt2) * 100
        total_gpt2 += n_gpt2
        total_avus += n_avus
        print(f"  {ex[:55]:<55} {n_gpt2:>6} {n_avus:>6} {savings:>7.1f}%")
    overall = (1 - total_avus / total_gpt2) * 100
    print(f"\n  Overall token reduction: {overall:.1f}%")
    print(f"  (same information, {overall:.0f}% fewer tokens = {overall:.0f}% less compute)\n")


class AvusBPETokenizer:
    """
    Drop-in replacement for AvusTokenizer that uses the trained BPE vocab.
    Falls back to GPT-2 tiktoken if the trained tokenizer isn't found.
    """
    TOKENIZER_PATH = "avus_tokenizer/tokenizer.json"

    def __init__(self, tokenizer_path: str = None):
        path = tokenizer_path or self.TOKENIZER_PATH
        if Path(path).exists():
            from tokenizers import Tokenizer as _HFTok
            self._tok = _HFTok.from_file(path)
            self._vocab_size = self._tok.get_vocab_size()
            self._mode = "bpe"
            print(f"[AvusBPETokenizer] Loaded custom BPE vocab "
                  f"({self._vocab_size:,} tokens) from {path}")
        else:
            # Fallback to GPT-2
            import tiktoken
            self._enc = tiktoken.get_encoding("gpt2")
            self._vocab_size = self._enc.max_token_value
            self._mode = "gpt2"
            print(f"[AvusBPETokenizer] Custom tokenizer not found at {path} "
                  f"— falling back to GPT-2. Run avus_tokenizer_trainer.py first.")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list:
        if self._mode == "bpe":
            return self._tok.encode(text).ids
        else:
            return self._enc.encode(
                text,
                allowed_special={t for t in SPECIAL_TOKENS}
            )

    def decode(self, tokens: list) -> str:
        if self._mode == "bpe":
            return self._tok.decode(tokens)
        else:
            valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
            try:
                return self._enc.decode(valid)
            except Exception:
                return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Avus BPE tokenizer")
    parser.add_argument("--vocab-size",  type=int, default=VOCAB_SIZE_DEFAULT)
    parser.add_argument("--from-jsonl",  type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default="avus_tokenizer")
    parser.add_argument("--samples-per", type=int, default=5_000)
    args = parser.parse_args()

    train_tokenizer(
        vocab_size=args.vocab_size,
        from_jsonl=args.from_jsonl,
        output_dir=args.output_dir,
        samples_per=args.samples_per,
    )
