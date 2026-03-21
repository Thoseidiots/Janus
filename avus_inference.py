"""
avus_inference.py
==================
Loads trained Avus weights and exposes a clean generate() interface
for the rest of the Janus system.

This is the single point of contact between the trained model and
all other Janus components. Nothing else should import model.py directly.

Usage:
    from avus_inference import AvusInference

    avus = AvusInference()
    avus.load()

    # Generate a 3D object description
    result = avus.generate("Generate a rusted metal pillar")

    # Generate a screen action
    result = avus.generate("A 'Submit' button is visible at (320, 480). Click it.")

    # Generate with semantic tag
    result = avus.generate("[SEM:dark_alleyway] Describe the lighting.")
"""

import os
import sys
import json
import importlib
import platform
from typing import Optional, List, Set
from pathlib import Path

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution — works on EliteDesk, Kaggle, and Lightning AI
# ─────────────────────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    """Walk up from this file's location to find the Janus repo root."""
    candidates = [
        Path(__file__).parent,                          # same dir as this file
        Path("/kaggle/working/Janus"),                  # Kaggle
        Path("/teamspace/studios/this_studio/Janus"),   # Lightning AI
        Path(os.path.expanduser("~/Janus")),            # local clone
        Path("C:/Users") / os.environ.get("USERNAME", "") / "Janus",  # Windows
    ]
    for p in candidates:
        if (p / "model.py").exists():
            return p
    # Fallback: current working directory
    return Path.cwd()


REPO_ROOT = _find_repo_root()

# Weight search order
WEIGHT_PATHS = [
    REPO_ROOT / "avus_1b_weights.pt",
    REPO_ROOT / "weights" / "avus_1b_weights.pt",
    Path("/kaggle/working/avus_1b_weights.pt"),
    Path("/kaggle/input/janus-avus-weights/avus_1b_weights.pt"),
    Path("/teamspace/studios/this_studio/Janus/avus_1b_weights.pt"),
]

CONFIG_PATHS = [
    REPO_ROOT / "config_avus_1b.json",
    Path("/kaggle/working/config_avus_1b.json"),
    Path("/kaggle/input/janus-avus-weights/config_avus_1b.json"),
]

DEFAULT_CONFIG = {
    "vocab_size":  50304,
    "dim":         768,
    "n_layers":    12,
    "n_heads":     12,
    "n_kv_heads":  4,
    "max_seq_len": 512,
}

# Special tokens used across all curricula
SPECIAL_TOKENS: Set[str] = {
    "<|startoftext|>", "<|endoftext|>",
    "[JSON_START]",    "[JSON_END]",
    "[ACT_START]",     "[ACT_END]",
    "[FRAME_START]",   "[FRAME_NEXT]",   "[/FRAME_END]",
    "[ZONE_PROTECT_START]", "[ZONE_PROTECT_END]",
    "[ZONE_SMOOTH_START]",  "[ZONE_SMOOTH_END]",
    "[RENDER_PARAMS]",      "[/RENDER_PARAMS]",
    "[GEO_BOUNDS]",         "[/GEO_BOUNDS]",
    "[POSE_START]",         "[/POSE_START]",
    "[FRAME_A]",            "[/FRAME_A]",
    "[FRAME_B_PREDICTION]", "[/FRAME_B_PREDICTION]",
    "[MOTION_VECTORS]",     "[/MOTION_VECTORS]",
    "[OCCLUSION_MASK]",     "[/OCCLUSION_MASK]",
    "[LIGHTING_VALID]",     "[/LIGHTING_VALID]",
    "[LIGHTING_INVALID]",   "[/LIGHTING_INVALID]",
}


# ─────────────────────────────────────────────────────────────────────────────
# Inline tokenizer (no reload issues)
# ─────────────────────────────────────────────────────────────────────────────

class _AvusTokenizer:
    def __init__(self):
        try:
            import tiktoken
        except ImportError:
            os.system("pip install tiktoken -q")
            import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=SPECIAL_TOKENS)

    def decode(self, tokens: List[int]) -> str:
        valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
        try:
            return self._enc.decode(valid)
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# AvusInference
# ─────────────────────────────────────────────────────────────────────────────

class AvusInference:
    """
    Wraps the trained Avus model with a simple generate() interface.

    All Janus components (orchestrator, game pipeline, screen interpreter,
    CEO loop) call this class. None of them import model.py directly.
    """

    def __init__(self, device: Optional[str] = None):
        self.model      = None
        self.config     = None
        self.tokenizer  = None
        self._loaded    = False

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, weights_path: Optional[str] = None,
             config_path: Optional[str] = None) -> bool:
        """
        Load Avus weights. Returns True on success, False on failure.
        Call this once at startup before any generate() calls.
        """
        # ── import model ──
        try:
            self._import_model()
        except Exception as e:
            print(f"[AvusInference] Failed to import model: {e}")
            return False

        # ── load config ──
        cfg = self._load_config(config_path)

        # ── build model ──
        try:
            from model import C, Avus
            self.config = C(**cfg)
            self.model  = Avus(self.config).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[AvusInference] Failed to build model: {e}")
            return False

        # ── load weights ──
        weight_file = self._find_weights(weights_path)
        if weight_file:
            try:
                sd   = torch.load(weight_file, map_location=self.device)
                # Strip DataParallel prefix if present
                sd   = {k.replace("module.", ""): v for k, v in sd.items()}
                # Drop shape-incompatible keys
                drop = [k for k in sd if any(x in k for x in
                        ("attn.mask", "attn.rope.c", "attn.rope.s"))]
                for k in drop:
                    del sd[k]
                self.model.load_state_dict(sd, strict=False)
                print(f"[AvusInference] Loaded weights from {weight_file}")
            except Exception as e:
                print(f"[AvusInference] Weight load failed: {e} — using random weights")
        else:
            print("[AvusInference] No weights found — using random initialisation")

        # ── tokenizer ──
        self.tokenizer = _AvusTokenizer()
        self._loaded   = True
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[AvusInference] Ready  |  params={total:,}  |  device={self.device}")
        return True

    def _import_model(self):
        """Add repo root to sys.path and import model.py."""
        repo = str(REPO_ROOT)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        if "model" in sys.modules:
            del sys.modules["model"]
        import model as _m
        importlib.reload(_m)

    def _load_config(self, config_path: Optional[str]) -> dict:
        paths = ([Path(config_path)] if config_path else []) + CONFIG_PATHS
        for p in paths:
            if p.exists():
                try:
                    with open(p) as f:
                        cfg = json.load(f)
                    cfg["max_seq_len"] = min(cfg.get("max_seq_len", 512), 512)
                    print(f"[AvusInference] Config from {p}")
                    return cfg
                except Exception:
                    continue
        print("[AvusInference] Using default config")
        return DEFAULT_CONFIG

    def _find_weights(self, weights_path: Optional[str]) -> Optional[Path]:
        paths = ([Path(weights_path)] if weights_path else []) + WEIGHT_PATHS
        for p in paths:
            if p.exists():
                return p
        return None

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:       str,
        max_new_tokens: int  = 256,
        temperature:  float  = 0.8,
        top_k:        int    = 50,
        stop_token:   Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input text. Can include semantic tags like [SEM:dark_alleyway]
            or curriculum tags like [JSON_START], [ACT_START] etc.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature. Lower = more deterministic.
        top_k : int
            Top-k sampling. 0 = disabled.
        stop_token : str, optional
            Stop generation when this token is produced.

        Returns
        -------
        str
            Generated text (prompt not included).
        """
        if not self._loaded:
            raise RuntimeError("Call avus.load() before generate().")

        # Prepend start token if not present
        if not prompt.startswith("<|startoftext|>"):
            prompt = f"<|startoftext|>{prompt}"

        tokens  = self.tokenizer.encode(prompt)
        input_t = torch.tensor([tokens], dtype=torch.long, device=self.device)

        end_id   = self.tokenizer.encode("<|endoftext|>")[0]
        stop_id  = (self.tokenizer.encode(stop_token)[0]
                    if stop_token else None)

        generated = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Trim context to max_seq_len
                ctx = input_t[:, -self.config.max_seq_len:]

                logits = self.model(ctx)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                # Take last token logits
                logits = logits[0, -1, :] / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0:
                    top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_vals[-1]] = float("-inf")

                probs    = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                tok_id   = next_tok.item()

                if tok_id == end_id:
                    break
                if stop_id is not None and tok_id == stop_id:
                    break

                generated.append(tok_id)
                input_t = torch.cat([input_t, next_tok.unsqueeze(0)], dim=1)

        return self.tokenizer.decode(generated)

    # ── specialised interfaces ────────────────────────────────────────────────

    def generate_3d_params(self, description: str) -> Optional[dict]:
        """
        Generate 3D generation parameters JSON from a natural language description.
        Returns parsed dict or None on failure.
        """
        prompt = f"{description} [JSON_START]"
        raw    = self.generate(prompt, max_new_tokens=300,
                               temperature=0.7, stop_token="[JSON_END]")
        try:
            # Extract JSON between tags
            if "[JSON_END]" in raw:
                raw = raw[:raw.index("[JSON_END]")]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    def generate_action(self, screen_description: str) -> Optional[dict]:
        """
        Generate a desktop action JSON from a screen description.
        Returns parsed dict or None on failure.
        """
        prompt = f"{screen_description} [ACT_START]"
        raw    = self.generate(prompt, max_new_tokens=128,
                               temperature=0.5, stop_token="[ACT_END]")
        try:
            if "[ACT_END]" in raw:
                raw = raw[:raw.index("[ACT_END]")]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    def generate_lighting(self, semantic_tag: str,
                          scene_description: str) -> Optional[dict]:
        """
        Generate a valid lighting configuration for a scene.
        Returns parsed dict or None on failure.
        """
        prompt = (f"[SEM:{semantic_tag}] {scene_description} "
                  f"[LIGHTING_VALID]")
        raw    = self.generate(prompt, max_new_tokens=200,
                               temperature=0.6, stop_token="[/LIGHTING_VALID]")
        try:
            if "[/LIGHTING_VALID]" in raw:
                raw = raw[:raw.index("[/LIGHTING_VALID]")]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    def generate_pose(self, archetype: str,
                      scene_tag: str) -> Optional[dict]:
        """
        Generate a valid character pose within geometric bounds.
        Returns parsed dict or None on failure.
        """
        prompt = (f"[SEM:{scene_tag}] Character archetype: {archetype}. "
                  f"[POSE_START]")
        raw    = self.generate(prompt, max_new_tokens=400,
                               temperature=0.6, stop_token="[/POSE_START]")
        try:
            if "[/POSE_START]" in raw:
                raw = raw[:raw.index("[/POSE_START]")]
            # Extract just the pose JSON (after GEO_BOUNDS)
            if "[POSE_VALID:" in raw:
                raw = raw[:raw.index("[POSE_VALID:")]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    # ── utilities ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"AvusInference(device={self.device}, status={status})"


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    avus = AvusInference()
    ok   = avus.load()

    if ok:
        print("\n── Test: free generation ───────────────────────────────")
        result = avus.generate("Generate a glowing crystal pillar", max_new_tokens=100)
        print(result)

        print("\n── Test: 3D params ─────────────────────────────────────")
        params = avus.generate_3d_params("Generate a rusty metal barrel")
        print(json.dumps(params, indent=2) if params else "Parse failed")

        print("\n── Test: screen action ─────────────────────────────────")
        action = avus.generate_action(
            "Chrome is open. A 'Login' button is at (640, 400). Click it.")
        print(json.dumps(action, indent=2) if action else "Parse failed")

        print("\n── Test: lighting ──────────────────────────────────────")
        light = avus.generate_lighting(
            "dark_alleyway",
            "A hooded figure stands in a narrow alley at night.")
        print(json.dumps(light, indent=2) if light else "Parse failed")
    else:
        print("Load failed — check model.py is in the repo root.")
