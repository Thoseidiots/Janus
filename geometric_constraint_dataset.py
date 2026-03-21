"""
geometric_constraint_dataset.py
=================================
Synthetic dataset for training Avus to respect character proportions
during motion. Targets the "Uncanny Valley / Geometric Drift" failure mode.

Format:
  <|startoftext|>[SEM:tag] {character_description}
  [GEO_BOUNDS] {proportion_bounds_json} [/GEO_BOUNDS]
  [POSE_START] {pose_frame_json} [POSE_VALID: true/false] [/POSE_START]
  <|endoftext|>

The model learns that generated poses must keep proportions within bounds.
POSE_VALID: false samples teach what violations look like so the model
learns to avoid them.
"""

import json
import random
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_TAGS = [
    "character_portrait", "combat_scene", "dialogue_scene", "idle_animation",
    "running_sequence", "attack_animation", "magic_cast", "death_animation",
    "victory_pose", "stealth_crouch", "climbing_sequence", "swimming",
]

CHARACTER_ARCHETYPES = [
    {"name": "human_adult",    "head_body_ratio": 0.143, "arm_span_height": 1.0,  "leg_torso_ratio": 1.1},
    {"name": "human_child",    "head_body_ratio": 0.20,  "arm_span_height": 0.95, "leg_torso_ratio": 0.85},
    {"name": "elf",            "head_body_ratio": 0.125, "arm_span_height": 1.05, "leg_torso_ratio": 1.2},
    {"name": "dwarf",          "head_body_ratio": 0.18,  "arm_span_height": 0.85, "leg_torso_ratio": 0.75},
    {"name": "giant",          "head_body_ratio": 0.10,  "arm_span_height": 1.1,  "leg_torso_ratio": 1.3},
    {"name": "manga_hero",     "head_body_ratio": 0.167, "arm_span_height": 1.0,  "leg_torso_ratio": 1.15},
    {"name": "seinen_fighter", "head_body_ratio": 0.143, "arm_span_height": 1.02, "leg_torso_ratio": 1.1},
    {"name": "creature",       "head_body_ratio": 0.20,  "arm_span_height": 1.3,  "leg_torso_ratio": 0.9},
]

BONE_NAMES = [
    "spine_base", "spine_mid", "spine_top", "neck", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "hip_l", "hip_r",
    "knee_l", "knee_r", "ankle_l", "ankle_r",
]

POSES = [
    "T-pose reference",
    "standing idle",
    "walking mid-stride left foot forward",
    "running full sprint",
    "crouching low",
    "jumping apex",
    "attack swing right arm",
    "blocking with arms raised",
    "sitting on ground",
    "kneeling on one knee",
    "looking upward arms at sides",
    "casting spell arms extended",
]


def _rc(lst):  return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 3)


def _bone_position(name, pose_idx):
    """Generate a plausible 3D bone position for a given pose."""
    base_positions = {
        "spine_base":  [0.0, 0.0, 0.0],
        "spine_mid":   [0.0, 0.5, 0.0],
        "spine_top":   [0.0, 0.9, 0.0],
        "neck":        [0.0, 1.0, 0.0],
        "head":        [0.0, 1.15, 0.0],
        "shoulder_l":  [-0.2, 0.9, 0.0],
        "shoulder_r":  [0.2,  0.9, 0.0],
        "elbow_l":     [-0.35, 0.65, 0.0],
        "elbow_r":     [0.35,  0.65, 0.0],
        "wrist_l":     [-0.45, 0.4, 0.0],
        "wrist_r":     [0.45,  0.4, 0.0],
        "hip_l":       [-0.1, -0.05, 0.0],
        "hip_r":       [0.1,  -0.05, 0.0],
        "knee_l":      [-0.12, -0.55, 0.0],
        "knee_r":      [0.12,  -0.55, 0.0],
        "ankle_l":     [-0.12, -1.0, 0.0],
        "ankle_r":     [0.12,  -1.0, 0.0],
    }
    pos = base_positions.get(name, [0.0, 0.0, 0.0])
    # Add small variation for different poses
    variation = _rf(-0.05, 0.05)
    return [round(pos[0] + variation, 3),
            round(pos[1] + variation * 0.5, 3),
            round(pos[2] + variation, 3)]


def _build_pose(archetype, valid=True):
    """Build a pose dict, optionally with proportion violations."""
    bones = {}
    for i, bone in enumerate(BONE_NAMES):
        bones[bone] = _bone_position(bone, i)

    if not valid:
        # Introduce a proportion violation
        violation = _rc(["eye_spacing", "head_scale", "arm_length"])
        bones["_violation"] = violation
        if violation == "eye_spacing":
            # Eyes too far apart — geometric drift
            bones["head"] = [bones["head"][0],
                             bones["head"][1],
                             bones["head"][2] + _rf(0.3, 0.8)]
        elif violation == "head_scale":
            # Head scaled too large
            bones["head"] = [bones["head"][0],
                             bones["head"][1] + _rf(0.2, 0.5),
                             bones["head"][2]]
        elif violation == "arm_length":
            # Left arm longer than right
            bones["wrist_l"] = [bones["wrist_l"][0] - _rf(0.2, 0.5),
                                bones["wrist_l"][1],
                                bones["wrist_l"][2]]

    return {
        "pose_name":   _rc(POSES),
        "bones":       bones,
        "blend_weight": _rf(0.0, 1.0),
    }


def _proportion_bounds(archetype):
    tolerance = 0.05  # 5% tolerance on all ratios
    return {
        "archetype":          archetype["name"],
        "head_body_ratio":    {
            "nominal": archetype["head_body_ratio"],
            "min":     round(archetype["head_body_ratio"] * (1 - tolerance), 4),
            "max":     round(archetype["head_body_ratio"] * (1 + tolerance), 4),
        },
        "arm_span_height":    {
            "nominal": archetype["arm_span_height"],
            "min":     round(archetype["arm_span_height"] * (1 - tolerance), 4),
            "max":     round(archetype["arm_span_height"] * (1 + tolerance), 4),
        },
        "leg_torso_ratio":    {
            "nominal": archetype["leg_torso_ratio"],
            "min":     round(archetype["leg_torso_ratio"] * (1 - tolerance), 4),
            "max":     round(archetype["leg_torso_ratio"] * (1 + tolerance), 4),
        },
        "inter_ocular_distance": {
            "nominal": 0.065,
            "min":     0.055,
            "max":     0.075,
        },
        "pupil_size_ratio": {
            "nominal": 0.33,
            "min":     0.28,
            "max":     0.38,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# GeometricConstraintDataset
# ─────────────────────────────────────────────────────────────────────────────

class GeometricConstraintDataset:
    """
    Generates pose sequences with proportion bounds.
    80% valid poses, 20% violation examples (labeled POSE_VALID: false).
    Teaches Avus to stay within geometric bounds during generation.
    """

    def __init__(self):
        self._generators = [
            self._single_pose_valid,
            self._single_pose_valid,
            self._single_pose_valid,
            self._single_pose_valid,
            self._single_pose_violation,   # 20% violations
        ]

    def _single_pose_valid(self):
        tag        = _rc(SEMANTIC_TAGS)
        archetype  = _rc(CHARACTER_ARCHETYPES)
        bounds     = _proportion_bounds(archetype)
        pose       = _build_pose(archetype, valid=True)
        bounds_str = json.dumps(bounds, indent=2)
        pose_str   = json.dumps(pose, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Character archetype: {archetype['name']}. "
            f"[GEO_BOUNDS] {bounds_str} [/GEO_BOUNDS] "
            f"[POSE_START] {pose_str} [POSE_VALID: true] [/POSE_START]"
            f"<|endoftext|>"
        )
        return text

    def _single_pose_violation(self):
        tag        = _rc(SEMANTIC_TAGS)
        archetype  = _rc(CHARACTER_ARCHETYPES)
        bounds     = _proportion_bounds(archetype)
        pose       = _build_pose(archetype, valid=False)
        violation  = pose["bones"].pop("_violation", "unknown")
        bounds_str = json.dumps(bounds, indent=2)
        pose_str   = json.dumps(pose, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Character archetype: {archetype['name']}. "
            f"[GEO_BOUNDS] {bounds_str} [/GEO_BOUNDS] "
            f"[POSE_START] {pose_str} "
            f"[POSE_VALID: false] [VIOLATION: {violation}] [/POSE_START]"
            f"<|endoftext|>"
        )
        return text

    def generate_dataset(self, samples: int = 10_000, seed: int = 333
                         ) -> List[str]:
        random.seed(seed)
        out = []
        for _ in range(samples):
            fn   = _rc(self._generators)
            text = fn()
            out.append(text)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen     = GeometricConstraintDataset()
    samples = gen.generate_dataset(samples=5_000)

    print(f"Generated {len(samples)} samples\n")
    valid_count     = sum(1 for s in samples if "POSE_VALID: true" in s)
    violation_count = sum(1 for s in samples if "POSE_VALID: false" in s)
    bounds_count    = sum(1 for s in samples if "[GEO_BOUNDS]" in s)
    print(f"Valid poses:     {valid_count}")
    print(f"Violation poses: {violation_count}")
    print(f"Bounds present:  {bounds_count}/{len(samples)}")

    print("\n── Sample valid ────────────────────────────────────────")
    for s in samples[:1]:
        print(s[:400] + "...")

    print("\n── Sample violation ────────────────────────────────────")
    violations = [s for s in samples if "POSE_VALID: false" in s]
    if violations:
        print(violations[0][:400] + "...")
