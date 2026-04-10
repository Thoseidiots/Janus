"""
temporal_consistency_dataset.py
=================================
Synthetic dataset for training Avus to maintain object permanence across frames.
Targets the "Squibbling" failure mode identified in DLSS 5 analysis.

Format:
  <|startoftext|>[SEM:tag] {scene_description}
  [FRAME_START] {frame_n_description} [STABLE: feature1, feature2, ...]
  [FRAME_NEXT] {frame_n+1_description} [STABLE: feature1, feature2, ...]
  [/FRAME_END]<|endoftext|>

The model learns that features listed in [STABLE:...] must remain consistent
across [FRAME_START] → [FRAME_NEXT] transitions.
"""

import json
import random
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_TAGS = [
    "forest_clearing", "dark_alleyway", "bright_exterior", "underground_cave",
    "medieval_castle", "futuristic_city", "desert_ruins", "ocean_shoreline",
    "mountain_peak", "interior_tavern", "space_station", "ancient_temple",
]

CHARACTERS = [
    "warrior", "mage", "rogue", "archer", "knight", "priestess",
    "merchant", "soldier", "elder", "child", "assassin", "ranger",
]

EYE_COLORS = ["amber", "blue", "green", "grey", "brown", "violet", "silver"]
HAIR_COLORS = ["black", "brown", "blonde", "red", "white", "silver", "auburn"]
SKIN_TONES  = ["pale", "fair", "tan", "bronze", "dark", "olive"]
ARMOR_TYPES = ["leather", "chainmail", "plate", "cloth robes", "fur-lined cloak", "light scout armor"]

MOTIONS = [
    "walking forward slowly",
    "turning to look left",
    "raising their right arm",
    "crouching down",
    "standing upright from a crouch",
    "drawing a weapon",
    "sheathing a weapon",
    "looking upward",
    "stepping backward",
    "nodding their head",
    "gesturing with their left hand",
    "turning to face the camera",
]

ENVIRONMENT_FEATURES = [
    "a stone wall with moss growing between the cracks",
    "a wooden door with iron hinges",
    "a flickering torch mounted on the wall",
    "a cobblestone floor with uneven stones",
    "a window with moonlight streaming through",
    "a pile of crates stacked in the corner",
    "a banner hanging from the ceiling",
    "a puddle of water reflecting the ceiling",
    "a table with scattered papers",
    "iron bars casting shadow stripes on the floor",
]

LIGHTING_CONDITIONS = {
    "dark_alleyway":    "dim ambient light, deep shadows, no direct highlights",
    "forest_clearing":  "dappled sunlight through leaves, soft green ambient",
    "bright_exterior":  "direct sunlight, sharp shadows, high contrast",
    "underground_cave": "faint bioluminescent glow, near total darkness",
    "medieval_castle":  "torch light, warm orange ambient, long shadows",
    "futuristic_city":  "neon reflections, cool blue ambient, sharp specular",
    "desert_ruins":     "harsh midday sun, minimal shadow, bleached tones",
    "ocean_shoreline":  "golden hour, warm diffuse, water reflections",
    "mountain_peak":    "cold blue sky light, crisp shadows, thin atmosphere",
    "interior_tavern":  "warm firelight, smoky haze, low contrast",
    "space_station":    "artificial white light, zero ambient, hard shadows",
    "ancient_temple":   "filtered light through cracks, dusty shafts, cool grey",
}


def _rc(lst): return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 2)


def _build_character():
    return {
        "type":       _rc(CHARACTERS),
        "eye_color":  _rc(EYE_COLORS),
        "hair_color": _rc(HAIR_COLORS),
        "skin_tone":  _rc(SKIN_TONES),
        "armor":      _rc(ARMOR_TYPES),
        "height":     _rc(["tall", "average height", "short", "slender", "broad-shouldered"]),
    }


def _stable_features(char):
    """Features that must remain constant across frames."""
    return [
        f"{char['eye_color']} eyes",
        f"{char['hair_color']} hair",
        f"{char['skin_tone']} skin tone",
        f"{char['armor']} armor",
        f"{char['height']} build",
    ]


def _frame_description(char, motion, env_feature, lighting):
    return (
        f"A {char['height']} {char['type']} with {char['eye_color']} eyes, "
        f"{char['hair_color']} hair, {char['skin_tone']} skin, wearing {char['armor']}. "
        f"They are {motion}. "
        f"Behind them: {env_feature}. "
        f"Lighting: {lighting}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TemporalConsistencyDataset
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConsistencyDataset:
    """
    Generates frame-pair sequences where stable features must persist
    across transitions. Teaches Avus object permanence.

    Each sample is a sequence of 2-8 frames with:
    - A character whose physical features are stable
    - Changing motion and environment details between frames
    - An explicit [STABLE:...] list the model must honour
    """

    def __init__(self):
        self._generators = [
            self._character_motion_sequence,
            self._environment_detail_sequence,
            self._combined_sequence,
        ]

    def _character_motion_sequence(self):
        """Character moves through multiple poses — features stay stable."""
        tag    = _rc(SEMANTIC_TAGS)
        char   = _build_character()
        stable = _stable_features(char)
        env    = _rc(ENVIRONMENT_FEATURES)
        light  = LIGHTING_CONDITIONS.get(tag, "neutral ambient lighting")

        n_frames = _ri(2, 8)
        motions  = random.sample(MOTIONS, min(n_frames, len(MOTIONS)))
        frames   = []

        for motion in motions:
            frames.append(_frame_description(char, motion, env, light))

        stable_str = ", ".join(stable)
        frames_text = ""
        for i, fd in enumerate(frames):
            if i == 0:
                frames_text += f"[FRAME_START] {fd} [STABLE: {stable_str}]\n"
            elif i == len(frames) - 1:
                frames_text += f"[FRAME_NEXT] {fd} [STABLE: {stable_str}] [/FRAME_END]"
            else:
                frames_text += f"[FRAME_NEXT] {fd} [STABLE: {stable_str}]\n"

        text = f"<|startoftext|>[SEM:{tag}] {frames_text}<|endoftext|>"
        return text

    def _environment_detail_sequence(self):
        """Environment details stay stable while lighting shifts slightly."""
        tag     = _rc(SEMANTIC_TAGS)
        char    = _build_character()
        stable  = _stable_features(char)
        env1    = _rc(ENVIRONMENT_FEATURES)
        env2    = _rc([e for e in ENVIRONMENT_FEATURES if e != env1])
        light   = LIGHTING_CONDITIONS.get(tag, "neutral ambient lighting")
        motion1 = _rc(MOTIONS)
        motion2 = _rc([m for m in MOTIONS if m != motion1])

        stable_str = ", ".join(stable + [env1])

        frame1 = _frame_description(char, motion1, env1, light)
        frame2 = _frame_description(char, motion2, env1, light)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"[FRAME_START] {frame1} [STABLE: {stable_str}]\n"
            f"[FRAME_NEXT] {frame2} [STABLE: {stable_str}] [/FRAME_END]"
            f"<|endoftext|>"
        )
        return text

    def _combined_sequence(self):
        """Multi-character scene — each character's features stay independent."""
        tag    = _rc(SEMANTIC_TAGS)
        char1  = _build_character()
        char2  = _build_character()
        light  = LIGHTING_CONDITIONS.get(tag, "neutral ambient lighting")
        env    = _rc(ENVIRONMENT_FEATURES)

        stable1 = _stable_features(char1)
        stable2 = _stable_features(char2)
        all_stable = stable1 + [f"second character: {s}" for s in stable2]
        stable_str = ", ".join(all_stable)

        motion1a = _rc(MOTIONS)
        motion1b = _rc([m for m in MOTIONS if m != motion1a])
        motion2a = _rc(MOTIONS)
        motion2b = _rc([m for m in MOTIONS if m != motion2a])

        frame1 = (
            f"Two characters are present. "
            f"First: {_frame_description(char1, motion1a, env, light)} "
            f"Second: {_frame_description(char2, motion2a, env, light)}"
        )
        frame2 = (
            f"Two characters are present. "
            f"First: {_frame_description(char1, motion1b, env, light)} "
            f"Second: {_frame_description(char2, motion2b, env, light)}"
        )

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"[FRAME_START] {frame1} [STABLE: {stable_str}]\n"
            f"[FRAME_NEXT] {frame2} [STABLE: {stable_str}] [/FRAME_END]"
            f"<|endoftext|>"
        )
        return text

    def generate_dataset(self, samples: int = 10_000, seed: int = 111
                         ) -> List[str]:
        """
        Returns a list of raw text strings (not pairs).
        Each string is a complete multi-frame training sequence.
        """
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
    gen     = TemporalConsistencyDataset()
    samples = gen.generate_dataset(samples=5_000)

    print(f"Generated {len(samples)} samples\n")
    print("── Sample outputs ──────────────────────────────────────")
    for i in random.sample(range(len(samples)), 3):
        print(f"\n[{i}]:\n{samples[i][:600]}...")

    # Check stable features always present
    stable_count = sum(1 for s in samples if "[STABLE:" in s)
    frame_count  = sum(1 for s in samples if "[FRAME_START]" in s)
    print(f"\n[STABLE:] present in {stable_count}/{len(samples)} samples")
    print(f"[FRAME_START] present in {frame_count}/{len(samples)} samples")
