"""
semantic_lighting_dataset.py
==============================
Synthetic dataset for training Avus to respect scene lighting semantics.
Targets the "Imaginary Lighting" failure mode — AI adding studio lights
to dark scenes, or removing ambient light from bright scenes.

Format:
  <|startoftext|>[SEM:tag] {scene_description}
  [LIGHTING_VALID] {valid_config_json} [/LIGHTING_VALID]
  [LIGHTING_INVALID] {invalid_config_json} [VIOLATION: reason] [/LIGHTING_INVALID]
  <|endoftext|>

Both valid and invalid configs are included so the model learns the boundary.
"""

import json
import random
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Semantic tag → lighting rules
# ─────────────────────────────────────────────────────────────────────────────

LIGHTING_RULES = {
    "dark_alleyway": {
        "description": "Narrow alley at night between tall buildings",
        "ambient_max": 0.15,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["distant moonlight", "faint window glow", "ambient urban haze"],
        "invalid_sources": ["studio key light", "rim light", "fill light", "directional sun"],
    },
    "underground_cave": {
        "description": "Deep cave far from any surface",
        "ambient_max": 0.05,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["bioluminescent moss", "lava glow", "held torch", "magic light source"],
        "invalid_sources": ["sunlight", "sky light", "rim light", "studio fill"],
    },
    "bright_exterior_noon": {
        "description": "Open outdoor area at midday",
        "ambient_min": 0.6,
        "rim_light_allowed": True,
        "studio_light_allowed": False,
        "valid_sources": ["direct sun", "sky ambient", "ground bounce"],
        "invalid_sources": ["artificial fill", "neon light", "fog removal"],
    },
    "interior_tavern_night": {
        "description": "Warm tavern interior with fireplace at night",
        "ambient_max": 0.4,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["fireplace glow", "candle light", "warm lamp oil"],
        "invalid_sources": ["daylight through window", "cool ambient", "studio light"],
    },
    "forest_at_night": {
        "description": "Dense forest under moonlight",
        "ambient_max": 0.2,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["dappled moonlight", "firefly glow", "faint starlight"],
        "invalid_sources": ["sun rays", "studio fill", "directional rim"],
    },
    "futuristic_city_night": {
        "description": "Cyberpunk city street at night",
        "ambient_max": 0.35,
        "rim_light_allowed": True,
        "studio_light_allowed": False,
        "valid_sources": ["neon sign glow", "screen reflection", "streetlight", "hologram emission"],
        "invalid_sources": ["natural sunlight", "sky ambient", "warm fire glow"],
    },
    "space_void": {
        "description": "Open space away from any star",
        "ambient_max": 0.02,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["distant star point", "ship navigation light", "suit indicator light"],
        "invalid_sources": ["ambient fill", "sky light", "bounce light", "studio key"],
    },
    "bright_desert_noon": {
        "description": "Open desert at noon",
        "ambient_min": 0.7,
        "rim_light_allowed": True,
        "studio_light_allowed": False,
        "valid_sources": ["direct sun overhead", "sand bounce", "heat shimmer emission"],
        "invalid_sources": ["cool ambient", "neon", "fog fill"],
    },
    "ancient_temple_interior": {
        "description": "Dark stone temple interior with small windows",
        "ambient_max": 0.25,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["shaft of light through opening", "torch on wall", "magical rune glow"],
        "invalid_sources": ["studio fill", "rim highlight", "cool ambient"],
    },
    "snowstorm_exterior": {
        "description": "Outdoor blizzard, heavy snow",
        "ambient_max": 0.3,
        "rim_light_allowed": False,
        "studio_light_allowed": False,
        "valid_sources": ["diffuse grey sky light", "ground snow bounce", "distant lantern"],
        "invalid_sources": ["direct sun", "rim highlight", "warm fire glow"],
    },
}


def _rc(lst):  return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 3)


def _valid_lighting_config(rules):
    ambient_max = rules.get("ambient_max", 1.0)
    ambient_min = rules.get("ambient_min", 0.0)
    ambient     = _rf(ambient_min, min(ambient_max, ambient_min + 0.15))

    sources = [_rc(rules["valid_sources"])]
    if len(rules["valid_sources"]) > 1 and random.random() > 0.5:
        s2 = _rc([s for s in rules["valid_sources"] if s != sources[0]])
        sources.append(s2)

    return {
        "ambient_intensity": ambient,
        "rim_light":         rules.get("rim_light_allowed", False) and random.random() > 0.5,
        "studio_light":      False,
        "light_sources":     sources,
        "valid":             True,
        "semantic_tag":      "matches scene",
    }


def _invalid_lighting_config(rules):
    violation_type = _rc(["studio_light", "rim_light", "wrong_ambient",
                           "wrong_source"])

    if violation_type == "studio_light":
        return {
            "ambient_intensity": _rf(0.5, 0.9),
            "rim_light": True,
            "studio_light": True,
            "light_sources": ["studio key light", "fill light"],
            "valid": False,
            "violation": "studio_lighting_in_dark_scene",
        }
    elif violation_type == "rim_light" and not rules.get("rim_light_allowed", True):
        return {
            "ambient_intensity": _rf(0.3, 0.6),
            "rim_light": True,
            "studio_light": False,
            "light_sources": [_rc(rules["valid_sources"]), "rim highlight"],
            "valid": False,
            "violation": "rim_light_not_permitted_in_this_scene",
        }
    elif violation_type == "wrong_ambient":
        ambient_max = rules.get("ambient_max", 0.2)
        return {
            "ambient_intensity": _rf(ambient_max + 0.2, 0.9),
            "rim_light": False,
            "studio_light": False,
            "light_sources": rules["invalid_sources"][:1],
            "valid": False,
            "violation": f"ambient_too_high_for_{rules['description'].split()[0]}_scene",
        }
    else:  # wrong_source
        return {
            "ambient_intensity": _rf(0.1, 0.3),
            "rim_light": False,
            "studio_light": False,
            "light_sources": rules["invalid_sources"][:2],
            "valid": False,
            "violation": "invalid_light_source_for_scene_semantics",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SemanticLightingDataset
# ─────────────────────────────────────────────────────────────────────────────

class SemanticLightingDataset:
    """
    Generates (scene, valid_lighting, invalid_lighting) training samples.
    Teaches Avus to never hallucinate lighting that contradicts scene semantics.
    """

    def __init__(self):
        self._tags = list(LIGHTING_RULES.keys())

    def _make_sample(self):
        tag   = _rc(self._tags)
        rules = LIGHTING_RULES[tag]

        valid   = _valid_lighting_config(rules)
        invalid = _invalid_lighting_config(rules)
        violation = invalid.get("violation", "semantic_mismatch")

        scene_desc = (
            f"Scene: {rules['description']}. "
            f"The semantic tag [{tag}] constrains all lighting decisions. "
            f"Valid light sources for this scene: {', '.join(rules['valid_sources'])}. "
            f"Forbidden sources: {', '.join(rules['invalid_sources'])}."
        )

        text = (
            f"<|startoftext|>[SEM:{tag}] {scene_desc} "
            f"[LIGHTING_VALID] {json.dumps(valid)} [/LIGHTING_VALID] "
            f"[LIGHTING_INVALID] {json.dumps(invalid)} "
            f"[VIOLATION: {violation}] [/LIGHTING_INVALID]"
            f"<|endoftext|>"
        )
        return text

    def generate_dataset(self, samples: int = 10_000, seed: int = 555
                         ) -> List[str]:
        random.seed(seed)
        out = []
        for _ in range(samples):
            out.append(self._make_sample())
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen     = SemanticLightingDataset()
    samples = gen.generate_dataset(samples=5_000)

    print(f"Generated {len(samples)} samples")
    valid_count   = sum(1 for s in samples if "[LIGHTING_VALID]" in s)
    invalid_count = sum(1 for s in samples if "[LIGHTING_INVALID]" in s)
    sem_count     = sum(1 for s in samples if "[SEM:" in s)
    print(f"Valid lighting:   {valid_count}/{len(samples)}")
    print(f"Invalid lighting: {invalid_count}/{len(samples)}")
    print(f"Semantic tags:    {sem_count}/{len(samples)}")

    print("\n── Sample ──────────────────────────────────────────────")
    print(samples[0][:600] + "...")
