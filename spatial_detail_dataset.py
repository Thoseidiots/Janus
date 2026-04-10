"""
spatial_detail_dataset.py
===========================
Synthetic dataset for training Avus to distinguish Protection Zones
(high-frequency detail) from Smooth Zones (low-frequency areas).
Targets the "Yosification" failure mode — over-smoothing of fine details.

Format:
  <|startoftext|>[SEM:tag] {scene_description}
  [ZONE_PROTECT_START] {high_detail_zone_description} [ZONE_PROTECT_END]
  [ZONE_SMOOTH_START] {low_detail_zone_description} [ZONE_SMOOTH_END]
  [RENDER_PARAMS] {json} [/RENDER_PARAMS]<|endoftext|>

The model learns to generate different detail densities for different zones.
"""

import json
import random
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_TAGS = [
    "portrait_closeup", "character_face", "armor_detail", "fabric_weave",
    "stone_texture", "wood_grain", "metal_surface", "landscape_wide",
    "crowd_scene", "interior_room", "sky_exterior", "underwater_scene",
]

PROTECTION_ZONE_TYPES = [
    "face", "eyes", "skin_pores", "iris_texture", "hair_strands",
    "fabric_weave", "armor_engraving", "inscription_text", "jewel_facets",
    "fingernails", "eyelashes", "lip_texture", "scar_detail", "tattoo",
]

SMOOTH_ZONE_TYPES = [
    "sky", "flat_wall", "calm_water_surface", "open_ground", "fog_layer",
    "plain_cloth_back", "background_foliage", "distant_mountains",
    "plain_floor", "empty_ceiling", "smooth_rock_face", "open_desert",
]

DETAIL_DESCRIPTORS = {
    "face":              ["individual pores visible", "fine wrinkle lines", "subtle colour variation", "subsurface scattering visible under skin"],
    "eyes":              ["iris fibre texture", "specular catchlight", "limbal ring definition", "sclera vein detail"],
    "skin_pores":        ["pore size variation", "micro-relief surface", "sebaceous follicle detail", "surface normal micro-variation"],
    "iris_texture":      ["radial fibre structure", "crypts and furrows", "pigment layer depth", "subtle colour rings"],
    "hair_strands":      ["individual strand separation", "anisotropic specular", "flyaway micro-strands", "root-to-tip colour gradient"],
    "fabric_weave":      ["individual thread crossing", "warp and weft pattern", "thread shadow casting", "loose fibre fraying at edges"],
    "armor_engraving":   ["chisel mark depth", "oxidation in recesses", "paint remnants in grooves", "edge highlight on raised relief"],
    "inscription_text":  ["letter edge sharpness", "carving depth shadow", "wear pattern on letters", "surface roughness variation"],
    "jewel_facets":      ["facet edge definition", "internal refraction colour", "surface scratch micro-detail", "light dispersion pattern"],
    "fingernails":       ["lunula boundary", "surface ridge lines", "translucency gradient", "cuticle edge detail"],
    "eyelashes":         ["individual lash curl", "lash root shadow", "tip taper", "inter-lash spacing"],
    "lip_texture":       ["vertical lip line pattern", "moisture micro-highlight", "pigment variation", "edge definition against skin"],
    "scar_detail":       ["tissue type boundary", "raised edge shadow", "colour desaturation in scar tissue", "pore distortion at edges"],
    "tattoo":            ["ink spread pattern", "line edge softness", "skin texture showing through ink", "colour saturation gradient"],
}

SMOOTH_DESCRIPTORS = {
    "sky":                   ["uniform colour gradient", "no surface texture", "soft cloud edge if present"],
    "flat_wall":             ["low frequency colour variation only", "no micro-detail", "smooth normal map"],
    "calm_water_surface":    ["planar reflection", "minimal surface ripple", "no foam detail"],
    "open_ground":           ["low resolution tiling acceptable", "no close-up pore detail", "smooth displacement"],
    "fog_layer":             ["volumetric density only", "no surface texture", "soft edge blending"],
    "plain_cloth_back":      ["uniform colour", "no weave visible", "smooth normal"],
    "background_foliage":    ["billboard acceptable", "colour blob", "no individual leaf"],
    "distant_mountains":     ["silhouette only", "no surface crack detail", "colour atmosphere only"],
    "plain_floor":           ["tiling low res", "no chip detail", "smooth specular"],
    "empty_ceiling":         ["flat colour", "no crack detail", "minimal texture"],
    "smooth_rock_face":      ["macro displacement only", "no micro-pore", "low freq normal"],
    "open_desert":           ["dune shape only", "no grain detail", "smooth sand normal"],
}


def _rc(lst): return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 2)


def _render_params(protect_zone, smooth_zone):
    return {
        "protection_zone": {
            "type": protect_zone,
            "native_scale": True,
            "detail_density": _rf(0.85, 1.0),
            "sharpening": _rf(0.7, 1.0),
            "normal_map_resolution": _rc([1024, 2048, 4096]),
        },
        "smooth_zone": {
            "type": smooth_zone,
            "native_scale": False,
            "detail_density": _rf(0.1, 0.3),
            "sharpening": _rf(0.0, 0.2),
            "normal_map_resolution": _rc([128, 256, 512]),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# SpatialDetailDataset
# ─────────────────────────────────────────────────────────────────────────────

class SpatialDetailDataset:
    """
    Generates (protection zone, smooth zone, render params) training samples.
    Teaches Avus to apply high detail density only where it matters.
    """

    def __init__(self):
        self._generators = [
            self._character_face_scene,
            self._armor_landscape_scene,
            self._interior_scene,
            self._nature_scene,
        ]

    def _character_face_scene(self):
        tag          = _rc(["portrait_closeup", "character_face"])
        protect_type = _rc(["face", "eyes", "skin_pores", "iris_texture",
                             "eyelashes", "lip_texture", "hair_strands"])
        smooth_type  = _rc(["sky", "background_foliage", "distant_mountains",
                             "flat_wall", "fog_layer"])

        protect_details = DETAIL_DESCRIPTORS.get(protect_type,
                          ["high frequency detail visible"])
        smooth_details  = SMOOTH_DESCRIPTORS.get(smooth_type,
                          ["low frequency only"])

        protect_desc = (f"The {protect_type.replace('_',' ')} region requires "
                        f"maximum detail preservation: "
                        f"{', '.join(protect_details)}.")
        smooth_desc  = (f"The {smooth_type.replace('_',' ')} region requires "
                        f"minimal detail: {', '.join(smooth_details)}.")

        params     = _render_params(protect_type, smooth_type)
        params_str = json.dumps(params, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"A scene requiring zone-aware detail rendering. "
            f"[ZONE_PROTECT_START] {protect_desc} [ZONE_PROTECT_END] "
            f"[ZONE_SMOOTH_START] {smooth_desc} [ZONE_SMOOTH_END] "
            f"[RENDER_PARAMS] {params_str} [/RENDER_PARAMS]"
            f"<|endoftext|>"
        )
        return text

    def _armor_landscape_scene(self):
        tag          = _rc(["armor_detail", "fabric_weave", "stone_texture",
                             "metal_surface", "wood_grain"])
        protect_type = _rc(["armor_engraving", "fabric_weave", "inscription_text",
                             "jewel_facets", "scar_detail", "tattoo"])
        smooth_type  = _rc(["open_ground", "sky", "plain_floor",
                             "empty_ceiling", "open_desert"])

        protect_details = DETAIL_DESCRIPTORS.get(protect_type,
                          ["high frequency detail visible"])
        smooth_details  = SMOOTH_DESCRIPTORS.get(smooth_type,
                          ["low frequency only"])

        protect_desc = (f"The {protect_type.replace('_',' ')} requires "
                        f"native-resolution processing: "
                        f"{', '.join(protect_details)}.")
        smooth_desc  = (f"The {smooth_type.replace('_',' ')} can be "
                        f"processed at reduced resolution: "
                        f"{', '.join(smooth_details)}.")

        params     = _render_params(protect_type, smooth_type)
        params_str = json.dumps(params, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Zone-aware rendering for material detail. "
            f"[ZONE_PROTECT_START] {protect_desc} [ZONE_PROTECT_END] "
            f"[ZONE_SMOOTH_START] {smooth_desc} [ZONE_SMOOTH_END] "
            f"[RENDER_PARAMS] {params_str} [/RENDER_PARAMS]"
            f"<|endoftext|>"
        )
        return text

    def _interior_scene(self):
        tag          = _rc(["interior_room", "portrait_closeup"])
        protect_type = _rc(["fingernails", "hair_strands", "fabric_weave",
                             "inscription_text"])
        smooth_type  = _rc(["flat_wall", "plain_floor", "empty_ceiling",
                             "plain_cloth_back"])

        protect_details = DETAIL_DESCRIPTORS.get(protect_type,
                          ["micro detail preserved"])
        smooth_details  = SMOOTH_DESCRIPTORS.get(smooth_type,
                          ["low frequency only"])

        protect_desc = (f"Interior close-up: {protect_type.replace('_',' ')} "
                        f"detail must be preserved at native scale. "
                        f"Key details: {', '.join(protect_details)}.")
        smooth_desc  = (f"Interior background: {smooth_type.replace('_',' ')} "
                        f"can use reduced resolution. "
                        f"Acceptable: {', '.join(smooth_details)}.")

        params     = _render_params(protect_type, smooth_type)
        params_str = json.dumps(params, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Interior scene with mixed detail requirements. "
            f"[ZONE_PROTECT_START] {protect_desc} [ZONE_PROTECT_END] "
            f"[ZONE_SMOOTH_START] {smooth_desc} [ZONE_SMOOTH_END] "
            f"[RENDER_PARAMS] {params_str} [/RENDER_PARAMS]"
            f"<|endoftext|>"
        )
        return text

    def _nature_scene(self):
        tag          = _rc(["landscape_wide", "sky_exterior", "underwater_scene"])
        protect_type = _rc(["fabric_weave", "armor_engraving", "hair_strands"])
        smooth_type  = _rc(["background_foliage", "distant_mountains",
                             "sky", "calm_water_surface", "open_ground"])

        protect_details = DETAIL_DESCRIPTORS.get(protect_type,
                          ["fine detail required"])
        smooth_details  = SMOOTH_DESCRIPTORS.get(smooth_type,
                          ["low frequency only"])

        protect_desc = (f"Foreground subject {protect_type.replace('_',' ')} "
                        f"in nature scene: {', '.join(protect_details)}.")
        smooth_desc  = (f"Natural background {smooth_type.replace('_',' ')}: "
                        f"{', '.join(smooth_details)}.")

        params     = _render_params(protect_type, smooth_type)
        params_str = json.dumps(params, indent=2)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Nature scene with foreground subject detail contrast. "
            f"[ZONE_PROTECT_START] {protect_desc} [ZONE_PROTECT_END] "
            f"[ZONE_SMOOTH_START] {smooth_desc} [ZONE_SMOOTH_END] "
            f"[RENDER_PARAMS] {params_str} [/RENDER_PARAMS]"
            f"<|endoftext|>"
        )
        return text

    def generate_dataset(self, samples: int = 10_000, seed: int = 222
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
    gen     = SpatialDetailDataset()
    samples = gen.generate_dataset(samples=5_000)

    print(f"Generated {len(samples)} samples\n")
    print("── Sample outputs ──────────────────────────────────────")
    for i in random.sample(range(len(samples)), 3):
        print(f"\n[{i}]:\n{samples[i][:500]}...")

    protect_count = sum(1 for s in samples if "[ZONE_PROTECT_START]" in s)
    smooth_count  = sum(1 for s in samples if "[ZONE_SMOOTH_START]" in s)
    render_count  = sum(1 for s in samples if "[RENDER_PARAMS]" in s)
    print(f"\nZone protect: {protect_count}/{len(samples)}")
    print(f"Zone smooth:  {smooth_count}/{len(samples)}")
    print(f"Render params: {render_count}/{len(samples)}")
