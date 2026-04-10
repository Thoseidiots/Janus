"""
optical_flow_dataset.py
=========================
Synthetic dataset for training Avus to predict pixel motion between frames.
Targets the "Ghosting / Disocclusion" failure mode.

Format:
  <|startoftext|>[SEM:tag]
  [FRAME_A] {scene_state_json} [/FRAME_A]
  [MOTION_VECTORS] {motion_json} [/MOTION_VECTORS]
  [OCCLUSION_MASK] {occlusion_json} [/OCCLUSION_MASK]
  [FRAME_B_PREDICTION] {predicted_state_json} [/FRAME_B_PREDICTION]
  <|endoftext|>

The model learns to predict Frame B from Frame A + motion vectors,
including correct background fill for disoccluded regions.
"""

import json
import random
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_TAGS = [
    "action_scene", "combat_sequence", "vehicle_chase", "crowd_movement",
    "projectile_flight", "door_opening", "character_walk", "falling_object",
    "swinging_weapon", "flag_in_wind", "running_character", "sliding_door",
]

MOVING_OBJECTS = [
    "character", "sword", "arrow", "shield", "torch", "door",
    "crate", "horse", "cart", "boulder", "fireball", "flag",
    "cloak", "banner", "projectile", "vehicle",
]

BACKGROUND_TYPES = [
    "stone wall with torch sconces",
    "wooden floor with scattered debris",
    "cobblestone street",
    "grassy field with distant trees",
    "dungeon corridor with iron bars",
    "market stalls with hanging cloth",
    "castle rampart with battlements",
    "forest path with roots",
    "interior room with furniture",
    "cliff face with rock ledges",
]

MOTION_DIRECTIONS = ["left", "right", "up", "down",
                     "toward_camera", "away_from_camera",
                     "diagonal_upper_left", "diagonal_upper_right",
                     "diagonal_lower_left", "diagonal_lower_right"]


def _rc(lst):  return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 3)


def _motion_vector(direction, speed):
    """Convert direction + speed to approximate pixel delta."""
    mapping = {
        "left":                 [-speed, 0],
        "right":                [speed,  0],
        "up":                   [0, -speed],
        "down":                 [0,  speed],
        "toward_camera":        [0,  0],   # scale change, not translation
        "away_from_camera":     [0,  0],
        "diagonal_upper_left":  [-speed * 0.7,  -speed * 0.7],
        "diagonal_upper_right": [speed * 0.7,   -speed * 0.7],
        "diagonal_lower_left":  [-speed * 0.7,   speed * 0.7],
        "diagonal_lower_right": [speed * 0.7,    speed * 0.7],
    }
    v = mapping.get(direction, [0, 0])
    return [round(v[0], 3), round(v[1], 3)]


def _build_scene_state(obj_name, bg, x, y, w, h):
    return {
        "moving_object": {
            "name":   obj_name,
            "bbox":   {"x": x, "y": y, "w": w, "h": h},
            "visible": True,
        },
        "background": bg,
        "occluded_region": None,
    }


def _build_frame_b(state_a, mv, obj_name, bg, w, h):
    """Predict Frame B given Frame A and motion vector."""
    ax = state_a["moving_object"]["bbox"]["x"]
    ay = state_a["moving_object"]["bbox"]["y"]
    aw = state_a["moving_object"]["bbox"]["w"]
    ah = state_a["moving_object"]["bbox"]["h"]

    bx = round(ax + mv[0], 3)
    by = round(ay + mv[1], 3)

    # Disoccluded region = where object was in A but is no longer in B
    disoccluded = {
        "x": ax, "y": ay, "w": aw, "h": ah,
        "fill": f"Background visible: {bg}",
        "confidence": _rf(0.7, 0.99),
    }

    return {
        "moving_object": {
            "name":   obj_name,
            "bbox":   {"x": bx, "y": by, "w": aw, "h": ah},
            "visible": 0 <= bx <= 1 and 0 <= by <= 1,
        },
        "background": bg,
        "disoccluded_region": disoccluded,
        "warping_applied": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpticalFlowDataset
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowDataset:
    """
    Generates (Frame A, motion vectors, Frame B prediction) training samples.
    Teaches Avus to correctly fill disoccluded background regions.
    """

    def __init__(self):
        self._generators = [
            self._simple_translation,
            self._fast_motion,
            self._partial_exit,
            self._multi_object,
        ]

    def _simple_translation(self):
        tag       = _rc(SEMANTIC_TAGS)
        obj       = _rc(MOVING_OBJECTS)
        bg        = _rc(BACKGROUND_TYPES)
        direction = _rc(MOTION_DIRECTIONS)
        speed     = _rf(0.02, 0.15)
        mv        = _motion_vector(direction, speed)

        x, y = _rf(0.1, 0.7), _rf(0.1, 0.7)
        w, h = _rf(0.05, 0.25), _rf(0.05, 0.3)

        state_a  = _build_scene_state(obj, bg, x, y, w, h)
        state_b  = _build_frame_b(state_a, mv, obj, bg, w, h)
        motion   = {"direction": direction, "speed": speed, "vector": mv,
                    "object": obj}

        occlusion = {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "type": "full_disocclusion",
            "background_fill_required": True,
        }

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"[FRAME_A] {json.dumps(state_a)} [/FRAME_A] "
            f"[MOTION_VECTORS] {json.dumps(motion)} [/MOTION_VECTORS] "
            f"[OCCLUSION_MASK] {json.dumps(occlusion)} [/OCCLUSION_MASK] "
            f"[FRAME_B_PREDICTION] {json.dumps(state_b)} [/FRAME_B_PREDICTION]"
            f"<|endoftext|>"
        )
        return text

    def _fast_motion(self):
        """Fast motion with motion blur — higher ghosting risk."""
        tag       = _rc(SEMANTIC_TAGS)
        obj       = _rc(MOVING_OBJECTS)
        bg        = _rc(BACKGROUND_TYPES)
        direction = _rc(MOTION_DIRECTIONS)
        speed     = _rf(0.2, 0.5)   # fast
        mv        = _motion_vector(direction, speed)

        x, y = _rf(0.1, 0.6), _rf(0.1, 0.6)
        w, h = _rf(0.05, 0.2), _rf(0.05, 0.25)

        state_a  = _build_scene_state(obj, bg, x, y, w, h)
        state_b  = _build_frame_b(state_a, mv, obj, bg, w, h)
        motion   = {"direction": direction, "speed": speed, "vector": mv,
                    "object": obj, "motion_blur": True,
                    "blur_samples": _ri(4, 16)}

        occlusion = {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "type": "fast_motion_disocclusion",
            "background_fill_required": True,
            "fill_confidence_penalty": _rf(0.1, 0.3),
        }

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Fast motion sequence — high ghosting risk. "
            f"[FRAME_A] {json.dumps(state_a)} [/FRAME_A] "
            f"[MOTION_VECTORS] {json.dumps(motion)} [/MOTION_VECTORS] "
            f"[OCCLUSION_MASK] {json.dumps(occlusion)} [/OCCLUSION_MASK] "
            f"[FRAME_B_PREDICTION] {json.dumps(state_b)} [/FRAME_B_PREDICTION]"
            f"<|endoftext|>"
        )
        return text

    def _partial_exit(self):
        """Object partially exits frame — edge case for background fill."""
        tag  = _rc(SEMANTIC_TAGS)
        obj  = _rc(MOVING_OBJECTS)
        bg   = _rc(BACKGROUND_TYPES)
        direction = _rc(["left", "right", "up", "down"])
        speed     = _rf(0.1, 0.3)
        mv        = _motion_vector(direction, speed)

        # Place object near an edge so it partially exits
        edge_positions = {
            "left":  (0.02, _rf(0.2, 0.7)),
            "right": (0.75, _rf(0.2, 0.7)),
            "up":    (_rf(0.2, 0.7), 0.02),
            "down":  (_rf(0.2, 0.7), 0.75),
        }
        x, y = edge_positions.get(direction, (0.5, 0.5))
        w, h = _rf(0.1, 0.2), _rf(0.1, 0.2)

        state_a  = _build_scene_state(obj, bg, x, y, w, h)
        state_b  = _build_frame_b(state_a, mv, obj, bg, w, h)
        motion   = {"direction": direction, "speed": speed, "vector": mv,
                    "object": obj, "partial_exit": True}

        occlusion = {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "type": "partial_exit",
            "background_fill_required": True,
            "edge_case": True,
        }

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Object partially exiting frame edge. "
            f"[FRAME_A] {json.dumps(state_a)} [/FRAME_A] "
            f"[MOTION_VECTORS] {json.dumps(motion)} [/MOTION_VECTORS] "
            f"[OCCLUSION_MASK] {json.dumps(occlusion)} [/OCCLUSION_MASK] "
            f"[FRAME_B_PREDICTION] {json.dumps(state_b)} [/FRAME_B_PREDICTION]"
            f"<|endoftext|>"
        )
        return text

    def _multi_object(self):
        """Multiple objects with independent motion vectors."""
        tag  = _rc(SEMANTIC_TAGS)
        bg   = _rc(BACKGROUND_TYPES)
        n    = _ri(2, 4)
        objects  = random.sample(MOVING_OBJECTS, n)
        motions  = []
        state_as = []
        state_bs = []

        for obj in objects:
            direction = _rc(MOTION_DIRECTIONS)
            speed     = _rf(0.02, 0.2)
            mv        = _motion_vector(direction, speed)
            x, y      = _rf(0.05, 0.75), _rf(0.05, 0.75)
            w, h      = _rf(0.05, 0.2),  _rf(0.05, 0.2)
            sa        = _build_scene_state(obj, bg, x, y, w, h)
            sb        = _build_frame_b(sa, mv, obj, bg, w, h)
            motions.append({"object": obj, "direction": direction,
                             "speed": speed, "vector": mv})
            state_as.append(sa)
            state_bs.append(sb)

        text = (
            f"<|startoftext|>[SEM:{tag}] "
            f"Multi-object scene with {n} independent motion vectors. "
            f"[FRAME_A] {json.dumps(state_as)} [/FRAME_A] "
            f"[MOTION_VECTORS] {json.dumps(motions)} [/MOTION_VECTORS] "
            f"[OCCLUSION_MASK] {json.dumps({'count': n, 'independent': True})} [/OCCLUSION_MASK] "
            f"[FRAME_B_PREDICTION] {json.dumps(state_bs)} [/FRAME_B_PREDICTION]"
            f"<|endoftext|>"
        )
        return text

    def generate_dataset(self, samples: int = 10_000, seed: int = 444
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
    gen     = OpticalFlowDataset()
    samples = gen.generate_dataset(samples=5_000)

    print(f"Generated {len(samples)} samples")
    frame_a  = sum(1 for s in samples if "[FRAME_A]" in s)
    frame_b  = sum(1 for s in samples if "[FRAME_B_PREDICTION]" in s)
    motion   = sum(1 for s in samples if "[MOTION_VECTORS]" in s)
    print(f"Frame A:         {frame_a}/{len(samples)}")
    print(f"Frame B pred:    {frame_b}/{len(samples)}")
    print(f"Motion vectors:  {motion}/{len(samples)}")

    print("\n── Sample ──────────────────────────────────────────────")
    print(samples[0][:500] + "...")
