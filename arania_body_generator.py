"""
arania_body_generator.py
========================
Generates a complete full-body 3D mesh for Arania (Janus avatar)
using the AAA game generation pipeline and Arania.mat material data.

Outputs:
    assets/arania_body.obj   - Full body mesh (head + torso + limbs + costume)
    assets/arania_body.mat   - Extended PBR material (references Arania.mat)
    assets/arania_world.ks   - KiroScene file for the OSS engine

Usage:
    python arania_body_generator.py
    python arania_body_generator.py --output-dir assets/ --scale 1.8
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

WORKSPACE = Path(__file__).parent
ASSETS_DIR = WORKSPACE / "assets"


# -- Material constants from Arania.mat ---------------------------------------

ARANIA_MAT = {
    "skin":      {"color": [0.784, 0.569, 0.416], "roughness": 0.42, "metallic": 0.0, "ao": 0.95, "sss": 0.28},
    "hair":      {"color": [0.102, 0.039, 0.008], "roughness": 0.35, "metallic": 0.0, "ao": 1.0},
    "robe":      {"color": [0.784, 0.627, 0.157], "roughness": 0.72, "metallic": 0.05, "ao": 0.9},
    "gold":      {"color": [0.941, 0.753, 0.188], "roughness": 0.18, "metallic": 0.95, "ao": 1.0},
    "eye_iris":  {"color": [1.0,   0.816, 0.376], "roughness": 0.05, "metallic": 0.0, "emissive": 0.9},
    "eye_white": {"color": [0.973, 0.941, 0.878], "roughness": 0.20, "metallic": 0.0},
    "lip":       {"color": [0.733, 0.439, 0.333], "roughness": 0.38, "metallic": 0.0},
}

LIGHTS = {
    "sun":  {"pos": [4.0, 8.0, 5.0],   "color": [1.0, 0.941, 0.816], "intensity": 2.2},
    "rim":  {"pos": [-4.0, 2.0, -3.0], "color": [0.376, 0.251, 0.753], "intensity": 0.45},
    "fill": {"pos": [0.0, -2.0, 3.0],  "color": [0.784, 0.565, 0.314], "intensity": 0.35},
}


# -- Geometry helpers ----------------------------------------------------------

def _sphere(cx, cy, cz, rx, ry, rz, segments=20, rings=14):
    """Generate ellipsoid vertices + faces."""
    verts, faces = [], []
    for ri in range(rings + 1):
        theta = math.pi * ri / rings
        for si in range(segments):
            phi = 2 * math.pi * si / segments
            x = cx + rx * math.sin(theta) * math.cos(phi)
            y = cy + ry * math.cos(theta)
            z = cz + rz * math.sin(theta) * math.sin(phi)
            verts.append((x, y, z))
    base = 0
    for ri in range(rings):
        for si in range(segments):
            a = base + ri * segments + si
            b = base + ri * segments + (si + 1) % segments
            c = base + (ri + 1) * segments + si
            d = base + (ri + 1) * segments + (si + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return verts, faces


def _capsule(cx, cy, cz, radius, half_height, segments=16, rings=8, material=None):
    """Generate capsule (cylinder + hemispherical caps)."""
    verts, faces = [], []

    # Top hemisphere
    for ri in range(rings + 1):
        theta = math.pi * 0.5 * ri / rings
        for si in range(segments):
            phi = 2 * math.pi * si / segments
            x = cx + radius * math.cos(theta) * math.cos(phi)
            y = cy + half_height + radius * math.sin(theta)
            z = cz + radius * math.cos(theta) * math.sin(phi)
            verts.append((x, y, z))

    # Bottom hemisphere
    for ri in range(rings + 1):
        theta = math.pi * 0.5 + math.pi * 0.5 * ri / rings
        for si in range(segments):
            phi = 2 * math.pi * si / segments
            x = cx + radius * math.cos(theta) * math.cos(phi)
            y = cy - half_height + radius * math.sin(theta)
            z = cz + radius * math.cos(theta) * math.sin(phi)
            verts.append((x, y, z))

    total_rings = (rings + 1) * 2
    for ri in range(total_rings - 1):
        for si in range(segments):
            a = ri * segments + si
            b = ri * segments + (si + 1) % segments
            c = (ri + 1) * segments + si
            d = (ri + 1) * segments + (si + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))

    return verts, faces


def _torus(cx, cy, cz, major_r, minor_r, segments=24, rings=12):
    """Generate torus (ring/collar)."""
    verts, faces = [], []
    for ri in range(rings):
        theta = 2 * math.pi * ri / rings
        for si in range(segments):
            phi = 2 * math.pi * si / segments
            x = cx + (major_r + minor_r * math.cos(phi)) * math.cos(theta)
            y = cy + minor_r * math.sin(phi)
            z = cz + (major_r + minor_r * math.cos(phi)) * math.sin(theta)
            verts.append((x, y, z))
    for ri in range(rings):
        for si in range(segments):
            a = ri * segments + si
            b = ri * segments + (si + 1) % segments
            c = ((ri + 1) % rings) * segments + si
            d = ((ri + 1) % rings) * segments + (si + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return verts, faces


def _merge(parts):
    """Merge list of (verts, faces, mat_name) into single OBJ data."""
    all_verts = []
    all_groups = []   # list of (mat_name, faces_absolute)
    offset = 0
    for verts, faces, mat in parts:
        abs_faces = [(f[0] + offset, f[1] + offset, f[2] + offset) for f in faces]
        all_verts.extend(verts)
        all_groups.append((mat, abs_faces))
        offset += len(verts)
    return all_verts, all_groups


# -- Body builder -------------------------------------------------------------

class AraniaBodyBuilder:
    """
    Builds a full-body mesh for Arania in character space:
        Y-up, Z-forward, feet at Y=0, top of head at Y-1.8 (metres).

    Body proportions follow stylised fantasy humanoid (7-head proportion).
    Head height = 0.22m, total height = 1.76m.
    """

    # Anatomical landmarks (Y-up, Z-forward)
    SCALE = 1.0           # change via --scale flag
    HEAD_Y   = 1.54       # chin height
    NECK_Y   = 1.50
    SHOULDER_Y = 1.38
    CHEST_Y  = 1.18
    WAIST_Y  = 0.92
    HIP_Y    = 0.80
    KNEE_Y   = 0.46
    ANKLE_Y  = 0.08

    def build(self) -> List[Tuple]:
        """Return list of (verts, faces, mat_name) parts."""
        parts = []
        s = self.SCALE

        # -- Head -------------------------------------------------------------
        # Slightly elongated ellipsoid
        hv, hf = _sphere(0, (self.HEAD_Y + 0.11) * s, 0,
                         0.11 * s, 0.135 * s, 0.11 * s, segments=32, rings=22)
        parts.append((hv, hf, "skin"))

        # -- Eyes -------------------------------------------------------------
        for xsign in (+1, -1):
            ev, ef = _sphere(xsign * 0.042 * s, (self.HEAD_Y + 0.06) * s, 0.095 * s,
                             0.018 * s, 0.018 * s, 0.018 * s, segments=14, rings=10)
            parts.append((ev, ef, "eye_white"))
            irv, irf = _sphere(xsign * 0.042 * s, (self.HEAD_Y + 0.06) * s, 0.109 * s,
                               0.010 * s, 0.010 * s, 0.005 * s, segments=14, rings=10)
            parts.append((irv, irf, "eye_iris"))

        # -- Hair cap ---------------------------------------------------------
        haircap_v, haircap_f = _sphere(0, (self.HEAD_Y + 0.12) * s, -0.01 * s,
                                        0.118 * s, 0.145 * s, 0.118 * s,
                                        segments=28, rings=12)
        # Keep only top half
        top_verts = [(x, y, z) for x, y, z in haircap_v if y >= (self.HEAD_Y + 0.09) * s]
        if top_verts:
            parts.append((top_verts, [], "hair"))  # No faces needed -- visual bulk

        # Full hair cap (separate volume)
        hcv, hcf = _sphere(0, (self.HEAD_Y + 0.135) * s, -0.008 * s,
                           0.116 * s, 0.14 * s, 0.116 * s, segments=24, rings=10)
        parts.append((hcv, hcf, "hair"))

        # Side hair (flowing strands)
        for xsign in (+1, -1):
            sv, sf = _capsule(xsign * 0.13 * s, (self.HEAD_Y - 0.05) * s, -0.01 * s,
                              0.055 * s, 0.15 * s, segments=14, rings=6)
            parts.append((sv, sf, "hair"))

        # Back hair mass
        bv, bf = _sphere(0, (self.HEAD_Y - 0.04) * s, -0.11 * s,
                         0.10 * s, 0.18 * s, 0.08 * s, segments=20, rings=14)
        parts.append((bv, bf, "hair"))

        # -- Neck -------------------------------------------------------------
        nv, nf = _capsule(0, self.NECK_Y * s, 0,
                          0.038 * s, 0.04 * s, segments=16, rings=4)
        parts.append((nv, nf, "skin"))

        # -- Gold collar ------------------------------------------------------
        tv, tf = _torus(0, (self.NECK_Y - 0.02) * s, 0,
                        0.05 * s, 0.012 * s, segments=24, rings=12)
        parts.append((tv, tf, "gold"))

        # -- Torso (chest) -----------------------------------------------------
        cv, cf = _capsule(0, self.CHEST_Y * s, 0,
                          0.14 * s, 0.12 * s, segments=22, rings=8)
        parts.append((cv, cf, "robe"))

        # -- Waist ------------------------------------------------------------
        wv, wf = _capsule(0, self.WAIST_Y * s, 0,
                          0.11 * s, 0.06 * s, segments=20, rings=6)
        parts.append((wv, wf, "robe"))

        # -- Hips / Skirt base -------------------------------------------------
        hpv, hpf = _capsule(0, self.HIP_Y * s, 0,
                            0.16 * s, 0.06 * s, segments=22, rings=6)
        parts.append((hpv, hpf, "robe"))

        # Full robe skirt (cone-ish ellipsoid)
        skv, skf = _sphere(0, (self.HIP_Y - 0.18) * s, 0,
                           0.20 * s, 0.22 * s, 0.18 * s, segments=28, rings=16)
        parts.append((skv, skf, "robe"))

        # Gold belt ring
        beltv, beltf = _torus(0, self.HIP_Y * s, 0,
                              0.13 * s, 0.014 * s, segments=28, rings=12)
        parts.append((beltv, beltf, "gold"))

        # -- Upper arms -------------------------------------------------------
        for xsign in (+1, -1):
            uav, uaf = _capsule(xsign * 0.19 * s, (self.SHOULDER_Y - 0.08) * s, 0,
                                0.046 * s, 0.085 * s, segments=14, rings=5)
            parts.append((uav, uaf, "skin"))
            # Sleeve
            slv, slf = _capsule(xsign * 0.20 * s, (self.SHOULDER_Y - 0.08) * s, 0,
                                0.052 * s, 0.082 * s, segments=14, rings=5)
            parts.append((slv, slf, "robe"))

        # -- Lower arms -------------------------------------------------------
        for xsign in (+1, -1):
            lav, laf = _capsule(xsign * 0.22 * s, (self.SHOULDER_Y - 0.27) * s, 0,
                                0.038 * s, 0.085 * s, segments=14, rings=5)
            parts.append((lav, laf, "skin"))

        # -- Hands ------------------------------------------------------------
        for xsign in (+1, -1):
            handv, handf = _sphere(xsign * 0.23 * s, (self.SHOULDER_Y - 0.42) * s, 0,
                                   0.038 * s, 0.048 * s, 0.025 * s, segments=14, rings=10)
            parts.append((handv, handf, "skin"))

        # -- Legs (under skirt -- partially visible) ---------------------------
        for xsign in (+1, -1):
            ulv, ulf = _capsule(xsign * 0.07 * s, self.KNEE_Y * s, 0,
                                0.06 * s, 0.16 * s, segments=16, rings=6)
            parts.append((ulv, ulf, "skin"))
            llv, llf = _capsule(xsign * 0.065 * s, self.ANKLE_Y * s, 0,
                                0.048 * s, 0.14 * s, segments=16, rings=6)
            parts.append((llv, llf, "skin"))

        # -- Feet / Shoes -----------------------------------------------------
        for xsign in (+1, -1):
            fv, ff = _sphere(xsign * 0.065 * s, 0.04 * s, 0.02 * s,
                             0.055 * s, 0.038 * s, 0.09 * s, segments=16, rings=10)
            parts.append((fv, ff, "gold"))

        return parts


# -- OBJ / MAT writer ---------------------------------------------------------

def write_obj(parts: List[Tuple], obj_path: Path, mat_path: Path) -> None:
    """Write merged mesh as Wavefront OBJ with usemtl groups."""
    all_verts, groups = _merge(parts)

    with open(obj_path, "w") as f:
        f.write("# Arania Full-Body Mesh -- generated by arania_body_generator.py\n")
        f.write(f"mtllib {mat_path.name}\n\n")

        for x, y, z in all_verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")

        for mat_name, faces in groups:
            if not faces:
                continue
            f.write(f"usemtl {mat_name}\n")
            for a, b, c in faces:
                # OBJ is 1-indexed
                f.write(f"f {a+1} {b+1} {c+1}\n")
            f.write("\n")

    print(f"[body] Wrote mesh  -> {obj_path}  ({len(all_verts)} verts)")


def write_mtl(mat_path: Path) -> None:
    """Write a Wavefront .mtl that mirrors Arania.mat PBR values."""
    with open(mat_path, "w") as f:
        f.write("# Arania PBR Materials\n")
        for name, props in ARANIA_MAT.items():
            r, g, b = props["color"]
            f.write(f"\nnewmtl {name}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            f.write(f"Ka 0.03 0.03 0.03\n")
            f.write(f"Ks {props.get('metallic', 0.0):.4f} "
                    f"{props.get('metallic', 0.0):.4f} "
                    f"{props.get('metallic', 0.0):.4f}\n")
            f.write(f"Ns {max(1.0, (1.0 - props.get('roughness', 0.5)) * 200):.1f}\n")
            f.write(f"d 1.0\n")
            if props.get("emissive", 0) > 0:
                e = props["emissive"]
                f.write(f"Ke {r*e:.4f} {g*e:.4f} {b*e:.4f}\n")

    print(f"[body] Wrote mat   -> {mat_path}")


def write_kiro_scene(obj_path: Path, mat_path: Path, ks_path: Path) -> None:
    """Write a KiroScene .ks file for the OSS engine to load."""
    rel_obj = obj_path.name
    rel_mat = mat_path.name

    content = f"""scene "arania_world" version 1

  # -- Arania character ------------------------------------------------------
  entity 1
    transform position 0.0 0.0 0.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    mesh_renderer mesh "assets/{rel_obj}" material "assets/{rel_mat}"
    animator controller "arania_controller" idle_anim "idle" walk_anim "walk"
    character_controller speed 1.4 turn_speed 3.0 nav_mode "screen_edge"

  # -- Desktop background plane -----------------------------------------------
  entity 2
    transform position 0.0 0.0 -5.0 rotation 0.0 0.0 0.0 1.0 scale 16.0 9.0 1.0
    mesh_renderer mesh "builtin/quad" material "builtin/desktop_capture"
    desktop_capture update_hz 30

  # -- Lights (from Arania.mat) -----------------------------------------------
  entity 3
    transform position {LIGHTS['sun']['pos'][0]} {LIGHTS['sun']['pos'][1]} {LIGHTS['sun']['pos'][2]} rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    directional_light color {LIGHTS['sun']['color'][0]:.3f} {LIGHTS['sun']['color'][1]:.3f} {LIGHTS['sun']['color'][2]:.3f} intensity {LIGHTS['sun']['intensity']}

  entity 4
    transform position {LIGHTS['rim']['pos'][0]} {LIGHTS['rim']['pos'][1]} {LIGHTS['rim']['pos'][2]} rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    point_light color {LIGHTS['rim']['color'][0]:.3f} {LIGHTS['rim']['color'][1]:.3f} {LIGHTS['rim']['color'][2]:.3f} intensity {LIGHTS['rim']['intensity']} radius 20.0

  entity 5
    transform position {LIGHTS['fill']['pos'][0]} {LIGHTS['fill']['pos'][1]} {LIGHTS['fill']['pos'][2]} rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    point_light color {LIGHTS['fill']['color'][0]:.3f} {LIGHTS['fill']['color'][1]:.3f} {LIGHTS['fill']['color'][2]:.3f} intensity {LIGHTS['fill']['intensity']} radius 14.0

  # -- Navigation waypoints (screen edges) -----------------------------------
  waypoints
    point 0  -6.0 0.0 0.0
    point 1  -6.0 0.0 4.0
    point 2   0.0 0.0 4.0
    point 3   6.0 0.0 4.0
    point 4   6.0 0.0 0.0
    point 5   6.0 0.0 -4.0
    point 6   0.0 0.0 -4.0
    point 7  -6.0 0.0 -4.0
"""
    ks_path.write_text(content)
    print(f"[body] Wrote scene -> {ks_path}")


def write_extended_mat_json(ks_dir: Path) -> None:
    """Write extended Arania material JSON (full Arania.mat for the engine renderer)."""
    mat = {
        "material_name": "Arania_PBR_FullBody",
        "shader": "oss://engine-renderer/pbr_cook_torrance",
        "shader_source": "engine-renderer/src/shaders.rs::PBR_FRAG_GLSL",
        "passes": ARANIA_MAT,
        "lights": LIGHTS,
        "blend_shapes": {
            "neutral": "assets/face_neutral.obj",
            "smile":   "assets/face_smile.obj",
        },
        "face_data": {
            "neutral": "assets/face_neutral.json",
            "smile":   "assets/face_smile.json",
        },
        "animation": {
            "idle":  {"head_sway": True, "breathe": True, "blink_interval_s": 4.0},
            "walk":  {"stride_length": 0.6, "arm_swing": True},
            "talk":  {"lip_sync": "janus_tts", "eye_dart": True},
        }
    }
    out = ks_dir / "arania_body.mat.json"
    out.write_text(json.dumps(mat, indent=2))
    print(f"[body] Wrote JSON  -> {out}")


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Arania full-body 3D asset")
    parser.add_argument("--output-dir", default=str(ASSETS_DIR),
                        help="Directory to write outputs (default: assets/)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Body scale multiplier (default: 1.0 = 1.76m tall)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    AraniaBodyBuilder.SCALE = args.scale

    print("[body] Building Arania full-body mesh...")
    builder = AraniaBodyBuilder()
    parts = builder.build()

    total_verts = sum(len(v) for v, f, m in parts)
    total_faces = sum(len(f) for v, f, m in parts)
    print(f"[body] Parts: {len(parts)}  |  Verts: {total_verts}  |  Faces: {total_faces}")

    obj_path = out_dir / "arania_body.obj"
    mat_path = out_dir / "arania_body.mtl"
    ks_path  = WORKSPACE / "arania_world.ks"

    write_obj(parts, obj_path, mat_path)
    write_mtl(mat_path)
    write_kiro_scene(obj_path, mat_path, ks_path)
    write_extended_mat_json(out_dir)

    print("\n[body] OK Arania body generation complete!")
    print(f"         Mesh   : {obj_path}")
    print(f"         Mat    : {mat_path}")
    print(f"         Scene  : {ks_path}")
    print(f"\n[body] Next step: python launch_arania.py")


if __name__ == "__main__":
    main()
