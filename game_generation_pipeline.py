"""
game_generation_pipeline.py
=============================
Connects AvusInference to the aaa_stack procedural generators.
A text prompt goes in, a game-ready asset comes out.

Pipeline:
  Text prompt
      ↓
  AvusInference.generate_3d_params()
      ↓
  Procedural3DGenerator  → mesh (vertices, faces)
  ProceduralPBRGenerator → textures (albedo, roughness, metallic, normal, ao)
      ↓
  GameAsset (combined output)

Usage:
    from game_generation_pipeline import GameGenerationPipeline

    pipeline = GameGenerationPipeline()
    pipeline.load()

    asset = pipeline.generate("A rusted metal barrel covered in moss")
    asset = pipeline.generate_terrain("A volcanic crater with obsidian edges")
    asset = pipeline.generate_character_prop("An ancient elven sword with runes")
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    candidates = [
        Path(__file__).parent,
        Path("/kaggle/working/Janus"),
        Path("/teamspace/studios/this_studio/Janus"),
        Path(os.path.expanduser("~/Janus")),
    ]
    for p in candidates:
        if (p / "model.py").exists():
            return p
    return Path.cwd()


REPO_ROOT = _find_repo_root()
AAA_STACK = REPO_ROOT / "game_ai_database" / "aaa_stack"


def _add_paths():
    for p in [str(REPO_ROOT), str(AAA_STACK)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# GameAsset — output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameAsset:
    """
    Complete game asset produced by the pipeline.
    Contains geometry, textures, metadata, and the parameters used.
    """
    name:          str
    prompt:        str
    avus_params:   Dict[str, Any]           # raw JSON from Avus
    geometry:      Optional[Dict]  = None   # {"vertices": np.array, "faces": np.array}
    textures:      Optional[Dict]  = None   # {"albedo": np.array, "roughness": ..., etc}
    generation_ms: float           = 0.0
    success:       bool            = False
    error:         Optional[str]   = None
    metadata:      Dict            = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Asset: {self.name}",
            f"Prompt: {self.prompt}",
            f"Success: {self.success}",
            f"Time: {self.generation_ms:.0f}ms",
        ]
        if self.success:
            if self.geometry:
                v = self.geometry.get("vertices")
                f = self.geometry.get("faces")
                if v is not None:
                    lines.append(f"Geometry: {len(v)} vertices, "
                                 f"{len(f) if f is not None else 0} faces")
            if self.textures:
                lines.append(f"Textures: {list(self.textures.keys())}")
        else:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialisable summary (numpy arrays excluded)."""
        return {
            "name":          self.name,
            "prompt":        self.prompt,
            "success":       self.success,
            "generation_ms": self.generation_ms,
            "error":         self.error,
            "avus_params":   self.avus_params,
            "has_geometry":  self.geometry is not None,
            "has_textures":  self.textures is not None,
            "texture_maps":  list(self.textures.keys()) if self.textures else [],
            "metadata":      self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GameGenerationPipeline
# ─────────────────────────────────────────────────────────────────────────────

class GameGenerationPipeline:
    """
    End-to-end pipeline: text prompt → Avus params → aaa_stack → GameAsset.

    Degrades gracefully:
    - If Avus weights not found: uses fallback params
    - If aaa_stack not found: returns params only (no geometry/textures)
    - Each stage is independently error-handled
    """

    # Default params when Avus parse fails
    FALLBACK_3D = {
        "object_name": "default_object",
        "geometry": {
            "primitive_type": "sphere",
            "geometry_params": {"radius": 1.0, "segments": 32}
        },
        "material": {
            "material_type": "stone",
            "material_params": {
                "resolution": [512, 512],
                "roughness_scale": 0.7,
                "metallic_scale": 0.0
            }
        }
    }

    def __init__(self):
        self.avus          = None
        self.gen3d         = None
        self.gen_pbr       = None
        self._avus_loaded  = False
        self._stack_loaded = False

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, weights_path: Optional[str] = None) -> bool:
        """Load Avus and aaa_stack generators. Returns True if at least
        one component loaded successfully."""
        self._load_avus(weights_path)
        self._load_aaa_stack()
        ready = self._avus_loaded or self._stack_loaded
        print(f"[Pipeline] avus={self._avus_loaded}  "
              f"aaa_stack={self._stack_loaded}  ready={ready}")
        return ready

    def _load_avus(self, weights_path):
        try:
            from avus_inference import AvusInference
            self.avus = AvusInference()
            self._avus_loaded = self.avus.load(weights_path)
        except Exception as e:
            print(f"[Pipeline] Avus load failed: {e}")
            self._avus_loaded = False

    def _load_aaa_stack(self):
        _add_paths()
        try:
            from procedural_3d import Procedural3DGenerator
            from procedural_pbr import ProceduralPBRGenerator
            self.gen3d     = Procedural3DGenerator()
            self.gen_pbr   = ProceduralPBRGenerator()
            self._stack_loaded = True
            print("[Pipeline] aaa_stack loaded.")
        except ImportError as e:
            print(f"[Pipeline] aaa_stack not found: {e}  "
                  f"(generate() will return params only)")
            self._stack_loaded = False

    # ── main API ──────────────────────────────────────────────────────────────

    def generate(self, prompt: str,
                 texture_resolution: int = 512) -> GameAsset:
        """
        Full pipeline: prompt → Avus → geometry + textures.

        Parameters
        ----------
        prompt : str
            Natural language description of the asset to generate.
        texture_resolution : int
            Resolution for PBR texture maps (128, 256, 512, 1024).

        Returns
        -------
        GameAsset
        """
        t0 = time.time()

        # ── Step 1: Avus → params ──
        params = self._get_params(prompt)

        asset = GameAsset(
            name       = params.get("object_name", "asset"),
            prompt     = prompt,
            avus_params= params,
        )

        # ── Step 2: Geometry ──
        asset.geometry = self._generate_geometry(params)

        # ── Step 3: Textures ──
        asset.textures = self._generate_textures(params, texture_resolution)

        asset.generation_ms = (time.time() - t0) * 1000
        asset.success       = (asset.geometry is not None or
                               asset.textures is not None)
        if not asset.success:
            asset.error = "Both geometry and texture generation failed."

        asset.metadata = {
            "avus_used":  self._avus_loaded,
            "stack_used": self._stack_loaded,
            "resolution": texture_resolution,
        }

        return asset

    def generate_terrain(self, prompt: str,
                         grid_size: int = 32) -> GameAsset:
        """Generate a terrain mesh from a description."""
        t0 = time.time()

        # Force terrain primitive
        if self._avus_loaded:
            raw = self.avus.generate_3d_params(prompt)
            params = raw if raw else {}
        else:
            params = {}

        params["object_name"] = params.get("object_name", "terrain")
        params["geometry"] = {
            "primitive_type": "terrain",
            "geometry_params": {
                "grid_size": [grid_size, grid_size],
                "height_scale": params.get("height_scale", 1.0),
                "octaves": params.get("octaves", 6),
                "scale": 0.1,
                "seed": params.get("seed", 42),
            }
        }
        params["material"] = params.get("material", {
            "material_type": "stone",
            "material_params": {"resolution": [512, 512],
                                "roughness_scale": 0.8,
                                "metallic_scale": 0.0}
        })

        asset = GameAsset(
            name        = "terrain",
            prompt      = prompt,
            avus_params = params,
        )
        asset.geometry     = self._generate_geometry(params)
        asset.textures     = self._generate_textures(params, 512)
        asset.generation_ms = (time.time() - t0) * 1000
        asset.success      = asset.geometry is not None
        return asset

    def generate_character_prop(self, prompt: str) -> GameAsset:
        """Generate a character prop (weapon, armour piece, accessory)."""
        t0     = time.time()
        params = self._get_params(prompt)

        # Character props prefer cylinder/box primitives
        geo    = params.get("geometry", {})
        ptype  = geo.get("primitive_type", "cylinder")
        if ptype not in ["box", "sphere", "cylinder", "torus"]:
            params["geometry"]["primitive_type"] = "cylinder"

        asset = GameAsset(
            name        = params.get("object_name", "prop"),
            prompt      = prompt,
            avus_params = params,
        )
        asset.geometry     = self._generate_geometry(params)
        asset.textures     = self._generate_textures(params, 1024)
        asset.generation_ms = (time.time() - t0) * 1000
        asset.success      = asset.geometry is not None
        return asset

    def generate_batch(self, prompts: list,
                       texture_resolution: int = 512) -> list:
        """Generate multiple assets from a list of prompts."""
        return [self.generate(p, texture_resolution) for p in prompts]

    # ── internal stages ───────────────────────────────────────────────────────

    def _get_params(self, prompt: str) -> Dict:
        """Get 3D params from Avus or fall back to defaults."""
        if self._avus_loaded:
            try:
                params = self.avus.generate_3d_params(prompt)
                if params:
                    return params
            except Exception as e:
                print(f"[Pipeline] Avus param generation failed: {e}")
        return dict(self.FALLBACK_3D)

    def _generate_geometry(self, params: Dict) -> Optional[Dict]:
        """Call Procedural3DGenerator with Avus params."""
        if not self._stack_loaded:
            return None
        try:
            geo_cfg  = params.get("geometry", {})
            ptype    = geo_cfg.get("primitive_type", "sphere")
            geo_params = geo_cfg.get("geometry_params", {})

            result   = self.gen3d.generate(ptype, **geo_params)
            return result
        except Exception as e:
            print(f"[Pipeline] Geometry generation failed: {e}")
            traceback.print_exc()
            return None

    def _generate_textures(self, params: Dict,
                           resolution: int) -> Optional[Dict]:
        """Call ProceduralPBRGenerator with Avus params."""
        if not self._stack_loaded:
            return None
        try:
            mat_cfg    = params.get("material", {})
            mtype      = mat_cfg.get("material_type", "stone")
            mat_params = mat_cfg.get("material_params", {})

            # Override resolution
            mat_params["resolution"] = [resolution, resolution]

            result = self.gen_pbr.generate(mtype, **mat_params)
            return result
        except Exception as e:
            print(f"[Pipeline] Texture generation failed: {e}")
            traceback.print_exc()
            return None

    # ── export helpers ────────────────────────────────────────────────────────

    def export_obj(self, asset: GameAsset,
                   output_path: str) -> bool:
        """
        Export asset geometry to a Wavefront .obj file.
        Returns True on success.
        """
        if not asset.geometry:
            print("[Pipeline] No geometry to export.")
            return False

        vertices = asset.geometry.get("vertices")
        faces    = asset.geometry.get("faces")

        if vertices is None or faces is None:
            print("[Pipeline] Geometry missing vertices or faces.")
            return False

        try:
            with open(output_path, "w") as f:
                f.write(f"# Janus Game Generation Pipeline\n")
                f.write(f"# Asset: {asset.name}\n")
                f.write(f"# Prompt: {asset.prompt}\n\n")
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write("\n")
                for face in faces:
                    # OBJ faces are 1-indexed
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            print(f"[Pipeline] Exported OBJ: {output_path}")
            return True
        except Exception as e:
            print(f"[Pipeline] OBJ export failed: {e}")
            return False

    def export_kiro_scene(self, asset: GameAsset, output_path: str) -> bool:
        """
        Export asset to a KiroScene format (.ks) for the OSS Game Engine.
        """
        try:
            from kiro_scene_exporter import export_to_kiro_scene
            return export_to_kiro_scene(asset, output_path)
        except Exception as e:
            print(f"[Pipeline] KiroScene export failed dynamically: {e}")
            return False

    def export_textures(self, asset: GameAsset,
                        output_dir: str) -> Dict[str, str]:
        """
        Export PBR texture maps as PNG files.
        Returns dict of {map_name: file_path}.
        """
        if not asset.textures:
            print("[Pipeline] No textures to export.")
            return {}

        try:
            import imageio
        except ImportError:
            os.system("pip install imageio -q")
            import imageio

        os.makedirs(output_dir, exist_ok=True)
        exported = {}

        for map_name, data in asset.textures.items():
            if not isinstance(data, np.ndarray):
                continue
            try:
                fname = f"{asset.name}_{map_name}.png"
                fpath = os.path.join(output_dir, fname)

                # Normalise to 0-255 uint8
                arr = data.copy()
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

                imageio.imwrite(fpath, arr)
                exported[map_name] = fpath
                print(f"[Pipeline] Exported {map_name}: {fpath}")
            except Exception as e:
                print(f"[Pipeline] Texture export failed for {map_name}: {e}")

        return exported

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._avus_loaded or self._stack_loaded

    def __repr__(self) -> str:
        return (f"GameGenerationPipeline("
                f"avus={self._avus_loaded}, "
                f"aaa_stack={self._stack_loaded})")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = GameGenerationPipeline()
    pipeline.load()

    print(f"\nPipeline: {pipeline}\n")

    prompts = [
        "Generate a rusted metal barrel covered in moss",
        "Create an ancient stone pillar with rune carvings",
        "Design a glowing crystal formation",
    ]

    for prompt in prompts:
        print(f"\n{'─'*60}")
        print(f"Prompt: {prompt}")
        asset = pipeline.generate(prompt)
        print(asset.summary())
        print(json.dumps(asset.to_dict(), indent=2))

    print(f"\n{'─'*60}")
    print("Terrain test:")
    terrain = pipeline.generate_terrain(
        "A volcanic crater with obsidian edges and steam vents")
    print(terrain.summary())

    print(f"\n{'─'*60}")
    print("Prop test:")
    prop = pipeline.generate_character_prop(
        "An ancient elven sword with glowing runes along the blade")
    print(prop.summary())
