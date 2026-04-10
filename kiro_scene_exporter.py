import os
from pathlib import Path
from game_generation_pipeline import GameAsset

def export_to_kiro_scene(asset: GameAsset, output_path: str) -> bool:
    """
    Takes a GameAsset from the pipeline and writes a valid .ks KiroScene file 
    that the OSS game engine can load directly.
    """
    if not asset.success:
        print(f"[KiroSceneExporter] Cannot export failed asset: {asset.name}")
        return False
        
    try:
        # We assume the engine expects glb and mat files under assets/
        # with the same base name.
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in asset.name)
        
        # Valid KiroScene Format output
        ks_content = (
            f'scene "{asset.name}" version 1\n'
            f'  entity 1\n'
            f'    transform position 0.0 0.0 0.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0\n'
            f'    mesh_renderer mesh "assets/{clean_name}.glb" material "assets/{clean_name}.mat"\n'
        )

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ks_content)
            
        print(f"[KiroSceneExporter] Exported KiroScene: {output_path}")
        return True
    except Exception as e:
        print(f"[KiroSceneExporter] KiroScene export failed: {e}")
        return False
