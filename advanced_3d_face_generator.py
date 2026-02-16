“””
Advanced 3D Face Generator for Janus
Generates realistic 3D face meshes with expressions, textures, and rigging
No external API dependencies - fully procedural generation
“””

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import hashlib
from datetime import datetime

@dataclass
class Vector3:
x: float
y: float
z: float

```
def to_list(self):
    return [self.x, self.y, self.z]

def __add__(self, other):
    return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

def __mul__(self, scalar):
    return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
```

@dataclass
class BlendShape:
“”“Morph target for facial expressions”””
name: str
deltas: List[Vector3]
weight: float = 0.0

@dataclass
class FacialFeatures:
“”“Parametric control for face generation”””
# Head shape
head_width: float = 1.0
head_height: float = 1.0
head_depth: float = 1.0
jaw_width: float = 1.0
chin_prominence: float = 0.5

```
# Eyes
eye_size: float = 1.0
eye_spacing: float = 1.0
eye_tilt: float = 0.0
eye_depth: float = 0.5

# Nose
nose_width: float = 1.0
nose_length: float = 1.0
nose_bridge_height: float = 0.5
nostril_size: float = 0.5

# Mouth
mouth_width: float = 1.0
lip_thickness: float = 0.5
mouth_height: float = 0.5

# Ears
ear_size: float = 1.0
ear_angle: float = 0.0

# Skin
skin_roughness: float = 0.5
skin_tone_r: float = 0.9
skin_tone_g: float = 0.7
skin_tone_b: float = 0.6

# Age markers
age: float = 0.5  # 0-1 scale
wrinkle_intensity: float = 0.0
```

class ProceduralFaceGenerator:
“”“Generates 3D face meshes procedurally with no external dependencies”””

```
def __init__(self):
    self.base_topology = self._create_base_topology()
    self.blend_shapes = self._create_blend_shapes()
    self.rig_points = self._create_rig_points()
    
def _create_base_topology(self) -> Dict:
    """Creates base face mesh topology"""
    vertices = []
    faces = []
    uv_coords = []
    
    # Create UV sphere-based head topology
    segments = 32
    rings = 24
    
    for ring in range(rings + 1):
        theta = (ring / rings) * np.pi
        for segment in range(segments):
            phi = (segment / segments) * 2 * np.pi
            
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)
            
            vertices.append(Vector3(x, y, z))
            uv_coords.append([segment / segments, ring / rings])
    
    # Create faces
    for ring in range(rings):
        for segment in range(segments):
            current = ring * segments + segment
            next_ring = (ring + 1) * segments + segment
            next_segment = ring * segments + ((segment + 1) % segments)
            next_both = (ring + 1) * segments + ((segment + 1) % segments)
            
            faces.append([current, next_ring, next_both])
            faces.append([current, next_both, next_segment])
    
    return {
        'vertices': vertices,
        'faces': faces,
        'uv_coords': uv_coords
    }

def _apply_facial_features(self, vertices: List[Vector3], features: FacialFeatures) -> List[Vector3]:
    """Applies parametric deformations based on facial features"""
    modified = []
    
    for v in vertices:
        x, y, z = v.x, v.y, v.z
        
        # Head shape deformations
        x *= features.head_width
        y *= features.head_height
        z *= features.head_depth
        
        # Jaw and chin shaping (affects lower half)
        if y < 0:
            jaw_factor = abs(y)
            x *= 1.0 + (features.jaw_width - 1.0) * jaw_factor
            z *= 1.0 + (features.chin_prominence - 0.5) * jaw_factor * 0.5
        
        # Eye region deformations (middle-upper face)
        if 0.3 < y < 0.7 and abs(z) < 0.3:
            eye_side = 1 if x > 0 else -1
            x_offset = features.eye_spacing * eye_side * 0.1
            x += x_offset
            
            # Eye depth
            if 0.4 < y < 0.6:
                z -= features.eye_depth * 0.1
        
        # Nose region (center, middle height)
        if -0.2 < y < 0.4 and abs(x) < 0.2:
            nose_factor = 1.0 - abs(x) / 0.2
            z += features.nose_length * nose_factor * 0.2
            
            # Nose bridge
            if y > 0:
                z += features.nose_bridge_height * nose_factor * 0.1
            
            # Nostril width
            if -0.1 < y < 0.1:
                x *= 1.0 + features.nose_width * 0.2
        
        # Mouth region (lower center)
        if -0.3 < y < 0 and abs(x) < 0.3 and z > 0:
            mouth_factor = 1.0 - abs(x) / 0.3
            x *= 1.0 + (features.mouth_width - 1.0) * mouth_factor
            z += features.lip_thickness * mouth_factor * 0.1
        
        modified.append(Vector3(x, y, z))
    
    return modified

def _create_blend_shapes(self) -> List[BlendShape]:
    """Creates blend shapes for facial expressions"""
    blend_shapes = []
    
    # Smile
    smile_deltas = []
    for v in self.base_topology['vertices']:
        delta = Vector3(0, 0, 0)
        # Move mouth corners up and out
        if -0.2 < v.y < 0.1 and 0.15 < abs(v.x) < 0.35:
            side = 1 if v.x > 0 else -1
            delta = Vector3(side * 0.1, 0.15, 0.05)
        smile_deltas.append(delta)
    
    blend_shapes.append(BlendShape("smile", smile_deltas))
    
    # Frown
    frown_deltas = []
    for v in self.base_topology['vertices']:
        delta = Vector3(0, 0, 0)
        if -0.2 < v.y < 0.1 and 0.15 < abs(v.x) < 0.35:
            side = 1 if v.x > 0 else -1
            delta = Vector3(side * 0.05, -0.1, 0)
        frown_deltas.append(delta)
    
    blend_shapes.append(BlendShape("frown", frown_deltas))
    
    # Surprised (eyes wide, mouth open)
    surprise_deltas = []
    for v in self.base_topology['vertices']:
        delta = Vector3(0, 0, 0)
        # Eyes wider
        if 0.4 < v.y < 0.6 and abs(v.z) < 0.2:
            delta = Vector3(0, 0.05, -0.05)
        # Mouth open
        if -0.15 < v.y < 0.05 and abs(v.x) < 0.2:
            delta = Vector3(0, -0.1, 0.05)
        surprise_deltas.append(delta)
    
    blend_shapes.append(BlendShape("surprise", surprise_deltas))
    
    # Angry (eyebrows down, eyes narrowed)
    angry_deltas = []
    for v in self.base_topology['vertices']:
        delta = Vector3(0, 0, 0)
        # Eyebrows down
        if 0.6 < v.y < 0.75 and abs(v.x) < 0.3:
            center_factor = 1.0 - abs(v.x) / 0.3
            delta = Vector3(0, -0.08 * center_factor, 0)
        # Eyes narrow
        if 0.45 < v.y < 0.55 and abs(v.z) < 0.2:
            delta = Vector3(0, 0, 0.03)
        angry_deltas.append(delta)
    
    blend_shapes.append(BlendShape("angry", angry_deltas))
    
    # Blink
    blink_deltas = []
    for v in self.base_topology['vertices']:
        delta = Vector3(0, 0, 0)
        if 0.45 < v.y < 0.55 and abs(v.x) < 0.25:
            upper_lid = v.y > 0.5
            if upper_lid:
                delta = Vector3(0, -0.1, 0)
            else:
                delta = Vector3(0, 0.05, 0)
        blink_deltas.append(delta)
    
    blend_shapes.append(BlendShape("blink", blink_deltas))
    
    return blend_shapes

def _create_rig_points(self) -> Dict[str, Vector3]:
    """Creates rig control points for animation"""
    return {
        'head_top': Vector3(0, 1, 0),
        'head_center': Vector3(0, 0, 0),
        'jaw': Vector3(0, -0.5, 0.2),
        'left_eye': Vector3(-0.2, 0.5, 0.3),
        'right_eye': Vector3(0.2, 0.5, 0.3),
        'nose_tip': Vector3(0, 0.2, 0.5),
        'mouth_center': Vector3(0, -0.1, 0.4),
        'left_mouth_corner': Vector3(-0.2, -0.1, 0.35),
        'right_mouth_corner': Vector3(0.2, -0.1, 0.35),
        'left_ear': Vector3(-0.6, 0.3, -0.1),
        'right_ear': Vector3(0.6, 0.3, -0.1),
    }

def _generate_procedural_texture(self, features: FacialFeatures, resolution: int = 512) -> np.ndarray:
    """Generates procedural skin texture"""
    texture = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    
    # Base skin tone
    base_color = np.array([
        features.skin_tone_r * 255,
        features.skin_tone_g * 255,
        features.skin_tone_b * 255
    ])
    
    # Create noise for skin variation
    np.random.seed(42)
    noise = np.random.normal(0, 10, (resolution, resolution, 3))
    
    # Add pores and skin texture
    pore_noise = np.random.normal(0, 5 * features.skin_roughness, (resolution, resolution, 3))
    
    # Combine
    texture = np.clip(base_color + noise + pore_noise, 0, 255).astype(np.uint8)
    
    # Add age-related features
    if features.age > 0.4:
        # Wrinkles (simplified as darker lines)
        wrinkle_mask = np.random.random((resolution, resolution)) < (features.wrinkle_intensity * 0.1)
        texture[wrinkle_mask] *= 0.8
    
    return texture

def generate_face(self, 
                 features: Optional[FacialFeatures] = None,
                 expressions: Optional[Dict[str, float]] = None) -> Dict:
    """
    Generates a complete 3D face with given features and expressions
    
    Args:
        features: FacialFeatures object controlling face shape
        expressions: Dict mapping expression name to weight (0-1)
    
    Returns:
        Dictionary containing mesh data, textures, and metadata
    """
    if features is None:
        features = FacialFeatures()
    
    if expressions is None:
        expressions = {}
    
    # Start with base topology
    vertices = self.base_topology['vertices'].copy()
    
    # Apply facial features
    vertices = self._apply_facial_features(vertices, features)
    
    # Apply blend shapes for expressions
    for blend_shape in self.blend_shapes:
        if blend_shape.name in expressions:
            weight = expressions[blend_shape.name]
            for i, delta in enumerate(blend_shape.deltas):
                vertices[i] = vertices[i] + (delta * weight)
    
    # Generate texture
    texture = self._generate_procedural_texture(features)
    
    # Create face data structure
    face_data = {
        'metadata': {
            'generator': 'Janus Advanced 3D Face Generator',
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'vertex_count': len(vertices),
            'face_count': len(self.base_topology['faces']),
        },
        'geometry': {
            'vertices': [v.to_list() for v in vertices],
            'faces': self.base_topology['faces'],
            'uv_coords': self.base_topology['uv_coords'],
            'normals': self._calculate_normals(vertices, self.base_topology['faces'])
        },
        'features': asdict(features),
        'expressions': expressions,
        'blend_shapes': [
            {
                'name': bs.name,
                'deltas': [d.to_list() for d in bs.deltas]
            }
            for bs in self.blend_shapes
        ],
        'rig': {k: v.to_list() for k, v in self.rig_points.items()},
        'texture': {
            'resolution': texture.shape[0],
            'format': 'RGB',
            'data': texture.tolist()
        }
    }
    
    return face_data

def _calculate_normals(self, vertices: List[Vector3], faces: List[List[int]]) -> List[List[float]]:
    """Calculate vertex normals for lighting"""
    normals = [Vector3(0, 0, 0) for _ in vertices]
    
    # Calculate face normals and accumulate
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Edge vectors
        e1 = Vector3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
        e2 = Vector3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
        
        # Cross product for normal
        nx = e1.y * e2.z - e1.z * e2.y
        ny = e1.z * e2.x - e1.x * e2.z
        nz = e1.x * e2.y - e1.y * e2.x
        
        normal = Vector3(nx, ny, nz)
        
        # Accumulate to vertices
        for idx in face:
            normals[idx] = normals[idx] + normal
    
    # Normalize
    normalized = []
    for n in normals:
        length = np.sqrt(n.x**2 + n.y**2 + n.z**2)
        if length > 0:
            normalized.append([n.x/length, n.y/length, n.z/length])
        else:
            normalized.append([0, 1, 0])
    
    return normalized

def export_to_obj(self, face_data: Dict, filepath: str):
    """Exports face mesh to Wavefront OBJ format"""
    with open(filepath, 'w') as f:
        f.write("# Janus 3D Face Generator\n")
        f.write(f"# Generated: {face_data['metadata']['timestamp']}\n\n")
        
        # Vertices
        for v in face_data['geometry']['vertices']:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        f.write("\n")
        
        # Normals
        for n in face_data['geometry']['normals']:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        f.write("\n")
        
        # UV coordinates
        for uv in face_data['geometry']['uv_coords']:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        
        f.write("\n")
        
        # Faces (OBJ is 1-indexed)
        for face in face_data['geometry']['faces']:
            f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                   f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                   f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")

def export_to_json(self, face_data: Dict, filepath: str):
    """Exports complete face data to JSON"""
    with open(filepath, 'w') as f:
        json.dump(face_data, f, indent=2)

def create_animation_sequence(self, 
                             features: FacialFeatures,
                             expression_timeline: List[Tuple[float, Dict[str, float]]],
                             fps: int = 30) -> List[Dict]:
    """
    Creates an animation sequence with expressions changing over time
    
    Args:
        features: Base facial features
        expression_timeline: List of (timestamp, expressions_dict) tuples
        fps: Frames per second
    
    Returns:
        List of frame data dictionaries
    """
    frames = []
    
    if not expression_timeline:
        return frames
    
    # Sort timeline by timestamp
    timeline = sorted(expression_timeline, key=lambda x: x[0])
    
    # Calculate total duration
    duration = timeline[-1][0]
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        time = frame_num / fps
        
        # Interpolate expressions at this time
        interpolated_expressions = {}
        
        # Find surrounding keyframes
        before = None
        after = None
        
        for i, (t, exprs) in enumerate(timeline):
            if t <= time:
                before = (t, exprs)
            if t >= time and after is None:
                after = (t, exprs)
                break
        
        if before and after and before[0] != after[0]:
            # Interpolate between keyframes
            t_before, exprs_before = before
            t_after, exprs_after = after
            alpha = (time - t_before) / (t_after - t_before)
            
            # Get all unique expression names
            all_expressions = set(exprs_before.keys()) | set(exprs_after.keys())
            
            for expr_name in all_expressions:
                val_before = exprs_before.get(expr_name, 0)
                val_after = exprs_after.get(expr_name, 0)
                interpolated_expressions[expr_name] = val_before + (val_after - val_before) * alpha
        
        elif before:
            interpolated_expressions = before[1].copy()
        
        # Generate frame
        frame_data = self.generate_face(features, interpolated_expressions)
        frame_data['metadata']['frame'] = frame_num
        frame_data['metadata']['time'] = time
        frames.append(frame_data)
    
    return frames
```

def main():
“”“Example usage and testing”””
generator = ProceduralFaceGenerator()

```
# Create a custom face with specific features
custom_features = FacialFeatures(
    head_width=1.1,
    head_height=0.95,
    eye_spacing=1.2,
    nose_length=1.1,
    mouth_width=0.9,
    skin_tone_r=0.85,
    skin_tone_g=0.65,
    skin_tone_b=0.55,
    age=0.6,
    wrinkle_intensity=0.3
)

# Generate neutral face
print("Generating neutral face...")
neutral_face = generator.generate_face(custom_features)
generator.export_to_json(neutral_face, 'face_neutral.json')
generator.export_to_obj(neutral_face, 'face_neutral.obj')

# Generate smiling face
print("Generating smiling face...")
smiling_face = generator.generate_face(
    custom_features,
    expressions={'smile': 0.8}
)
generator.export_to_json(smiling_face, 'face_smile.json')
generator.export_to_obj(smiling_face, 'face_smile.obj')

# Create animation: neutral -> smile -> surprise -> neutral
print("Generating animation sequence...")
animation_timeline = [
    (0.0, {}),  # Neutral
    (1.0, {'smile': 0.8}),  # Smile
    (2.0, {'surprise': 1.0}),  # Surprised
    (3.0, {}),  # Back to neutral
]

frames = generator.create_animation_sequence(
    custom_features,
    animation_timeline,
    fps=30
)

print(f"Generated {len(frames)} animation frames")

# Save animation data
animation_data = {
    'metadata': {
        'fps': 30,
        'duration': 3.0,
        'frame_count': len(frames)
    },
    'frames': frames
}

with open('face_animation.json', 'w') as f:
    json.dump(animation_data, f)

print("\nGeneration complete!")
print("Files created:")
print("  - face_neutral.json (full data)")
print("  - face_neutral.obj (3D mesh)")
print("  - face_smile.json")
print("  - face_smile.obj")
print("  - face_animation.json (animation sequence)")
```

if **name** == ‘**main**’:
main()