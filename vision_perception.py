“””
Janus Vision Perception System
Real-time camera input, object detection, scene understanding, and visual memory
No external API dependencies - uses local models only
“””

import cv2
import numpy as np
import json
import threading
import queue
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import time

@dataclass
class DetectedObject:
“”“Represents a detected object in the visual field”””
object_id: str
class_name: str
confidence: float
bbox: Tuple[int, int, int, int]  # x, y, w, h
center: Tuple[int, int]
timestamp: str
features: Optional[Dict[str, Any]] = None

@dataclass
class SceneContext:
“”“High-level scene understanding”””
scene_id: str
timestamp: str
objects: List[DetectedObject]
scene_type: str  # ‘indoor’, ‘outdoor’, ‘workspace’, etc.
lighting: str  # ‘bright’, ‘dim’, ‘natural’
dominant_colors: List[Tuple[int, int, int]]
motion_detected: bool
people_count: int
text_detected: List[str]
summary: str

@dataclass
class VisualMemory:
“”“Visual episodic memory entry”””
memory_id: str
timestamp: str
scene: SceneContext
importance: float  # 0-1 scale
emotional_valence: float  # -1 to 1
linked_events: List[str]

class ObjectDetector:
“”“Local object detection using OpenCV DNN”””

```
def __init__(self, model_type: str = 'mobilenet'):
    self.model_type = model_type
    self.classes = self._load_coco_classes()
    self.net = self._load_model()
    self.confidence_threshold = 0.5
    self.nms_threshold = 0.4

def _load_coco_classes(self) -> List[str]:
    """Load COCO class names"""
    # Common COCO classes
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

def _load_model(self):
    """Load pre-trained detection model"""
    # For demonstration - in production, download and use actual models
    # This is a placeholder that does basic blob detection
    return None

def detect(self, frame: np.ndarray) -> List[DetectedObject]:
    """Detect objects in frame"""
    objects = []
    
    # Simple color-based detection as fallback
    # In production, use actual DNN models
    height, width = frame.shape[:2]
    
    # Convert to different color spaces for detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces (basic Haar cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for i, (x, y, w, h) in enumerate(faces):
        obj = DetectedObject(
            object_id=f"person_{i}_{int(time.time())}",
            class_name="person",
            confidence=0.8,
            bbox=(int(x), int(y), int(w), int(h)),
            center=(int(x + w/2), int(y + h/2)),
            timestamp=datetime.now().isoformat(),
            features={'type': 'face'}
        )
        objects.append(obj)
    
    # Detect contours for general objects
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small detections
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify based on shape
            obj_class = self._classify_by_shape(contour, area)
            
            obj = DetectedObject(
                object_id=f"{obj_class}_{i}_{int(time.time())}",
                class_name=obj_class,
                confidence=0.6,
                bbox=(int(x), int(y), int(w), int(h)),
                center=(int(x + w/2), int(y + h/2)),
                timestamp=datetime.now().isoformat(),
                features={'area': int(area)}
            )
            objects.append(obj)
    
    return objects

def _classify_by_shape(self, contour, area) -> str:
    """Simple shape-based classification"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return "triangle_object"
    elif num_vertices == 4:
        return "rectangular_object"
    elif num_vertices > 6:
        # Check circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            return "circular_object"
    
    return "unknown_object"
```

class SceneAnalyzer:
“”“Analyzes overall scene context”””

```
def analyze_scene(self, frame: np.ndarray, objects: List[DetectedObject]) -> SceneContext:
    """Perform scene-level analysis"""
    
    # Analyze lighting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness > 160:
        lighting = "bright"
    elif mean_brightness > 80:
        lighting = "normal"
    else:
        lighting = "dim"
    
    # Extract dominant colors
    dominant_colors = self._get_dominant_colors(frame, k=3)
    
    # Detect motion (requires previous frame)
    motion_detected = False  # Placeholder
    
    # Count people
    people_count = sum(1 for obj in objects if obj.class_name == "person")
    
    # Classify scene type
    scene_type = self._classify_scene_type(objects, frame)
    
    # OCR for text detection (basic)
    text_detected = self._detect_text(frame)
    
    # Generate summary
    summary = self._generate_scene_summary(objects, scene_type, people_count)
    
    scene = SceneContext(
        scene_id=f"scene_{int(time.time())}",
        timestamp=datetime.now().isoformat(),
        objects=objects,
        scene_type=scene_type,
        lighting=lighting,
        dominant_colors=dominant_colors,
        motion_detected=motion_detected,
        people_count=people_count,
        text_detected=text_detected,
        summary=summary
    )
    
    return scene

def _get_dominant_colors(self, frame: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
    """Extract dominant colors using k-means"""
    # Reshape image to be a list of pixels
    pixels = frame.reshape(-1, 3).astype(np.float32)
    
    # Sample for performance
    sample_size = min(10000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    pixels = pixels[indices]
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert to int tuples
    colors = [tuple(map(int, center)) for center in centers]
    return colors

def _classify_scene_type(self, objects: List[DetectedObject], frame: np.ndarray) -> str:
    """Classify type of scene"""
    object_classes = [obj.class_name for obj in objects]
    
    # Indoor indicators
    indoor_objects = {'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop'}
    indoor_count = sum(1 for cls in object_classes if cls in indoor_objects)
    
    # Outdoor indicators
    outdoor_objects = {'car', 'bicycle', 'tree', 'sky'}
    outdoor_count = sum(1 for cls in object_classes if cls in outdoor_objects)
    
    # Workspace indicators
    workspace_objects = {'laptop', 'keyboard', 'mouse', 'monitor', 'book'}
    workspace_count = sum(1 for cls in object_classes if cls in workspace_objects)
    
    if workspace_count >= 2:
        return "workspace"
    elif indoor_count > outdoor_count:
        return "indoor"
    elif outdoor_count > indoor_count:
        return "outdoor"
    else:
        return "unknown"

def _detect_text(self, frame: np.ndarray) -> List[str]:
    """Basic text detection (placeholder for OCR)"""
    # In production, use pytesseract or similar
    # For now, return empty list
    return []

def _generate_scene_summary(self, objects: List[DetectedObject], 
                            scene_type: str, people_count: int) -> str:
    """Generate natural language scene summary"""
    
    if not objects:
        return f"Empty {scene_type} scene with no detected objects"
    
    obj_counts = {}
    for obj in objects:
        obj_counts[obj.class_name] = obj_counts.get(obj.class_name, 0) + 1
    
    summary_parts = [f"{scene_type.capitalize()} scene"]
    
    if people_count > 0:
        summary_parts.append(f"with {people_count} {'person' if people_count == 1 else 'people'}")
    
    # Add notable objects
    notable = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    if notable:
        obj_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                             for name, count in notable])
        summary_parts.append(f"containing {obj_desc}")
    
    return " ".join(summary_parts)
```

class VisionPerceptionSystem:
“”“Main vision perception system with real-time processing”””

```
def __init__(self, camera_id: int = 0, memory_dir: str = "/tmp/janus_vision"):
    self.camera_id = camera_id
    self.memory_dir = Path(memory_dir)
    self.memory_dir.mkdir(exist_ok=True, parents=True)
    
    self.detector = ObjectDetector()
    self.analyzer = SceneAnalyzer()
    
    self.capture = None
    self.running = False
    self.frame_queue = queue.Queue(maxsize=10)
    self.perception_thread = None
    
    self.visual_memories: List[VisualMemory] = []
    self.current_scene: Optional[SceneContext] = None
    
    self.frame_count = 0
    self.fps = 0
    
    # Callbacks
    self.on_scene_change = None
    self.on_object_detected = None
    self.on_person_detected = None

def start(self):
    """Start camera capture and perception loop"""
    self.capture = cv2.VideoCapture(self.camera_id)
    
    if not self.capture.isOpened():
        raise RuntimeError(f"Cannot open camera {self.camera_id}")
    
    self.running = True
    self.perception_thread = threading.Thread(target=self._perception_loop)
    self.perception_thread.daemon = True
    self.perception_thread.start()
    
    print(f"Vision perception started on camera {self.camera_id}")

def stop(self):
    """Stop perception system"""
    self.running = False
    
    if self.perception_thread:
        self.perception_thread.join(timeout=2)
    
    if self.capture:
        self.capture.release()
    
    cv2.destroyAllWindows()
    print("Vision perception stopped")

def _perception_loop(self):
    """Main perception processing loop"""
    last_time = time.time()
    frame_times = []
    
    while self.running:
        ret, frame = self.capture.read()
        
        if not ret:
            continue
        
        # Calculate FPS
        current_time = time.time()
        frame_times.append(current_time - last_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        self.fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        last_time = current_time
        
        # Process frame
        self._process_frame(frame)
        
        self.frame_count += 1
        
        # Process at ~10 FPS to reduce CPU load
        time.sleep(0.1)

def _process_frame(self, frame: np.ndarray):
    """Process a single frame"""
    # Detect objects
    objects = self.detector.detect(frame)
    
    # Analyze scene
    scene = self.analyzer.analyze_scene(frame, objects)
    
    # Check for scene changes
    if self._is_scene_change(scene):
        if self.on_scene_change:
            self.on_scene_change(scene)
        
        # Store as memory if important
        self._maybe_store_memory(scene, frame)
    
    # Trigger callbacks
    if objects and self.on_object_detected:
        self.on_object_detected(objects)
    
    person_detected = any(obj.class_name == "person" for obj in objects)
    if person_detected and self.on_person_detected:
        self.on_person_detected(objects)
    
    self.current_scene = scene
    
    # Add to queue for display
    if not self.frame_queue.full():
        # Annotate frame
        annotated = self._annotate_frame(frame.copy(), scene)
        self.frame_queue.put(annotated)

def _is_scene_change(self, new_scene: SceneContext) -> bool:
    """Detect significant scene changes"""
    if not self.current_scene:
        return True
    
    # Check if scene type changed
    if new_scene.scene_type != self.current_scene.scene_type:
        return True
    
    # Check if object count changed significantly
    old_count = len(self.current_scene.objects)
    new_count = len(new_scene.objects)
    if abs(new_count - old_count) > 2:
        return True
    
    # Check if people count changed
    if new_scene.people_count != self.current_scene.people_count:
        return True
    
    return False

def _maybe_store_memory(self, scene: SceneContext, frame: np.ndarray):
    """Store scene as episodic memory if important"""
    # Calculate importance
    importance = self._calculate_importance(scene)
    
    if importance > 0.5:  # Threshold for storage
        memory = VisualMemory(
            memory_id=f"vmem_{len(self.visual_memories)}",
            timestamp=scene.timestamp,
            scene=scene,
            importance=importance,
            emotional_valence=0.0,  # Placeholder
            linked_events=[]
        )
        
        self.visual_memories.append(memory)
        
        # Save snapshot
        snapshot_path = self.memory_dir / f"{memory.memory_id}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        
        # Save metadata
        metadata_path = self.memory_dir / f"{memory.memory_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(memory), f, indent=2)

def _calculate_importance(self, scene: SceneContext) -> float:
    """Calculate importance score for memory storage"""
    score = 0.0
    
    # People are important
    score += min(scene.people_count * 0.3, 0.6)
    
    # Motion is interesting
    if scene.motion_detected:
        score += 0.2
    
    # Text detection is notable
    if scene.text_detected:
        score += 0.2
    
    # Many objects = complex scene
    score += min(len(scene.objects) * 0.05, 0.3)
    
    return min(score, 1.0)

def _annotate_frame(self, frame: np.ndarray, scene: SceneContext) -> np.ndarray:
    """Annotate frame with detection results"""
    # Draw bounding boxes
    for obj in scene.objects:
        x, y, w, h = obj.bbox
        
        # Color based on class
        if obj.class_name == "person":
            color = (0, 255, 0)  # Green for people
        else:
            color = (255, 0, 0)  # Blue for objects
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Label
        label = f"{obj.class_name}: {obj.confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Scene info overlay
    info_y = 30
    cv2.putText(frame, f"Scene: {scene.scene_type}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    info_y += 25
    cv2.putText(frame, f"Objects: {len(scene.objects)}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    info_y += 25
    cv2.putText(frame, f"People: {scene.people_count}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    info_y += 25
    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def get_current_scene(self) -> Optional[SceneContext]:
    """Get current scene context"""
    return self.current_scene

def get_visual_memories(self, limit: int = 10) -> List[VisualMemory]:
    """Get recent visual memories"""
    return self.visual_memories[-limit:]

def query_memories(self, query: str) -> List[VisualMemory]:
    """Query visual memories by content"""
    results = []
    query_lower = query.lower()
    
    for memory in self.visual_memories:
        # Search in summary
        if query_lower in memory.scene.summary.lower():
            results.append(memory)
            continue
        
        # Search in detected objects
        for obj in memory.scene.objects:
            if query_lower in obj.class_name.lower():
                results.append(memory)
                break
    
    return results

def display_live_feed(self):
    """Display live annotated feed (blocking)"""
    print("Displaying live feed. Press 'q' to quit.")
    
    while self.running:
        try:
            frame = self.frame_queue.get(timeout=1)
            cv2.imshow('Janus Vision Perception', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            continue
    
    cv2.destroyAllWindows()
```

def main():
“”“Demo of vision perception system”””
print(”=== Janus Vision Perception System Demo ===\n”)

```
# Create perception system
vision = VisionPerceptionSystem(camera_id=0)

# Setup callbacks
def on_scene_change(scene):
    print(f"\n[SCENE CHANGE] {scene.summary}")

def on_person_detected(objects):
    people = [obj for obj in objects if obj.class_name == "person"]
    print(f"[PERSON DETECTED] {len(people)} person(s) in view")

vision.on_scene_change = on_scene_change
vision.on_person_detected = on_person_detected

# Start perception
vision.start()

try:
    # Display live feed
    vision.display_live_feed()
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    vision.stop()

# Show statistics
print(f"\nProcessed {vision.frame_count} frames")
print(f"Stored {len(vision.visual_memories)} visual memories")

# Show recent memories
if vision.visual_memories:
    print("\nRecent visual memories:")
    for mem in vision.visual_memories[-5:]:
        print(f"  - {mem.timestamp}: {mem.scene.summary} (importance: {mem.importance:.2f})")
```

if **name** == ‘**main**’:
main()