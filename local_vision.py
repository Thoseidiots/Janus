"""
Local Vision & Action Inference - No External APIs
Uses YOLO, OpenCV, and Tesseract for completely offline operation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger("janus_vision")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("Tesseract not installed - OCR disabled")


class LocalVisionAnalyzer:
    """
    Analyze screenshots locally without APIs.
    Detects UI elements, text, changes, and infers actions.
    """
    
    def __init__(self):
        self.last_frame = None
        self.ui_elements_cache = []
        logger.info("Local Vision Analyzer initialized")
    
    def detect_ui_elements(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect clickable UI elements: buttons, text boxes, etc.
        Uses edge detection and contours.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        elements = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (ignore too small/large)
            if 20 < w < 500 and 20 < h < 500:
                # Calculate center
                cx, cy = x + w // 2, y + h // 2
                
                element = {
                    "type": self._classify_element(frame[y:y+h, x:x+w]),
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center_x": cx,
                    "center_y": cy,
                    "area": w * h
                }
                elements.append(element)
        
        # Remove duplicates (merge overlapping elements)
        elements = self._merge_overlapping(elements)
        self.ui_elements_cache = elements
        return elements
    
    def _classify_element(self, roi: np.ndarray) -> str:
        """Classify UI element type based on visual characteristics."""
        
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            return "unknown"
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Check for rectangular shape (button)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        if np.sum(binary) / binary.size > 0.7:
            return "button"
        
        # Check for text-like patterns
        if self._has_text_pattern(gray):
            return "text_field"
        
        return "element"
    
    def _has_text_pattern(self, gray: np.ndarray) -> bool:
        """Check if ROI contains text-like patterns."""
        # Simple heuristic: text has many transitions between light/dark
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges) / edges.size
        return edge_pixels > 0.05
    
    def _merge_overlapping(self, elements: List[Dict]) -> List[Dict]:
        """Merge overlapping detected elements."""
        if not elements:
            return []
        
        merged = []
        used = set()
        
        for i, elem1 in enumerate(elements):
            if i in used:
                continue
            
            # Find overlapping elements
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if j in used:
                    continue
                
                # Check overlap
                if self._rectangles_overlap(elem1, elem2):
                    # Merge
                    elem1["x"] = min(elem1["x"], elem2["x"])
                    elem1["y"] = min(elem1["y"], elem2["y"])
                    elem1["width"] = max(elem1["x"] + elem1["width"], elem2["x"] + elem2["width"]) - elem1["x"]
                    elem1["height"] = max(elem1["y"] + elem1["height"], elem2["y"] + elem2["height"]) - elem1["y"]
                    used.add(j)
            
            merged.append(elem1)
        
        return merged
    
    def _rectangles_overlap(self, rect1: Dict, rect2: Dict) -> bool:
        """Check if two rectangles overlap."""
        r1_right = rect1["x"] + rect1["width"]
        r2_right = rect2["x"] + rect2["width"]
        r1_bottom = rect1["y"] + rect1["height"]
        r2_bottom = rect2["y"] + rect2["height"]
        
        return not (r1_right < rect2["x"] or r2_right < rect1["x"] or
                   r1_bottom < rect2["y"] or r2_bottom < rect1["y"])
    
    def detect_text(self, frame: np.ndarray) -> List[Dict]:
        """Detect text regions and extract text using Tesseract."""
        
        if not HAS_TESSERACT:
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Preprocess for OCR
            gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
            
            # Extract text with position info
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            texts = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 50:  # Confidence threshold
                    texts.append({
                        "text": data['text'][i],
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                        "confidence": int(data['conf'][i])
                    })
            
            return texts
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return []
    
    def detect_frame_changes(self, new_frame: np.ndarray) -> Dict:
        """Detect what changed between frames."""
        
        if self.last_frame is None:
            self.last_frame = new_frame
            return {"change_detected": False, "magnitude": 0}
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate change magnitude
        change_pixels = np.sum(thresh > 0)
        total_pixels = diff.shape[0] * diff.shape[1]
        change_percentage = (change_pixels / total_pixels) * 100
        
        # Find changed region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        changed_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Ignore noise
                changed_regions.append({"x": x, "y": y, "width": w, "height": h})
        
        self.last_frame = new_frame
        
        return {
            "change_detected": change_percentage > 2,  # 2% threshold
            "magnitude": change_percentage,
            "changed_regions": changed_regions
        }
    
    def infer_action(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[Dict]:
        """
        Infer what action was performed between two frames.
        Returns: click, scroll, type, wait, etc.
        """
        
        change_info = self.detect_frame_changes(curr_frame)
        
        if not change_info["change_detected"]:
            return {"action": "wait", "confidence": 0.9}
        
        regions = change_info["changed_regions"]
        if not regions:
            return {"action": "wait", "confidence": 0.8}
        
        # Analyze change pattern
        largest_region = max(regions, key=lambda r: r["width"] * r["height"])
        
        # Heuristic: small localized change = click
        if largest_region["width"] < 100 and largest_region["height"] < 100:
            return {
                "action": "click",
                "x": largest_region["x"] + largest_region["width"] // 2,
                "y": largest_region["y"] + largest_region["height"] // 2,
                "confidence": 0.7
            }
        
        # Heuristic: vertical change = scroll
        if largest_region["height"] > largest_region["width"]:
            return {
                "action": "scroll",
                "direction": "down" if largest_region["y"] > curr_frame.shape[0] // 2 else "up",
                "confidence": 0.6
            }
        
        # Heuristic: horizontal change = scroll or resize
        if largest_region["width"] > largest_region["height"]:
            return {
                "action": "scroll",
                "direction": "right" if largest_region["x"] > curr_frame.shape[1] // 2 else "left",
                "confidence": 0.5
            }
        
        return {
            "action": "unknown",
            "change_magnitude": change_info["magnitude"],
            "confidence": 0.4
        }
    
    def get_screenshot_summary(self, frame: np.ndarray) -> Dict:
        """Get comprehensive summary of current screen state."""
        
        elements = self.detect_ui_elements(frame)
        texts = self.detect_text(frame)
        
        return {
            "ui_elements": len(elements),
            "elements": elements[:10],  # Top 10
            "text_regions": len(texts),
            "texts": texts[:5],  # Top 5
            "frame_shape": frame.shape
        }


class ActionSequenceBuilder:
    """Build executable action sequences from observations."""
    
    def __init__(self):
        self.sequence = []
        self.observations = []
    
    def add_observation(self, frame: np.ndarray, label: str = ""):
        """Add a frame observation."""
        self.observations.append({
            "frame_id": len(self.observations),
            "label": label,
            "shape": frame.shape
        })
    
    def infer_sequence(self, analyzer: LocalVisionAnalyzer, frames: List[np.ndarray]) -> List[Dict]:
        """Infer action sequence from frame sequence."""
        
        actions = []
        
        for i in range(len(frames) - 1):
            action = analyzer.infer_action(frames[i], frames[i+1])
            if action:
                actions.append({
                    "frame_index": i,
                    "action": action,
                    "timestamp": f"{i/30:.1f}s"  # Assuming 30fps
                })
        
        self.sequence = actions
        return actions
    
    def to_executable(self) -> List[Dict]:
        """Convert to executable action format."""
        return [
            {
                "type": action["action"].get("action", "unknown"),
                "params": {k: v for k, v in action["action"].items() if k != "action"}
            }
            for action in self.sequence
        ]


if __name__ == "__main__":
    print("Local Vision Analyzer")
    print("=" * 50)
    
    analyzer = LocalVisionAnalyzer()
    print("✓ Initialized")
    print("Features:")
    print("  • UI element detection (buttons, text fields)")
    print("  • OCR text extraction" + (" (Tesseract)" if HAS_TESSERACT else " (disabled)"))
    print("  • Frame change detection")
    print("  • Action inference (click, scroll, wait)")
    print("  • No external APIs needed")
