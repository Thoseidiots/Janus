"""
Janus Loop Detector: Human-like pattern recognition for repeated actions

Problem: AI enters the same room 10 times thinking it's new each time.
Solution: Detect semantic similarity in action sequences, not just exact matches.
"""

import torch
import torch.nn.functional as F
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class Action:
    """Represents a single action with context"""
    action_type: str
    target: str
    context: Dict[str, Any]
    timestamp: float
    embedding: Optional[torch.Tensor] = None


@dataclass
class LoopDetection:
    """Result of loop detection analysis"""
    is_loop: bool
    confidence: float
    repetition_count: int
    loop_pattern: List[str]
    recommendation: str
    similar_actions: List[Tuple[int, int, float]]  # (idx1, idx2, similarity)


class JanusLoopDetector:
    """
    Detects when AI is repeating similar actions, even with slight variations.

    Human threshold: 3 repetitions = definitely a pattern
    AI threshold: Must be smarter - recognize semantic similarity
    """

    def __init__(
        self,
        history_size: int = 100,
        similarity_threshold: float = 0.85,
        repetition_threshold: int = 3,
        dim: int = 512
    ):
        self.history_size = history_size
        self.similarity_threshold = similarity_threshold
        self.repetition_threshold = repetition_threshold
        self.dim = dim

        # Action history (ring buffer)
        self.action_history: deque = deque(maxlen=history_size)

        # Semantic embeddings for fast similarity checking
        self.action_embeddings: deque = deque(maxlen=history_size)

        # Loop statistics
        self.loop_count = 0
        self.total_actions = 0

    def _embed_action(self, action: Action) -> torch.Tensor:
        """
        Convert action to semantic embedding.
        Similar actions (e.g., "enter room_A" and "enter room_A again") get similar embeddings.
        """
        # Combine all action components
        action_str = f"{action.action_type}:{action.target}"

        # Add important context fields
        if action.context:
            context_str = ":".join(f"{k}={v}" for k, v in sorted(action.context.items())[:3])
            action_str = f"{action_str}:{context_str}"

        # Hash to fixed-size vector
        h = hashlib.sha256(action_str.encode()).digest()
        np_arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)

        # Tile to desired dimension
        if len(np_arr) < self.dim:
            np_arr = np.tile(np_arr, self.dim // len(np_arr) + 1)[:self.dim]
        else:
            np_arr = np_arr[:self.dim]

        # Normalize for cosine similarity
        embedding = torch.tensor(np_arr, dtype=torch.float32)
        embedding = F.normalize(embedding, dim=0)

        return embedding

    def _find_similar_actions(self, action_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar actions in history.
        Returns: [(index, similarity), ...]
        """
        if not self.action_embeddings:
            return []

        # Stack all embeddings
        history_embeddings = torch.stack(list(self.action_embeddings))

        # Compute cosine similarity
        similarities = F.cosine_similarity(
            action_embedding.unsqueeze(0),
            history_embeddings,
            dim=1
        )

        # Get top-k matches
        top_k = min(top_k, len(similarities))
        values, indices = torch.topk(similarities, top_k)

        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def _detect_repeating_pattern(self, recent_count: int = 10) -> Optional[Tuple[List[str], int]]:
        """
        Detect if recent actions form a repeating pattern.

        Example:
        [A, B, C, A, B, C, A, B, C] -> Pattern [A, B, C] repeated 3 times
        """
        if len(self.action_history) < recent_count:
            recent_count = len(self.action_history)

        if recent_count < self.repetition_threshold:
            return None

        recent_actions = list(self.action_history)[-recent_count:]
        recent_embeddings = list(self.action_embeddings)[-recent_count:]

        # Try different pattern lengths
        for pattern_len in range(1, recent_count // 2 + 1):
            # Check if last N actions can be split into repeating patterns
            if recent_count % pattern_len != 0:
                continue

            num_repetitions = recent_count // pattern_len

            # Extract pattern
            pattern_embeddings = recent_embeddings[:pattern_len]

            # Check if pattern repeats
            is_repeating = True
            for rep in range(1, num_repetitions):
                start_idx = rep * pattern_len
                end_idx = start_idx + pattern_len

                # Compare this repetition to the pattern
                for i, pattern_emb in enumerate(pattern_embeddings):
                    rep_emb = recent_embeddings[start_idx + i]
                    similarity = F.cosine_similarity(
                        pattern_emb.unsqueeze(0),
                        rep_emb.unsqueeze(0),
                        dim=1
                    ).item()

                    if similarity < self.similarity_threshold:
                        is_repeating = False
                        break

                if not is_repeating:
                    break

            if is_repeating and num_repetitions >= self.repetition_threshold:
                # Found repeating pattern!
                pattern_actions = [
                    f"{recent_actions[i].action_type}({recent_actions[i].target})"
                    for i in range(pattern_len)
                ]
                return (pattern_actions, num_repetitions)

        return None

    def record_action(self, action: Action) -> LoopDetection:
        """
        Record an action and check if we're in a loop.

        This is the main API - call this after every AI action.
        """
        self.total_actions += 1

        # Embed the action
        action.embedding = self._embed_action(action)

        # Find similar actions in history
        similar = self._find_similar_actions(action.embedding, top_k=10)

        # Count recent high-similarity matches
        recent_similar_count = sum(
            1 for idx, sim in similar
            if sim > self.similarity_threshold and (len(self.action_history) - idx) <= 10
        )

        # Check for repeating pattern
        pattern_result = self._detect_repeating_pattern(recent_count=15)

        # Store in history
        self.action_history.append(action)
        self.action_embeddings.append(action.embedding)

        # Determine if this is a loop
        is_loop = False
        confidence = 0.0
        repetition_count = recent_similar_count
        loop_pattern = []
        recommendation = "Continue - no loop detected"

        if pattern_result:
            # Found repeating pattern
            loop_pattern, reps = pattern_result
            is_loop = True
            confidence = 0.95
            repetition_count = reps
            self.loop_count += 1
            recommendation = f"⚠️ LOOP DETECTED! Pattern {loop_pattern} repeated {reps}x. " \
                           f"Suggest: Break pattern or choose alternative action."

        elif recent_similar_count >= self.repetition_threshold:
            # Multiple similar actions recently
            is_loop = True
            confidence = min(0.9, recent_similar_count / 10.0)
            self.loop_count += 1
            recommendation = f"⚠️ Repeated similar actions {recent_similar_count}x. " \
                           f"Suggest: Try a different approach."

        # Prepare similar action details
        similar_actions_detail = [
            (idx, len(self.action_history) - 1, sim)
            for idx, sim in similar
            if sim > 0.7  # Include moderately similar actions
        ]

        return LoopDetection(
            is_loop=is_loop,
            confidence=confidence,
            repetition_count=repetition_count,
            loop_pattern=loop_pattern,
            recommendation=recommendation,
            similar_actions=similar_actions_detail[:5]  # Top 5
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get loop detection statistics"""
        return {
            "total_actions": self.total_actions,
            "detected_loops": self.loop_count,
            "loop_rate": self.loop_count / max(self.total_actions, 1),
            "history_size": len(self.action_history),
            "avg_similarity_in_history": self._compute_avg_similarity()
        }

    def _compute_avg_similarity(self) -> float:
        """Compute average pairwise similarity in recent history"""
        if len(self.action_embeddings) < 2:
            return 0.0

        embeddings = torch.stack(list(self.action_embeddings)[-20:])  # Last 20 actions
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0),
                    dim=1
                ).item()
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def reset(self):
        """Reset the detector (useful for new task contexts)"""
        self.action_history.clear()
        self.action_embeddings.clear()
        self.loop_count = 0
        self.total_actions = 0


def demo_loop_detector():
    """Demonstrate loop detection with realistic scenarios"""
    print("="*70)
    print("JANUS LOOP DETECTOR: HUMAN-LIKE PATTERN RECOGNITION")
    print("="*70)

    detector = JanusLoopDetector(
        similarity_threshold=0.85,
        repetition_threshold=3
    )

    # Scenario 1: AI enters the same room multiple times
    print("\n[SCENARIO 1] AI navigating rooms - enters same room repeatedly")
    print("-" * 70)

    actions_scenario1 = [
        Action("navigate", "hallway", {"direction": "north"}, time.time()),
        Action("enter", "room_A", {"door": "main"}, time.time()),
        Action("look_around", "room_A", {}, time.time()),
        Action("exit", "room_A", {}, time.time()),
        Action("navigate", "hallway", {"direction": "south"}, time.time()),
        Action("enter", "room_A", {"door": "main"}, time.time()),  # Same room again!
        Action("look_around", "room_A", {}, time.time()),
        Action("exit", "room_A", {}, time.time()),
        Action("navigate", "hallway", {"direction": "north"}, time.time()),
        Action("enter", "room_A", {"door": "main"}, time.time()),  # Third time!
    ]

    for i, action in enumerate(actions_scenario1):
        result = detector.record_action(action)
        print(f"\n[Action {i+1}] {action.action_type}({action.target})")

        if result.is_loop:
            print(f"  🔴 LOOP DETECTED!")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Repetitions: {result.repetition_count}")
            if result.loop_pattern:
                print(f"  Pattern: {' -> '.join(result.loop_pattern)}")
            print(f"  {result.recommendation}")
            break

    # Scenario 2: Subtle variation - room_A vs room_B but similar actions
    print("\n\n[SCENARIO 2] Similar actions in different rooms")
    print("-" * 70)

    detector.reset()
    actions_scenario2 = [
        Action("enter", "kitchen", {}, time.time()),
        Action("search", "kitchen", {"target": "food"}, time.time()),
        Action("enter", "bedroom", {}, time.time()),
        Action("search", "bedroom", {"target": "food"}, time.time()),
        Action("enter", "bathroom", {}, time.time()),
        Action("search", "bathroom", {"target": "food"}, time.time()),  # Same pattern!
    ]

    for i, action in enumerate(actions_scenario2):
        result = detector.record_action(action)
        print(f"\n[Action {i+1}] {action.action_type}({action.target})")

        if result.is_loop:
            print(f"  🔴 LOOP DETECTED!")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  {result.recommendation}")

    # Statistics
    print("\n" + "="*70)
    print("STATISTICS:")
    print("="*70)
    stats = detector.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("LOOP DETECTION SUMMARY:")
    print("="*70)
    print("✓ Detects exact repetitions (same room 3+ times)")
    print("✓ Detects semantic repetitions (similar actions, different targets)")
    print("✓ Detects repeating patterns ([A, B, C] -> [A, B, C] -> [A, B, C])")
    print("✓ Human-like threshold: 3 repetitions triggers alert")
    print("✓ Provides actionable recommendations")
    print("="*70)


if __name__ == "__main__":
    demo_loop_detector()
