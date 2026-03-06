"""
Long-Term Memory System
Persistent learning that compounds across sessions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("long_term_memory")


class Memory:
    """Single memory entry with recency tracking."""
    
    def __init__(self, content: str, category: str, importance: float = 0.5):
        self.id = f"mem_{datetime.now().timestamp()}"
        self.content = content
        self.category = category  # "lesson", "pattern", "decision", "outcome", "relationship"
        self.importance = importance  # 0-1
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.reinforcement_count = 0  # How many times this was confirmed
    
    def access(self):
        """Access memory - increases relevance."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def reinforce(self):
        """Memory confirmed - strengthen it."""
        self.reinforcement_count += 1
        self.importance = min(1.0, self.importance + 0.05)
    
    def decay_score(self) -> float:
        """
        Score memory by recency and importance.
        Recent + important = higher score.
        Old + unimportant = lower score.
        """
        days_old = (datetime.now() - self.last_accessed).days
        recency = 1.0 / (1.0 + days_old * 0.1)  # Decay over time
        return (self.importance * 0.6 + recency * 0.4) * (1.0 + self.reinforcement_count * 0.1)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
            "decay_score": self.decay_score()
        }


class LongTermMemory:
    """
    Persistent learning system that compounds across sessions.
    Memories strengthen with repetition and relevance.
    """
    
    def __init__(self, memory_file: str = "long_term_memory.json"):
        self.memory_file = memory_file
        self.memories: Dict[str, Memory] = {}
        self.patterns = defaultdict(int)  # Track recurring patterns
        self.learned_lessons = []
        self.relationship_history = defaultdict(list)
        
        self.load_memories()
        logger.info("Long-Term Memory System initialized")
    
    def load_memories(self):
        """Load memories from persistent storage."""
        if Path(self.memory_file).exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for mem_data in data.get("memories", []):
                        mem = Memory(
                            mem_data["content"],
                            mem_data["category"],
                            mem_data["importance"]
                        )
                        mem.id = mem_data["id"]
                        mem.created_at = datetime.fromisoformat(mem_data["created_at"])
                        mem.last_accessed = datetime.fromisoformat(mem_data["last_accessed"])
                        mem.access_count = mem_data["access_count"]
                        mem.reinforcement_count = mem_data["reinforcement_count"]
                        self.memories[mem.id] = mem
                
                logger.info(f"Loaded {len(self.memories)} memories")
            except Exception as e:
                logger.warning(f"Could not load memories: {e}")
    
    def save_memories(self):
        """Save memories to persistent storage."""
        data = {
            "saved_at": datetime.now().isoformat(),
            "total_memories": len(self.memories),
            "memories": [m.to_dict() for m in self.memories.values()]
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.memories)} memories")
    
    def add_memory(self, content: str, category: str, importance: float = 0.5) -> Memory:
        """Add a new memory."""
        mem = Memory(content, category, importance)
        self.memories[mem.id] = mem
        
        # Track patterns
        self.patterns[category] += 1
        
        logger.debug(f"Memory added: {content[:50]}... (importance: {importance})")
        return mem
    
    def recall_similar(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Memory]:
        """
        Recall similar memories.
        Uses simple keyword matching + importance/recency scoring.
        """
        query_lower = query.lower()
        scores = []
        
        for mem in self.memories.values():
            # Filter by category if specified
            if category and mem.category != category:
                continue
            
            # Simple keyword matching
            keyword_match = sum(1 for word in query_lower.split() if word in mem.content.lower())
            
            if keyword_match > 0:
                # Score by keyword match + decay
                score = (keyword_match / len(query_lower.split())) * mem.decay_score()
                scores.append((mem, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        recalled = [mem for mem, _ in scores[:top_k]]
        
        # Update access counts
        for mem in recalled:
            mem.access()
        
        return recalled
    
    def learn_lesson(self, situation: str, outcome: str, lesson: str, importance: float = 0.7):
        """
        Learn a lesson from experience.
        Stores the situation, outcome, and generalized lesson.
        """
        
        lesson_memory = {
            "situation": situation,
            "outcome": outcome,
            "lesson": lesson,
            "learned_at": datetime.now().isoformat(),
            "confidence": importance
        }
        
        self.learned_lessons.append(lesson_memory)
        
        # Add to memory system with high importance
        self.add_memory(
            f"Lesson: {lesson} (from: {situation})",
            category="lesson",
            importance=importance
        )
        
        logger.info(f"Lesson learned: {lesson}")
    
    def pattern_recognition(self) -> Dict[str, int]:
        """Identify recurring patterns across memories."""
        return dict(self.patterns)
    
    def get_lessons_for_situation(self, situation: str) -> List[Dict]:
        """
        Get lessons relevant to a current situation.
        Helps avoid repeating mistakes.
        """
        relevant_lessons = []
        
        for lesson in self.learned_lessons:
            # Simple keyword matching
            if any(word in situation.lower() for word in lesson["situation"].lower().split()):
                relevant_lessons.append(lesson)
        
        # Sort by recency and confidence
        relevant_lessons.sort(
            key=lambda x: datetime.fromisoformat(x["learned_at"]),
            reverse=True
        )
        
        return relevant_lessons
    
    def track_relationship(self, person: str, interaction: str, sentiment: float = 0.0):
        """
        Track relationship history with a person.
        Accumulates context for better interactions.
        """
        self.relationship_history[person].append({
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction,
            "sentiment": sentiment  # -1 to 1
        })
        
        # Add to memory system
        self.add_memory(
            f"Interaction with {person}: {interaction}",
            category="relationship",
            importance=max(0.3, abs(sentiment))
        )
    
    def get_relationship_summary(self, person: str) -> Dict:
        """Get summary of relationship history."""
        interactions = self.relationship_history.get(person, [])
        
        if not interactions:
            return {"person": person, "interactions": 0, "history": []}
        
        # Calculate average sentiment
        sentiments = [i["sentiment"] for i in interactions]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Get trends
        recent = interactions[-5:]
        recent_avg = sum(i["sentiment"] for i in recent) / len(recent)
        
        return {
            "person": person,
            "interactions": len(interactions),
            "avg_sentiment": avg_sentiment,
            "recent_trend": "improving" if recent_avg > avg_sentiment else "declining" if recent_avg < avg_sentiment else "stable",
            "history": recent  # Last 5 interactions
        }
    
    def get_memory_insights(self) -> Dict:
        """Generate insights from memory system."""
        
        # Most important memories
        top_memories = sorted(
            self.memories.values(),
            key=lambda m: m.decay_score(),
            reverse=True
        )[:5]
        
        # Most accessed memories
        most_accessed = sorted(
            self.memories.values(),
            key=lambda m: m.access_count,
            reverse=True
        )[:3]
        
        # Patterns
        categories = defaultdict(int)
        for mem in self.memories.values():
            categories[mem.category] += 1
        
        return {
            "total_memories": len(self.memories),
            "memory_categories": dict(categories),
            "lessons_learned": len(self.learned_lessons),
            "top_memories": [m.to_dict() for m in top_memories],
            "most_accessed": [m.to_dict() for m in most_accessed],
            "recurring_patterns": self.pattern_recognition(),
            "memory_health": {
                "avg_reinforcement": sum(m.reinforcement_count for m in self.memories.values()) / max(len(self.memories), 1),
                "avg_access_count": sum(m.access_count for m in self.memories.values()) / max(len(self.memories), 1)
            }
        }
    
    def demonstrate_learning(self, new_situation: str) -> Dict:
        """
        Show how past memories inform current decisions.
        This is "learning" in the sense of applying accumulated experience.
        """
        
        # Recall relevant memories
        relevant_memories = self.recall_similar(new_situation, top_k=3)
        
        # Get applicable lessons
        applicable_lessons = self.get_lessons_for_situation(new_situation)
        
        # Synthesize recommendation
        recommendation = {
            "situation": new_situation,
            "relevant_memories": [m.to_dict() for m in relevant_memories],
            "applicable_lessons": applicable_lessons,
            "reasoning": self._synthesize_recommendation(relevant_memories, applicable_lessons),
            "confidence": len(applicable_lessons) * 0.3 + len(relevant_memories) * 0.2  # Based on supporting evidence
        }
        
        return recommendation
    
    def _synthesize_recommendation(self, memories: List[Memory], lessons: List[Dict]) -> str:
        """Create recommendation from memories and lessons."""
        
        if lessons:
            return f"Based on {len(lessons)} similar lessons learned: " + ". ".join(l["lesson"] for l in lessons[:2])
        elif memories:
            return f"Based on {len(memories)} relevant past experiences: " + "; ".join(m.content[:30] + "..." for m in memories[:2])
        else:
            return "No direct precedent, but framework applies from general experience"


if __name__ == "__main__":
    print("Long-Term Memory System")
    print("=" * 50)
    
    mem = LongTermMemory()
    
    # Add some memories
    print("\n[Adding memories from experience]")
    mem.add_memory("Client X became upset when deadlines were missed", "lesson", 0.8)
    mem.add_memory("Revenue grew 40% when we focused on core product", "lesson", 0.9)
    mem.add_memory("Team morale dropped with unclear communication", "lesson", 0.7)
    
    # Learn lessons
    print("\n[Learning lessons]")
    mem.learn_lesson(
        situation="Missed deadline with client",
        outcome="Client unhappy, relationship damaged",
        lesson="Always communicate delays proactively",
        importance=0.9
    )
    
    # Track relationships
    print("\n[Tracking relationships]")
    mem.track_relationship("Client A", "Delivered on time", sentiment=0.8)
    mem.track_relationship("Team Lead", "Good planning meeting", sentiment=0.7)
    mem.track_relationship("Investor", "Missed milestone", sentiment=-0.6)
    
    # Demonstrate learning
    print("\n[Testing memory recall]")
    recall = mem.recall_similar("deadline issue", top_k=3)
    print(f"Found {len(recall)} relevant memories:")
    for m in recall:
        print(f"  • {m.content[:50]}... (importance: {m.importance})")
    
    # Get insights
    print("\n[Memory Insights]")
    insights = mem.get_memory_insights()
    print(json.dumps({
        "total_memories": insights["total_memories"],
        "lessons_learned": insights["lessons_learned"],
        "memory_categories": insights["memory_categories"]
    }, indent=2))
    
    # Save
    mem.save_memories()
    print("\n✓ Memories saved to long_term_memory.json")
