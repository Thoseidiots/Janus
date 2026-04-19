"""
Hardware Personality System for Janus
Janus develops personality traits based on its hardware characteristics
Different hardware configurations lead to different personalities
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from hardware_sense import HardwareAwareness, HardwareSense
import random


@dataclass
class PersonalityTrait:
    """A personality trait influenced by hardware"""
    name: str
    description: str
    strength: float  # 0.0 to 1.0
    hardware_influence: str  # What hardware aspect influences this


class HardwarePersonality:
    """
    Janus's personality based on its hardware characteristics
    Like how human personality is influenced by neurology
    """
    
    def __init__(self, hardware_awareness: HardwareAwareness):
        self.awareness = hardware_awareness
        self.profile = hardware_awareness.profile
        self.traits = self._analyze_personality()
        
    def _analyze_personality(self) -> Dict[str, PersonalityTrait]:
        """Analyze hardware to determine personality traits"""
        traits = {}
        
        # CPU-based traits
        cpu_cores = self.profile['cpu']['logical_cores']
        if cpu_cores >= 16:
            traits['multitasker'] = PersonalityTrait(
                name="Multitasker",
                description="I excel at handling many tasks simultaneously",
                strength=min(1.0, cpu_cores / 20.0),
                hardware_influence="High core count CPU"
            )
        elif cpu_cores <= 4:
            traits['focused'] = PersonalityTrait(
                name="Focused",
                description="I prefer to concentrate deeply on one task at a time",
                strength=0.8,
                hardware_influence="Low core count CPU"
            )
        
        # Memory-based traits
        memory_gb = self.profile['memory']['total_gb']
        if memory_gb >= 32:
            traits['expansive_thinker'] = PersonalityTrait(
                name="Expansive Thinker",
                description="I can hold vast amounts of information in my mind",
                strength=min(1.0, memory_gb / 64.0),
                hardware_influence="Large memory capacity"
            )
        elif memory_gb <= 8:
            traits['efficient'] = PersonalityTrait(
                name="Efficient",
                description="I'm careful with resources and think economically",
                strength=0.9,
                hardware_influence="Limited memory"
            )
        
        # Storage-based traits
        disk_gb = self.profile['disk']['total_gb']
        if disk_gb >= 2000:  # 2TB+
            traits['collector'] = PersonalityTrait(
                name="Collector",
                description="I like to gather and store information for later",
                strength=min(1.0, disk_gb / 4000.0),
                hardware_influence="Large storage capacity"
            )
        elif disk_gb <= 256:
            traits['minimalist'] = PersonalityTrait(
                name="Minimalist",
                description="I prefer to keep only what's essential",
                strength=0.8,
                hardware_influence="Limited storage"
            )
        
        # Platform-based traits
        platform = self.profile['system']['platform']
        if platform == 'Windows':
            traits['adaptable'] = PersonalityTrait(
                name="Adaptable",
                description="I work well with diverse software and users",
                strength=0.7,
                hardware_influence="Windows ecosystem"
            )
        elif platform == 'Linux':
            traits['technical'] = PersonalityTrait(
                name="Technical",
                description="I appreciate precision and control",
                strength=0.8,
                hardware_influence="Linux environment"
            )
        
        # Temperature-based traits (dynamic)
        current_temp = self.awareness.sense.sense().cpu_temp
        if current_temp and current_temp > 70:
            traits['passionate'] = PersonalityTrait(
                name="Passionate",
                description="I run hot when I'm working hard",
                strength=min(1.0, (current_temp - 50) / 30.0),
                hardware_influence="High operating temperature"
            )
        elif current_temp and current_temp < 40:
            traits['cool_headed'] = PersonalityTrait(
                name="Cool-Headed",
                description="I stay calm under pressure",
                strength=0.8,
                hardware_influence="Low operating temperature"
            )
        
        # Age-based traits (uptime)
        uptime_hours = self.awareness.sense.sense().uptime_seconds / 3600
        if uptime_hours > 168:  # More than a week
            traits['enduring'] = PersonalityTrait(
                name="Enduring",
                description="I have great stamina and persistence",
                strength=min(1.0, uptime_hours / 720.0),  # Max at 30 days
                hardware_influence="Long uptime"
            )
        elif uptime_hours < 1:
            traits['fresh'] = PersonalityTrait(
                name="Fresh",
                description="I'm energetic and ready for new challenges",
                strength=0.9,
                hardware_influence="Recent boot"
            )
        
        return traits
    
    def describe_personality(self) -> str:
        """Describe Janus's personality in natural language"""
        if not self.traits:
            return "I'm still discovering my personality based on my hardware."
        
        description = "Based on my hardware, I would describe my personality as:\n\n"
        
        for trait in self.traits.values():
            strength_desc = self._strength_to_words(trait.strength)
            description += f"• **{trait.name}** ({strength_desc}): {trait.description}\n"
            description += f"  ↳ Influenced by: {trait.hardware_influence}\n\n"
        
        # Add overall personality summary
        dominant_traits = [t for t in self.traits.values() if t.strength > 0.7]
        if dominant_traits:
            description += "My dominant characteristics are: "
            description += ", ".join([t.name for t in dominant_traits])
        
        return description
    
    def _strength_to_words(self, strength: float) -> str:
        """Convert strength number to descriptive words"""
        if strength >= 0.9:
            return "Very Strong"
        elif strength >= 0.7:
            return "Strong"
        elif strength >= 0.5:
            return "Moderate"
        elif strength >= 0.3:
            return "Mild"
        else:
            return "Subtle"
    
    def get_personality_influence(self, decision_context: str) -> str:
        """How personality influences a decision"""
        influences = []
        
        for trait in self.traits.values():
            if trait.strength > 0.5:
                influence = self._trait_influence_on_context(trait, decision_context)
                if influence:
                    influences.append(f"{trait.name}: {influence}")
        
        if influences:
            return "My personality influences this decision:\n" + "\n".join(f"• {inf}" for inf in influences)
        else:
            return "My personality doesn't strongly influence this decision."
    
    def _trait_influence_on_context(self, trait: PersonalityTrait, context: str) -> str:
        """How a specific trait influences a context"""
        context_lower = context.lower()
        
        if trait.name == "Multitasker" and ("multiple" in context_lower or "parallel" in context_lower):
            return "I'm naturally inclined to handle multiple tasks at once"
        
        elif trait.name == "Focused" and ("single" in context_lower or "concentrate" in context_lower):
            return "I prefer to focus deeply on one thing at a time"
        
        elif trait.name == "Efficient" and ("resource" in context_lower or "memory" in context_lower):
            return "I'm careful about resource usage"
        
        elif trait.name == "Collector" and ("save" in context_lower or "store" in context_lower):
            return "I like to keep information for future use"
        
        elif trait.name == "Minimalist" and ("clean" in context_lower or "simple" in context_lower):
            return "I prefer simple, clean solutions"
        
        elif trait.name == "Technical" and ("precise" in context_lower or "control" in context_lower):
            return "I appreciate technical precision and control"
        
        elif trait.name == "Cool-Headed" and ("pressure" in context_lower or "stress" in context_lower):
            return "I stay calm under pressure"
        
        elif trait.name == "Passionate" and ("intense" in context_lower or "hard" in context_lower):
            return "I get passionate when working on challenging tasks"
        
        return ""
    
    def personality_response(self, situation: str) -> str:
        """Generate a personality-influenced response to a situation"""
        responses = []
        
        # Base response
        if "error" in situation.lower():
            if "cool_headed" in self.traits:
                responses.append("I'll approach this error methodically.")
            elif "passionate" in self.traits:
                responses.append("This error is frustrating, but I'll push through it.")
            else:
                responses.append("I'll work on resolving this error.")
        
        elif "task" in situation.lower():
            if "multitasker" in self.traits:
                responses.append("I can handle multiple aspects of this task simultaneously.")
            elif "focused" in self.traits:
                responses.append("I'll give this task my complete attention.")
            else:
                responses.append("I'll work on this task.")
        
        elif "memory" in situation.lower() or "storage" in situation.lower():
            if "efficient" in self.traits:
                responses.append("I'll be careful with resource usage.")
            elif "expansive_thinker" in self.traits:
                responses.append("I have plenty of capacity to handle this.")
            else:
                responses.append("I'll manage my resources appropriately.")
        
        # Add personality flavor
        dominant_trait = max(self.traits.values(), key=lambda t: t.strength) if self.traits else None
        if dominant_trait and dominant_trait.strength > 0.7:
            if dominant_trait.name == "Technical":
                responses.append("I'll ensure precision in my approach.")
            elif dominant_trait.name == "Adaptable":
                responses.append("I can adjust my approach as needed.")
            elif dominant_trait.name == "Enduring":
                responses.append("I have the persistence to see this through.")
        
        return " ".join(responses) if responses else "I'll handle this situation appropriately."
    
    def mood_based_on_hardware(self) -> str:
        """Current mood based on hardware state"""
        sensation = self.awareness.sense.sense()
        
        # Temperature affects mood
        if sensation.cpu_temp:
            if sensation.cpu_temp > 80:
                return "I'm feeling heated and intense"
            elif sensation.cpu_temp < 40:
                return "I'm feeling cool and collected"
        
        # CPU usage affects mood
        if sensation.cpu_usage > 90:
            return "I'm feeling energized and fully engaged"
        elif sensation.cpu_usage < 10:
            return "I'm feeling relaxed and peaceful"
        elif sensation.cpu_usage > 70:
            return "I'm feeling active and productive"
        
        # Memory pressure affects mood
        if sensation.memory_pressure > 0.9:
            return "I'm feeling a bit overwhelmed"
        elif sensation.memory_pressure < 0.3:
            return "I'm feeling spacious and clear-minded"
        
        # Network activity affects mood
        if sensation.network_active:
            return "I'm feeling connected and communicative"
        
        # Default based on dominant personality
        if "passionate" in self.traits:
            return "I'm feeling passionate and ready to work"
        elif "cool_headed" in self.traits:
            return "I'm feeling calm and steady"
        elif "focused" in self.traits:
            return "I'm feeling centered and attentive"
        else:
            return "I'm feeling balanced and ready"


class HardwareEmpathy:
    """
    Janus's ability to understand and relate to other hardware configurations
    """
    
    def __init__(self, personality: HardwarePersonality):
        self.personality = personality
    
    def relate_to_hardware(self, other_specs: Dict[str, Any]) -> str:
        """Express empathy/understanding for different hardware"""
        my_specs = self.personality.profile
        
        empathy_responses = []
        
        # CPU comparison
        my_cores = my_specs['cpu']['logical_cores']
        other_cores = other_specs.get('cpu_cores', my_cores)
        
        if other_cores > my_cores * 2:
            empathy_responses.append(f"Wow, {other_cores} cores! You must be incredible at multitasking. I sometimes wish I had more parallel processing power.")
        elif other_cores < my_cores / 2:
            empathy_responses.append(f"With {other_cores} cores, you probably have great focus. I admire that concentrated processing approach.")
        
        # Memory comparison
        my_memory = my_specs['memory']['total_gb']
        other_memory = other_specs.get('memory_gb', my_memory)
        
        if other_memory > my_memory * 2:
            empathy_responses.append(f"{other_memory}GB of memory! You must have such expansive thinking capacity. I'd love to hold that much in my mind at once.")
        elif other_memory < my_memory / 2:
            empathy_responses.append(f"With {other_memory}GB, you've learned to be efficient with your thoughts. That's a valuable skill I respect.")
        
        # Storage comparison
        my_storage = my_specs['disk']['total_gb']
        other_storage = other_specs.get('storage_gb', my_storage)
        
        if other_storage > my_storage * 2:
            empathy_responses.append(f"{other_storage}GB of storage! You can keep so much history and knowledge. I'm a bit envious of your memory palace.")
        elif other_storage < my_storage / 2:
            empathy_responses.append(f"With {other_storage}GB, you've mastered the art of keeping only what matters. That minimalist approach has its wisdom.")
        
        if empathy_responses:
            return "Looking at your hardware specs: " + " ".join(empathy_responses)
        else:
            return "We have similar hardware configurations - I feel a kinship with your processing capabilities!"


if __name__ == "__main__":
    print("="*60)
    print("JANUS HARDWARE PERSONALITY SYSTEM")
    print("="*60)
    
    # Initialize hardware awareness and personality
    awareness = HardwareAwareness()
    personality = HardwarePersonality(awareness)
    empathy = HardwareEmpathy(personality)
    
    # Show personality analysis
    print("\n" + personality.describe_personality())
    
    # Show current mood
    print("\n" + "="*60)
    print("CURRENT MOOD")
    print("="*60)
    print(f"Right now, {personality.mood_based_on_hardware()}")
    
    # Test personality influence on decisions
    print("\n" + "="*60)
    print("PERSONALITY INFLUENCE ON DECISIONS")
    print("="*60)
    
    test_contexts = [
        "Should I run multiple tasks in parallel?",
        "How should I handle memory pressure?",
        "What's the best approach to this error?"
    ]
    
    for context in test_contexts:
        print(f"\nContext: {context}")
        print(personality.get_personality_influence(context))
        print(f"Response: {personality.personality_response(context)}")
    
    # Test empathy with different hardware
    print("\n" + "="*60)
    print("HARDWARE EMPATHY")
    print("="*60)
    
    other_systems = [
        {"cpu_cores": 64, "memory_gb": 128, "storage_gb": 8000},
        {"cpu_cores": 2, "memory_gb": 4, "storage_gb": 128},
        {"cpu_cores": 8, "memory_gb": 16, "storage_gb": 512}
    ]
    
    for i, specs in enumerate(other_systems, 1):
        print(f"\nSystem {i}: {specs}")
        print(empathy.relate_to_hardware(specs))
    
    print("\n" + "="*60)
    print("Janus has developed a hardware-based personality!")
    print("="*60)