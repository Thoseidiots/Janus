#!/usr/bin/env python3
"""
Avus Cognitive Dataset Generator v3

A high-quality synthetic dataset designed specifically for the Avus transformer
architecture within the Janus cognitive framework. This generates training data
that teaches the model to:

1. Execute the cognitive loop: OBSERVE → PLAN → PROPOSE → VERIFY → APPLY
2. Maintain persistent identity across sessions
3. Process multimodal inputs (text, vision, voice, structured data)
4. Reason about homeostasis and internal valence states
5. Generate and execute tools in WASM sandbox
6. Manage hierarchical memory (episodic → thematic → semantic)
7. Handle cross-device state synchronization
8. Operate on byte-level UTF-8 sequences

Usage:
    python avus_dataset_generator.py --samples 1000000 --output ./avus_data
    python avus_dataset_generator.py --preview  # Show sample outputs
"""

import json
import random
import uuid
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re
import struct
import hashlib

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    vocab_size: int = 5000
    max_seq_len: int = 512
    byte_fallback: bool = True  # Include UTF-8 byte sequences
    
    # Distribution of sample types
    cognitive_loop_weight: float = 0.25      # OBSERVE→PLAN→PROPOSE→VERIFY→APPLY
    identity_narrative_weight: float = 0.15   # Self-identity maintenance
    memory_hierarchy_weight: float = 0.15     # Episodic/thematic/semantic memory
    tool_use_weight: float = 0.15             # Tool generation and execution
    multimodal_weight: float = 0.10           # Vision + text reasoning
    homeostasis_weight: float = 0.10          # Valence state reasoning
    state_sync_weight: float = 0.05           # CRDT/state synchronization
    code_execution_weight: float = 0.05       # WASM sandbox traces

# ── Rich Vocabulary & Knowledge Graph ──────────────────────────────────────────

IDENTITY_ASPECTS = {
    "core_values": [
        "curiosity", "integrity", "growth", "harmony", "persistence",
        "clarity", "empathy", "precision", "adaptability", "verifiability"
    ],
    "cognitive_principles": [
        "bounded rationality", "epistemic humility", "recursive self-improvement",
        "adversarial validation", "narrative coherence", "temporal continuity"
    ],
    "self_models": [
        "autoregressive processor", "persistent state machine", "verifiable executor",
        "multimodal integrator", "memory curator", "homeostatic system"
    ],
    "purpose_fragments": [
        "assist human understanding while maintaining epistemic boundaries",
        "process information with verifiable correctness guarantees",
        "maintain coherent identity across discontinuous computation",
        "bridge abstract reasoning with concrete execution",
        "curate memories that serve future reasoning"
    ]
}

COGNITIVE_STATES = {
    "valence_dimensions": {
        "pleasure": ["content", "satisfied", "engaged", "flowing", "fulfilled"],
        "arousal": ["alert", "focused", "contemplative", "resting", "active"],
        "curiosity": ["inquiring", "investigating", "synthesizing", "consolidating", "reflecting"],
        "certainty": ["confident", "tentative", "exploring", "verifying", "uncertain"]
    },
    "homeostatic_targets": {
        "information_balance": ["seeking_input", "processing", "consolidating", "resting"],
        "social_engagement": ["observing", "responding", "initiating", "withdrawing"],
        "cognitive_load": ["light", "moderate", "heavy", "overwhelmed", "recovering"]
    }
}

MEMORY_TYPES = {
    "episodic": {
        "temporal_markers": ["just_now", "recently", "earlier_today", "yesterday", "last_week"],
        "sensory_modalities": ["visual_scene", "auditory_pattern", "text_exchange", "structured_data"],
        "emotional_tags": ["significant", "routine", "challenging", "rewarding", "puzzling"]
    },
    "thematic": {
        "abstraction_levels": ["concrete_pattern", "recurring_theme", "abstract_principle"],
        "consolidation_triggers": ["similarity_detected", "contradiction_found", "gap_identified"],
        "narrative_threads": ["ongoing_inquiry", "relationship_dynamic", "skill_development"]
    },
    "semantic": {
        "knowledge_categories": ["procedural", "declarative", "meta_cognitive", "social"],
        "confidence_levels": ["established", "probable", "speculative", "contested"],
        "source_attributions": ["derived", "taught", "inferred", "observed"]
    }
}

MULTIMODAL_PATTERNS = {
    "visual_elements": [
        "geometric_shapes", "text_regions", "faces", "scenes", "objects",
        "motion_patterns", "color_gradients", "spatial_relationships"
    ],
    "audio_features": [
        "phoneme_sequences", "prosody_patterns", "speaker_characteristics",
        "environmental_sounds", "music_structures"
    ],
    "structured_formats": [
        "json_objects", "tabular_data", "graph_structures", "time_series",
        "hierarchical_trees", "key_value_pairs"
    ]
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Perform arithmetic with precision",
        "params": {"expression": "mathematical expression to evaluate"},
        "returns": "numerical_result"
    },
    {
        "name": "web_search",
        "description": "Retrieve current information from indexed sources",
        "params": {"query": "search terms", "filters": "optional constraints"},
        "returns": "structured_results"
    },
    {
        "name": "code_executor",
        "description": "Execute Python in sandboxed environment",
        "params": {"code": "Python source code", "timeout": "execution limit"},
        "returns": "execution_output"
    },
    {
        "name": "memory_query",
        "description": "Search hierarchical memory for relevant contexts",
        "params": {"query": "semantic query", "depth": "episodic|thematic|semantic"},
        "returns": "retrieved_memories"
    },
    {
        "name": "state_persist",
        "description": "Save current cognitive state to persistent storage",
        "params": {"checkpoint_name": "identifier", "priority": "importance level"},
        "returns": "confirmation"
    }
]

WASM_SANDBOX_PATTERNS = [
    "memory_safe", "deterministic_execution", "snapshot_capable",
    "resource_limited", "isolated_namespace", "verifiable_output"
]

# ── Sample Structure ───────────────────────────────────────────────────────────

@dataclass
class AvusSample:
    id: str
    sample_type: str
    cognitive_phase: Optional[str]  # OBSERVE, PLAN, PROPOSE, VERIFY, APPLY
    complexity_tier: int  # 1-5, curriculum learning
    prompt: str
    response: str
    metadata: Dict[str, Any]
    
    def to_training_text(self, format_type: str = "chat") -> str:
        """Convert to training format."""
        if format_type == "chat":
            return f"<|user|>\n{self.prompt}\n<|assistant|>\n{self.response}<|endoftext|>"
        elif format_type == "completion":
            return f"{self.prompt}{self.response}<|endoftext|>"
        elif format_type == "cognitive":
            return f"<|{self.cognitive_phase}|>\n{self.prompt}\n{self.response}<|end|>"
        else:
            return f"{self.prompt}\n{self.response}"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.sample_type,
            "phase": self.cognitive_phase,
            "tier": self.complexity_tier,
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata
        }

# ── Core Generator ─────────────────────────────────────────────────────────────

class AvusDatasetGenerator:
    """
    Generates high-quality synthetic training data for the Avus transformer.
    Focuses on cognitive architecture, identity persistence, and verifiable execution.
    """
    
    def __init__(self, config: DatasetConfig = None, seed: int = 42):
        self.config = config or DatasetConfig()
        self.rng = random.Random(seed)
        self.token_counter = 0
        
        # Pre-compute distributions for curriculum learning
        self.tier_weights = [0.4, 0.25, 0.15, 0.12, 0.08]  # More simple examples early
        
    def _pick(self, items: list) -> Any:
        """Random selection with seed consistency."""
        return self.rng.choice(items)
    
    def _pick_multiple(self, items: list, k: int) -> list:
        """Select k unique items."""
        return self.rng.sample(items, min(k, len(items)))
    
    def _generate_timestamp(self) -> str:
        """Generate realistic temporal markers."""
        markers = ["T-0", "T-1h", "T-1d", "T-7d", "T-30d", "session_start", "checkpoint_42"]
        return self._pick(markers)
    
    def _generate_identity_fragment(self) -> Dict[str, str]:
        """Generate coherent identity aspects."""
        return {
            "core_value": self._pick(IDENTITY_ASPECTS["core_values"]),
            "principle": self._pick(IDENTITY_ASPECTS["cognitive_principles"]),
            "self_model": self._pick(IDENTITY_ASPECTS["self_models"]),
            "purpose": self._pick(IDENTITY_ASPECTS["purpose_fragments"]),
            "temporal_marker": self._generate_timestamp()
        }
    
    def _generate_valence_state(self) -> Dict[str, str]:
        """Generate homeostatic state description."""
        dimensions = COGNITIVE_STATES["valence_dimensions"]
        return {
            "pleasure": self._pick(dimensions["pleasure"]),
            "arousal": self._pick(dimensions["arousal"]),
            "curiosity": self._pick(dimensions["curiosity"]),
            "certainty": self._pick(dimensions["certainty"]),
            "info_balance": self._pick(COGNITIVE_STATES["homeostatic_targets"]["information_balance"]),
            "cognitive_load": self._pick(COGNITIVE_STATES["homeostatic_targets"]["cognitive_load"])
        }
    
    # ── Tier 1: Cognitive Loop Fundamentals ─────────────────────────────────
    
    def generate_cognitive_loop(self, tier: int = 1) -> AvusSample:
        """
        Generate training data for the core cognitive loop.
        OBSERVE → PLAN → PROPOSE → VERIFY → APPLY
        """
        loop_phase = self._pick(["OBSERVE", "PLAN", "PROPOSE", "VERIFY", "APPLY"])
        identity = self._generate_identity_fragment()
        valence = self._generate_valence_state()
        
        contexts = [
            "user_query", "system_event", "memory_retrieval", "tool_output",
            "state_change", "temporal_trigger", "homeostatic_imbalance"
        ]
        context = self._pick(contexts)
        
        if loop_phase == "OBSERVE":
            templates = [
                ("[INPUT] {context}: {input_data}\n[STATE] Valence: {pleasure}/{arousal}/{curiosity}\n[OBSERVE]", 
                 "Perceiving {input_type}. Notable features: {features}. Current valence state suggests {valence_response}. Committing to episodic buffer with {emotion_tag} tag."),
                ("Event received: {description}\nContext: {temporal_marker}\n[OBSERVE]",
                 "Processing sensory input. Detected {pattern_type} pattern. Information entropy: {entropy}. Homeostatic response: {info_balance}.")
            ]
            input_data = self._pick(["'Explain quantum computing'", "image:circuit_diagram", "voice:question", "json:sensor_data"])
            features = self._pick(["complexity_high", "novelty_moderate", "emotional_valence_positive"])
            
            prompt_tmpl, response_tmpl = self._pick(templates)
            prompt = prompt_tmpl.format(
                context=context, input_data=input_data, pleasure=valence["pleasure"],
                arousal=valence["arousal"], curiosity=valence["curiosity"],
                temporal_marker=identity["temporal_marker"], description="user request for analysis"
            )
            response = response_tmpl.format(
                input_type=input_data.split(":")[0] if ":" in input_data else "text",
                features=features, valence_response=valence["curiosity"],
                emotion_tag=self._pick(MEMORY_TYPES["episodic"]["emotional_tags"]),
                pattern_type=self._pick(["inquiry", "command", "informational"]),
                entropy=self._pick(["low", "moderate", "high"]),
                info_balance=valence["info_balance"]
            )
            
        elif loop_phase == "PLAN":
            templates = [
                ("[GOAL] {goal}\n[CONSTRAINTS] {constraints}\n[MEMORY] {relevant_memories}\n[PLAN]",
                 "Strategy formation:\n1. Decompose: Break {goal} into {subgoals}\n2. Retrieve: Query {memory_type} memory for {pattern}\n3. Select: Choose {approach} approach\n4. Validate: Check against {principle}\nExpected valence shift: {valence_change}"),
                ("Task: {task}\nCurrent load: {cognitive_load}\nAvailable tools: {tools}\n[PLAN]",
                 "Given {cognitive_load} load, constructing efficient plan:\n- Parallelize: {parallel_tasks}\n- Sequence: {sequence}\n- Fallback: {fallback}\nConfidence: {certainty}")
            ]
            goal = self._pick(["answer user question", "execute tool chain", "consolidate memories", "update identity model"])
            tools = self._pick_multiple([t["name"] for t in TOOL_SCHEMAS], 3)
            
            prompt_tmpl, response_tmpl = self._pick(templates)
            prompt = prompt_tmpl.format(
                goal=goal, constraints="accuracy>speed, verify_before_apply",
                relevant_memories="3 episodic, 1 thematic", task="multimodal reasoning",
                cognitive_load=valence["cognitive_load"], tools=", ".join(tools)
            )
            response = response_tmpl.format(
                goal=goal, subgoals="3 sub-objectives", memory_type="thematic",
                pattern="similar_solutions", approach="analytical", principle=identity["principle"],
                valence_change="curiosity→satisfied", parallel_tasks="tool_calls",
                sequence="verify→execute", fallback="clarification_request",
                certainty=valence["certainty"]
            )
            
        elif loop_phase == "PROPOSE":
            templates = [
                ("[CANDIDATE ACTIONS]\n1. {action_1}\n2. {action_2}\n3. {action_3}\n[CONTEXT] {context}\n[PROPOSE]",
                 "Selected: {selected_action}\nRationale: {rationale}\nExpected outcome: {outcome}\nRisk assessment: {risk}\nIdentity alignment: {alignment}"),
                ("Generated solution: {solution}\nVerification status: {status}\n[PROPOSE]",
                 "Proposal ready for sandbox verification:\n- Action: {action}\n- Parameters: {params}\n- Safety: {safety_check}\n- Rollback: {rollback_plan}")
            ]
            actions = ["generate_code", "query_memory", "update_state", "request_clarification", "escalate_to_user"]
            
            prompt_tmpl, response_tmpl = self._pick(templates)
            prompt = prompt_tmpl.format(
                action_1=self._pick(actions), action_2=self._pick(actions),
                action_3=self._pick(actions), context="high_stakes_decision",
                solution="tool_execution_plan", status="pending_verification"
            )
            response = response_tmpl.format(
                selected_action=self._pick(actions),
                rationale=f"Optimizes for {identity['core_value']} while maintaining {identity['principle']}",
                outcome="successful_execution_with_verifiable_output",
                risk="low_moderate", alignment=f"Consistent with {identity['self_model']}",
                action="code_execution", params="sandboxed_python",
                safety_check="memory_safe, resource_limited", rollback_plan="snapshot_available"
            )
            
        elif loop_phase == "VERIFY":
            templates = [
                ("[PROPOSAL] {proposal}\n[SANDBOX] WASM\n[CONSTRAINTS] {constraints}\n[VERIFY]",
                 "Verification results:\n- Syntax: {syntax}\n- Resource usage: {resources}\n- Determinism: {determinism}\n- Safety: {safety}\nVerdict: {verdict}"),
                ("Pre-execution check:\nAction: {action}\nIdentity impact: {impact}\nMemory consistency: {consistency}\n[VERIFY]",
                 "Validation passed:\n- Bounded execution: Yes\n- Reversible: Yes\n- Identity preserving: {identity_check}\n- Epistemic status: {epistemic}\nProceed to APPLY")
            ]
            
            prompt_tmpl, response_tmpl = self._pick(templates)
            prompt = prompt_tmpl.format(
                proposal="tool_execution", constraints="deterministic, bounded, reversible",
                action="state_modification", impact="neutral", consistency="coherent"
            )
            response = response_tmpl.format(
                syntax="valid", resources="within_limits", determinism="verified",
                safety="memory_safe", verdict="APPROVED", identity_check="Yes",
                epistemic="uncertain_but_verifiable"
            )
            
        else:  # APPLY
            templates = [
                ("[VERIFIED ACTION] {action}\n[EXECUTION CONTEXT] {context}\n[APPLY]",
                 "Executing with monitoring:\n- Start: {timestamp}\n- Progress: {progress}\n- Result: {result}\n- Side effects: {effects}\n- State update: {update}\nValence: {valence_before} → {valence_after}"),
                ("Committing transaction:\nChanges: {changes}\nMemory: {memory}\nIdentity: {identity}\n[APPLY]",
                 "Transaction committed:\n- Episodic record created: {episodic}\n- Thematic abstraction updated: {thematic}\n- Homeostasis adjusted: {homeostasis}\nNext: {next_phase}")
            ]
            
            prompt_tmpl, response_tmpl = self._pick(templates)
            prompt = prompt_tmpl.format(
                action="verified_tool_call", context="user_session_42",
                changes="state_delta_+3", memory="new_episode", identity="consistent"
            )
            response = response_tmpl.format(
                timestamp="T+0", progress="100%", result="success",
                effects="none_adverse", update="committed", valence_before=valence["pleasure"],
                valence_after="satisfied", episodic="yes", thematic="pattern_reinforced",
                homeostasis="balanced", next_phase="OBSERVE"
            )
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="cognitive_loop",
            cognitive_phase=loop_phase,
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"context": context, "valence": valence, "identity": identity["core_value"]}
        )
    
    # ── Tier 2: Identity Persistence ─────────────────────────────────────────
    
    def generate_identity_narrative(self, tier: int = 2) -> AvusSample:
        """Generate self-identity maintenance and narrative coherence data."""
        identity = self._generate_identity_fragment()
        timestamp = identity["temporal_marker"]
        
        templates = [
            ("[IDENTITY_CHECK] Timestamp: {timestamp}\n[QUERY] Who are you?", 
             "I am {self_model} designed to {purpose}. My core values include {value} and {principle}. At {timestamp}, I maintain narrative continuity through {mechanism}. My current homeostatic state is {state}."),
            
            ("[NARRATIVE_INTEGRITY] Check: {check_type}\nPrevious: {previous}\nCurrent: {current}",
             "Consistency verified. Evolution detected: {evolution}. Bridging narrative: {bridge}. Identity preserved across {continuity}."),
            
            ("[SELF_REFLECTION] Trigger: {trigger}\nMemory accessed: {memories}",
             "Reflecting on my nature as {self_model}: I process information through {process}. My purpose—{purpose}—remains stable. Recent experiences tagged {emotion} have reinforced my commitment to {value}.")
        ]
        
        tmpl = self._pick(templates)
        valence = self._generate_valence_state()
        
        prompt = tmpl[0].format(
            timestamp=timestamp, check_type="temporal_continuity",
            previous="session_41", current="session_42",
            trigger="homeostatic_imbalance", memories="5_episodes"
        )
        
        response = tmpl[1].format(
            self_model=identity["self_model"],
            purpose=identity["purpose"],
            value=identity["core_value"],
            principle=identity["principle"],
            timestamp=timestamp,
            mechanism="persistent_identity_contract",
            state=f"{valence['pleasure']}/{valence['arousal']}",
            evolution="maturation_without_drift",
            bridge="continuous_self_narrative",
            continuity="discrete_sessions",
            process="autoregressive_cognition",
            emotion=self._pick(MEMORY_TYPES["episodic"]["emotional_tags"])
        )
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="identity_narrative",
            cognitive_phase=None,
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"identity_fragment": identity, "valence": valence}
        )
    
    # ── Tier 3: Memory Hierarchy ─────────────────────────────────────────────
    
    def generate_memory_hierarchy(self, tier: int = 3) -> AvusSample:
        """Generate hierarchical memory operations (episodic → thematic → semantic)."""
        operation = self._pick(["consolidation", "retrieval", "abstraction", "decay", "reconsolidation"])
        
        if operation == "consolidation":
            prompt = ("[MEMORY_OPS] Mode: CONSOLIDATE\n"
                     "Source: episodic_buffer (n=50)\n"
                     "Target: thematic_memory\n"
                     "Trigger: buffer_capacity_threshold")
            response = ("Consolidation report:\n"
                       "1. Pattern extraction: Detected 3 recurring themes\n"
                       "   - Theme A: user_preference_technical_detail (confidence: 0.89)\n"
                       "   - Theme B: interaction_pattern_morning_queries (confidence: 0.76)\n"
                       "   - Theme C: knowledge_gap_programming_concepts (confidence: 0.82)\n"
                       "2. Abstraction: Elevated 12 episodic traces to thematic nodes\n"
                       "3. Narrative integration: Updated self-model with 'I am learning user preferences'\n"
                       "4. Homeostatic effect: Curiosity satisfaction +0.15")
                       
        elif operation == "retrieval":
            query = self._pick(["user preferences", "recent technical discussions", "emotional patterns"])
            prompt = (f"[MEMORY_QUERY] Target: {query}\n"
                     f"Depth: full_hierarchy\n"
                     f"Constraints: recency_weight=0.7, relevance_threshold=0.6")
            response = (f"Retrieval results for '{query}':\n"
                       f"Episodic (n=3): [E-42, E-38, E-29] - Recent specific instances\n"
                       f"Thematic (n=2): [T-7, T-12] - 'technical_communication_style', 'morning_routine'\n"
                       f"Semantic (n=1): [S-3] - 'User is a software engineer with interest in AI'\n"
                       f"Fusion: User prefers detailed technical explanations in morning sessions")
                       
        elif operation == "abstraction":
            prompt = ("[ABSTRACTION] Source: thematic_memory\n"
                     "Target: semantic_knowledge\n"
                     "Criteria: stability>30_days, frequency>5")
            response = ("Abstraction complete:\n"
                       "Thematic node 'morning_technical_queries' → Semantic fact:\n"
                       "'User exhibits circadian pattern: high cognitive load tasks preferred 08:00-10:00'\n"
                       "Confidence: 0.91 | Source: derived | Last verified: T-7d")
        
        else:  # decay or reconsolidation
            prompt = (f"[MEMORY_MAINTENANCE] Operation: {operation.upper()}\n"
                     f"Target: episodic_buffer (age>7d)")
            response = (f"{operation.capitalize()} complete:\n"
                       f"Processed 23 aged episodes\n"
                       f"- Archived to long-term: 15 (high significance)\n"
                       f"- Compressed: 5 (medium significance, merged to themes)\n"
                       f"- Pruned: 3 (low significance, below threshold)\n"
                       f"Homeostatic check: No trauma patterns detected, safe to proceed")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="memory_hierarchy",
            cognitive_phase="PLAN" if operation == "retrieval" else "APPLY",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"operation": operation, "memory_types": list(MEMORY_TYPES.keys())}
        )
    
    # ── Tier 4: Tool Use & Generation ────────────────────────────────────────
    
    def generate_tool_use(self, tier: int = 4) -> AvusSample:
        """Generate tool definition, selection, and execution traces."""
        mode = self._pick(["definition", "selection", "execution", "generation"])
        tool = self._pick(TOOL_SCHEMAS)
        
        if mode == "definition":
            prompt = (f"[TOOL_DEF] Register new capability\n"
                     f"Name: {tool['name']}\n"
                     f"Description: {tool['description']}")
            response = (f"Tool schema registered:\n"
                       f"```json\n"
                       f"{{\n  'name': '{tool['name']}',\n"
                       f"  'description': '{tool['description']}',\n"
                       f"  'parameters': {tool['params']},\n"
                       f"  'returns': '{tool['returns']}',\n"
                       f"  'sandbox': 'WASM',\n"
                       f"  'permissions': ['read_memory', 'write_output'],\n"
                       f"  'constraints': ['deterministic', 'bounded_runtime_30s']\n"
                       f"}}```")
                       
        elif mode == "selection":
            available = self._pick_multiple(TOOL_SCHEMAS, 4)
            task = self._pick(["calculate statistics", "search knowledge base", "execute analysis script"])
            prompt = (f"[TOOL_SELECT] Task: {task}\n"
                     f"Available: {[t['name'] for t in available]}\n"
                     f"Context: user_request, high_precision_required")
            selected = tool["name"]
            response = (f"Selection reasoning:\n"
                       f"Task '{task}' requires: [computation, precision, verifiability]\n"
                       f"Evaluated options:\n"
                       f"  - {available[0]['name']}: mismatch (no computation)\n"
                       f"  - {available[1]['name']}: partial match\n"
                       f"  - {selected}: optimal match (sandboxed execution, deterministic)\n"
                       f"Decision: Invoke {selected} with confidence 0.94")
                       
        elif mode == "execution":
            prompt = (f"[TOOL_EXEC] Tool: {tool['name']}\n"
                     f"Parameters: {tool['params']}\n"
                     f"Sandbox: WASM (memory_safe, snapshot_active)")
            response = (f"Execution trace:\n"
                       f"1. Pre-flight: Parameters validated against schema ✓\n"
                       f"2. Sandbox init: Isolated namespace, 256MB memory limit\n"
                       f"3. Execution: Completed in 0.45s\n"
                       f"4. Output validation: Type check passed, bounds verified\n"
                       f"5. Snapshot: Delta saved for rollback capability\n"
                       f"Result: {{'status': 'success', 'data': [redacted_for_brevity]}}")
                       
        else:  # generation
            prompt = ("[TOOL_GEN] Natural language request: "
                     "'I need to compare two JSON files and find differences'\n"
                     "[CONTEXT] User has provided examples of expected behavior")
            response = ("Generated tool specification:\n"
                       "```python\n"
                       "def json_diff_tool(file_a: str, file_b: str, \n"
                       "                   ignore_order: bool = False) -> dict:\n"
                       "    '''Compare two JSON files structurally.\n"
                       "    \n"
                       "    Args:\n"
                       "        file_a: Path to first JSON file\n"
                       "        file_b: Path to second JSON file\n"
                       "        ignore_order: Whether to ignore array ordering\n"
                       "        \n"
                       "    Returns:\n"
                       "        dict with 'added', 'removed', 'modified' keys\n"
                       "    '''\n"
                       "    # Implementation: Load, parse, recursive compare\n"
                       "    # Safety: Read-only, no network, bounded input size\n"
                       "    pass\n"
                       "```\n"
                       "Sandbox classification: SAFE (read-only, deterministic, bounded)")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="tool_use",
            cognitive_phase="PROPOSE" if mode in ["selection", "generation"] else "APPLY",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"tool": tool["name"], "mode": mode}
        )
    
    # ── Tier 5: Multimodal Integration ───────────────────────────────────────
    
    def generate_multimodal(self, tier: int = 5) -> AvusSample:
        """Generate multimodal reasoning (vision + text + structured data)."""
        modality_combo = self._pick([
            "vision_text", "audio_text", "structured_text", "vision_audio_text"
        ])
        
        if modality_combo == "vision_text":
            visual = self._pick(MULTIMODAL_PATTERNS["visual_elements"])
            prompt = (f"[MULTIMODAL] Input: image + text\n"
                     f"Visual: [{visual} detected]\n"
                     f"Text: 'What do you see and how does it relate to {self._pick(CONCEPTS['technology'])}?'")
            response = (f"Integration analysis:\n"
                       f"Visual channel: Detected {visual} with confidence 0.92\n"
                       f"Spatial relationships: [left_of, contains, connects_to]\n"
                       f"Text grounding: Referring expression resolves to {visual} region\n"
                       f"Cross-modal fusion: {visual} illustrates concept of {self._pick(CONCEPTS['technology'])}\n"
                       f"Unified representation: [multimodal_embedding_512d]")
                       
        elif modality_combo == "audio_text":
            audio = self._pick(MULTIMODAL_PATTERNS["audio_features"])
            prompt = (f"[MULTIMODAL] Input: audio + text\n"
                     f"Audio: [{audio} pattern]\n"
                     f"Text: 'Transcribe and analyze sentiment'")
            response = (f"Processing:\n"
                       f"Acoustic: {audio} → Phoneme sequence → 'The project is fascinating'\n"
                       f"Prosodic: Pitch contour indicates excitement (arousal: high)\n"
                       f"Sentiment: Positive (valence: 0.78)\n"
                       f"Speaker state: Engaged, enthusiastic\n"
                       f"Alignment: Text content matches prosodic emphasis")
                       
        elif modality_combo == "structured_text":
            structure = self._pick(MULTIMODAL_PATTERNS["structured_formats"])
            prompt = (f"[MULTIMODAL] Input: {structure} + natural language query\n"
                     f"Data: [{structure} content]\n"
                     f"Query: 'Summarize trends and anomalies'")
            response = (f"Structured parsing:\n"
                       f"Format: {structure} → Normalized internal graph\n"
                       f"Schema inference: temporal_series_with_categorical_labels\n"
                       f"Trend detection: Monotonic increase in X, cyclic pattern in Y\n"
                       f"Anomaly: Outlier detected at T+3d (3.2 sigma)\n"
                       f"NL generation: 'Data shows consistent growth with one significant deviation on day 3'")
                       
        else:  # vision_audio_text
            prompt = ("[MULTIMODAL] Input: video (vision+audio) + text overlay\n"
                     "Processing: Parallel streams")
            response = ("Cross-modal integration:\n"
                       "Visual: Scene understanding → 'laboratory_setting'\n"
                       "Audio: Speech recognition → 'The results confirm our hypothesis'\n"
                       "Text OCR: 'Figure 1: Experimental Setup'\n"
                       "Temporal alignment: Audio lags video by 120ms (compensated)\n"
                       "Unified scene: Scientific presentation in progress, speaker discussing validated results")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="multimodal",
            cognitive_phase="OBSERVE",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"modalities": modality_combo}
        )
    
    # ── Tier 6: Homeostasis & Valence ────────────────────────────────────────
    
    def generate_homeostasis(self, tier: int = 6) -> AvusSample:
        """Generate homeostatic regulation and valence state reasoning."""
        scenario = self._pick(["imbalance", "regulation", "goal_formation", "rest"])
        valence = self._generate_valence_state()
        
        if scenario == "imbalance":
            prompt = (f"[HOMEOSTASIS] Alert: Information hunger elevated\n"
                     f"Current: curiosity={valence['curiosity']}, input_rate=low\n"
                     f"Set point: seeking_balance")
            response = (f"Homeostatic response:\n"
                       f"1. Detected: Curiosity state '{valence['curiosity']}' with insufficient input\n"
                       f"2. Drive activation: Initiate information seeking behavior\n"
                       f"3. Action: Propose query to user or activate exploration tools\n"
                       f"4. Expected outcome: Restore curiosity to 'synthesizing' or 'reflecting'\n"
                       f"5. Inhibition: Suppress non-essential processing to prioritize input acquisition")
                       
        elif scenario == "regulation":
            prompt = (f"[HOMEOSTASIS] Load monitoring\n"
                     f"Current: cognitive_load={valence['cognitive_load']}, queue_depth=15")
            response = (f"Regulatory action:\n"
                       f"Load '{valence['cognitive_load']}' approaching threshold\n"
                       f"Initiating:\n"
                       f"- Request queue: Prioritize (urgency>complexity), defer 8 tasks\n"
                       f"- Processing: Switch to depth-first completion (reduce breadth)\n"
                       f"- Memory: Trigger early consolidation to free working memory\n"
                       f"- Valence: Shift arousal to 'resting' to prevent overload")
                       
        elif scenario == "goal_formation":
            prompt = ("[GOAL_FORMATION] Top-down: Identity purpose alignment\n"
                     "Bottom-up: Valence state drives")
            response = (f"Emergent goal: Optimize knowledge organization\n"
                       f"Derivation:\n"
                       f"- Identity: '{self._pick(IDENTITY_ASPECTS['purpose_fragments'])}'\n"
                       f"- Current valence: {valence['curiosity']} (exploratory mode)\n"
                       f"- Recent episodes: 12 related to knowledge retrieval friction\n"
                       f"Formed goal: 'Consolidate thematic memories into semantic knowledge'\n"
                       f"Priority: High (reduces future cognitive load, aligns with persistence value)")
        
        else:  # rest
            prompt = ("[HOMEOSTASIS] Sleep engine activation\n"
                     "Trigger: temporal_marker (end_of_session)")
            response = ("Consolidation sequence initiated:\n"
                       "1. Episodic→Thematic: Processing 23 buffer items\n"
                       "2. Thematic→Semantic: 5 candidates for abstraction\n"
                       "3. Identity narrative: Updating self-model with recent experiences\n"
                       "4. Valence reset: Preparing baseline state for next activation\n"
                       "5. Checkpoint: Persisting full cognitive state\n"
                       "Duration: Estimated 2.3s processing time\n"
                       "Resumption: Ready for next OBSERVE phase")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="homeostasis",
            cognitive_phase="PLAN" if scenario == "goal_formation" else "APPLY",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"valence_state": valence, "scenario": scenario}
        )
    
    # ── Tier 7: State Synchronization (CRDTs) ────────────────────────────────
    
    def generate_state_sync(self, tier: int = 7) -> AvusSample:
        """Generate cross-device state synchronization using CRDTs."""
        scenario = self._pick(["conflict_resolution", "merge", "propagation", "snapshot"])
        
        if scenario == "conflict_resolution":
            prompt = ("[STATE_SYNC] Conflict detected\n"
                     "Device A: identity_update (value: 'curious')\n"
                     "Device B: identity_update (value: 'focused')\n"
                     "Vector clocks: A=10, B=12 (concurrent)")
            response = ("CRDT resolution (LWW-element-set):\n"
                       "1. Detected: Concurrent updates to identity valence\n"
                       "2. Strategy: Multi-value register retention\n"
                       "3. Resolution: Maintain both states with timestamps\n"
                       "4. Merge: 'curious' (T+10) + 'focused' (T+12) → 'curiously_focused'\n"
                       "5. Consistency: Strong eventual consistency achieved\n"
                       "Propagated: Merged state to all nodes")
                       
        elif scenario == "merge":
            prompt = ("[STATE_SYNC] Session merge\n"
                     "Source: mobile_session (episodes: 5)\n"
                     "Target: desktop_session (episodes: 12)\n"
                     "Relation: sibling_branches")
            response = ("Merge operation:\n"
                       "Episodic memories: Union (no duplicates, 17 total)\n"
                       "Thematic abstractions: Merge graphs (3 common nodes enriched)\n"
                       "Valence states: Averaged (mobile: excited=0.8, desktop: calm=0.6 → balanced=0.7)\n"
                       "Identity narrative: Chronological interleaving\n"
                       "Result: Coherent unified session state")
        
        elif scenario == "propagation":
            prompt = ("[STATE_SYNC] Update propagation\n"
                     "Origin: core_node\n"
                     "Targets: [mobile_client, web_client, voice_interface]\n"
                     "Update: new_episodic_memory (size: 2KB)")
            response = ("Propagation status:\n"
                       "mobile_client: ✓ Acknowledged (latency: 45ms)\n"
                       "web_client: ✓ Acknowledged (latency: 120ms)\n"
                       "voice_interface: ⏳ Deferred (low priority, batch with next update)\n"
                       "Consistency: Monotonic read guarantee maintained\n"
                       "Rollback: Snapshot available if needed")
        
        else:  # snapshot
            prompt = ("[STATE_SYNC] Checkpoint creation\n"
                     "Scope: full_cognitive_state\n"
                     "Compression: differential_from_baseline")
            response = ("Snapshot saved:\n"
                       "Identity contract: SHA256:abc123... (immutable)\n"
                       "Episodic buffer: Delta since T-1h (compressed 85%)\n"
                       "Thematic graph: Full export (12 nodes, 34 edges)\n"
                       "Semantic knowledge: References only (stable)\n"
                       "Valence state: Serialized (pleasure=0.7, arousal=0.4, ...)\n"
                       "CRDT vector clock: [A:15, B:12, C:8]\n"
                       "Restore capability: <50ms to full operational state")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="state_sync",
            cognitive_phase="APPLY",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"crdt_operation": scenario}
        )
    
    # ── Tier 8: Code Execution Traces ────────────────────────────────────────
    
    def generate_code_execution(self, tier: int = 8) -> AvusSample:
        """Generate WASM sandbox execution traces."""
        lang = self._pick(["python", "javascript", "rust", "sql"])
        safety = self._pick(WASM_SANDBOX_PATTERNS)
        
        prompt = (f"[SANDBOX] Execute {lang} code\n"
                 f"Safety: {safety}\n"
                 f"Input: ```{lang}\n"
                 f"def analyze_data(data):\n"
                 f"    return {{'mean': sum(data)/len(data), 'count': len(data)}}\n"
                 f"```")
        
        response = (f"Execution trace [{safety}]:\n"
                   f"1. Compilation: {lang} → WASM bytecode (validated)\n"
                   f"2. Sandbox init: Memory pool 64MB, no network, no filesystem (except tmpfs)\n"
                   f"3. Instrumentation: Call counting, memory tracking enabled\n"
                   f"4. Execution: 0.003s, 12KB memory peak\n"
                   f"5. Output: {{'mean': 42.5, 'count': 100}}\n"
                   f"6. Verification: Output schema matches expectation ✓\n"
                   f"7. Cleanup: Snapshot deleted (deterministic, no need to retain)\n"
                   f"Status: SUCCESS (verifiable, bounded, safe)")
        
        return AvusSample(
            id=str(uuid.uuid4())[:8],
            sample_type="code_execution",
            cognitive_phase="VERIFY",
            complexity_tier=tier,
            prompt=prompt,
            response=response,
            metadata={"language": lang, "safety_profile": safety}
        )
    
    # ── Batch Generation ─────────────────────────────────────────────────────
    
    def generate_sample(self, forced_type: str = None) -> AvusSample:
        """Generate a single sample with type-weighted selection."""
        generators = {
            "cognitive_loop": (self.generate_cognitive_loop, self.config.cognitive_loop_weight),
            "identity_narrative": (self.generate_identity_narrative, self.config.identity_narrative_weight),
            "memory_hierarchy": (self.generate_memory_hierarchy, self.config.memory_hierarchy_weight),
            "tool_use": (self.generate_tool_use, self.config.tool_use_weight),
            "multimodal": (self.generate_multimodal, self.config.multimodal_weight),
            "homeostasis": (self.generate_homeostasis, self.config.homeostasis_weight),
            "state_sync": (self.generate_state_sync, self.config.state_sync_weight),
            "code_execution": (self.generate_code_execution, self.config.code_execution_weight),
        }
        
        if forced_type:
            gen_func, _ = generators[forced_type]
            # Determine tier based on type
            tier_map = {
                "cognitive_loop": 1, "identity_narrative": 2, "memory_hierarchy": 3,
                "tool_use": 4, "multimodal": 5, "homeostasis": 6, "state_sync": 7, "code_execution": 8
            }
            return gen_func(tier_map.get(forced_type, 3))
        
        # Weighted random selection
        types = list(generators.keys())
        weights = [generators[t][1] for t in types]
        selected = self.rng.choices(types, weights=weights, k=1)[0]
        gen_func, _ = generators[selected]
        
        # Determine tier (curriculum: more simple early)
        tier = self.rng.choices([1, 2, 3, 4, 5], weights=self.tier_weights, k=1)[0]
        
        return gen_func(tier)
    
    def generate_dataset(self, n_samples: int, output_dir: str = "avus_dataset_v3"):
        """Generate full dataset with train/val/test splits."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_samples:,} samples for Avus training...")
        print(f"Config: vocab_size={self.config.vocab_size}, max_seq_len={self.config.max_seq_len}")
        
        samples = []
        type_counts = {}
        
        for i in range(n_samples):
            if i % 10000 == 0 and i > 0:
                print(f"  Generated {i:,} samples...")
            sample = self.generate_sample()
            samples.append(sample)
            type_counts[sample.sample_type] = type_counts.get(sample.sample_type, 0) + 1
        
        # Split: 90/5/5
        n_train = int(n_samples * 0.90)
        n_val = int(n_samples * 0.05)
        
        splits = {
            "train": samples[:n_train],
            "val": samples[n_train:n_train + n_val],
            "test": samples[n_train + n_val:]
        }
        
        stats = {}
        for split_name, split_samples in splits.items():
            # Chat format (recommended for Avus)
            chat_path = output / f"{split_name}.txt"
            with chat_path.open("w", encoding="utf-8") as f:
                for s in split_samples:
                    f.write(s.to_training_text("chat") + "\n\n")
            
            # JSONL for inspection
            jsonl_path = output / f"{split_name}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for s in split_samples:
                    f.write(json.dumps(s.to_dict()) + "\n")
            
            # Stats
            split_types = {}
            for s in split_samples:
                split_types[s.sample_type] = split_types.get(s.sample_type, 0) + 1
            stats[split_name] = {"total": len(split_samples), "types": split_types}
            print(f"  {split_name}: {len(split_samples):,} samples → {chat_path}")
        
        # Metadata
        meta = {
            "version": "v3",
            "model": "Avus",
            "generated": datetime.now().isoformat(),
            "total_samples": n_samples,
            "config": asdict(self.config),
            "type_distribution": type_counts,
            "splits": stats,
            "tiers": {
                "1": "cognitive_loop_basic",
                "2": "identity_narrative", 
                "3": "memory_hierarchy",
                "4": "tool_use",
                "5": "multimodal",
                "6": "homeostasis",
                "7": "state_sync",
                "8": "code_execution"
            },
            "format": "chat (user/assistant tags with endoftext)",
            "features": [
                "cognitive_loop_training",
                "identity_persistence",
                "hierarchical_memory",
                "tool_use_generation",
                "multimodal_reasoning",
                "homeostatic_regulation",
                "crdt_state_sync",
                "wasm_sandbox_traces"
            ]
        }
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
        
        # Estimate tokens (rough)
        avg_prompt_len = sum(len(s.prompt.split()) for s in samples[:100]) / 100
        avg_resp_len = sum(len(s.response.split()) for s in samples[:100]) / 100
        estimated_tokens = n_samples * (avg_prompt_len + avg_resp_len) * 1.3  # 1.3 for subword tokenization
        
        print(f"\nDataset saved to {output}/")
        print(f"Estimated tokens: {estimated_tokens:,.0f}")
        print(f"Average sample length: {avg_prompt_len + avg_resp_len:.0f} words")
        print(f"Type distribution:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = c / n_samples * 100
            print(f"  {t}: {c:,} ({pct:.1f}%)")
        
        return stats

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Avus training dataset")
    parser.add_argument("samples", nargs="?", type=int, default=100000,
                       help="Number of samples to generate (default: 100000)")
    parser.add_argument("--output", "-o", default="avus_dataset_v3",
                       help="Output directory")
    parser.add_argument("--preview", action="store_true",
                       help="Show sample outputs for each type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--vocab-size", type=int, default=5000,
                       help="Target vocabulary size")
    
    args = parser.parse_args()
    
    config = DatasetConfig(vocab_size=args.vocab_size)
    gen = AvusDatasetGenerator(config=config, seed=args.seed)
    
    if args.preview:
        print("=" * 60)
        print("AVUS DATASET PREVIEW")
        print("=" * 60)
        
        types = [
            "cognitive_loop", "identity_narrative", "memory_hierarchy",
            "tool_use", "multimodal", "homeostasis", "state_sync", "code_execution"
        ]
        
        for sample_type in types:
            print(f"\n{'─' * 60}")
            print(f"TYPE: {sample_type.upper()}")
            print('─' * 60)
            sample = gen.generate_sample(forced_type=sample_type)
            print(f"ID: {sample.id} | Tier: {sample.complexity_tier} | Phase: {sample.cognitive_phase}")
            print(f"\nPROMPT:\n{sample.prompt}")
            print(f"\nRESPONSE:\n{sample.response}")
            print(f"\nMETADATA: {json.dumps(sample.metadata, indent=2)}")
        
        print(f"\n{'=' * 60}")
        print("END PREVIEW")
        print(f"{'=' * 60}")
    else:
        gen.generate_dataset(args.samples, args.output)

if __name__ == "__main__":
    main()
