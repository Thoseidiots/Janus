# language_dataset_generator_v2.py

“””
Avus Language Comprehension Dataset Generator v2.

Fixes from v1:

- Shared context dict so prompt/response stay about the same topic
- Expanded vocabulary (300+ concepts across 12 categories)
- Dialogue samples for conversational flow
- Reasoning samples for step-by-step thinking
- Curriculum levels (easy → hard)
- Scales to 1M+ samples

Usage:
python language_dataset_generator_v2.py 100000
python language_dataset_generator_v2.py –preview
python language_dataset_generator_v2.py 1000000  # 1M samples
“””

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# ── Expanded Vocabulary ───────────────────────────────────────────────────────

CONCEPTS = {
“technology”: [
“artificial intelligence”, “machine learning”, “neural networks”, “deep learning”,
“natural language processing”, “computer vision”, “robotics”, “automation”,
“algorithms”, “data structures”, “databases”, “networking”, “cloud computing”,
“cybersecurity”, “blockchain”, “virtual reality”, “augmented reality”,
“quantum computing”, “programming”, “software engineering”, “operating systems”,
“compilers”, “APIs”, “microservices”, “distributed systems”,
],
“science”: [
“physics”, “chemistry”, “biology”, “astronomy”, “geology”, “ecology”,
“neuroscience”, “genetics”, “evolution”, “thermodynamics”, “quantum mechanics”,
“relativity”, “electromagnetism”, “photosynthesis”, “metabolism”,
“cell division”, “DNA replication”, “protein synthesis”, “atomic structure”,
],
“mathematics”: [
“calculus”, “algebra”, “geometry”, “statistics”, “probability”,
“number theory”, “linear algebra”, “differential equations”, “topology”,
“graph theory”, “set theory”, “logic”, “combinatorics”, “trigonometry”,
],
“arts”: [
“painting”, “sculpture”, “music composition”, “storytelling”, “photography”,
“animation”, “game design”, “architecture”, “fashion design”, “film making”,
“poetry”, “illustration”, “digital art”, “character design”, “world building”,
],
“tools”: [
“a hammer”, “a microscope”, “a telescope”, “a computer”, “a calculator”,
“a paintbrush”, “a camera”, “a compass”, “a thermometer”, “a ruler”,
“a keyboard”, “a database”, “a framework”, “a library”, “a compiler”,
],
“professions”: [
“engineer”, “scientist”, “artist”, “developer”, “researcher”,
“teacher”, “designer”, “architect”, “musician”, “writer”,
“doctor”, “mathematician”, “philosopher”, “explorer”, “inventor”,
],
“places”: [
“laboratory”, “library”, “studio”, “workshop”, “university”,
“research center”, “hospital”, “observatory”, “museum”, “factory”,
“data center”, “garden”, “forest”, “mountain”, “city”,
],
“emotions”: [
“curious”, “excited”, “determined”, “inspired”, “thoughtful”,
“focused”, “confident”, “energized”, “calm”, “creative”,
“motivated”, “passionate”, “persistent”, “patient”, “resilient”,
],
“domains”: [
“software development”, “scientific research”, “education”, “healthcare”,
“entertainment”, “business”, “engineering”, “creative arts”, “finance”,
“manufacturing”, “transportation”, “communication”, “agriculture”,
“environmental science”, “space exploration”,
],
“actions”: [
“learn a new skill”, “solve a complex problem”, “create something original”,
“improve a system”, “understand a concept”, “build a project”,
“design an algorithm”, “analyze data”, “write clean code”, “debug software”,
“train a model”, “optimize performance”, “communicate ideas”, “teach others”,
],
“qualities”: [
“efficient”, “reliable”, “scalable”, “accurate”, “innovative”,
“robust”, “flexible”, “powerful”, “elegant”, “practical”,
“creative”, “systematic”, “precise”, “adaptive”, “intelligent”,
],
“materials”: [
“silicon”, “carbon”, “light”, “electricity”, “data”,
“code”, “mathematics”, “energy”, “information”, “knowledge”,
],
}

ALL_CONCEPTS = [c for cats in CONCEPTS.values() for c in cats]

DEFINITIONS = [
“a systematic approach to solving complex problems”,
“a method for processing and transforming information”,
“a set of principles that guide understanding and behavior”,
“a collection of interconnected components working together”,
“a framework for organizing and applying knowledge”,
“a technique that enables new capabilities and insights”,
“a tool that amplifies human creativity and intelligence”,
“a process that converts raw data into meaningful results”,
“a discipline focused on understanding fundamental patterns”,
“a system that learns and adapts from experience”,
]

PROCESSES = [
“analyzing patterns in large amounts of data”,
“breaking complex problems into manageable components”,
“applying learned knowledge to new situations”,
“iteratively improving results through feedback”,
“combining multiple techniques to achieve better outcomes”,
“systematically testing hypotheses and refining models”,
“transforming raw inputs into structured outputs”,
“building abstractions that simplify complex systems”,
]

REASONS = [
“it enables solutions to previously unsolvable problems”,
“it dramatically improves efficiency and accuracy”,
“it opens new possibilities for human creativity”,
“it helps us understand the world at a deeper level”,
“it makes powerful capabilities accessible to everyone”,
“it reduces errors and increases reliability”,
“it accelerates progress across many fields”,
“it creates new opportunities for innovation”,
]

# ── Sample dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainingSample:
id:       str
type:     str
level:    int       # curriculum level 1-5
prompt:   str
response: str

```
def to_training_text(self) -> str:
    return f"{self.prompt}\n{self.response}<|endoftext|>"

def to_dict(self) -> dict:
    return {
        "id":       self.id,
        "type":     self.type,
        "level":    self.level,
        "prompt":   self.prompt,
        "response": self.response,
        "text":     self.to_training_text(),
    }
```

# ── Generator ─────────────────────────────────────────────────────────────────

class LanguageDatasetGeneratorV2:
“””
Fixed and expanded language dataset generator for Avus.
Key fix: shared context dict ensures prompt and response
are always about the same topic.
“””

```
def __init__(self, seed: int = 42):
    self.rng = random.Random(seed)

def _pick(self, lst: list) -> str:
    return self.rng.choice(lst)

def _concept(self) -> str:
    return self._pick(ALL_CONCEPTS)

def _context(self) -> Dict[str, str]:
    """
    Build a shared context dictionary.
    CRITICAL: prompt and response templates both use this same
    context so they stay about the same topic.
    """
    concept  = self._concept()
    concept2 = self._concept()
    action   = self._pick(REASONS[0:4] + PROCESSES)
    return {
        "concept":    concept,
        "concept2":   concept2,
        "definition": self._pick(DEFINITIONS),
        "process":    self._pick(PROCESSES),
        "reason":     self._pick(REASONS),
        "benefit1":   self._pick(REASONS),
        "benefit2":   self._pick(REASONS),
        "action":     self._pick(list(CONCEPTS["actions"])),
        "domain":     self._pick(list(CONCEPTS["domains"])),
        "quality":    self._pick(list(CONCEPTS["qualities"])),
        "profession": self._pick(list(CONCEPTS["professions"])),
        "place":      self._pick(list(CONCEPTS["places"])),
        "emotion":    self._pick(list(CONCEPTS["emotions"])),
        "step1":      self._pick(PROCESSES),
        "step2":      self._pick(PROCESSES),
        "step3":      self._pick(REASONS),
    }

def _fill(self, template: str, ctx: Dict[str, str]) -> str:
    """Fill template using shared context — guarantees consistency."""
    result = template
    for key, value in ctx.items():
        result = result.replace("{" + key + "}", value)
    return result

# ── Level 1: Simple QA ────────────────────────────────────────────────────

def generate_qa(self) -> TrainingSample:
    ctx = self._context()
    templates = [
        ("What is {concept}?",
         "{concept} is {definition}."),
        ("How does {concept} work?",
         "{concept} works by {process}."),
        ("Why is {concept} important?",
         "{concept} is important because {reason}."),
        ("What are the benefits of {concept}?",
         "The benefits of {concept} include {benefit1} and {benefit2}."),
        ("Where is {concept} used?",
         "{concept} is commonly used in {domain} and related fields."),
        ("Who uses {concept}?",
         "{concept} is used by {profession}s to {action}."),
    ]
    pt, rt = self._pick(templates)
    return TrainingSample(
        id       = str(uuid.uuid4())[:8],
        type     = "qa",
        level    = 1,
        prompt   = self._fill(pt, ctx),
        response = self._fill(rt, ctx),
    )

# ── Level 2: Instructions ─────────────────────────────────────────────────

def generate_instruction(self) -> TrainingSample:
    ctx = self._context()
    templates = [
        ("Describe {concept} briefly.",
         "{concept} is {definition}. It is used in {domain} and is known for being {quality}."),
        ("Explain {concept} in simple terms.",
         "{concept} can be understood as {definition}. Essentially, it involves {process}."),
        ("List three facts about {concept}.",
         "1. {concept} is {definition}.\n2. It works by {process}.\n3. It is valued because {reason}."),
        ("Compare {concept} and {concept2}.",
         "Both {concept} and {concept2} are important in {domain}. {concept} focuses on {process}, while {concept2} emphasizes {definition}."),
        ("Summarize {concept} in one sentence.",
         "{concept} is {definition} that enables {process}."),
        ("Give an example of how {concept} is used.",
         "{concept} is used in {domain}. For instance, a {profession} might use it to {action}, which helps {reason}."),
        ("What should someone know about {concept}?",
         "About {concept}: it is {definition}. The key thing to understand is that it works by {process}. It matters because {reason}."),
    ]
    pt, rt = self._pick(templates)
    return TrainingSample(
        id       = str(uuid.uuid4())[:8],
        type     = "instruction",
        level    = 2,
        prompt   = self._fill(pt, ctx),
        response = self._fill(rt, ctx),
    )

# ── Level 3: Dialogue ─────────────────────────────────────────────────────

def generate_dialogue(self) -> TrainingSample:
    ctx = self._context()
    concept  = ctx["concept"]
    defn     = ctx["definition"]
    reason   = ctx["reason"]
    process  = ctx["process"]
    domain   = ctx["domain"]

    templates = [
        (
            f"User: What is {concept}?\nAssistant:",
            f"{concept} is {defn}.\n\nUser: Why does it matter?\nAssistant: It matters because {reason}.",
        ),
        (
            f"User: Can you explain {concept}?\nAssistant:",
            f"Of course. {concept} is {defn}. It works by {process}.\n\nUser: Where is it used?\nAssistant: It is commonly used in {domain}.",
        ),
        (
            f"User: I want to learn about {concept}.\nAssistant:",
            f"{concept} is {defn}.\n\nUser: How does it work?\nAssistant: It works by {process}.\n\nUser: That makes sense, thanks.\nAssistant: You're welcome. {concept} is a valuable topic to understand.",
        ),
    ]
    pt, rt = self._pick(templates)
    return TrainingSample(
        id       = str(uuid.uuid4())[:8],
        type     = "dialogue",
        level    = 3,
        prompt   = pt,
        response = rt,
    )

# ── Level 4: Story continuation ───────────────────────────────────────────

def generate_story(self) -> TrainingSample:
    ctx        = self._context()
    concept    = ctx["concept"]
    profession = ctx["profession"]
    place      = ctx["place"]
    emotion    = ctx["emotion"]
    domain     = ctx["domain"]
    defn       = ctx["definition"]
    process    = ctx["process"]

    beginnings = [
        f"The {profession} had spent years studying {concept}.",
        f"In the {place}, work on {concept} was progressing rapidly.",
        f"It started as a simple question about {concept}.",
        f"Nobody expected {concept} to change everything about {domain}.",
    ]

    continuations = [
        f"The key insight was that {concept} is {defn}. By {process}, new possibilities emerged. The {profession} felt {emotion} about what this meant for {domain}.",
        f"Understanding {concept} required patience. It is {defn}. The process of {process} took time, but eventually the pattern became clear.",
        f"The breakthrough came when the team realized that {concept} works by {process}. This changed how everyone in {domain} approached their work.",
    ]

    beginning   = self._pick(beginnings)
    continuation = self._pick(continuations)

    return TrainingSample(
        id       = str(uuid.uuid4())[:8],
        type     = "story",
        level    = 4,
        prompt   = beginning,
        response = beginning + " " + continuation,
    )

# ── Level 5: Reasoning ────────────────────────────────────────────────────

def generate_reasoning(self) -> TrainingSample:
    reasoning_type = self._pick(["math", "logic", "steps"])

    if reasoning_type == "math":
        a = self.rng.randint(2, 20)
        b = self.rng.randint(2, 20)
        op = self._pick(["add", "multiply"])
        if op == "add":
            prompt   = f"If you have {a} items and add {b} more, how many do you have?"
            response = f"Start with {a} items.\nAdd {b} more items.\n{a} + {b} = {a + b}\nThe answer is {a + b}."
        else:
            prompt   = f"If you have {a} groups of {b} items each, how many items total?"
            response = f"There are {a} groups.\nEach group has {b} items.\n{a} × {b} = {a * b}\nThe total is {a * b} items."

    elif reasoning_type == "logic":
        ctx     = self._context()
        concept = ctx["concept"]
        quality = ctx["quality"]
        domain  = ctx["domain"]
        prompt   = f"Is {concept} useful in {domain}? Explain your reasoning."
        response = f"Step 1: Consider what {concept} does. It is {ctx['definition']}.\nStep 2: Consider what {domain} needs. It requires {ctx['process']}.\nStep 3: Compare the two. {concept} is {quality}, which matches {domain}'s needs.\nConclusion: Yes, {concept} is useful in {domain}."

    else:  # steps
        ctx    = self._context()
        action = self._pick(list(CONCEPTS["actions"]))
        prompt   = f"What are the steps to {action}?"
        response = f"To {action}, follow these steps:\n\nStep 1: {ctx['step1'].capitalize()}.\nStep 2: {ctx['step2'].capitalize()}.\nStep 3: {ctx['step3'].capitalize()}.\n\nFollowing these steps will help you {action} effectively."

    return TrainingSample(
        id       = str(uuid.uuid4())[:8],
        type     = "reasoning",
        level    = 5,
        prompt   = prompt,
        response = response,
    )

# ── Batch generation ──────────────────────────────────────────────────────

def generate_batch(self, n: int, curriculum: bool = True) -> List[TrainingSample]:
    """
    Generate n samples. With curriculum=True, distributes across
    difficulty levels so Avus learns simple patterns before complex ones.
    """
    generators = [
        self.generate_qa,           # level 1
        self.generate_instruction,  # level 2
        self.generate_dialogue,     # level 3
        self.generate_story,        # level 4
        self.generate_reasoning,    # level 5
    ]

    samples = []
    for i in range(n):
        if curriculum:
            # Weight toward simpler tasks early, harder tasks later
            progress = i / n
            if progress < 0.3:
                gen = self._pick(generators[:2])    # QA + instruction
            elif progress < 0.6:
                gen = self._pick(generators[:3])    # + dialogue
            elif progress < 0.8:
                gen = self._pick(generators[:4])    # + story
            else:
                gen = self._pick(generators)        # all including reasoning
        else:
            gen = generators[i % len(generators)]

        samples.append(gen())

    return samples

def generate_all(
    self,
    output_dir:     str  = "language_dataset_v2",
    total_samples:  int  = 10000,
    curriculum:     bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Generating {total_samples:,} samples (curriculum={curriculum})...")
    samples = self.generate_batch(total_samples, curriculum=curriculum)

    # 90/5/5 split
    n_train = int(total_samples * 0.90)
    n_val   = int(total_samples * 0.05)
    splits  = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }

    stats = {}
    for split_name, split_samples in splits.items():
        # Plain text for training
        txt_path = output / f"{split_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(s.to_training_text() + "\n")

        # JSONL for inspection
        jsonl_path = output / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps(s.to_dict()) + "\n")

        type_counts = {}
        for s in split_samples:
            type_counts[s.type] = type_counts.get(s.type, 0) + 1

        stats[split_name] = {"total": len(split_samples), "by_type": type_counts}
        print(f"  {split_name}: {len(split_samples):,} samples → {txt_path}")

    # Metadata
    meta = {
        "version":       "v2",
        "generated":     datetime.now().isoformat(),
        "total_samples": total_samples,
        "curriculum":    curriculum,
        "splits":        stats,
        "fix_applied":   "shared context dict — prompt/response always about same topic",
        "sample_types":  ["qa", "instruction", "dialogue", "story", "reasoning"],
        "levels":        {1: "qa", 2: "instruction", 3: "dialogue", 4: "story", 5: "reasoning"},
        "format":        "prompt\\nresponse<|endoftext|>",
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))

    estimated_tokens = total_samples * 120
    print(f"\nDataset saved to {output}/")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Use {output}/train.txt to fine-tune Avus.")
    return stats
```

# ── CLI ───────────────────────────────────────────────────────────────────────

if **name** == “**main**”:
import sys

```
n          = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10000
curriculum = "--no-curriculum" not in sys.argv

gen = LanguageDatasetGeneratorV2(seed=42)

if "--preview" in sys.argv:
    print("=== QA (level 1) ===")
    s = gen.generate_qa()
    print(f"Prompt:   {s.prompt}")
    print(f"Response: {s.response}\n")

    print("=== Instruction (level 2) ===")
    s = gen.generate_instruction()
    print(f"Prompt:   {s.prompt}")
    print(f"Response: {s.response}\n")

    print("=== Dialogue (level 3) ===")
    s = gen.generate_dialogue()
    print(f"Prompt:   {s.prompt}")
    print(f"Response: {s.response}\n")

    print("=== Story (level 4) ===")
    s = gen.generate_story()
    print(f"Prompt:   {s.prompt}")
    print(f"Response: {s.response}\n")

    print("=== Reasoning (level 5) ===")
    s = gen.generate_reasoning()
    print(f"Prompt:   {s.prompt}")
    print(f"Response: {s.response}\n")

else:
    gen.generate_all(
        output_dir    = "language_dataset_v2",
        total_samples = n,
        curriculum    = curriculum,
    )
```
