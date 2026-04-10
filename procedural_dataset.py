"""
procedural_dataset.py -- No Man's Sky style procedural training data generator for Avus.

Key principle: combinatorially infinite unique samples, never repeats, gets harder over time,
seeded for reproducibility but practically infinite.

All data is 100% synthetic. No external sources required. Python stdlib only.
"""

import hashlib
import random
import itertools
import math
from typing import Generator, List, Dict, Tuple, Any, Optional

SOT = "<|startoftext|>"
EOT = "<|endoftext|>"


# -----------------------------------------------------------------------------
# 1. ProceduralSeed
# -----------------------------------------------------------------------------

class ProceduralSeed:
    """
    Deterministic pseudo-random seed engine.
    Every call to next() advances internal state; fork() creates isolated child streams.
    """

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFFFFFFFFFF
        self._counter = 0

    def _mix(self, value: int) -> int:
        # xorshift64 + multiplicative hash
        value ^= value >> 33
        value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 33
        value = (value * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 33
        return value

    def next(self) -> float:
        """Return a deterministic float in [0, 1)."""
        self._counter += 1
        raw = self._mix(self._state ^ (self._counter * 0x9E3779B97F4A7C15))
        return raw / 0xFFFFFFFFFFFFFFFF

    def next_int(self, lo: int, hi: int) -> int:
        """Return a deterministic int in [lo, hi]."""
        return lo + int(self.next() * (hi - lo + 1)) % (hi - lo + 1)

    def choice(self, seq: list) -> Any:
        return seq[self.next_int(0, len(seq) - 1)]

    def sample(self, seq: list, k: int) -> list:
        seq = list(seq)
        result = []
        for _ in range(min(k, len(seq))):
            idx = self.next_int(0, len(seq) - 1)
            result.append(seq.pop(idx))
        return result

    def shuffle(self, seq: list) -> list:
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self.next_int(0, i)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    def fork(self, key: str) -> "ProceduralSeed":
        """Create a child seed deterministically derived from this seed + a string key."""
        digest = hashlib.sha256(f"{self._state}:{key}".encode()).hexdigest()
        child_seed = int(digest[:16], 16)
        return ProceduralSeed(child_seed)


# -----------------------------------------------------------------------------
# 2. ConceptGraph
# -----------------------------------------------------------------------------

class ConceptGraph:
    """
    Large vocabulary pools. Combinations are astronomically large.
    50+ items per pool -> C(50,3) = 19,600 triples alone; with relations the space is millions+.
    """

    ENTITIES = [
        "photon", "neuron", "crystal", "vortex", "membrane", "lattice", "filament",
        "resonator", "catalyst", "substrate", "conduit", "matrix", "prism", "node",
        "beacon", "shard", "nexus", "pulse", "echo", "fragment", "core", "shell",
        "bridge", "anchor", "lens", "mirror", "gate", "thread", "wave", "particle",
        "cluster", "field", "orbit", "spiral", "arc", "loop", "chain", "grid",
        "vault", "spire", "basin", "ridge", "canopy", "root", "stem", "branch",
        "seed", "bloom", "drift", "current", "tide", "flux", "charge", "signal",
        "cipher", "token", "vector", "tensor", "manifold", "topology", "gradient",
    ]

    PROPERTIES = [
        "luminous", "dense", "volatile", "stable", "recursive", "adaptive", "rigid",
        "fluid", "resonant", "dormant", "active", "latent", "emergent", "entropic",
        "coherent", "chaotic", "symmetric", "asymmetric", "bounded", "unbounded",
        "compressed", "expanded", "inverted", "layered", "fractured", "unified",
        "polarized", "neutral", "charged", "decaying", "regenerating", "oscillating",
        "converging", "diverging", "saturated", "sparse", "opaque", "transparent",
        "elastic", "brittle", "magnetic", "inert", "reactive", "catalytic",
        "self-similar", "hierarchical", "distributed", "centralized", "stochastic",
        "deterministic", "periodic", "aperiodic", "transient", "persistent",
        "amplified", "attenuated", "modulated", "quantized", "continuous",
    ]

    RELATIONS = [
        "absorbs", "emits", "transforms", "contains", "disrupts", "stabilizes",
        "amplifies", "attenuates", "generates", "consumes", "reflects", "refracts",
        "catalyzes", "inhibits", "propagates", "collapses", "expands", "contracts",
        "encodes", "decodes", "filters", "routes", "bridges", "severs", "merges",
        "splits", "orbits", "anchors", "repels", "attracts", "synchronizes",
        "desynchronizes", "modulates", "demodulates", "compresses", "decompresses",
        "encrypts", "decrypts", "maps", "unmaps", "binds", "releases", "activates",
        "deactivates", "reinforces", "weakens", "channels", "disperses", "focuses",
        "scatters", "aligns", "misaligns", "couples", "decouples", "resonates with",
    ]

    ACTIONS = [
        "traverse", "analyze", "reconstruct", "optimize", "simulate", "predict",
        "classify", "cluster", "encode", "decode", "compress", "expand", "search",
        "sort", "filter", "transform", "aggregate", "distribute", "synchronize",
        "validate", "verify", "infer", "deduce", "extrapolate", "interpolate",
        "decompose", "compose", "map", "reduce", "fold", "unfold", "iterate",
        "recurse", "branch", "merge", "split", "route", "schedule", "allocate",
        "balance", "calibrate", "tune", "probe", "sample", "measure", "estimate",
        "approximate", "converge", "diverge", "stabilize", "perturb", "evolve",
        "mutate", "select", "crossover", "propagate", "backpropagate", "update",
    ]

    DOMAINS = [
        "thermodynamics", "quantum mechanics", "graph theory", "topology", "ecology",
        "linguistics", "cryptography", "neuroscience", "cosmology", "genetics",
        "fluid dynamics", "information theory", "game theory", "complexity theory",
        "signal processing", "materials science", "evolutionary biology", "logic",
        "number theory", "combinatorics", "probability theory", "set theory",
        "category theory", "control theory", "chaos theory", "network science",
        "cognitive science", "epistemology", "formal languages", "automata theory",
        "distributed systems", "computational geometry", "optimization theory",
        "statistical mechanics", "electrodynamics", "acoustics", "optics",
        "biochemistry", "geophysics", "climatology", "economics", "sociology",
        "anthropology", "archaeology", "philosophy of mind", "ethics", "aesthetics",
        "semiotics", "rhetoric", "narrative theory", "systems biology", "proteomics",
        "genomics", "metabolomics", "pharmacology", "immunology", "virology",
    ]

    CONTEXTS = [
        "under extreme pressure", "in a vacuum", "at absolute zero", "near a singularity",
        "during phase transition", "at equilibrium", "far from equilibrium", "at scale",
        "in isolation", "in a network", "under observation", "without measurement",
        "in a closed system", "in an open system", "over geological time", "instantaneously",
        "recursively", "iteratively", "stochastically", "deterministically",
        "in parallel", "sequentially", "reversibly", "irreversibly", "locally",
        "globally", "asymptotically", "transiently", "periodically", "aperiodically",
        "in the presence of noise", "in a noiseless channel", "under adversarial conditions",
        "cooperatively", "competitively", "hierarchically", "flatly", "emergently",
        "reductively", "holistically", "probabilistically", "categorically",
        "continuously", "discretely", "in high dimensions", "in low dimensions",
        "across scales", "within a single scale", "under symmetry breaking",
        "preserving invariants", "violating conservation laws", "at criticality",
        "below threshold", "above threshold", "in steady state", "in transient state",
    ]

    def combine(self, seed: ProceduralSeed, n: int = 3) -> str:
        """Pick n items from different pools and combine into a novel concept description."""
        s = seed.fork("combine")
        entity = s.choice(self.ENTITIES)
        prop = s.choice(self.PROPERTIES)
        domain = s.choice(self.DOMAINS)
        context = s.choice(self.CONTEXTS)
        action = s.choice(self.ACTIONS)

        templates = [
            f"a {prop} {entity} that {action}s {context} within {domain}",
            f"the {action} of a {prop} {entity} as studied in {domain}, {context}",
            f"a {domain}-derived {prop} {entity} capable of {action}ing {context}",
            f"{prop} {entity}-{action} dynamics in {domain} {context}",
            f"emergent {prop} behavior of a {entity} {action}ing in {domain} {context}",
        ]
        return s.choice(templates)

    def relate(self, seed: ProceduralSeed, concept_a: str, concept_b: str) -> str:
        """Generate a relationship statement between two concepts."""
        s = seed.fork("relate")
        relation = s.choice(self.RELATIONS)
        context = s.choice(self.CONTEXTS)
        qualifier = s.choice(["directly", "indirectly", "partially", "fully", "conditionally"])
        templates = [
            f"The {concept_a} {qualifier} {relation} the {concept_b} {context}.",
            f"When {context}, the {concept_a} {relation}s the {concept_b}.",
            f"It is observed that the {concept_a} {qualifier} {relation}s the {concept_b} {context}.",
            f"The interaction between {concept_a} and {concept_b} shows that the former {relation}s the latter {context}.",
        ]
        return s.choice(templates)

    def random_concept(self, seed: ProceduralSeed) -> str:
        s = seed.fork("rc")
        return f"{s.choice(self.PROPERTIES)} {s.choice(self.ENTITIES)}"


# -----------------------------------------------------------------------------
# 3. ProceduralReasoningChain
# -----------------------------------------------------------------------------

class ProceduralReasoningChain:
    """
    Generates multi-step reasoning chains. Difficulty controls length and complexity.
    """

    CONNECTORS = [
        "Therefore", "It follows that", "Consequently", "This implies",
        "Building on this", "Given the above", "As a result", "We can deduce that",
        "This leads to the conclusion that", "Extending this reasoning",
        "Applying this principle", "By analogy", "Contrastingly",
        "However, if we consider", "Furthermore",
    ]

    QUESTION_FRAMES = [
        "What happens when {concept_a} interacts with {concept_b} {context}?",
        "How does {concept_a} influence {concept_b} in the domain of {domain}?",
        "Explain the relationship between {concept_a} and {concept_b} {context}.",
        "Given {concept_a}, what can be inferred about {concept_b}?",
        "Analyze the behavior of {concept_a} when subjected to {concept_b} {context}.",
        "Under what conditions does {concept_a} {action} {concept_b}?",
        "Describe the emergent properties when {concept_a} and {concept_b} interact {context}.",
    ]

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def generate(self, seed: ProceduralSeed, difficulty: int = 1) -> Dict[str, Any]:
        s = seed.fork("reasoning")
        steps = max(2, min(8, 2 + difficulty))
        complexity_words = max(1, difficulty // 2)

        # Build concepts for this chain
        concepts = [self.graph.combine(s.fork(f"c{i}"), n=complexity_words + 1)
                    for i in range(steps + 1)]
        domain = s.choice(ConceptGraph.DOMAINS)
        context = s.choice(ConceptGraph.CONTEXTS)
        action = s.choice(ConceptGraph.ACTIONS)

        # Build question
        q_template = s.choice(self.QUESTION_FRAMES)
        question = q_template.format(
            concept_a=concepts[0],
            concept_b=concepts[1],
            context=context,
            domain=domain,
            action=action,
        )

        # Build reasoning steps
        chain_steps = []
        premise = f"We begin by observing that {concepts[0]} exhibits behavior consistent with {domain} principles {context}."
        chain_steps.append(("Step 1 [Premise]", premise))

        for i in range(1, steps):
            connector = s.choice(self.CONNECTORS)
            relation_stmt = self.graph.relate(s.fork(f"rel{i}"), concepts[i - 1], concepts[i])
            elaboration = self._elaborate(s.fork(f"elab{i}"), concepts[i], domain, difficulty)
            step_text = f"{connector}, {relation_stmt} {elaboration}"
            chain_steps.append((f"Step {i + 1}", step_text))

        # Conclusion
        conclusion_concept = concepts[-1]
        conclusion = (
            f"In conclusion, the interaction of {concepts[0]} and {concepts[1]} "
            f"within {domain} {context} ultimately leads to {conclusion_concept}, "
            f"demonstrating a {s.choice(ConceptGraph.PROPERTIES)} outcome."
        )
        chain_steps.append(("Conclusion", conclusion))

        return {
            "type": "reasoning_chain",
            "difficulty": difficulty,
            "question": question,
            "steps": chain_steps,
            "domain": domain,
            "num_steps": len(chain_steps),
        }

    def _elaborate(self, seed: ProceduralSeed, concept: str, domain: str, difficulty: int) -> str:
        s = seed
        prop = s.choice(ConceptGraph.PROPERTIES)
        action = s.choice(ConceptGraph.ACTIONS)
        context = s.choice(ConceptGraph.CONTEXTS)
        if difficulty <= 3:
            return f"This is because {concept} is {prop} {context}."
        elif difficulty <= 6:
            return (f"This is because {concept} is {prop} {context}, "
                    f"which causes it to {action} according to {domain} principles.")
        else:
            extra_prop = s.choice(ConceptGraph.PROPERTIES)
            extra_action = s.choice(ConceptGraph.ACTIONS)
            return (f"This is because {concept} is simultaneously {prop} and {extra_prop} {context}, "
                    f"causing it to {action} and subsequently {extra_action}, "
                    f"a non-trivial consequence of {domain} at this scale.")

    def format(self, chain: Dict[str, Any]) -> str:
        lines = [
            f"[Reasoning Chain | Difficulty {chain['difficulty']} | Domain: {chain['domain']}]",
            f"",
            f"Question: {chain['question']}",
            f"",
            f"Reasoning:",
        ]
        for label, text in chain["steps"]:
            lines.append(f"  {label}: {text}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 4. ProceduralScenario
# -----------------------------------------------------------------------------

class ProceduralScenario:
    """
    Generates world scenarios with agents, goals, obstacles, and outcomes.
    """

    AGENT_ARCHETYPES = [
        "explorer", "guardian", "architect", "analyst", "mediator", "disruptor",
        "synthesizer", "observer", "catalyst", "regulator", "navigator", "harvester",
        "sentinel", "propagator", "optimizer", "decoder", "encoder", "mapper",
        "curator", "arbitrator",
    ]

    GOAL_TEMPLATES = [
        "achieve stable {property} state in the {entity} system",
        "maximize {property} output of the {entity} network",
        "prevent {property} collapse of the {entity} structure",
        "decode the {property} pattern within the {entity} field",
        "establish a {property} equilibrium between competing {entity} forces",
        "reconstruct the {property} {entity} from fragmented data",
        "navigate the {property} {entity} topology without triggering instability",
        "extract the {property} signal from the {entity} noise",
        "synchronize {property} oscillations across the {entity} lattice",
        "minimize entropy in the {property} {entity} manifold",
    ]

    OBSTACLE_TEMPLATES = [
        "a {property} {entity} that {relation}s all progress {context}",
        "cascading {property} failures in the {entity} network {context}",
        "a {property} feedback loop within the {entity} system {context}",
        "interference from a {property} {entity} operating {context}",
        "resource depletion caused by a {property} {entity} {context}",
        "a {property} {entity} that {relation}s critical pathways {context}",
        "temporal instability in the {property} {entity} {context}",
        "adversarial {property} {entity} agents acting {context}",
    ]

    OUTCOME_TEMPLATES = [
        "The agent's {property} attribute overcomes the obstacle's {property2} resistance, achieving the goal.",
        "The obstacle's {property} nature proves too strong; the agent must adapt its strategy.",
        "A partial success: the agent achieves {goal_fragment} but the {property} obstacle persists.",
        "Unexpected synergy: the {property} agent and {property2} obstacle reach a stable coexistence.",
        "The agent's {property} capability transforms the obstacle into a resource.",
        "Critical failure: the {property} obstacle triggers a cascade that invalidates the goal.",
        "The agent discovers a {property} bypass, circumventing the obstacle entirely.",
        "Stalemate: neither the {property} agent nor the {property2} obstacle can dominate {context}.",
    ]

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def _make_agent(self, seed: ProceduralSeed, difficulty: int) -> Dict[str, Any]:
        s = seed
        archetype = s.choice(self.AGENT_ARCHETYPES)
        num_props = max(1, min(5, difficulty // 2 + 1))
        properties = s.sample(ConceptGraph.PROPERTIES, num_props)
        goal_template = s.choice(self.GOAL_TEMPLATES)
        goal = goal_template.format(
            property=s.choice(ConceptGraph.PROPERTIES),
            entity=s.choice(ConceptGraph.ENTITIES),
        )
        constraints = []
        if difficulty >= 4:
            constraints.append(f"must operate {s.choice(ConceptGraph.CONTEXTS)}")
        if difficulty >= 7:
            constraints.append(f"cannot use {s.choice(ConceptGraph.ACTIONS)} actions")
        return {
            "archetype": archetype,
            "properties": properties,
            "goal": goal,
            "constraints": constraints,
            "capability_score": round(sum(hash(p) % 10 for p in properties) / (num_props * 10), 2),
        }

    def _make_obstacle(self, seed: ProceduralSeed, difficulty: int) -> Dict[str, Any]:
        s = seed
        template = s.choice(self.OBSTACLE_TEMPLATES)
        desc = template.format(
            property=s.choice(ConceptGraph.PROPERTIES),
            entity=s.choice(ConceptGraph.ENTITIES),
            relation=s.choice(ConceptGraph.RELATIONS),
            context=s.choice(ConceptGraph.CONTEXTS),
        )
        resistance = round(0.3 + (difficulty / 10) * 0.6 + s.next() * 0.1, 2)
        return {
            "description": desc,
            "resistance": min(1.0, resistance),
            "type": s.choice(ConceptGraph.PROPERTIES),
        }

    def generate(self, seed: ProceduralSeed, difficulty: int = 1) -> Dict[str, Any]:
        s = seed.fork("scenario")
        num_agents = max(1, min(5, 1 + difficulty // 3))
        num_obstacles = max(1, min(6, 1 + difficulty // 2))

        agents = [self._make_agent(s.fork(f"agent{i}"), difficulty) for i in range(num_agents)]
        obstacles = [self._make_obstacle(s.fork(f"obs{i}"), difficulty) for i in range(num_obstacles)]

        # Determine outcomes per agent
        outcomes = []
        for i, agent in enumerate(agents):
            obs = obstacles[i % len(obstacles)]
            cap = agent["capability_score"]
            res = obs["resistance"]
            outcome_template = s.choice(self.OUTCOME_TEMPLATES)
            props = agent["properties"]
            outcome = outcome_template.format(
                property=props[0] if props else "adaptive",
                property2=obs["type"],
                goal_fragment=agent["goal"][:40] + "...",
                context=s.choice(ConceptGraph.CONTEXTS),
            )
            outcomes.append({
                "agent": agent["archetype"],
                "success": cap > res,
                "outcome": outcome,
            })

        world_context = s.choice(ConceptGraph.CONTEXTS)
        domain = s.choice(ConceptGraph.DOMAINS)

        return {
            "type": "scenario",
            "difficulty": difficulty,
            "world_context": world_context,
            "domain": domain,
            "agents": agents,
            "obstacles": obstacles,
            "outcomes": outcomes,
        }

    def format(self, scenario: Dict[str, Any]) -> str:
        lines = [
            f"[Scenario | Difficulty {scenario['difficulty']} | Domain: {scenario['domain']}]",
            f"World Context: {scenario['world_context']}",
            f"",
            f"Agents ({len(scenario['agents'])}):",
        ]
        for i, agent in enumerate(scenario["agents"]):
            lines.append(f"  Agent {i+1} [{agent['archetype']}]")
            lines.append(f"    Properties: {', '.join(agent['properties'])}")
            lines.append(f"    Goal: {agent['goal']}")
            if agent["constraints"]:
                lines.append(f"    Constraints: {'; '.join(agent['constraints'])}")
            lines.append(f"    Capability Score: {agent['capability_score']}")

        lines.append(f"")
        lines.append(f"Obstacles ({len(scenario['obstacles'])}):")
        for i, obs in enumerate(scenario["obstacles"]):
            lines.append(f"  Obstacle {i+1}: {obs['description']}")
            lines.append(f"    Resistance: {obs['resistance']} | Type: {obs['type']}")

        lines.append(f"")
        lines.append(f"Outcomes:")
        for out in scenario["outcomes"]:
            status = "SUCCESS" if out["success"] else "FAILURE"
            lines.append(f"  [{status}] {out['agent']}: {out['outcome']}")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 5. ProceduralCodePuzzle
# -----------------------------------------------------------------------------

class ProceduralCodePuzzle:
    """
    Generates novel algorithm problems from combinations of data structures,
    operations, and constraints. Produces problem description, buggy solution,
    and correct solution.
    """

    DATA_STRUCTURES = [
        ("list", "a sequence of elements", "[]"),
        ("stack", "a LIFO collection", "[]"),
        ("queue", "a FIFO collection", "deque()"),
        ("binary tree", "a hierarchical node structure", "None"),
        ("graph", "nodes connected by edges", "{}"),
        ("hash map", "key-value pairs", "{}"),
        ("heap", "a priority-ordered collection", "[]"),
        ("trie", "a prefix tree", "{}"),
        ("matrix", "a 2D grid of values", "[[]]"),
        ("linked list", "nodes with next pointers", "None"),
        ("set", "unique unordered elements", "set()"),
        ("deque", "double-ended queue", "deque()"),
        ("segment tree", "range query structure", "[]"),
        ("union-find", "disjoint set structure", "{}"),
        ("circular buffer", "fixed-size ring buffer", "[]"),
    ]

    OPERATIONS = [
        ("find the maximum", "max_val", "track running maximum"),
        ("find the minimum", "min_val", "track running minimum"),
        ("compute the sum", "total", "accumulate values"),
        ("count occurrences", "count", "use a frequency map"),
        ("detect a cycle", "has_cycle", "use visited set or slow/fast pointers"),
        ("find duplicates", "duplicates", "use a seen set"),
        ("reverse the order", "reversed_result", "swap elements or use a stack"),
        ("sort by frequency", "sorted_result", "count then sort"),
        ("find the median", "median", "use two heaps or sort"),
        ("compute the depth", "depth", "use BFS or DFS with level tracking"),
        ("find all paths", "paths", "use DFS with backtracking"),
        ("check balance", "is_balanced", "compare left and right subtree heights"),
        ("merge intervals", "merged", "sort then scan for overlaps"),
        ("find longest subsequence", "longest", "use dynamic programming"),
        ("compute prefix sums", "prefix", "accumulate from left"),
        ("partition elements", "partitioned", "two-pointer technique"),
        ("find k-th element", "kth", "use a heap of size k"),
        ("validate structure", "is_valid", "check invariants recursively"),
    ]

    CONSTRAINTS = [
        "in O(n) time",
        "in O(n log n) time",
        "using O(1) extra space",
        "using O(n) extra space",
        "without modifying the input",
        "handling negative values",
        "handling duplicate values",
        "for an empty input",
        "where values can be None",
        "in a single pass",
        "using recursion",
        "without recursion",
        "for very large inputs",
        "where the structure may be cyclic",
        "preserving relative order",
        "returning all valid solutions",
        "returning only the first solution",
        "where elements are unsorted",
    ]

    BUG_TYPES = [
        ("off-by-one error", "uses `<` instead of `<=` in loop bound"),
        ("wrong base case", "returns incorrect value for empty input"),
        ("missing initialization", "forgets to initialize accumulator variable"),
        ("incorrect update", "updates the wrong variable in the loop body"),
        ("wrong comparison", "uses `>` instead of `>=` causing missed edge case"),
        ("index out of bounds", "accesses index without checking length"),
        ("wrong return value", "returns intermediate result instead of final answer"),
        ("mutation bug", "modifies input list while iterating over it"),
        ("missing edge case", "does not handle single-element input"),
        ("wrong loop direction", "iterates forward when backward is needed"),
        ("incorrect condition", "uses `and` instead of `or` in conditional"),
        ("type mismatch", "compares int to string without conversion"),
    ]

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def generate(self, seed: ProceduralSeed, difficulty: int = 1) -> Dict[str, Any]:
        s = seed.fork("codepuzzle")

        ds_name, ds_desc, ds_init = s.choice(self.DATA_STRUCTURES)
        op_name, op_var, op_hint = s.choice(self.OPERATIONS)
        num_constraints = max(1, min(4, difficulty // 2 + 1))
        constraints = s.sample(self.CONSTRAINTS, num_constraints)
        bug_type, bug_desc = s.choice(self.BUG_TYPES)

        # Generate thematic flavor from concept graph
        flavor_entity = s.choice(ConceptGraph.ENTITIES)
        flavor_domain = s.choice(ConceptGraph.DOMAINS)

        problem = self._build_problem(s, ds_name, ds_desc, op_name, constraints,
                                      flavor_entity, flavor_domain, difficulty)
        correct = self._build_correct_solution(s, ds_name, op_name, op_var, op_hint,
                                                constraints, difficulty)
        buggy = self._inject_bug(correct, bug_type, bug_desc, s)

        return {
            "type": "code_puzzle",
            "difficulty": difficulty,
            "data_structure": ds_name,
            "operation": op_name,
            "constraints": constraints,
            "problem": problem,
            "buggy_solution": buggy,
            "correct_solution": correct,
            "bug_type": bug_type,
            "bug_description": bug_desc,
        }

    def _build_problem(self, s, ds_name, ds_desc, op_name, constraints,
                       flavor_entity, flavor_domain, difficulty):
        constraint_str = ", ".join(constraints)
        intro_styles = [
            f"In the study of {flavor_domain}, a {flavor_entity} system stores data as {ds_desc} ({ds_name}).",
            f"A {flavor_entity}-based {flavor_domain} model uses {ds_desc} ({ds_name}) to represent state.",
            f"Given a {ds_name} ({ds_desc}) derived from {flavor_domain} observations of {flavor_entity} behavior,",
        ]
        intro = s.choice(intro_styles)
        task = f"implement a function to {op_name} from the given {ds_name}."
        req = f"Requirements: {constraint_str}."
        example_note = "Your function should handle all edge cases including empty inputs."
        if difficulty >= 7:
            extra = (f" Additionally, the solution must be robust to {s.choice(self.CONSTRAINTS)} "
                     f"and {s.choice(self.CONSTRAINTS)}.")
        elif difficulty >= 4:
            extra = f" The solution must also handle {s.choice(self.CONSTRAINTS)}."
        else:
            extra = ""
        return f"{intro} {task} {req}{extra} {example_note}"

    def _build_correct_solution(self, s, ds_name, op_name, op_var, op_hint,
                                 constraints, difficulty):
        fn_name = op_name.replace(" ", "_").replace("-", "_")
        use_recursion = "using recursion" in constraints
        single_pass = "in a single pass" in constraints
        no_modify = "without modifying the input" in constraints

        if "maximum" in op_name or "minimum" in op_name:
            cmp = ">" if "maximum" in op_name else "<"
            init = "float('-inf')" if "maximum" in op_name else "float('inf')"
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} from {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return None
    result = {init}
    for item in data:
        if item {cmp} result:
            result = item
    return result"""

        elif "sum" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} of {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return 0
    total = 0
    for item in data:
        total += item
    return total"""

        elif "count" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} in {ds_name}.
    Hint: {op_hint}
    \"\"\"
    freq = {{}}
    for item in data:
        freq[item] = freq.get(item, 0) + 1
    return freq"""

        elif "reverse" in op_name:
            if no_modify:
                code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} of {ds_name} without modifying input.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return []
    result = []
    for i in range(len(data) - 1, -1, -1):
        result.append(data[i])
    return result"""
            else:
                code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} of {ds_name}.
    Hint: {op_hint}
    \"\"\"
    left, right = 0, len(data) - 1
    data = list(data)
    while left < right:
        data[left], data[right] = data[right], data[left]
        left += 1
        right -= 1
    return data"""

        elif "duplicate" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} in {ds_name}.
    Hint: {op_hint}
    \"\"\"
    seen = set()
    duplicates = []
    for item in data:
        if item in seen:
            if item not in duplicates:
                duplicates.append(item)
        else:
            seen.add(item)
    return duplicates"""

        elif "prefix" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} of {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return []
    prefix = []
    running = 0
    for item in data:
        running += item
        prefix.append(running)
    return prefix"""

        elif "median" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} of {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]"""

        elif "sort by frequency" in op_name:
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} in {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return []
    freq = {{}}
    for item in data:
        freq[item] = freq.get(item, 0) + 1
    return sorted(data, key=lambda x: freq[x], reverse=True)"""

        elif "merge intervals" in op_name:
            code = f"""def {fn_name}(intervals):
    \"\"\"
    {op_name.capitalize()} in {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged"""

        elif "k-th" in op_name:
            code = f"""import heapq

def {fn_name}(data, k):
    \"\"\"
    {op_name.capitalize()} from {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data or k < 1 or k > len(data):
        return None
    heap = []
    for item in data:
        heapq.heappush(heap, item)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0] if heap else None"""

        else:
            # Generic fallback
            code = f"""def {fn_name}(data):
    \"\"\"
    {op_name.capitalize()} from {ds_name}.
    Hint: {op_hint}
    \"\"\"
    if not data:
        return None
    result = None
    for i, item in enumerate(data):
        if result is None or item > result:
            result = item
    return result"""

        return code

    def _inject_bug(self, correct_code: str, bug_type: str, bug_desc: str,
                    seed: ProceduralSeed) -> str:
        """Inject a plausible bug into the correct solution."""
        lines = correct_code.split("\n")
        buggy = list(lines)

        if "off-by-one" in bug_type:
            for i, line in enumerate(buggy):
                if "range(len" in line and "- 1" in line:
                    buggy[i] = line.replace("- 1", "")
                    break
                elif "range(len" in line and "- 1" not in line:
                    buggy[i] = line.replace(")", " - 1)")
                    break

        elif "wrong base case" in bug_type:
            for i, line in enumerate(buggy):
                if "return []" in line or "return 0" in line or "return None" in line:
                    buggy[i] = line.replace("return []", "return [0]").replace(
                        "return 0", "return -1")
                    break

        elif "missing initialization" in bug_type:
            for i, line in enumerate(buggy):
                if "running = 0" in line or "total = 0" in line:
                    buggy[i] = "    # " + line.strip() + "  # BUG: missing initialization"
                    break

        elif "incorrect update" in bug_type:
            for i, line in enumerate(buggy):
                if "+= item" in line:
                    buggy[i] = line.replace("+= item", "+= 1")
                    break

        elif "wrong comparison" in bug_type:
            for i, line in enumerate(buggy):
                if " > result" in line:
                    buggy[i] = line.replace(" > result", " >= result")
                    break
                elif " < result" in line:
                    buggy[i] = line.replace(" < result", " <= result")
                    break

        elif "wrong return value" in bug_type:
            # Return from inside loop instead of after
            for i in range(len(buggy) - 1, -1, -1):
                if "return result" in buggy[i] or "return merged" in buggy[i]:
                    # Find the loop body and add an early return
                    for j in range(i - 1, -1, -1):
                        if "for " in buggy[j] or "while " in buggy[j]:
                            indent = len(buggy[j]) - len(buggy[j].lstrip())
                            buggy.insert(j + 2, " " * (indent + 4) + "return result  # BUG: early return")
                            break
                    break

        elif "mutation bug" in bug_type:
            for i, line in enumerate(buggy):
                if "data = list(data)" in line:
                    buggy[i] = "    # " + line.strip() + "  # BUG: removed copy"
                    break

        # Add bug comment at top of function
        for i, line in enumerate(buggy):
            if line.strip().startswith("def "):
                buggy.insert(i + 1, f'    # NOTE: This solution contains a {bug_type} ({bug_desc})')
                break

        return "\n".join(buggy)

    def format(self, puzzle: Dict[str, Any]) -> str:
        lines = [
            f"[Code Puzzle | Difficulty {puzzle['difficulty']}]",
            f"Data Structure: {puzzle['data_structure']}",
            f"Operation: {puzzle['operation']}",
            f"Constraints: {', '.join(puzzle['constraints'])}",
            f"",
            f"Problem:",
            puzzle["problem"],
            f"",
            f"--- Buggy Solution (contains: {puzzle['bug_type']}) ---",
            "```python",
            puzzle["buggy_solution"],
            "```",
            f"",
            f"--- Correct Solution ---",
            "```python",
            puzzle["correct_solution"],
            "```",
        ]
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 6. ProceduralDialogue
# -----------------------------------------------------------------------------

class ProceduralDialogue:
    """
    Generates multi-turn conversations between agents with distinct personalities,
    goals, and evolving knowledge states.
    """

    PERSONALITIES = [
        ("skeptic", "questions every claim", "demands evidence"),
        ("enthusiast", "embraces new ideas eagerly", "may overlook flaws"),
        ("pragmatist", "focuses on practical outcomes", "dismisses theory"),
        ("theorist", "prefers abstract frameworks", "may miss practical details"),
        ("mediator", "seeks common ground", "avoids conflict"),
        ("contrarian", "challenges consensus", "plays devil's advocate"),
        ("empiricist", "trusts only observed data", "distrusts speculation"),
        ("rationalist", "derives truth from first principles", "may ignore data"),
        ("synthesizer", "combines opposing views", "seeks unified models"),
        ("minimalist", "prefers simple explanations", "resists complexity"),
        ("maximalist", "embraces full complexity", "resists simplification"),
        ("historicist", "grounds claims in precedent", "may resist novelty"),
        ("futurist", "extrapolates trends forward", "may ignore constraints"),
        ("reductionist", "breaks problems into parts", "may miss emergent effects"),
        ("holist", "considers the whole system", "may miss local details"),
    ]

    KNOWLEDGE_DOMAINS = [
        "the behavior of {property} {entity} systems",
        "the principles of {domain}",
        "the relationship between {entity} and {property} phenomena",
        "the limits of {action} in {domain}",
        "the history of {property} discoveries in {domain}",
        "the mathematical structure of {property} {entity} models",
        "the practical applications of {domain} to {entity} problems",
        "the unresolved questions in {domain} regarding {property} {entity}",
    ]

    TURN_OPENERS = [
        "I believe that {claim}.",
        "Consider this: {claim}.",
        "The evidence suggests that {claim}.",
        "From my perspective, {claim}.",
        "It seems clear that {claim}.",
        "One must acknowledge that {claim}.",
        "The data indicates that {claim}.",
        "In my experience, {claim}.",
    ]

    RESPONSES = {
        "skeptic": [
            "That's a bold claim. What evidence supports {claim_fragment}?",
            "I'm not convinced. How do you account for {counter}?",
            "Interesting, but {claim_fragment} seems to contradict {counter}.",
            "Can you demonstrate {claim_fragment} empirically?",
        ],
        "enthusiast": [
            "Fascinating! And if {claim_fragment}, then perhaps {extension}!",
            "Yes! This aligns perfectly with {extension}.",
            "Exactly -- and it also implies {extension}.",
            "This opens up so many possibilities, especially regarding {extension}.",
        ],
        "pragmatist": [
            "Fine, but how does {claim_fragment} help us solve real problems?",
            "Theory is nice, but what's the practical application of {claim_fragment}?",
            "Let's focus: what does {claim_fragment} actually change in practice?",
            "I need concrete results, not just {claim_fragment}.",
        ],
        "theorist": [
            "This connects to a deeper principle: {extension}.",
            "From first principles, {claim_fragment} follows from {extension}.",
            "The abstract framework here is {extension}.",
            "We should formalize {claim_fragment} as {extension}.",
        ],
        "mediator": [
            "Both perspectives have merit. Perhaps {claim_fragment} and {counter} can coexist.",
            "Let's find common ground: {extension}.",
            "I think we agree more than we disagree -- {extension}.",
            "The synthesis here might be {extension}.",
        ],
        "contrarian": [
            "Actually, the opposite may be true: {counter}.",
            "I'd argue that {counter} undermines {claim_fragment}.",
            "What if {counter} is the real explanation?",
            "Devil's advocate: {counter}.",
        ],
        "empiricist": [
            "Show me the data. Without measurement, {claim_fragment} is speculation.",
            "Observed results suggest {counter}, not {claim_fragment}.",
            "I'll accept {claim_fragment} only when it's been replicated.",
            "The empirical record on {claim_fragment} is mixed at best.",
        ],
        "rationalist": [
            "Logically, {claim_fragment} must follow from {extension}.",
            "By deduction: if {extension}, then {claim_fragment} is necessarily true.",
            "The rational basis for {claim_fragment} is {extension}.",
            "Reason alone tells us {extension}.",
        ],
        "synthesizer": [
            "Combining {claim_fragment} with {counter} gives us {extension}.",
            "The unified view is {extension}.",
            "Both {claim_fragment} and {counter} are partial truths; {extension} is the whole.",
            "Integration of these views yields {extension}.",
        ],
        "minimalist": [
            "Simpler explanation: {extension}.",
            "We're overcomplicating this. {extension} suffices.",
            "Occam's razor: {extension} explains {claim_fragment} more parsimoniously.",
            "Strip away the complexity and you get {extension}.",
        ],
    }

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def _make_claim(self, seed: ProceduralSeed) -> str:
        s = seed
        entity = s.choice(ConceptGraph.ENTITIES)
        prop = s.choice(ConceptGraph.PROPERTIES)
        relation = s.choice(ConceptGraph.RELATIONS)
        entity2 = s.choice(ConceptGraph.ENTITIES)
        context = s.choice(ConceptGraph.CONTEXTS)
        return f"the {prop} {entity} {relation}s the {entity2} {context}"

    def _make_extension(self, seed: ProceduralSeed) -> str:
        s = seed
        domain = s.choice(ConceptGraph.DOMAINS)
        prop = s.choice(ConceptGraph.PROPERTIES)
        action = s.choice(ConceptGraph.ACTIONS)
        entity = s.choice(ConceptGraph.ENTITIES)
        return f"this implies a {prop} {action} mechanism in {domain} involving {entity}"

    def _make_counter(self, seed: ProceduralSeed) -> str:
        s = seed
        prop = s.choice(ConceptGraph.PROPERTIES)
        entity = s.choice(ConceptGraph.ENTITIES)
        relation = s.choice(ConceptGraph.RELATIONS)
        context = s.choice(ConceptGraph.CONTEXTS)
        return f"the {prop} {entity} actually {relation}s the system {context}"

    def generate(self, seed: ProceduralSeed, difficulty: int = 1) -> Dict[str, Any]:
        s = seed.fork("dialogue")
        num_turns = max(2, min(12, 2 + difficulty))
        num_agents = 2 if difficulty < 5 else min(4, 2 + difficulty // 4)

        # Pick agents
        agents = []
        personality_pool = list(self.PERSONALITIES)
        for i in range(num_agents):
            idx = s.next_int(0, len(personality_pool) - 1)
            p = personality_pool.pop(idx)
            name = f"Agent-{chr(65 + i)}"  # Agent-A, Agent-B, ...
            kd_template = s.choice(self.KNOWLEDGE_DOMAINS)
            knowledge = kd_template.format(
                property=s.choice(ConceptGraph.PROPERTIES),
                entity=s.choice(ConceptGraph.ENTITIES),
                domain=s.choice(ConceptGraph.DOMAINS),
                action=s.choice(ConceptGraph.ACTIONS),
            )
            agents.append({
                "name": name,
                "personality": p[0],
                "style": p[1],
                "bias": p[2],
                "knowledge": knowledge,
                "goal": self._make_claim(s.fork(f"goal{i}")),
            })

        # Generate conversation
        topic = self._make_claim(s.fork("topic"))
        turns = []
        current_claim = topic

        for t in range(num_turns):
            agent = agents[t % len(agents)]
            personality = agent["personality"]

            if t == 0:
                opener = s.choice(self.TURN_OPENERS)
                text = opener.format(claim=current_claim)
            else:
                response_pool = self.RESPONSES.get(personality, self.RESPONSES["skeptic"])
                template = s.choice(response_pool)
                claim_fragment = " ".join(current_claim.split()[:6]) + "..."
                counter = self._make_counter(s.fork(f"counter{t}"))
                extension = self._make_extension(s.fork(f"ext{t}"))
                text = template.format(
                    claim_fragment=claim_fragment,
                    counter=counter,
                    extension=extension,
                )
                # Evolve the claim
                if t % 3 == 0:
                    current_claim = self._make_claim(s.fork(f"evolve{t}"))

            turns.append({
                "agent": agent["name"],
                "personality": personality,
                "text": text,
            })

        # Resolution
        resolution_styles = [
            f"The conversation concludes with unresolved tension between {agents[0]['name']} and {agents[-1]['name']}.",
            f"{agents[0]['name']} and {agents[-1]['name']} reach a tentative agreement on {self._make_extension(s.fork('res'))}.",
            f"The discussion surfaces a deeper question: {self._make_claim(s.fork('deep'))}.",
            f"Both parties agree to revisit the topic with more data on {self._make_counter(s.fork('revisit'))}.",
        ]
        resolution = s.choice(resolution_styles)

        return {
            "type": "dialogue",
            "difficulty": difficulty,
            "topic": topic,
            "agents": agents,
            "turns": turns,
            "resolution": resolution,
        }

    def format(self, dialogue: Dict[str, Any]) -> str:
        lines = [
            f"[Dialogue | Difficulty {dialogue['difficulty']}]",
            f"Topic: {dialogue['topic']}",
            f"",
            f"Participants:",
        ]
        for agent in dialogue["agents"]:
            lines.append(f"  {agent['name']} [{agent['personality']}]: {agent['style']}; knows {agent['knowledge']}")

        lines.append("")
        lines.append("Conversation:")
        for turn in dialogue["turns"]:
            lines.append(f"  {turn['agent']} ({turn['personality']}): {turn['text']}")

        lines.append("")
        lines.append(f"Resolution: {dialogue['resolution']}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 7. ProceduralDataset
# -----------------------------------------------------------------------------

class ProceduralDataset:
    """
    Master generator. Combines all sub-generators.
    Infinite, seeded, difficulty-scaled training data for Avus.
    """

    SAMPLE_TYPES = ["reasoning", "scenario", "code_puzzle", "dialogue"]

    def __init__(self):
        self.graph = ConceptGraph()
        self.reasoning_gen = ProceduralReasoningChain(self.graph)
        self.scenario_gen = ProceduralScenario(self.graph)
        self.code_gen = ProceduralCodePuzzle(self.graph)
        self.dialogue_gen = ProceduralDialogue(self.graph)

    def get_difficulty_for_skill_level(self, skill_confidence: float) -> int:
        """
        Map skill confidence [0.0, 1.0] to difficulty [1, 10].
        Uses a sigmoid-like curve so early gains are fast, later gains are harder.
        """
        skill_confidence = max(0.0, min(1.0, skill_confidence))
        # Sigmoid mapping: difficulty ramps slowly at first, then accelerates
        x = skill_confidence * 10.0 - 5.0  # shift to [-5, 5]
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        difficulty = 1 + int(sigmoid * 9)
        return max(1, min(10, difficulty))

    def _generate_one(self, seed: ProceduralSeed, difficulty: int,
                       sample_type: Optional[str] = None) -> str:
        s = seed.fork("dispatch")
        if sample_type is None:
            sample_type = s.choice(self.SAMPLE_TYPES)

        if sample_type == "reasoning":
            data = self.reasoning_gen.generate(seed.fork("r"), difficulty)
            body = self.reasoning_gen.format(data)
        elif sample_type == "scenario":
            data = self.scenario_gen.generate(seed.fork("s"), difficulty)
            body = self.scenario_gen.format(data)
        elif sample_type == "code_puzzle":
            data = self.code_gen.generate(seed.fork("c"), difficulty)
            body = self.code_gen.format(data)
        elif sample_type == "dialogue":
            data = self.dialogue_gen.generate(seed.fork("d"), difficulty)
            body = self.dialogue_gen.format(data)
        else:
            body = f"[Unknown sample type: {sample_type}]"

        return f"{SOT}\n{body}\n{EOT}"

    def generate(self, n: int, difficulty: int = 1, seed: int = 42) -> List[str]:
        """
        Generate n samples at a given difficulty from a given seed.
        Same seed + difficulty always produces identical output.
        """
        root = ProceduralSeed(seed)
        samples = []
        for i in range(n):
            child = root.fork(f"sample_{i}")
            sample_type = self.SAMPLE_TYPES[i % len(self.SAMPLE_TYPES)]
            samples.append(self._generate_one(child, difficulty, sample_type))
        return samples

    def stream(self, difficulty: int = 1, seed: int = 42) -> Generator[str, None, None]:
        """
        Yield samples infinitely. Each sample is unique.
        Difficulty can be updated externally by wrapping this generator.
        """
        root = ProceduralSeed(seed)
        counter = 0
        while True:
            child = root.fork(f"stream_{counter}")
            sample_type = self.SAMPLE_TYPES[counter % len(self.SAMPLE_TYPES)]
            yield self._generate_one(child, difficulty, sample_type)
            counter += 1

    def stream_with_curriculum(self, seed: int = 42,
                                skill_confidence_fn=None) -> Generator[str, None, None]:
        """
        Stream samples where difficulty is dynamically determined by a skill confidence
        function. skill_confidence_fn() should return a float in [0, 1].
        If None, difficulty stays at 1.
        """
        root = ProceduralSeed(seed)
        counter = 0
        while True:
            if skill_confidence_fn is not None:
                confidence = skill_confidence_fn()
                difficulty = self.get_difficulty_for_skill_level(confidence)
            else:
                difficulty = 1
            child = root.fork(f"curriculum_{counter}")
            sample_type = self.SAMPLE_TYPES[counter % len(self.SAMPLE_TYPES)]
            yield self._generate_one(child, difficulty, sample_type)
            counter += 1


# -----------------------------------------------------------------------------
# 8. Main demonstration block
# -----------------------------------------------------------------------------

def _separator(title: str = "") -> str:
    line = "=" * 72
    if title:
        return f"\n{line}\n  {title}\n{line}"
    return f"\n{line}"


def main():
    dataset = ProceduralDataset()

    # -- Demo 1: 5 samples from 3 different seeds ------------------------------
    print(_separator("DEMO 1: 5 samples from 3 different seeds"))
    for seed_val in [1001, 2002, 3003]:
        print(f"\n{'-'*60}")
        print(f"  Seed: {seed_val}")
        print(f"{'-'*60}")
        samples = dataset.generate(n=5, difficulty=3, seed=seed_val)
        for i, sample in enumerate(samples):
            print(f"\n[Sample {i+1}]")
            print(sample)

    # -- Demo 2: Reproducibility test -----------------------------------------
    print(_separator("DEMO 2: Reproducibility -- same seed always produces same output"))
    seed_a = 9999
    run1 = dataset.generate(n=2, difficulty=2, seed=seed_a)
    run2 = dataset.generate(n=2, difficulty=2, seed=seed_a)
    for i in range(len(run1)):
        match = run1[i] == run2[i]
        print(f"\nSample {i+1} identical across runs: {match}")
        if not match:
            print("  WARNING: Reproducibility failure!")
        else:
            print(run1[i])

    # -- Demo 3: Difficulty scaling --------------------------------------------
    print(_separator("DEMO 3: Difficulty scaling (difficulty 1, 5, 10)"))
    for diff in [1, 5, 10]:
        print(f"\n{'-'*60}")
        print(f"  Difficulty: {diff}")
        print(f"{'-'*60}")
        samples = dataset.generate(n=1, difficulty=diff, seed=42)
        print(samples[0])

    # -- Demo 4: Skill-level to difficulty mapping -----------------------------
    print(_separator("DEMO 4: Skill confidence -> difficulty mapping"))
    print(f"{'Skill Confidence':>20} | {'Difficulty':>10}")
    print(f"{'-'*20}-+-{'-'*10}")
    for conf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        diff = dataset.get_difficulty_for_skill_level(conf)
        bar = "#" * diff
        print(f"{conf:>20.1f} | {diff:>3}  {bar}")

    # -- Demo 5: Infinite stream (first 3 items) -------------------------------
    print(_separator("DEMO 5: Infinite stream -- first 3 items from stream()"))
    gen = dataset.stream(difficulty=4, seed=777)
    for i in range(3):
        print(f"\n[Stream item {i+1}]")
        print(next(gen))

    print(_separator("END OF DEMONSTRATION"))
    print("\nProceduralDataset is ready. Call dataset.generate(n, difficulty, seed)")
    print("or dataset.stream(difficulty, seed) for infinite unique training samples.")


if __name__ == "__main__":
    main()
