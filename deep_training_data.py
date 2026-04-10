"""
deep_training_data.py
Synthetic deep training data generator for Avus AI.
All data is 100% synthetic - generated from templates, logic, and randomization.
"""

import random
import json
import itertools
import math
from typing import List

SOT = "<|startoftext|>"
EOT = "<|endoftext|>"

def wrap(text: str) -> str:
    return f"{SOT}\n{text.strip()}\n{EOT}"


# ---------------------------------------------------------------------------
# 1. ReasoningGenerator
# ---------------------------------------------------------------------------

class ReasoningGenerator:
    """
    Generates multi-step logic chains, syllogisms, contradictions, cause-effect.
    Difficulty 1-5:
      1 - simple if-then
      2 - two-step chains
      3 - syllogisms
      4 - cause-effect with multiple branches
      5 - multi-step contradictions requiring resolution
    """

    ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    PROPERTIES = ["tall", "fast", "smart", "honest", "brave", "kind", "strong", "clever"]
    OBJECTS = ["apple", "book", "key", "coin", "map", "stone", "ring", "lamp"]
    ACTIONS = ["runs", "jumps", "reads", "writes", "builds", "breaks", "finds", "loses"]
    CAUSES = [
        "the temperature dropped", "the power went out", "the road was blocked",
        "the supply ran out", "the signal was lost", "the door was locked",
        "the timer expired", "the connection failed"
    ]
    EFFECTS = [
        "the project was delayed", "the team had to adapt", "a new plan was needed",
        "resources were reallocated", "the deadline was missed", "an alert was triggered",
        "the system restarted", "a backup was activated"
    ]

    def _level1(self) -> str:
        a = random.choice(self.ENTITIES)
        prop = random.choice(self.PROPERTIES)
        obj = random.choice(self.OBJECTS)
        return (
            f"Premise: If {a} is {prop}, then {a} can find the {obj}.\n"
            f"Fact: {a} is {prop}.\n"
            f"Question: Can {a} find the {obj}?\n"
            f"Answer: Yes. By modus ponens, since {a} is {prop}, {a} can find the {obj}."
        )

    def _level2(self) -> str:
        a, b = random.sample(self.ENTITIES, 2)
        p1, p2 = random.sample(self.PROPERTIES, 2)
        return (
            f"Premise 1: All {p1} people are {p2}.\n"
            f"Premise 2: {a} is {p1}.\n"
            f"Premise 3: {b} is not {p1}.\n"
            f"Question: Is {a} {p2}? Is {b} {p2}?\n"
            f"Answer: {a} is {p2} (follows from premises 1 and 2). "
            f"We cannot conclude {b} is {p2} from these premises alone."
        )

    def _level3(self) -> str:
        a, b, c = random.sample(self.ENTITIES, 3)
        p1, p2, p3 = random.sample(self.PROPERTIES, 3)
        return (
            f"Syllogism:\n"
            f"  Major premise: All {p1} beings are {p2}.\n"
            f"  Minor premise: All {p2} beings are {p3}.\n"
            f"  Fact: {a} is {p1}. {b} is {p2} but not {p1}. {c} is neither {p1} nor {p2}.\n"
            f"Question: Which of {a}, {b}, {c} must be {p3}?\n"
            f"Answer: {a} must be {p3} (chain: {p1}→{p2}→{p3}). "
            f"{b} must also be {p3} (direct: {p2}→{p3}). "
            f"{c} cannot be determined from these premises."
        )

    def _level4(self) -> str:
        cause = random.choice(self.CAUSES)
        eff1, eff2 = random.sample(self.EFFECTS, 2)
        branch_cond = random.choice(self.PROPERTIES)
        a = random.choice(self.ENTITIES)
        return (
            f"Causal chain analysis:\n"
            f"  Event: {cause}.\n"
            f"  Direct effect: {eff1}.\n"
            f"  Branch: If the team is {branch_cond}, then {eff2} is also triggered.\n"
            f"  Fact: {a}'s team is {branch_cond}.\n"
            f"Question: What are all the effects?\n"
            f"Answer: Primary effect is '{eff1}'. "
            f"Because {a}'s team is {branch_cond}, the secondary effect '{eff2}' is also triggered. "
            f"Total effects: [{eff1}] and [{eff2}]."
        )

    def _level5(self) -> str:
        a, b = random.sample(self.ENTITIES, 2)
        p1, p2 = random.sample(self.PROPERTIES, 2)
        obj = random.choice(self.OBJECTS)
        return (
            f"Contradiction resolution:\n"
            f"  Statement 1: {a} is always {p1}.\n"
            f"  Statement 2: No one who is {p1} can own a {obj}.\n"
            f"  Statement 3: {a} owns a {obj}.\n"
            f"  Statement 4: {b} claims {a} is not {p1}.\n"
            f"Question: Is there a contradiction? How can it be resolved?\n"
            f"Answer: Yes, statements 1, 2, and 3 form a contradiction. "
            f"If {a} is always {p1} (S1) and no {p1} person can own a {obj} (S2), "
            f"then {a} cannot own a {obj} — but S3 says they do. "
            f"Resolution options: (a) S1 is false — {a} is not always {p1}; "
            f"(b) S2 has exceptions; (c) S3 is false. "
            f"{b}'s claim (S4) supports resolution (a): {a} is not {p1}, making S3 consistent with S2."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Reasoning | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# 2. LanguageGenerator
# ---------------------------------------------------------------------------

class LanguageGenerator:
    """
    Generates nested sentences, pronoun resolution, long-range dependencies,
    ambiguous references.
    Difficulty 1-5:
      1 - simple pronoun resolution
      2 - nested relative clauses
      3 - long-range dependency
      4 - ambiguous pronoun with context clues
      5 - multi-layer ambiguity requiring full discourse analysis
    """

    NAMES = ["Maria", "James", "Priya", "Luca", "Yuki", "Omar", "Sofia", "Chen"]
    VERBS = ["told", "helped", "warned", "praised", "criticized", "followed", "ignored", "trusted"]
    NOUNS = ["report", "letter", "package", "message", "document", "gift", "note", "plan"]
    ADJECTIVES = ["important", "urgent", "confusing", "detailed", "brief", "encrypted", "missing", "final"]

    def _level1(self) -> str:
        a, b = random.sample(self.NAMES, 2)
        v = random.choice(self.VERBS)
        noun = random.choice(self.NOUNS)
        return (
            f"Sentence: {a} {v} {b} about the {noun}. She was grateful.\n"
            f"Question: Who was grateful?\n"
            f"Answer: '{b}' was grateful. 'She' refers to {b}, "
            f"the recipient of {a}'s action."
        )

    def _level2(self) -> str:
        a, b, c = random.sample(self.NAMES, 3)
        v1, v2 = random.sample(self.VERBS, 2)
        noun = random.choice(self.NOUNS)
        adj = random.choice(self.ADJECTIVES)
        return (
            f"Sentence: {a}, who {v1} {b} last week, {v2} {c} about the {adj} {noun} "
            f"that {b} had originally written.\n"
            f"Question: Who wrote the {noun}?\n"
            f"Answer: {b} wrote the {noun}. The relative clause 'that {b} had originally written' "
            f"modifies '{noun}' and attributes authorship to {b}."
        )

    def _level3(self) -> str:
        names = random.sample(self.NAMES, 4)
        a, b, c, d = names
        v1, v2, v3 = random.sample(self.VERBS, 3)
        noun = random.choice(self.NOUNS)
        return (
            f"Passage: {a} {v1} {b}. Later, {b} met {c}, who had already spoken with {d}. "
            f"The {noun} that {d} had prepared was eventually given to the person {a} first contacted.\n"
            f"Question: Who received the {noun}?\n"
            f"Answer: {b} received the {noun}. '{a} first contacted' refers to {b} "
            f"(the first person {a} interacted with), so the {noun} went to {b}."
        )

    def _level4(self) -> str:
        a, b = random.sample(self.NAMES, 2)
        v = random.choice(self.VERBS)
        noun = random.choice(self.NOUNS)
        adj = random.choice(self.ADJECTIVES)
        return (
            f"Passage: {a} and {b} worked on the {adj} {noun} together. "
            f"When it was finished, {a} {v} the manager. "
            f"They said it was the best work they had ever seen.\n"
            f"Question: Who said it was the best work?\n"
            f"Answer: 'They' is ambiguous — it could refer to the manager (responding to {a}) "
            f"or to {a} and {b} collectively. "
            f"Context clue: {a} {v} the manager, so the manager's reaction follows naturally. "
            f"Most likely interpretation: the manager said it was the best work."
        )

    def _level5(self) -> str:
        a, b, c = random.sample(self.NAMES, 3)
        v1, v2 = random.sample(self.VERBS, 2)
        n1, n2 = random.sample(self.NOUNS, 2)
        adj = random.choice(self.ADJECTIVES)
        return (
            f"Passage: {a} gave {b} the {adj} {n1} that {c} had requested, "
            f"but {b} passed it to someone who needed it more. "
            f"Later, {a} asked {c} whether they had received it, "
            f"and {c} said that the person {b} gave it to had already shared it with them.\n"
            f"Question: Did {c} end up with the {n1}?\n"
            f"Answer: Yes. Trace: {a}→{b}→unknown person→{c}. "
            f"'{c} said the person {b} gave it to had shared it with them' confirms {c} received it, "
            f"though indirectly. The chain of possession: {a} gave it to {b}, "
            f"{b} gave it to a third party, who then shared it with {c}."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Language | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# 3. CodeGenerator
# ---------------------------------------------------------------------------

class CodeGenerator:
    """
    Generates Python code with intentional bugs and correct fixes.
    Difficulty 1-5:
      1 - syntax errors
      2 - name/type errors
      3 - off-by-one / index errors
      4 - logic errors in simple algorithms
      5 - logic bugs in complex algorithms
    """

    def _level1(self) -> str:
        var = random.choice(["x", "y", "count", "total", "result"])
        val = random.randint(1, 100)
        return (
            f"Task: Fix the syntax error in the following Python code.\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"{var} = {val}\n"
            f"if {var} > 10\n"
            f"    print('big')\n"
            f"```\n\n"
            f"Bug: Missing colon after the `if` condition.\n\n"
            f"Fixed code:\n"
            f"```python\n"
            f"{var} = {val}\n"
            f"if {var} > 10:\n"
            f"    print('big')\n"
            f"```"
        )

    def _level2(self) -> str:
        items = [random.randint(1, 9) for _ in range(4)]
        return (
            f"Task: Fix the NameError in the following code.\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"numbers = {items}\n"
            f"total = 0\n"
            f"for num in numbers:\n"
            f"    totl += num\n"
            f"print(total)\n"
            f"```\n\n"
            f"Bug: `totl` is a typo — the variable is named `total`.\n\n"
            f"Fixed code:\n"
            f"```python\n"
            f"numbers = {items}\n"
            f"total = 0\n"
            f"for num in numbers:\n"
            f"    total += num\n"
            f"print(total)\n"
            f"```"
        )

    def _level3(self) -> str:
        size = random.randint(4, 7)
        items = list(range(1, size + 1))
        return (
            f"Task: Fix the off-by-one error.\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"data = {items}\n"
            f"for i in range(1, len(data) + 1):\n"
            f"    print(data[i])\n"
            f"```\n\n"
            f"Bug: `range(1, len(data) + 1)` causes an IndexError on the last iteration "
            f"because valid indices are 0 to {size - 1}.\n\n"
            f"Fixed code:\n"
            f"```python\n"
            f"data = {items}\n"
            f"for i in range(len(data)):\n"
            f"    print(data[i])\n"
            f"```"
        )

    def _level4(self) -> str:
        items = sorted(random.sample(range(1, 20), 6))
        target = random.choice(items)
        return (
            f"Task: Fix the logic error in this linear search function.\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"def find(lst, target):\n"
            f"    for i, val in enumerate(lst):\n"
            f"        if val == target:\n"
            f"            return i\n"
            f"    return i  # bug: returns last index instead of -1\n"
            f"\n"
            f"print(find({items}, {target}))\n"
            f"```\n\n"
            f"Bug: When the target is not found, `return i` returns the last loop index "
            f"instead of a sentinel value like -1.\n\n"
            f"Fixed code:\n"
            f"```python\n"
            f"def find(lst, target):\n"
            f"    for i, val in enumerate(lst):\n"
            f"        if val == target:\n"
            f"            return i\n"
            f"    return -1\n"
            f"\n"
            f"print(find({items}, {target}))\n"
            f"```"
        )

    def _level5(self) -> str:
        return (
            f"Task: Fix the logic bug in this binary search implementation.\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"def binary_search(arr, target):\n"
            f"    low, high = 0, len(arr)\n"
            f"    while low < high:\n"
            f"        mid = (low + high) // 2\n"
            f"        if arr[mid] == target:\n"
            f"            return mid\n"
            f"        elif arr[mid] < target:\n"
            f"            low = mid\n"
            f"        else:\n"
            f"            high = mid\n"
            f"    return -1\n"
            f"```\n\n"
            f"Bugs:\n"
            f"1. `high = len(arr)` should be `len(arr) - 1` (last valid index).\n"
            f"2. `low = mid` causes infinite loop when `arr[mid] < target`; "
            f"should be `low = mid + 1`.\n"
            f"3. `high = mid` should be `high = mid - 1` to avoid re-checking mid.\n\n"
            f"Fixed code:\n"
            f"```python\n"
            f"def binary_search(arr, target):\n"
            f"    low, high = 0, len(arr) - 1\n"
            f"    while low <= high:\n"
            f"        mid = (low + high) // 2\n"
            f"        if arr[mid] == target:\n"
            f"            return mid\n"
            f"        elif arr[mid] < target:\n"
            f"            low = mid + 1\n"
            f"        else:\n"
            f"            high = mid - 1\n"
            f"    return -1\n"
            f"```"
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Code | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# 4. PlanningGenerator
# ---------------------------------------------------------------------------

class PlanningGenerator:
    """
    Generates goal decomposition, ordered steps, failure recovery.
    Difficulty 1-5:
      1 - 2-step goal, 1 failure scenario
      2 - 3-step goal, 1 failure scenario
      3 - 4-step goal, 2 failure scenarios
      4 - 5-step goal with dependencies, 2 failures
      5 - 6-step goal with dependencies, conditional branches, 3 failures
    """

    GOALS = [
        "deploy a web application",
        "organize a team meeting",
        "publish a research paper",
        "launch a new product",
        "migrate a database",
        "train a machine learning model",
        "set up a CI/CD pipeline",
        "onboard a new employee",
        "conduct a security audit",
        "build a mobile app prototype",
    ]

    STEP_TEMPLATES = [
        "Define requirements and scope",
        "Gather necessary resources",
        "Set up the environment",
        "Implement the core functionality",
        "Write and run tests",
        "Review and get approval",
        "Deploy or deliver the output",
        "Monitor and collect feedback",
        "Document the process",
        "Notify all stakeholders",
    ]

    FAILURE_TEMPLATES = [
        ("resources are unavailable", "identify alternative resources or adjust scope"),
        ("a dependency fails", "roll back to the previous stable state and debug"),
        ("approval is denied", "revise the plan based on feedback and resubmit"),
        ("tests reveal critical bugs", "pause deployment, fix bugs, and re-run tests"),
        ("the environment is misconfigured", "restore from backup configuration and retry"),
        ("a key team member is unavailable", "redistribute tasks among remaining members"),
        ("the deadline is moved up", "prioritize critical path tasks and defer non-essentials"),
    ]

    def _make_plan(self, goal: str, num_steps: int, num_failures: int) -> str:
        steps = random.sample(self.STEP_TEMPLATES, num_steps)
        failures = random.sample(self.FAILURE_TEMPLATES, min(num_failures, len(self.FAILURE_TEMPLATES)))

        lines = [f"Goal: {goal}\n", "Steps:"]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")

        lines.append("\nFailure Recovery:")
        for i, (fail, recovery) in enumerate(failures, 1):
            step_idx = random.randint(1, num_steps)
            lines.append(f"  Scenario {i}: If step {step_idx} fails because {fail},")
            lines.append(f"    → Recovery: {recovery}.")

        return "\n".join(lines)

    def _level1(self) -> str:
        return self._make_plan(random.choice(self.GOALS), 2, 1)

    def _level2(self) -> str:
        return self._make_plan(random.choice(self.GOALS), 3, 1)

    def _level3(self) -> str:
        return self._make_plan(random.choice(self.GOALS), 4, 2)

    def _level4(self) -> str:
        goal = random.choice(self.GOALS)
        num_steps = 5
        steps = random.sample(self.STEP_TEMPLATES, num_steps)
        failures = random.sample(self.FAILURE_TEMPLATES, 2)

        lines = [f"Goal: {goal}\n", "Steps (with dependencies):"]
        for i, step in enumerate(steps, 1):
            dep = f" [depends on step {i-1}]" if i > 1 else ""
            lines.append(f"  {i}. {step}{dep}")

        lines.append("\nFailure Recovery:")
        for i, (fail, recovery) in enumerate(failures, 1):
            step_idx = random.randint(2, num_steps)
            lines.append(f"  Scenario {i}: If step {step_idx} fails because {fail},")
            lines.append(f"    → Recovery: {recovery}.")
            lines.append(f"    → Downstream impact: steps {step_idx+1} to {num_steps} must be re-evaluated.")

        return "\n".join(lines)

    def _level5(self) -> str:
        goal = random.choice(self.GOALS)
        num_steps = 6
        steps = random.sample(self.STEP_TEMPLATES, num_steps)
        failures = random.sample(self.FAILURE_TEMPLATES, 3)
        branch_step = random.randint(2, 4)

        lines = [f"Goal: {goal}\n", "Steps (with dependencies and conditional branches):"]
        for i, step in enumerate(steps, 1):
            dep = f" [depends on step {i-1}]" if i > 1 else ""
            if i == branch_step:
                lines.append(f"  {i}. {step}{dep}")
                lines.append(f"     Branch A: If condition X is met → proceed to step {i+1}")
                lines.append(f"     Branch B: If condition X is not met → skip to step {i+2}")
            else:
                lines.append(f"  {i}. {step}{dep}")

        lines.append("\nFailure Recovery:")
        for i, (fail, recovery) in enumerate(failures, 1):
            step_idx = random.randint(1, num_steps)
            lines.append(f"  Scenario {i}: If step {step_idx} fails because {fail},")
            lines.append(f"    → Recovery: {recovery}.")
            lines.append(f"    → Re-enter plan at step {max(1, step_idx - 1)} after recovery.")

        return "\n".join(lines)

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Planning | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# 5. MemoryRecallGenerator
# ---------------------------------------------------------------------------

class MemoryRecallGenerator:
    """
    Generates long context followed by questions about early details.
    Difficulty scales with context length and question specificity.
      1 - short context (3 facts), simple recall
      2 - medium context (6 facts), direct recall
      3 - medium context (9 facts), indirect recall
      4 - long context (12 facts), multi-hop recall
      5 - very long context (16 facts), specific detail + inference
    """

    FACT_TEMPLATES = [
        ("{name} was born in {city}.", "{name}", "city", "{city}"),
        ("{name} works as a {job}.", "{name}", "job", "{job}"),
        ("{name} owns a {color} {vehicle}.", "{name}", "vehicle color", "{color}"),
        ("{name} has {n} siblings.", "{name}", "number of siblings", "{n}"),
        ("{name} studied at {university}.", "{name}", "university", "{university}"),
        ("{name} speaks {language} fluently.", "{name}", "language", "{language}"),
        ("{name}'s favorite food is {food}.", "{name}", "favorite food", "{food}"),
        ("{name} lives on {street} Street.", "{name}", "street", "{street}"),
        ("{name} has a pet {animal}.", "{name}", "pet", "{animal}"),
        ("{name} was hired in {year}.", "{name}", "year hired", "{year}"),
        ("{name} drives to work every {day}.", "{name}", "commute day", "{day}"),
        ("{name} is allergic to {substance}.", "{name}", "allergy", "{substance}"),
        ("{name} volunteers at a local {place}.", "{name}", "volunteer location", "{place}"),
        ("{name} completed a marathon in {time} hours.", "{name}", "marathon time", "{time}"),
        ("{name} reads {n2} books per month.", "{name}", "books per month", "{n2}"),
        ("{name} recently visited {country}.", "{name}", "recent travel destination", "{country}"),
    ]

    NAMES = ["Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Drew", "Quinn"]
    CITIES = ["Berlin", "Tokyo", "Lagos", "Sydney", "Montreal", "Nairobi", "Oslo", "Lima"]
    JOBS = ["engineer", "teacher", "doctor", "designer", "analyst", "chef", "pilot", "writer"]
    COLORS = ["red", "blue", "silver", "black", "green", "white", "orange", "yellow"]
    VEHICLES = ["car", "motorcycle", "truck", "van", "scooter", "bicycle"]
    UNIVERSITIES = ["Northgate", "Westfield", "Eastbrook", "Southmoor", "Lakeside", "Hillcrest"]
    LANGUAGES = ["Spanish", "Mandarin", "French", "Arabic", "Swahili", "Portuguese", "Hindi"]
    FOODS = ["sushi", "tacos", "pasta", "curry", "dumplings", "falafel", "ramen", "jollof rice"]
    STREETS = ["Maple", "Oak", "Cedar", "Pine", "Elm", "Birch", "Walnut", "Ash"]
    ANIMALS = ["cat", "dog", "parrot", "rabbit", "hamster", "turtle", "fish"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    SUBSTANCES = ["peanuts", "gluten", "shellfish", "dairy", "pollen", "dust"]
    PLACES = ["library", "shelter", "hospital", "school", "community center", "food bank"]
    COUNTRIES = ["Brazil", "Japan", "Kenya", "Iceland", "Vietnam", "Peru", "Greece"]

    def _fill(self, template: str, name: str) -> dict:
        return {
            "{name}": name,
            "{city}": random.choice(self.CITIES),
            "{job}": random.choice(self.JOBS),
            "{color}": random.choice(self.COLORS),
            "{vehicle}": random.choice(self.VEHICLES),
            "{n}": str(random.randint(0, 5)),
            "{n2}": str(random.randint(1, 8)),
            "{university}": random.choice(self.UNIVERSITIES),
            "{language}": random.choice(self.LANGUAGES),
            "{food}": random.choice(self.FOODS),
            "{street}": random.choice(self.STREETS),
            "{animal}": random.choice(self.ANIMALS),
            "{year}": str(random.randint(2005, 2023)),
            "{day}": random.choice(self.DAYS),
            "{substance}": random.choice(self.SUBSTANCES),
            "{place}": random.choice(self.PLACES),
            "{time}": str(round(random.uniform(3.5, 6.5), 1)),
            "{country}": random.choice(self.COUNTRIES),
        }

    def _apply(self, template: str, fills: dict) -> str:
        for k, v in fills.items():
            template = template.replace(k, v)
        return template

    def _build(self, num_facts: int, num_questions: int) -> str:
        name = random.choice(self.NAMES)
        selected = random.sample(self.FACT_TEMPLATES, num_facts)
        fills = self._fill("", name)

        facts = [self._apply(t[0], fills) for t in selected]
        context = f"Context about {name}:\n" + "\n".join(f"  - {f}" for f in facts)

        # Pick questions from early facts (first half)
        early = selected[:max(1, num_facts // 2)]
        q_facts = random.sample(early, min(num_questions, len(early)))

        qa_lines = ["\nQuestions and Answers:"]
        for t in q_facts:
            question = f"What is {name}'s {self._apply(t[2], fills)}?"
            answer = self._apply(t[3], fills)
            qa_lines.append(f"  Q: {question}")
            qa_lines.append(f"  A: {answer}")

        return context + "\n" + "\n".join(qa_lines)

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        params = {1: (3, 1), 2: (6, 2), 3: (9, 2), 4: (12, 3), 5: (16, 4)}
        num_facts, num_q = params[max(1, min(5, difficulty))]
        return [wrap(f"[MemoryRecall | Difficulty {difficulty}]\n\n" + self._build(num_facts, num_q))
                for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# 6. VisionDescriptionGenerator
# ---------------------------------------------------------------------------

class VisionDescriptionGenerator:
    """
    Generates spatial scene descriptions with relationships, occlusion, lighting.
    Difficulty 1-5:
      1 - 2 objects, simple spatial relation
      2 - 3 objects, basic occlusion
      3 - 4 objects, lighting + shadow
      4 - 5 objects, multiple occlusions + perspective
      5 - 6+ objects, complex lighting, shadows, reflections, depth
    """

    OBJECTS = ["chair", "table", "lamp", "vase", "book", "cup", "plant", "mirror",
               "clock", "painting", "box", "candle", "window", "door", "rug", "shelf"]
    COLORS = ["red", "blue", "wooden", "white", "black", "glass", "metallic", "green"]
    POSITIONS = ["to the left of", "to the right of", "in front of", "behind",
                 "on top of", "beneath", "next to", "diagonally across from"]
    LIGHT_SOURCES = ["a ceiling light", "a window", "a desk lamp", "sunlight", "candlelight"]
    LIGHT_DIRS = ["from the left", "from above", "from the right", "from behind", "from below"]
    SHADOW_DIRS = ["to the right", "to the left", "behind it", "in front of it"]

    def _obj(self) -> str:
        return f"{random.choice(self.COLORS)} {random.choice(self.OBJECTS)}"

    def _level1(self) -> str:
        o1, o2 = self._obj(), self._obj()
        pos = random.choice(self.POSITIONS)
        return (
            f"Scene description:\n"
            f"  A {o1} is {pos} a {o2}.\n\n"
            f"Spatial query: What is the relationship between the two objects?\n"
            f"Answer: The {o1} is {pos} the {o2}."
        )

    def _level2(self) -> str:
        o1, o2, o3 = self._obj(), self._obj(), self._obj()
        pos1 = random.choice(self.POSITIONS)
        return (
            f"Scene description:\n"
            f"  A {o1} is {pos1} a {o2}.\n"
            f"  A {o3} is partially hidden behind the {o2.split()[-1]}.\n\n"
            f"Occlusion query: Which object is occluded and by what?\n"
            f"Answer: The {o3} is partially occluded by the {o2}. "
            f"Only part of the {o3.split()[-1]} is visible from the viewer's perspective."
        )

    def _level3(self) -> str:
        objs = [self._obj() for _ in range(4)]
        light = random.choice(self.LIGHT_SOURCES)
        light_dir = random.choice(self.LIGHT_DIRS)
        shadow_dir = random.choice(self.SHADOW_DIRS)
        pos1 = random.choice(self.POSITIONS)
        pos2 = random.choice(self.POSITIONS)
        return (
            f"Scene description:\n"
            f"  A {objs[0]} is {pos1} a {objs[1]}.\n"
            f"  A {objs[2]} is {pos2} the {objs[1].split()[-1]}.\n"
            f"  A {objs[3]} sits in the corner.\n"
            f"  Lighting: {light} shines {light_dir}, casting shadows {shadow_dir}.\n\n"
            f"Lighting query: Which object likely casts the longest shadow?\n"
            f"Answer: The {objs[0]} — it is closest to the light source direction "
            f"and its shadow extends {shadow_dir}. "
            f"The {objs[3]} in the corner receives indirect light."
        )

    def _level4(self) -> str:
        objs = [self._obj() for _ in range(5)]
        light = random.choice(self.LIGHT_SOURCES)
        light_dir = random.choice(self.LIGHT_DIRS)
        return (
            f"Scene description:\n"
            f"  Foreground: A {objs[0]} and a {objs[1]} are side by side.\n"
            f"  Midground: A {objs[2]} is partially behind the {objs[0].split()[-1]}, "
            f"occluding a {objs[3]} further back.\n"
            f"  Background: A {objs[4]} is barely visible through the gap between "
            f"the {objs[0].split()[-1]} and {objs[1].split()[-1]}.\n"
            f"  Lighting: {light} {light_dir}.\n\n"
            f"Depth query: List objects from nearest to farthest.\n"
            f"Answer: 1. {objs[0]} and {objs[1]} (foreground, same depth plane). "
            f"2. {objs[2]} (midground, partially occluded). "
            f"3. {objs[3]} (behind {objs[2]}). "
            f"4. {objs[4]} (background, barely visible)."
        )

    def _level5(self) -> str:
        objs = [self._obj() for _ in range(6)]
        light1, light2 = random.sample(self.LIGHT_SOURCES, 2)
        dir1, dir2 = random.sample(self.LIGHT_DIRS, 2)
        return (
            f"Scene description:\n"
            f"  A {objs[0]} sits at the center of the room.\n"
            f"  A {objs[1]} is to its left, partially occluded by a {objs[2]}.\n"
            f"  A {objs[3]} is reflected in a nearby mirror, appearing to be "
            f"to the right even though it is actually to the left.\n"
            f"  A {objs[4]} hangs above, casting a circular shadow on the {objs[0].split()[-1]}.\n"
            f"  A {objs[5]} is in the far background, barely distinguishable.\n"
            f"  Dual lighting: {light1} {dir1} and {light2} {dir2}, "
            f"creating overlapping shadow regions.\n\n"
            f"Complex query: Where is the {objs[3].split()[-1]} actually located, "
            f"and what creates the illusion?\n"
            f"Answer: The {objs[3]} is actually to the LEFT of center. "
            f"The mirror reflection creates the illusion that it is to the right. "
            f"This is a mirror-reversal effect — the reflected image appears laterally inverted. "
            f"The dual lighting creates two shadow sets, making depth estimation harder."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Vision | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# FinancialReasoningGenerator
# ---------------------------------------------------------------------------

class FinancialReasoningGenerator:
    """Financial reasoning: budgets, percentages, compound interest, affordability."""

    ITEMS = ["laptop", "phone", "car", "vacation", "course", "monitor", "desk", "camera"]
    CATEGORIES = ["rent", "food", "transport", "entertainment", "savings", "utilities"]

    def _level1(self) -> str:
        item = random.choice(self.ITEMS)
        price = random.randint(100, 2000)
        budget = random.randint(price + 50, price + 500)
        return (
            f"Budget: ${budget}. The {item} costs ${price}.\n"
            f"Question: Can I afford the {item}?\n"
            f"Answer: Yes. ${budget} - ${price} = ${budget - price} remaining after purchase."
        )

    def _level2(self) -> str:
        income = random.randint(2000, 6000)
        cats = random.sample(self.CATEGORIES, 4)
        amounts = [random.randint(200, 800) for _ in cats]
        total = sum(amounts)
        remaining = income - total
        breakdown = ", ".join(f"{c}: ${a}" for c, a in zip(cats, amounts))
        return (
            f"Monthly income: ${income}.\n"
            f"Expenses: {breakdown}.\n"
            f"Question: How much is left for savings?\n"
            f"Answer: Total expenses = ${total}. Remaining = ${income} - ${total} = ${remaining}."
        )

    def _level3(self) -> str:
        principal = random.randint(1000, 10000)
        rate = round(random.uniform(2, 8), 1)
        years = random.randint(2, 10)
        simple = round(principal * rate / 100 * years, 2)
        total = principal + simple
        return (
            f"Investment: ${principal} at {rate}% simple interest for {years} years.\n"
            f"Question: What is the total value?\n"
            f"Answer: Interest = ${principal} × {rate}% × {years} = ${simple}. "
            f"Total = ${principal} + ${simple} = ${total}."
        )

    def _level4(self) -> str:
        principal = random.randint(1000, 5000)
        rate = round(random.uniform(3, 10), 1)
        years = random.randint(3, 10)
        compound = round(principal * (1 + rate/100) ** years, 2)
        gain = round(compound - principal, 2)
        return (
            f"Investment: ${principal} at {rate}% compound interest for {years} years.\n"
            f"Question: What is the final value and total gain?\n"
            f"Answer: Final = ${principal} × (1 + {rate/100})^{years} = ${compound}. "
            f"Gain = ${compound} - ${principal} = ${gain}."
        )

    def _level5(self) -> str:
        income = random.randint(3000, 8000)
        item = random.choice(self.ITEMS)
        item_cost = random.randint(500, 3000)
        monthly_expenses = random.randint(1500, income - 500)
        savings_rate = round(random.uniform(10, 30), 0)
        monthly_savings = round(income * savings_rate / 100, 2)
        months_needed = math.ceil(item_cost / monthly_savings)
        can_afford_now = (income - monthly_expenses) >= item_cost
        return (
            f"Income: ${income}/month. Expenses: ${monthly_expenses}/month. "
            f"Savings rate: {savings_rate}%. {item.capitalize()} costs ${item_cost}.\n"
            f"Question: Can I buy the {item} now? If not, how many months to save?\n"
            f"Answer: Monthly savings = ${monthly_savings}. "
            f"{'Can afford now — surplus covers cost.' if can_afford_now else f'Cannot afford now. Months needed = ceil({item_cost}/{monthly_savings}) = {months_needed} months.'}"
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Financial | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# SchedulingGenerator
# ---------------------------------------------------------------------------

class SchedulingGenerator:
    """Time and scheduling: meetings, travel time, deadlines, conflicts."""

    ACTIVITIES = ["meeting", "doctor appointment", "gym session", "lunch", "call",
                  "workshop", "interview", "class", "presentation", "commute"]
    LOCATIONS = ["downtown", "the office", "the airport", "the clinic",
                 "the gym", "the conference center", "home", "the client site"]

    def _level1(self) -> str:
        start_h = random.randint(8, 16)
        start_m = random.choice([0, 15, 30, 45])
        duration = random.choice([30, 45, 60, 90])
        end_m = start_m + duration
        end_h = start_h + end_m // 60
        end_m = end_m % 60
        act = random.choice(self.ACTIVITIES)
        return (
            f"A {act} starts at {start_h}:{start_m:02d} and lasts {duration} minutes.\n"
            f"Question: When does it end?\n"
            f"Answer: {start_h}:{start_m:02d} + {duration} min = {end_h}:{end_m:02d}."
        )

    def _level2(self) -> str:
        event_h = random.randint(10, 17)
        travel = random.randint(20, 90)
        prep = random.randint(10, 30)
        total = travel + prep
        leave_m = event_h * 60 - total
        leave_h = leave_m // 60
        leave_min = leave_m % 60
        loc = random.choice(self.LOCATIONS)
        return (
            f"Event at {event_h}:00 at {loc}. Travel time: {travel} min. Prep time: {prep} min.\n"
            f"Question: When should I leave?\n"
            f"Answer: Total buffer = {travel} + {prep} = {total} min. "
            f"Leave by {leave_h}:{leave_min:02d}."
        )

    def _level3(self) -> str:
        acts = random.sample(self.ACTIVITIES, 3)
        times = sorted([random.randint(8, 17) for _ in range(3)])
        durations = [random.choice([30, 60, 90]) for _ in range(3)]
        conflicts = []
        for i in range(len(times) - 1):
            end = times[i] * 60 + durations[i]
            start_next = times[i+1] * 60
            if end > start_next:
                conflicts.append(f"{acts[i]} overlaps with {acts[i+1]}")
        conflict_str = "; ".join(conflicts) if conflicts else "No conflicts"
        schedule = "; ".join(f"{acts[i]} at {times[i]}:00 ({durations[i]}min)"
                             for i in range(3))
        return (
            f"Schedule: {schedule}.\n"
            f"Question: Are there any scheduling conflicts?\n"
            f"Answer: {conflict_str}."
        )

    def _level4(self) -> str:
        deadline_h = random.randint(14, 18)
        tasks = [(f"task_{i+1}", random.randint(20, 90)) for i in range(4)]
        total_time = sum(t[1] for t in tasks)
        current_h = random.randint(8, 12)
        available = (deadline_h - current_h) * 60
        feasible = total_time <= available
        task_str = ", ".join(f"{t[0]}({t[1]}min)" for t in tasks)
        return (
            f"Current time: {current_h}:00. Deadline: {deadline_h}:00. "
            f"Tasks: {task_str}.\n"
            f"Question: Can all tasks be completed before the deadline?\n"
            f"Answer: Total time needed = {total_time} min. "
            f"Available = {available} min. "
            f"{'Yes, feasible.' if feasible else f'No — {total_time - available} min short.'}"
        )

    def _level5(self) -> str:
        n_meetings = random.randint(3, 5)
        meetings = []
        t = random.randint(8, 10)
        for i in range(n_meetings):
            dur = random.choice([30, 45, 60])
            gap = random.choice([0, 15, 30])
            meetings.append((f"Meeting {i+1}", t, dur))
            t += dur + gap
        lunch_start = 12
        lunch_end = 13
        conflicts_with_lunch = [m for m in meetings
                                 if m[1] < lunch_end and m[1] + m[2]/60 > lunch_start]
        free_slots = []
        prev_end = 8
        for m in meetings:
            if m[1] > prev_end:
                free_slots.append(f"{prev_end}:00-{m[1]}:00")
            prev_end = m[1] + m[2] // 60
        sched = "; ".join(f"{m[0]} {m[1]}:00 ({m[2]}min)" for m in meetings)
        return (
            f"Schedule: {sched}.\n"
            f"Question: What are the free slots and does anything conflict with lunch (12-1pm)?\n"
            f"Answer: Free slots: {', '.join(free_slots) if free_slots else 'none'}. "
            f"Lunch conflicts: {', '.join(m[0] for m in conflicts_with_lunch) if conflicts_with_lunch else 'none'}."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Scheduling | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# DecisionTreeGenerator
# ---------------------------------------------------------------------------

class DecisionTreeGenerator:
    """Decision trees: options with tradeoffs, pick best with reasoning."""

    DECISIONS = [
        "buy vs rent a home", "take job offer A vs job offer B",
        "repair vs replace the laptop", "hire a contractor vs do it yourself",
        "invest in stocks vs bonds", "take the highway vs side streets",
        "study now vs study later", "buy in bulk vs buy as needed",
    ]
    CRITERIA = ["cost", "time", "risk", "quality", "flexibility", "long-term value"]

    def _make_option(self, name: str, n_criteria: int) -> dict:
        criteria = random.sample(self.CRITERIA, n_criteria)
        scores = {c: random.randint(1, 10) for c in criteria}
        return {"name": name, "scores": scores}

    def _level1(self) -> str:
        decision = random.choice(self.DECISIONS)
        a_score = random.randint(4, 10)
        b_score = random.randint(1, 9)
        winner = "A" if a_score > b_score else "B"
        return (
            f"Decision: {decision}.\n"
            f"Option A score: {a_score}/10. Option B score: {b_score}/10.\n"
            f"Question: Which option is better?\n"
            f"Answer: Option {winner} with score {max(a_score, b_score)}/10."
        )

    def _level2(self) -> str:
        decision = random.choice(self.DECISIONS)
        criteria = random.sample(self.CRITERIA, 2)
        a = {c: random.randint(3, 10) for c in criteria}
        b = {c: random.randint(3, 10) for c in criteria}
        a_total = sum(a.values())
        b_total = sum(b.values())
        winner = "A" if a_total >= b_total else "B"
        a_str = ", ".join(f"{c}={v}" for c, v in a.items())
        b_str = ", ".join(f"{c}={v}" for c, v in b.items())
        return (
            f"Decision: {decision}.\n"
            f"Option A: {a_str} (total={a_total}).\n"
            f"Option B: {b_str} (total={b_total}).\n"
            f"Question: Which option wins on combined score?\n"
            f"Answer: Option {winner} with total {max(a_total, b_total)}."
        )

    def _level3(self) -> str:
        decision = random.choice(self.DECISIONS)
        criteria = random.sample(self.CRITERIA, 3)
        weights = [random.randint(1, 5) for _ in criteria]
        a = [random.randint(3, 10) for _ in criteria]
        b = [random.randint(3, 10) for _ in criteria]
        a_weighted = sum(s * w for s, w in zip(a, weights))
        b_weighted = sum(s * w for s, w in zip(b, weights))
        winner = "A" if a_weighted >= b_weighted else "B"
        crit_str = ", ".join(f"{c}(w={w})" for c, w in zip(criteria, weights))
        return (
            f"Decision: {decision}. Criteria (weighted): {crit_str}.\n"
            f"Option A scores: {a}. Weighted total: {a_weighted}.\n"
            f"Option B scores: {b}. Weighted total: {b_weighted}.\n"
            f"Question: Which option wins on weighted score?\n"
            f"Answer: Option {winner} with weighted score {max(a_weighted, b_weighted)}."
        )

    def _level4(self) -> str:
        decision = random.choice(self.DECISIONS)
        options = ["A", "B", "C"]
        criteria = random.sample(self.CRITERIA, 3)
        scores = {o: {c: random.randint(2, 10) for c in criteria} for o in options}
        totals = {o: sum(scores[o].values()) for o in options}
        winner = max(totals, key=lambda k: totals[k])
        score_str = "; ".join(
            f"Option {o}: " + ", ".join(f"{c}={scores[o][c]}" for c in criteria) +
            f" (total={totals[o]})" for o in options
        )
        return (
            f"Decision: {decision}.\n{score_str}.\n"
            f"Question: Which of the three options is best overall?\n"
            f"Answer: Option {winner} with total score {totals[winner]}. "
            f"It leads on combined criteria."
        )

    def _level5(self) -> str:
        decision = random.choice(self.DECISIONS)
        criteria = random.sample(self.CRITERIA, 4)
        weights = [random.randint(1, 5) for _ in criteria]
        options = ["A", "B", "C"]
        scores = {o: [random.randint(2, 10) for _ in criteria] for o in options}
        weighted = {o: sum(s * w for s, w in zip(scores[o], weights)) for o in options}
        winner = max(weighted, key=lambda k: weighted[k])
        # Find which criteria the winner leads on
        winner_leads = [criteria[i] for i in range(len(criteria))
                        if scores[winner][i] == max(scores[o][i] for o in options)]
        crit_str = ", ".join(f"{c}(w={w})" for c, w in zip(criteria, weights))
        return (
            f"Decision: {decision}. Weighted criteria: {crit_str}.\n"
            f"Scores: " + "; ".join(
                f"Option {o}={scores[o]} (weighted={weighted[o]})" for o in options
            ) + ".\n"
            f"Question: Which option wins and why?\n"
            f"Answer: Option {winner} wins with weighted score {weighted[winner]}. "
            f"It leads on: {', '.join(winner_leads) if winner_leads else 'overall balance'}."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[Decision | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# SelfCorrectionGenerator
# ---------------------------------------------------------------------------

class SelfCorrectionGenerator:
    """Self-correction: model makes a mistake, catches it, corrects it."""

    MISTAKE_TYPES = [
        ("arithmetic", "calculation error"),
        ("logic", "invalid inference"),
        ("factual", "incorrect assumption"),
        ("unit", "wrong unit conversion"),
        ("sign", "sign error in calculation"),
    ]

    def _level1(self) -> str:
        a, b = random.randint(10, 99), random.randint(10, 99)
        wrong = a + b + random.choice([-1, 1, 2, -2])
        correct = a + b
        return (
            f"Question: What is {a} + {b}?\n"
            f"Initial answer: {wrong}.\n"
            f"Wait — let me recheck. {a} + {b} = {correct}, not {wrong}.\n"
            f"Corrected answer: {correct}."
        )

    def _level2(self) -> str:
        items = random.randint(3, 8)
        price = random.randint(5, 50)
        wrong_total = items * price + random.choice([-price, price])
        correct_total = items * price
        return (
            f"Question: {items} items at ${price} each. Total cost?\n"
            f"Initial answer: ${wrong_total}.\n"
            f"Wait — I multiplied incorrectly. {items} × ${price} = ${correct_total}.\n"
            f"Corrected answer: ${correct_total}."
        )

    def _level3(self) -> str:
        km = random.randint(5, 100)
        wrong_miles = round(km * 0.5, 2)   # wrong conversion
        correct_miles = round(km * 0.621371, 2)
        return (
            f"Question: Convert {km} km to miles.\n"
            f"Initial answer: {wrong_miles} miles (used factor 0.5).\n"
            f"Wait — the correct conversion factor is 0.621371, not 0.5.\n"
            f"Corrected answer: {km} × 0.621371 = {correct_miles} miles."
        )

    def _level4(self) -> str:
        p = random.randint(100, 1000)
        r = random.randint(2, 15)
        t = random.randint(1, 5)
        wrong = round(p * r * t, 2)   # forgot to divide by 100
        correct = round(p * r / 100 * t, 2)
        return (
            f"Question: Simple interest on ${p} at {r}% for {t} years.\n"
            f"Initial answer: ${wrong} (forgot to divide rate by 100).\n"
            f"Wait — interest = P × r/100 × t = ${p} × {r}/100 × {t} = ${correct}.\n"
            f"Corrected answer: ${correct}."
        )

    def _level5(self) -> str:
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        c = random.randint(2, 9)
        wrong = a * b + c   # wrong order of operations
        correct = a * (b + c)
        return (
            f"Question: Calculate {a} × ({b} + {c}).\n"
            f"Initial answer: {wrong} (applied multiplication before addition incorrectly).\n"
            f"Wait — parentheses first: ({b} + {c}) = {b+c}, then {a} × {b+c} = {correct}.\n"
            f"Corrected answer: {correct}."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[SelfCorrection | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# ToolUseGenerator
# ---------------------------------------------------------------------------

class ToolUseGenerator:
    """Tool use patterns: given a goal, select and call the right tool."""

    TOOLS = [
        {"name": "search_web",      "params": ["query"],
         "desc": "search the internet for information"},
        {"name": "read_file",       "params": ["path"],
         "desc": "read contents of a file"},
        {"name": "write_file",      "params": ["path", "content"],
         "desc": "write content to a file"},
        {"name": "run_code",        "params": ["code", "language"],
         "desc": "execute a code snippet"},
        {"name": "send_email",      "params": ["to", "subject", "body"],
         "desc": "send an email"},
        {"name": "get_calendar",    "params": ["date"],
         "desc": "retrieve calendar events for a date"},
        {"name": "set_reminder",    "params": ["time", "message"],
         "desc": "set a reminder"},
        {"name": "calculate",       "params": ["expression"],
         "desc": "evaluate a mathematical expression"},
        {"name": "get_weather",     "params": ["location"],
         "desc": "get current weather for a location"},
        {"name": "translate_text",  "params": ["text", "target_language"],
         "desc": "translate text to another language"},
    ]

    GOALS = [
        "find the latest news about AI",
        "check if I have any meetings tomorrow",
        "remind me to call mom at 6pm",
        "calculate my monthly budget",
        "send a follow-up email to the client",
        "translate this message to Spanish",
        "check the weather before I leave",
        "run this Python script",
        "save my notes to a file",
        "look up the definition of holographic memory",
    ]

    def _make_call(self, tool: dict, goal: str) -> str:
        params = {p: f"<{p}_value>" for p in tool["params"]}
        return f"{tool['name']}({', '.join(f'{k}={v}' for k, v in params.items())})"

    def _level1(self) -> str:
        tool = random.choice(self.TOOLS)
        goal = random.choice(self.GOALS)
        call = self._make_call(tool, goal)
        return (
            f"Goal: {goal}.\n"
            f"Available tool: {tool['name']} — {tool['desc']}.\n"
            f"Question: How do I use this tool to achieve the goal?\n"
            f"Answer: Call {call}."
        )

    def _level2(self) -> str:
        tools = random.sample(self.TOOLS, 3)
        goal = random.choice(self.GOALS)
        correct = random.choice(tools)
        tool_list = "; ".join(f"{t['name']} ({t['desc']})" for t in tools)
        call = self._make_call(correct, goal)
        return (
            f"Goal: {goal}.\n"
            f"Available tools: {tool_list}.\n"
            f"Question: Which tool should I use and how?\n"
            f"Answer: Use {correct['name']} because it {correct['desc']}. "
            f"Call: {call}."
        )

    def _level3(self) -> str:
        tool1, tool2 = random.sample(self.TOOLS, 2)
        goal = f"{random.choice(self.GOALS)} and then {random.choice(self.GOALS)}"
        call1 = self._make_call(tool1, goal)
        call2 = self._make_call(tool2, goal)
        return (
            f"Goal: {goal}.\n"
            f"Tools: {tool1['name']} ({tool1['desc']}), "
            f"{tool2['name']} ({tool2['desc']}).\n"
            f"Question: What is the correct sequence of tool calls?\n"
            f"Answer: Step 1: {call1}. Step 2: {call2}."
        )

    def _level4(self) -> str:
        tools = random.sample(self.TOOLS, 4)
        goal = random.choice(self.GOALS)
        correct = tools[0]
        wrong_tools = tools[1:]
        why_wrong = [f"{t['name']} is for {t['desc']}, not relevant here"
                     for t in wrong_tools[:2]]
        call = self._make_call(correct, goal)
        return (
            f"Goal: {goal}.\n"
            f"Available tools: {', '.join(t['name'] for t in tools)}.\n"
            f"Question: Which tool is correct and why are the others wrong?\n"
            f"Answer: Use {correct['name']}. "
            f"{'; '.join(why_wrong)}. "
            f"Call: {call}."
        )

    def _level5(self) -> str:
        tools = random.sample(self.TOOLS, 3)
        steps = random.randint(3, 4)
        goal = f"complete a {steps}-step workflow: " + " then ".join(
            random.choice(self.GOALS) for _ in range(steps)
        )
        calls = [self._make_call(random.choice(tools), goal)
                 for _ in range(steps)]
        fallback_tool = random.choice(tools)
        fallback_call = self._make_call(fallback_tool, "retry")
        return (
            f"Goal: {goal}.\n"
            f"Tools: {', '.join(t['name'] for t in tools)}.\n"
            f"Question: Plan the full tool call sequence with error handling.\n"
            f"Answer:\n"
            + "\n".join(f"  Step {i+1}: {c}" for i, c in enumerate(calls)) +
            f"\n  On failure: {fallback_call} to retry."
        )

    def generate(self, n: int, difficulty: int = 1) -> List[str]:
        fn = {1: self._level1, 2: self._level2, 3: self._level3,
              4: self._level4, 5: self._level5}[max(1, min(5, difficulty))]
        return [wrap(f"[ToolUse | Difficulty {difficulty}]\n\n" + fn()) for _ in range(n)]

    def generate_curriculum(self, n: int) -> List[str]:
        out = []
        per = max(1, n // 5)
        for d in range(1, 6):
            out.extend(self.generate(per, difficulty=d))
        random.shuffle(out)
        return out[:n]


# ---------------------------------------------------------------------------
# CombinedDeepDataset
# ---------------------------------------------------------------------------

class CombinedDeepDataset:
    """Calls all generators and returns a combined shuffled list."""

    def __init__(self):
        self.generators = [
            ReasoningGenerator(),
            LanguageGenerator(),
            CodeGenerator(),
            PlanningGenerator(),
            MemoryRecallGenerator(),
            VisionDescriptionGenerator(),
            FinancialReasoningGenerator(),
            SchedulingGenerator(),
            DecisionTreeGenerator(),
            SelfCorrectionGenerator(),
            ToolUseGenerator(),
        ]

    def generate(self, n_per_generator: int, difficulty: int = 1) -> List[str]:
        out = []
        for gen in self.generators:
            out.extend(gen.generate(n_per_generator, difficulty=difficulty))
        random.shuffle(out)
        return out

    def generate_curriculum(self, n_per_generator: int) -> List[str]:
        out = []
        for gen in self.generators:
            out.extend(gen.generate_curriculum(n_per_generator))
        random.shuffle(out)
        return out


# ---------------------------------------------------------------------------
# Main preview block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    generators = {
        "ReasoningGenerator": ReasoningGenerator(),
        "LanguageGenerator": LanguageGenerator(),
        "CodeGenerator": CodeGenerator(),
        "PlanningGenerator": PlanningGenerator(),
        "MemoryRecallGenerator": MemoryRecallGenerator(),
        "VisionDescriptionGenerator": VisionDescriptionGenerator(),
    }

    print("=" * 70)
    print("Avus Deep Synthetic Training Data — Preview (100 samples each)")
    print("=" * 70)

    all_samples = []
    for name, gen in generators.items():
        samples = gen.generate_curriculum(100)
        all_samples.extend(samples)
        print(f"\n[{name}] Generated {len(samples)} curriculum samples.")
        print("  First sample preview:")
        preview = samples[0][:300].replace("\n", " ")
        print(f"  {preview}...")

    print(f"\nTotal samples across all generators: {len(all_samples)}")

    # Combined dataset
    combined = CombinedDeepDataset()
    combined_samples = combined.generate_curriculum(n_per_generator=50)
    print(f"\nCombinedDeepDataset curriculum (50 per generator): {len(combined_samples)} total samples")

    # Save a small JSON preview
    preview_data = {
        "total": len(all_samples),
        "generators": list(generators.keys()),
        "sample_count_each": 100,
        "preview_first_5": all_samples[:5],
    }
    with open("deep_training_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview_data, f, indent=2, ensure_ascii=False)
    print("\nPreview saved to deep_training_preview.json")
    print("=" * 70)
