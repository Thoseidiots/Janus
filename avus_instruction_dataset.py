"""
avus_instruction_dataset.py
============================
Generates instruction-following training data for Avus.
No API keys. No external services. Pure synthetic data built from
templates grounded in Janus's own domain: planning, decisions,
task execution, financial reasoning, and self-reflection.

The goal is to teach Avus to:
  1. Follow instructions (question → answer format)
  2. Reason step-by-step (chain-of-thought)
  3. Make decisions with justification
  4. Plan multi-step tasks
  5. Summarize and report

Usage:
    from avus_instruction_dataset import InstructionDatasetGenerator
    gen = InstructionDatasetGenerator()
    pairs = gen.generate(samples=50_000)
    gen.save(pairs, "training_data/instructions.jsonl")
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# ── Data pair ─────────────────────────────────────────────────────────────────

@dataclass
class InstructionPair:
    instruction: str
    response: str
    category: str

    def to_training_text(self) -> str:
        """Format for causal LM training."""
        return (
            f"<|startoftext|>"
            f"[INST] {self.instruction} [/INST]\n"
            f"{self.response}"
            f"<|endoftext|>"
        )

    def to_dict(self) -> dict:
        return {
            "id": str(uuid.uuid4())[:8],
            "category": self.category,
            "instruction": self.instruction,
            "response": self.response,
        }


# ── Vocabulary pools ──────────────────────────────────────────────────────────

_TASKS = [
    "write a client proposal", "analyze market data", "draft a project plan",
    "review a contract", "prepare a financial report", "schedule team meetings",
    "research competitors", "create a marketing strategy", "evaluate a job candidate",
    "negotiate a vendor deal", "set quarterly goals", "audit expenses",
    "design a product roadmap", "write a performance review", "plan a product launch",
]

_SKILLS = [
    "data analysis", "copywriting", "financial modeling", "project management",
    "customer support", "software development", "graphic design", "SEO optimization",
    "social media management", "content creation", "video editing", "bookkeeping",
]

_SERVICES = [
    "freelance writing", "web development", "logo design", "virtual assistance",
    "data entry", "translation", "tutoring", "consulting", "photography editing",
    "podcast editing", "resume writing", "social media management",
]

_PROBLEMS = [
    "revenue is declining", "team morale is low", "a key client is unhappy",
    "costs are too high", "a deadline was missed", "a competitor launched a similar product",
    "a key employee resigned", "a project is behind schedule", "cash flow is tight",
    "a product has a critical bug", "marketing isn't converting", "a supplier raised prices",
]

_METRICS = [
    "monthly revenue", "customer acquisition cost", "churn rate", "profit margin",
    "employee satisfaction", "project completion rate", "client retention",
    "average deal size", "conversion rate", "net promoter score",
]

_AMOUNTS = [500, 1000, 2500, 5000, 10000, 25000, 50000]
_TIMEFRAMES = ["this week", "this month", "this quarter", "in 30 days", "in 90 days"]
_PRIORITIES = ["critical", "high", "medium", "low"]


def _rc(lst):
    return random.choice(lst)

def _ri(a, b):
    return random.randint(a, b)


# ── Generator categories ──────────────────────────────────────────────────────

class InstructionDatasetGenerator:
    """
    Generates diverse instruction-following pairs across 8 categories.
    All data is synthetic — no external dependencies.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._generators = [
            self._task_planning,
            self._decision_making,
            self._financial_reasoning,
            self._goal_setting,
            self._problem_solving,
            self._summarization,
            self._step_by_step,
            self._self_reflection,
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, samples: int = 50_000) -> List[InstructionPair]:
        pairs = []
        for _ in range(samples):
            fn = _rc(self._generators)
            pairs.append(fn())
        return pairs

    def save(self, pairs: List[InstructionPair], path: str):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p.to_dict()) + "\n")
        print(f"[InstructionDataset] Saved {len(pairs)} pairs to {out}")

    def save_as_text(self, pairs: List[InstructionPair], path: str):
        """Save as raw training text (one sample per line)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(p.to_training_text() + "\n")
        print(f"[InstructionDataset] Saved {len(pairs)} text samples to {out}")

    # ── Category 1: Task Planning ─────────────────────────────────────────────

    def _task_planning(self) -> InstructionPair:
        task = _rc(_TASKS)
        timeframe = _rc(_TIMEFRAMES)
        n_steps = _ri(3, 6)

        instruction = f"Create a step-by-step plan to {task} {timeframe}."

        steps = []
        step_templates = [
            "Define the scope and objectives clearly.",
            "Research relevant background information and constraints.",
            "Identify the key stakeholders and their requirements.",
            "Break the work into concrete deliverables.",
            "Assign responsibilities and set deadlines for each deliverable.",
            "Set up a tracking system to monitor progress.",
            "Schedule a review checkpoint at the halfway mark.",
            "Prepare a risk mitigation plan for the top 3 risks.",
            "Draft the initial version and gather feedback.",
            "Finalize and deliver the completed work.",
        ]
        chosen = random.sample(step_templates, n_steps)
        for i, s in enumerate(chosen, 1):
            steps.append(f"{i}. {s}")

        response = (
            f"Here is a {n_steps}-step plan to {task} {timeframe}:\n\n"
            + "\n".join(steps)
            + f"\n\nKey success metric: completion of all steps by the end of {timeframe} "
            f"with documented outcomes for each."
        )

        return InstructionPair(instruction, response, "task_planning")

    # ── Category 2: Decision Making ───────────────────────────────────────────

    def _decision_making(self) -> InstructionPair:
        options = random.sample(_SERVICES, 3)
        context = _rc([
            "with limited starting capital",
            "working 10 hours per week",
            "with no prior experience",
            "targeting remote clients",
            "aiming for passive income",
        ])

        instruction = (
            f"I need to choose between these three income options {context}: "
            f"{options[0]}, {options[1]}, or {options[2]}. "
            f"Which should I pursue first and why?"
        )

        scores = []
        for opt in options:
            startup = _ri(1, 5)
            income_potential = _ri(1, 5)
            time_to_first_dollar = _ri(1, 5)
            score = startup + income_potential + time_to_first_dollar
            scores.append((opt, startup, income_potential, time_to_first_dollar, score))

        scores.sort(key=lambda x: x[4], reverse=True)
        best = scores[0]

        response = (
            f"Evaluating each option {context}:\n\n"
            f"**{scores[0][0]}**: Startup ease {scores[0][1]}/5, "
            f"Income potential {scores[0][2]}/5, Speed to first payment {scores[0][3]}/5. "
            f"Total score: {scores[0][4]}/15.\n"
            f"**{scores[1][0]}**: Startup ease {scores[1][1]}/5, "
            f"Income potential {scores[1][2]}/5, Speed to first payment {scores[1][3]}/5. "
            f"Total score: {scores[1][4]}/15.\n"
            f"**{scores[2][0]}**: Startup ease {scores[2][1]}/5, "
            f"Income potential {scores[2][2]}/5, Speed to first payment {scores[2][3]}/5. "
            f"Total score: {scores[2][4]}/15.\n\n"
            f"Recommendation: Start with **{best[0]}**. It scores highest overall "
            f"and is the most practical given your constraints. Once you have consistent "
            f"income from it, you can layer in {scores[1][0]} as a secondary stream."
        )

        return InstructionPair(instruction, response, "decision_making")

    # ── Category 3: Financial Reasoning ──────────────────────────────────────

    def _financial_reasoning(self) -> InstructionPair:
        revenue = _rc(_AMOUNTS)
        expenses = int(revenue * random.uniform(0.3, 0.8))
        profit = revenue - expenses
        margin = round(profit / revenue * 100, 1)
        goal = _rc(_AMOUNTS)
        timeframe = _rc(_TIMEFRAMES)

        instruction = (
            f"My business made ${revenue:,} in revenue last month with ${expenses:,} in expenses. "
            f"I want to reach ${goal:,} in monthly profit {timeframe}. "
            f"What should I focus on?"
        )

        gap = goal - profit
        revenue_increase_needed = round(gap / (1 - expenses / revenue))

        response = (
            f"Current financial snapshot:\n"
            f"- Revenue: ${revenue:,}\n"
            f"- Expenses: ${expenses:,}\n"
            f"- Profit: ${profit:,} ({margin}% margin)\n\n"
            f"To reach ${goal:,} profit {timeframe}, you have a gap of ${gap:,}.\n\n"
            f"Two levers to pull:\n\n"
            f"1. **Increase revenue**: You need approximately ${revenue_increase_needed:,} "
            f"in additional monthly revenue (assuming expenses stay flat). "
            f"Focus on upselling existing clients or adding a new service line.\n\n"
            f"2. **Reduce expenses**: Audit your ${expenses:,} in costs. "
            f"A 20% reduction saves ${int(expenses * 0.2):,}/month, closing "
            f"{round(int(expenses * 0.2) / gap * 100)}% of the gap immediately.\n\n"
            f"Fastest path: Combine both — target ${int(revenue_increase_needed * 0.6):,} "
            f"in new revenue AND cut ${int(expenses * 0.15):,} in expenses."
        )

        return InstructionPair(instruction, response, "financial_reasoning")

    # ── Category 4: Goal Setting ──────────────────────────────────────────────

    def _goal_setting(self) -> InstructionPair:
        metric = _rc(_METRICS)
        current = _ri(10, 80)
        target = current + _ri(20, 60)
        timeframe = _rc(_TIMEFRAMES)

        instruction = (
            f"Help me set a SMART goal to improve my {metric} "
            f"from {current}% to {target}% {timeframe}."
        )

        weekly_improvement = round((target - current) / 4, 1)

        response = (
            f"SMART Goal for {metric}:\n\n"
            f"**Specific**: Increase {metric} from {current}% to {target}% "
            f"by implementing targeted improvements.\n\n"
            f"**Measurable**: Track {metric} weekly. "
            f"Target: +{weekly_improvement}% per week.\n\n"
            f"**Achievable**: A {target - current}% improvement {timeframe} is realistic "
            f"with focused effort. Similar benchmarks show 15-25% gains are common.\n\n"
            f"**Relevant**: Improving {metric} directly impacts overall business health "
            f"and long-term sustainability.\n\n"
            f"**Time-bound**: Achieve {target}% by end of {timeframe}.\n\n"
            f"Weekly milestones:\n"
            f"- Week 1: {current + weekly_improvement:.0f}%\n"
            f"- Week 2: {current + weekly_improvement * 2:.0f}%\n"
            f"- Week 3: {current + weekly_improvement * 3:.0f}%\n"
            f"- Week 4: {target}%\n\n"
            f"Review progress every Friday and adjust tactics if you fall behind pace."
        )

        return InstructionPair(instruction, response, "goal_setting")

    # ── Category 5: Problem Solving ───────────────────────────────────────────

    def _problem_solving(self) -> InstructionPair:
        problem = _rc(_PROBLEMS)
        urgency = _rc(_PRIORITIES)

        instruction = f"My business has a {urgency} priority problem: {problem}. What should I do?"

        causes = random.sample([
            "insufficient market research",
            "pricing misalignment",
            "communication breakdown",
            "process inefficiency",
            "resource constraints",
            "external market shift",
            "skill gap in the team",
            "unclear ownership",
        ], 3)

        actions = random.sample([
            "Conduct a root cause analysis with the team within 48 hours.",
            "Communicate transparently with affected stakeholders immediately.",
            "Implement a quick fix to stop the bleeding while planning a permanent solution.",
            "Reassign resources to address the highest-impact area first.",
            "Set up daily check-ins until the situation stabilizes.",
            "Document the issue and create a post-mortem to prevent recurrence.",
            "Bring in an outside perspective if internal solutions aren't working.",
            "Negotiate a short-term extension or adjustment with affected parties.",
        ], 4)

        response = (
            f"Addressing '{problem}' ({urgency} priority):\n\n"
            f"**Likely root causes:**\n"
            + "\n".join(f"- {c}" for c in causes)
            + f"\n\n**Immediate actions ({urgency} priority means act now):**\n"
            + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
            + f"\n\n**30-day recovery plan:**\n"
            f"Week 1: Stabilize — stop the situation from getting worse.\n"
            f"Week 2: Diagnose — identify the true root cause with data.\n"
            f"Week 3: Fix — implement the permanent solution.\n"
            f"Week 4: Monitor — confirm the fix is holding and document lessons learned."
        )

        return InstructionPair(instruction, response, "problem_solving")

    # ── Category 6: Summarization ─────────────────────────────────────────────

    def _summarization(self) -> InstructionPair:
        n_items = _ri(4, 8)
        items = random.sample(_TASKS + _SKILLS, n_items)
        context = _rc(["last week", "this sprint", "Q3", "the past month"])

        instruction = (
            f"Summarize the following completed work from {context} into a brief executive report:\n"
            + "\n".join(f"- {item}" for item in items)
        )

        highlights = items[:2]
        total_value = _rc(_AMOUNTS)

        response = (
            f"Executive Summary — {context.title()}\n\n"
            f"**Overview**: Completed {n_items} key initiatives this period, "
            f"delivering an estimated ${total_value:,} in value.\n\n"
            f"**Highlights**:\n"
            f"- {highlights[0].title()}: Delivered on time, exceeded quality targets.\n"
            f"- {highlights[1].title()}: Completed ahead of schedule, "
            f"enabling early deployment.\n\n"
            f"**Full Completion List** ({n_items} items):\n"
            + "\n".join(f"✓ {item.title()}" for item in items)
            + f"\n\n**Next Period Focus**: Build on momentum from {highlights[0]} "
            f"and begin planning the next phase of {highlights[1]}."
        )

        return InstructionPair(instruction, response, "summarization")

    # ── Category 7: Step-by-Step Reasoning ───────────────────────────────────

    def _step_by_step(self) -> InstructionPair:
        skill = _rc(_SKILLS)
        service = _rc(_SERVICES)
        amount = _rc(_AMOUNTS)

        instruction = (
            f"Walk me through exactly how to earn ${amount:,} per month "
            f"offering {service} using my {skill} skills. Think step by step."
        )

        hourly = _ri(25, 150)
        hours_needed = round(amount / hourly)
        clients_needed = _ri(2, 8)
        per_client = round(amount / clients_needed)

        response = (
            f"Step-by-step path to ${amount:,}/month with {service}:\n\n"
            f"**Step 1 — Set your rate**\n"
            f"Based on {skill} market rates, charge ${hourly}/hour or "
            f"${per_client:,}/month per client on retainer. "
            f"You need {clients_needed} clients at ${per_client:,} each.\n\n"
            f"**Step 2 — Build a simple portfolio**\n"
            f"Create 2-3 sample projects showcasing your {skill}. "
            f"These don't need to be paid work — personal projects count.\n\n"
            f"**Step 3 — Find your first client**\n"
            f"Post on Fiverr, Upwork, or LinkedIn. "
            f"Offer a discounted first project to get a review. "
            f"Target businesses that clearly need {service}.\n\n"
            f"**Step 4 — Deliver and get a testimonial**\n"
            f"Over-deliver on the first project. Ask for a written testimonial. "
            f"This becomes your most valuable marketing asset.\n\n"
            f"**Step 5 — Scale to {clients_needed} clients**\n"
            f"Use the testimonial to land the next client. "
            f"Repeat until you have {clients_needed} paying clients. "
            f"At ${per_client:,}/client, that's ${amount:,}/month.\n\n"
            f"**Timeline**: Realistic first client in 2-4 weeks. "
            f"Full ${amount:,}/month in 3-6 months with consistent effort."
        )

        return InstructionPair(instruction, response, "step_by_step")

    # ── Category 8: Self-Reflection / Status ──────────────────────────────────

    def _self_reflection(self) -> InstructionPair:
        completed = _ri(2, 8)
        pending = _ri(1, 5)
        revenue = _rc(_AMOUNTS)
        goal_revenue = int(revenue * random.uniform(1.2, 2.5))

        instruction = _rc([
            "What is your current status and what should we focus on next?",
            "Give me a brief status report on operations.",
            "What have you accomplished and what are the priorities going forward?",
            "Summarize where we stand and recommend next actions.",
        ])

        response = (
            f"Status Report:\n\n"
            f"**Completed this cycle**: {completed} tasks finished successfully.\n"
            f"**Pending**: {pending} tasks in queue, prioritized by impact.\n"
            f"**Revenue tracked**: ${revenue:,} confirmed.\n"
            f"**Goal progress**: ${revenue:,} of ${goal_revenue:,} target "
            f"({round(revenue / goal_revenue * 100)}% complete).\n\n"
            f"**Recommended next actions**:\n"
            f"1. Execute the highest-priority pending task to maintain momentum.\n"
            f"2. Follow up on any outstanding client deliverables.\n"
            f"3. Review expenses to identify any optimization opportunities.\n"
            f"4. Plan the next revenue-generating activity to close the gap "
            f"to the ${goal_revenue:,} goal.\n\n"
            f"**Assessment**: Operations are running. "
            f"The path to goal is clear — consistent execution is the key variable."
        )

        return InstructionPair(instruction, response, "self_reflection")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Avus instruction-tuning dataset")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--output", type=str, default="training_data/instructions.jsonl")
    parser.add_argument("--text-output", type=str, default="training_data/instructions_text.txt")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    gen = InstructionDatasetGenerator()

    if args.preview:
        pairs = gen.generate(samples=8)
        for p in pairs:
            print(f"\n[{p.category.upper()}]")
            print(f"INST: {p.instruction[:100]}...")
            print(f"RESP: {p.response[:150]}...")
        import sys; sys.exit(0)

    print(f"Generating {args.samples:,} instruction pairs...")
    pairs = gen.generate(samples=args.samples)
    gen.save(pairs, args.output)
    gen.save_as_text(pairs, args.text_output)

    # Show category distribution
    from collections import Counter
    cats = Counter(p.category for p in pairs)
    print("\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,} ({count/len(pairs)*100:.1f}%)")
