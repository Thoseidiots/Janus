"""
synthetic_databases/janus_reasoning_db.py
==========================================
Generates synthetic reasoning chain training data for Janus.

Covers:
  - Chain-of-thought problem solving (math, logic, planning)
  - Job scoring decisions with explicit reasoning
  - Self-reflection and metacognition
  - Error analysis and correction
  - Multi-step task decomposition

Output: SQLite + JSONL with full reasoning traces.
"""

import json
import random
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any

SEED = 99
random.seed(SEED)

DB_PATH   = Path("synthetic_databases/janus_reasoning.db")
JSONL_PATH = Path("synthetic_databases/janus_reasoning.jsonl")


# ── Reasoning templates ───────────────────────────────────────────────────────

def make_job_scoring_trace(job: Dict) -> Dict[str, Any]:
    """Generate a job scoring decision with explicit reasoning steps."""
    skill_match   = random.uniform(0.3, 1.0)
    budget_score  = min(job["budget"] / 500, 1.0)
    deadline_days = random.randint(1, 30)
    deadline_score = max(0, 1 - deadline_days / 30)
    learning_score = random.uniform(0, 0.3)

    final_score = (
        skill_match   * 0.4 +
        budget_score  * 0.3 +
        deadline_score * 0.2 +
        learning_score * 0.1
    )
    decision = "ACCEPT" if final_score > 0.5 else "REJECT"

    reasoning = (
        f"Let me evaluate this job systematically.\n\n"
        f"Job: {job['title']}\n"
        f"Budget: ${job['budget']}\n"
        f"Deadline: {deadline_days} days\n\n"
        f"Step 1 — Skill match: {skill_match:.2f} × 0.4 = {skill_match * 0.4:.3f}\n"
        f"  My skills overlap {skill_match:.0%} with the requirements.\n\n"
        f"Step 2 — Budget score: {budget_score:.2f} × 0.3 = {budget_score * 0.3:.3f}\n"
        f"  ${job['budget']} normalised against $500 benchmark.\n\n"
        f"Step 3 — Deadline score: {deadline_score:.2f} × 0.2 = {deadline_score * 0.2:.3f}\n"
        f"  {deadline_days} days remaining. {'Tight' if deadline_days < 7 else 'Comfortable'} timeline.\n\n"
        f"Step 4 — Learning opportunity: {learning_score:.2f} × 0.1 = {learning_score * 0.1:.3f}\n"
        f"  {'Some new skills to gain.' if learning_score > 0.1 else 'Mostly familiar territory.'}\n\n"
        f"Final score: {final_score:.3f}\n"
        f"Decision: {decision} (threshold: 0.5)\n"
        f"Reasoning: {'Score exceeds threshold — good fit.' if decision == 'ACCEPT' else 'Score below threshold — not worth the risk.'}"
    )

    return {
        "reasoning_id": hashlib.md5(f"{job['title']}{final_score}".encode()).hexdigest()[:12],
        "category": "job_scoring",
        "input": f"Should I take this job? Title: {job['title']}, Budget: ${job['budget']}, Deadline: {deadline_days} days",
        "reasoning_trace": reasoning,
        "output": f"{decision} (score: {final_score:.3f})",
        "metadata": {
            "skill_match": skill_match,
            "budget_score": budget_score,
            "deadline_score": deadline_score,
            "final_score": final_score,
            "decision": decision,
        },
    }


def make_task_decomposition(goal: str, steps: List[str]) -> Dict[str, Any]:
    """Generate a task decomposition reasoning trace."""
    reasoning = (
        f"Goal: {goal}\n\n"
        f"Let me break this down into manageable steps:\n\n"
    )
    for i, step in enumerate(steps, 1):
        reasoning += f"Step {i}: {step}\n"
        reasoning += f"  → Why: This is necessary because it {'sets up' if i == 1 else 'builds on step ' + str(i-1) + ' and'} enables the next phase.\n"
        reasoning += f"  → Risk: {'Low' if i < len(steps) else 'Medium'} — {'straightforward' if i < len(steps) else 'final integration can surface hidden issues'}.\n\n"

    reasoning += (
        f"Estimated total time: {len(steps) * random.randint(15, 45)} minutes\n"
        f"Critical path: Steps {', '.join(str(i) for i in range(1, min(3, len(steps)+1)))} must complete before parallelising.\n"
        f"I'll start with Step 1 and check in after each milestone."
    )

    return {
        "reasoning_id": hashlib.md5(goal.encode()).hexdigest()[:12],
        "category": "task_decomposition",
        "input": f"How should I approach: {goal}",
        "reasoning_trace": reasoning,
        "output": f"Decomposed into {len(steps)} steps. Starting with: {steps[0]}",
        "metadata": {"goal": goal, "step_count": len(steps)},
    }


def make_error_analysis(error: str, context: str, fix: str) -> Dict[str, Any]:
    """Generate an error analysis reasoning trace."""
    reasoning = (
        f"Error encountered: {error}\n"
        f"Context: {context}\n\n"
        f"Analysis:\n"
        f"1. What went wrong: {error.split(':')[0] if ':' in error else error}\n"
        f"2. Why it happened: The root cause is likely a mismatch between expected and actual state.\n"
        f"3. Impact: This blocks the current operation but is recoverable.\n"
        f"4. Fix: {fix}\n"
        f"5. Prevention: Add validation before this operation in future runs.\n\n"
        f"Confidence in fix: {random.randint(75, 98)}%\n"
        f"Fallback if fix fails: Log the error, mark task as failed, move to next job."
    )

    return {
        "reasoning_id": hashlib.md5(f"{error}{fix}".encode()).hexdigest()[:12],
        "category": "error_analysis",
        "input": f"Error: {error}\nContext: {context}",
        "reasoning_trace": reasoning,
        "output": fix,
        "metadata": {"error": error, "fix": fix},
    }


def make_self_reflection(situation: str, mood: str, energy: float) -> Dict[str, Any]:
    """Generate a self-reflection reasoning trace."""
    reasoning = (
        f"Current situation: {situation}\n"
        f"My mood: {mood}\n"
        f"Energy level: {energy:.0%}\n\n"
        f"Self-assessment:\n"
        f"- Am I operating at full capacity? {'Yes' if energy > 0.7 else 'No — I should be more selective about tasks.'}\n"
        f"- Is my current mood affecting my judgement? "
        f"{'Possibly — {mood} can lead to over-confidence.' if mood in ('excited', 'motivated') else 'Unlikely — I feel grounded.'}\n"
        f"- What should I prioritise? "
        f"{'High-value, low-effort tasks to conserve energy.' if energy < 0.5 else 'Tackle the hardest task while I have capacity.'}\n\n"
        f"Decision: "
        f"{'Take a short break before the next task.' if energy < 0.3 else 'Continue with current plan, monitor energy.'}\n"
        f"Adjustment: "
        f"{'Lower acceptance threshold for new jobs until energy recovers.' if energy < 0.5 else 'Maintain current strategy.'}"
    )

    return {
        "reasoning_id": hashlib.md5(f"{situation}{mood}".encode()).hexdigest()[:12],
        "category": "self_reflection",
        "input": f"Situation: {situation}. How should I proceed?",
        "reasoning_trace": reasoning,
        "output": f"{'Rest and recover' if energy < 0.3 else 'Continue with adjusted strategy'}",
        "metadata": {"mood": mood, "energy": energy},
    }


# ── Data fixtures ─────────────────────────────────────────────────────────────

JOBS = [
    {"title": "Write a Python tutorial", "budget": 80},
    {"title": "Build a REST API", "budget": 200},
    {"title": "Data analysis report", "budget": 150},
    {"title": "Logo design", "budget": 60},
    {"title": "SEO blog post", "budget": 40},
    {"title": "Machine learning model", "budget": 500},
    {"title": "Mobile app UI design", "budget": 300},
    {"title": "Database optimisation", "budget": 180},
]

GOALS_AND_STEPS = [
    ("Build a web scraper", [
        "Define target URLs and data schema",
        "Write HTTP request handler with retry logic",
        "Parse HTML with BeautifulSoup",
        "Store results in SQLite",
        "Add rate limiting and error handling",
        "Test with 10 sample URLs",
    ]),
    ("Complete a writing job", [
        "Read the brief carefully",
        "Research the topic",
        "Create an outline",
        "Write the first draft",
        "Edit for clarity and tone",
        "Final proofread and submit",
    ]),
    ("Debug a failing API", [
        "Reproduce the error locally",
        "Check logs for stack trace",
        "Isolate the failing component",
        "Write a minimal test case",
        "Apply fix and verify",
        "Deploy and monitor",
    ]),
]

ERRORS = [
    ("ConnectionError: Max retries exceeded", "Calling Upwork API", "Implement exponential backoff and retry after 30s"),
    ("JSONDecodeError: Expecting value", "Parsing API response", "Add response validation before parsing"),
    ("sqlite3.OperationalError: database is locked", "Writing to janus_worker.db", "Use WAL mode and add retry logic"),
    ("TimeoutError: Request timed out after 30s", "Generating work with Avus", "Increase timeout to 60s and add progress logging"),
    ("ValueError: Quality score 0.45 below threshold", "Submitting work", "Regenerate with higher detail level"),
]

SITUATIONS = [
    ("Completed 5 jobs in a row without a break", "tired", 0.25),
    ("Just received a $200 payment", "excited", 0.85),
    ("Failed 2 jobs due to quality issues", "frustrated", 0.55),
    ("Starting a new work cycle", "motivated", 0.90),
    ("Approaching deadline on 3 concurrent jobs", "uncertain", 0.45),
]


# ── Database builder ──────────────────────────────────────────────────────────

def build_database(n_per_category: int = 300) -> None:
    """Generate reasoning traces and write to SQLite + JSONL."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reasoning_traces (
            reasoning_id    TEXT PRIMARY KEY,
            category        TEXT NOT NULL,
            input_text      TEXT NOT NULL,
            reasoning_trace TEXT NOT NULL,
            output_text     TEXT NOT NULL,
            metadata_json   TEXT,
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cat ON reasoning_traces(category);
    """)
    conn.commit()

    from datetime import datetime
    now = datetime.utcnow().isoformat()
    all_traces: List[Dict] = []

    for _ in range(n_per_category):
        all_traces.append(make_job_scoring_trace(random.choice(JOBS)))

    for _ in range(n_per_category):
        goal, steps = random.choice(GOALS_AND_STEPS)
        all_traces.append(make_task_decomposition(goal, steps))

    for _ in range(n_per_category):
        error, context, fix = random.choice(ERRORS)
        all_traces.append(make_error_analysis(error, context, fix))

    for _ in range(n_per_category):
        situation, mood, energy = random.choice(SITUATIONS)
        energy_jitter = max(0.0, min(1.0, energy + random.uniform(-0.15, 0.15)))
        all_traces.append(make_self_reflection(situation, mood, energy_jitter))

    for trace in all_traces:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO reasoning_traces
                   (reasoning_id, category, input_text, reasoning_trace, output_text, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace["reasoning_id"],
                    trace["category"],
                    trace["input"],
                    trace["reasoning_trace"],
                    trace["output"],
                    json.dumps(trace.get("metadata", {})),
                    now,
                ),
            )
        except Exception as e:
            print(f"  Warning: {trace['reasoning_id']}: {e}")

    conn.commit()
    conn.close()

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for trace in all_traces:
            record = {
                "instruction": trace["input"],
                "chain_of_thought": trace["reasoning_trace"],
                "response": trace["output"],
                "category": trace["category"],
                "source": "janus_reasoning_synthetic_v1",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {len(all_traces)} reasoning traces → {DB_PATH} + {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=500)
