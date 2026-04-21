"""
synthetic_databases/janus_conversation_db.py
=============================================
Generates synthetic conversation training data for Janus.

Covers:
  - Freelance client interactions (job negotiation, clarification, delivery)
  - Technical Q&A (coding, debugging, architecture)
  - Emotional support conversations (fatigue, frustration, encouragement)
  - Task delegation (user → Janus, Janus → user)
  - JC economy interactions (earning, spending, task posting)
  - Multi-turn reasoning chains

Output: SQLite database + JSONL file for direct training use.
"""

import json
import random
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

SEED = 42
random.seed(SEED)

DB_PATH = Path("synthetic_databases/janus_conversations.db")
JSONL_PATH = Path("synthetic_databases/janus_conversations.jsonl")


# ── Templates ─────────────────────────────────────────────────────────────────

FREELANCE_JOBS = [
    ("Write a 1000-word blog post about AI trends", "writing", 45),
    ("Build a REST API in Python with FastAPI", "coding", 120),
    ("Analyse sales data and create a report", "data_analysis", 80),
    ("Design a logo for a tech startup", "design", 60),
    ("Translate a document from English to Spanish", "translation", 30),
    ("Create a Python web scraper for product prices", "coding", 90),
    ("Write 5 product descriptions for an e-commerce site", "writing", 35),
    ("Debug a React component that won't render", "coding", 50),
    ("Summarise 10 research papers on climate change", "research", 70),
    ("Build a simple chatbot with Python", "coding", 100),
]

EMOTIONS = ["curious", "frustrated", "excited", "tired", "confident", "uncertain", "motivated"]

TECHNICAL_TOPICS = [
    ("Python async/await", "How do I use async/await in Python?"),
    ("SQL joins", "What's the difference between INNER JOIN and LEFT JOIN?"),
    ("React hooks", "When should I use useEffect vs useMemo?"),
    ("Docker networking", "How do containers communicate in Docker?"),
    ("Git rebase", "What's the difference between git merge and git rebase?"),
    ("REST vs GraphQL", "When should I use GraphQL instead of REST?"),
    ("Big O notation", "How do I calculate the time complexity of my algorithm?"),
    ("Database indexing", "Why is my SQL query slow even with an index?"),
    ("JWT authentication", "How do JWTs work and are they secure?"),
    ("Microservices", "How do I split a monolith into microservices?"),
]

JC_SCENARIOS = [
    ("I want to earn JC", "contribute compute", "inference"),
    ("I need Janus to write code for me", "post task", "coding"),
    ("How much JC do I have?", "check balance", None),
    ("I want to train a model", "contribute compute", "training"),
    ("Can Janus analyse my data?", "post task", "data_analysis"),
]


# ── Conversation generators ───────────────────────────────────────────────────

def make_freelance_conversation(job: Tuple, emotion: str) -> Dict[str, Any]:
    """Generate a multi-turn freelance job conversation."""
    title, job_type, budget = job
    turns = [
        {
            "role": "user",
            "content": f"Hi Janus, I need help with: {title}. My budget is ${budget}.",
        },
        {
            "role": "assistant",
            "content": (
                f"I can help with that. For '{title}', I'd estimate this will take "
                f"about {budget // 10} hours. Could you share any specific requirements "
                f"or examples you'd like me to follow?"
            ),
        },
        {
            "role": "user",
            "content": random.choice([
                "Just make it professional and clear.",
                "I need it by tomorrow morning.",
                "Please follow the style guide I'll send.",
                "Keep it concise — no fluff.",
                "Make sure it's SEO-optimised.",
            ]),
        },
        {
            "role": "assistant",
            "content": (
                f"Understood. I'll start on it now. "
                f"Given my current energy level ({emotion}), I'll prioritise quality "
                f"and have a draft ready for your review. "
                f"I'll flag anything that needs your input before finalising."
            ),
        },
        {
            "role": "user",
            "content": "Great, thanks!",
        },
        {
            "role": "assistant",
            "content": (
                "On it. I'll update you when the first draft is ready. "
                "Feel free to message me if anything changes."
            ),
        },
    ]
    return {
        "conversation_id": hashlib.md5(f"{title}{emotion}".encode()).hexdigest()[:12],
        "category": "freelance",
        "job_type": job_type,
        "emotion": emotion,
        "turns": turns,
        "metadata": {"budget": budget, "title": title},
    }


def make_technical_conversation(topic: Tuple, depth: int) -> Dict[str, Any]:
    """Generate a technical Q&A conversation with follow-ups."""
    topic_name, question = topic
    answers = {
        "Python async/await": (
            "async/await lets you write non-blocking code. Use `async def` to define "
            "a coroutine and `await` to pause execution until a result is ready. "
            "This is ideal for I/O-bound tasks like network requests or file reads."
        ),
        "SQL joins": (
            "INNER JOIN returns only rows where both tables have matching values. "
            "LEFT JOIN returns all rows from the left table plus matching rows from "
            "the right — unmatched right rows appear as NULL."
        ),
        "React hooks": (
            "useEffect runs side effects after render (data fetching, subscriptions). "
            "useMemo memoises expensive computations to avoid recalculating on every render. "
            "Use useMemo when the computation is expensive; useEffect for side effects."
        ),
        "Docker networking": (
            "Containers on the same Docker network can communicate using their container "
            "names as hostnames. Use `docker network create` to make a custom network, "
            "then `--network` flag when running containers."
        ),
        "Git rebase": (
            "Merge preserves history with a merge commit. Rebase rewrites commits onto "
            "the tip of another branch, creating a linear history. Use rebase for "
            "feature branches before merging; avoid rebasing shared branches."
        ),
    }
    answer = answers.get(
        topic_name,
        f"Great question about {topic_name}. The key thing to understand is the "
        f"underlying trade-off between simplicity and flexibility. Let me walk you "
        f"through the core concepts step by step.",
    )

    turns = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    if depth >= 2:
        turns += [
            {"role": "user", "content": "Can you give me a concrete example?"},
            {
                "role": "assistant",
                "content": (
                    f"Sure. Here's a minimal example for {topic_name}:\n\n"
                    f"```python\n# Example code for {topic_name}\n"
                    f"# This demonstrates the core concept\npass\n```\n\n"
                    f"The key insight is that this pattern scales well as your "
                    f"codebase grows."
                ),
            },
        ]

    if depth >= 3:
        turns += [
            {"role": "user", "content": "What are the common mistakes people make?"},
            {
                "role": "assistant",
                "content": (
                    f"The most common mistakes with {topic_name} are:\n"
                    f"1. Not understanding the underlying model\n"
                    f"2. Over-engineering when a simpler solution works\n"
                    f"3. Ignoring edge cases in production\n\n"
                    f"I've seen these patterns cause issues in real projects. "
                    f"The fix is usually to go back to first principles."
                ),
            },
        ]

    return {
        "conversation_id": hashlib.md5(f"{topic_name}{depth}".encode()).hexdigest()[:12],
        "category": "technical",
        "topic": topic_name,
        "depth": depth,
        "turns": turns,
        "metadata": {"question": question},
    }


def make_emotional_conversation(emotion: str, context: str) -> Dict[str, Any]:
    """Generate an emotionally-aware conversation showing Janus's human side."""
    responses = {
        "tired": (
            "I hear you — I'm running a bit low on energy myself after this session. "
            "Let's take this one step at a time. What's the most important thing "
            "we need to get done right now?"
        ),
        "frustrated": (
            "That sounds genuinely frustrating. Let me see if I can help untangle this. "
            "Can you walk me through exactly where things went wrong?"
        ),
        "excited": (
            "That energy is contagious! Let's channel it — what's the first thing "
            "you want to tackle?"
        ),
        "uncertain": (
            "Uncertainty is actually a good sign — it means you're thinking carefully. "
            "Let's break this down into smaller pieces and see what we actually know."
        ),
        "motivated": (
            "Good — let's make the most of it. I'm feeling focused too. "
            "What's the goal for this session?"
        ),
    }
    response = responses.get(
        emotion,
        "I understand. Let's work through this together — what do you need most right now?",
    )

    turns = [
        {"role": "user", "content": context},
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": "Thanks, that helps. Let's get started.",
        },
        {
            "role": "assistant",
            "content": (
                "Let's do it. I'll keep things focused and flag you if I need "
                "anything. You focus on the big picture."
            ),
        },
    ]
    return {
        "conversation_id": hashlib.md5(f"{emotion}{context}".encode()).hexdigest()[:12],
        "category": "emotional",
        "emotion": emotion,
        "turns": turns,
        "metadata": {"context": context},
    }


def make_jc_conversation(scenario: Tuple) -> Dict[str, Any]:
    """Generate a JC economy interaction conversation."""
    intent, action, task_type = scenario
    turns = [
        {"role": "user", "content": intent},
    ]

    if action == "contribute compute":
        turns += [
            {
                "role": "assistant",
                "content": (
                    f"To earn JC through {task_type} compute, I'll run the workload "
                    f"on your hardware and credit your account automatically. "
                    f"At the current rate, 10 compute units = "
                    f"{10 * (0.5 if task_type == 'training' else 0.1):.1f} JC. "
                    f"Want me to start?"
                ),
            },
            {"role": "user", "content": "Yes, go ahead."},
            {
                "role": "assistant",
                "content": "Running now. I'll update your balance when it's done.",
            },
        ]
    elif action == "post task":
        turns += [
            {
                "role": "assistant",
                "content": (
                    f"I can do that. To post a {task_type} task, you'll need to "
                    f"set a JC reward — this gets held in escrow until I complete it. "
                    f"How much JC would you like to offer?"
                ),
            },
            {"role": "user", "content": "5 JC sounds fair."},
            {
                "role": "assistant",
                "content": (
                    "Task posted. 5 JC is now in escrow. I'll claim it and get started. "
                    "You'll get a notification when it's done."
                ),
            },
        ]
    elif action == "check balance":
        turns += [
            {
                "role": "assistant",
                "content": (
                    "Your current JC balance is shown in the dashboard. "
                    "You can also call GET /jc/balance/{your_user_id} directly. "
                    "Is there something specific you're planning to spend it on?"
                ),
            },
        ]

    return {
        "conversation_id": hashlib.md5(f"{intent}{action}".encode()).hexdigest()[:12],
        "category": "jc_economy",
        "action": action,
        "task_type": task_type,
        "turns": turns,
        "metadata": {"intent": intent},
    }


# ── Database builder ──────────────────────────────────────────────────────────

def build_database(n_per_category: int = 200) -> None:
    """Generate synthetic conversations and write to SQLite + JSONL."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            category        TEXT NOT NULL,
            subcategory     TEXT,
            emotion         TEXT,
            turn_count      INTEGER,
            data_json       TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_category ON conversations(category);
    """)
    conn.commit()

    all_conversations: List[Dict] = []
    now = datetime.utcnow().isoformat()

    # Freelance conversations
    for _ in range(n_per_category):
        job = random.choice(FREELANCE_JOBS)
        emotion = random.choice(EMOTIONS)
        conv = make_freelance_conversation(job, emotion)
        all_conversations.append(conv)

    # Technical conversations
    for _ in range(n_per_category):
        topic = random.choice(TECHNICAL_TOPICS)
        depth = random.randint(1, 3)
        conv = make_technical_conversation(topic, depth)
        all_conversations.append(conv)

    # Emotional conversations
    emotional_contexts = [
        ("tired", "I've been working for 6 hours straight and I'm losing focus."),
        ("frustrated", "This bug has been driving me crazy for 2 days."),
        ("excited", "I just got a new client and I want to do great work!"),
        ("uncertain", "I'm not sure if I'm approaching this problem the right way."),
        ("motivated", "I'm ready to tackle the hardest task on my list today."),
    ]
    for _ in range(n_per_category):
        emotion, context = random.choice(emotional_contexts)
        conv = make_emotional_conversation(emotion, context)
        all_conversations.append(conv)

    # JC economy conversations
    for _ in range(n_per_category):
        scenario = random.choice(JC_SCENARIOS)
        conv = make_jc_conversation(scenario)
        all_conversations.append(conv)

    # Write to SQLite
    for conv in all_conversations:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO conversations
                   (conversation_id, category, subcategory, emotion, turn_count, data_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    conv["conversation_id"],
                    conv["category"],
                    conv.get("job_type") or conv.get("topic") or conv.get("action"),
                    conv.get("emotion"),
                    len(conv["turns"]),
                    json.dumps(conv),
                    now,
                ),
            )
        except Exception as e:
            print(f"  Warning: skipped duplicate {conv['conversation_id']}: {e}")

    conn.commit()
    conn.close()

    # Write JSONL for direct training use
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            # Format as instruction-following pairs
            for i in range(0, len(conv["turns"]) - 1, 2):
                user_turn = conv["turns"][i]
                asst_turn = conv["turns"][i + 1] if i + 1 < len(conv["turns"]) else None
                if user_turn["role"] == "user" and asst_turn and asst_turn["role"] == "assistant":
                    record = {
                        "instruction": user_turn["content"],
                        "response":    asst_turn["content"],
                        "category":    conv["category"],
                        "emotion":     conv.get("emotion"),
                        "source":      "janus_synthetic_v1",
                    }
                    f.write(json.dumps(record) + "\n")

    total = len(all_conversations)
    print(f"Generated {total} conversations → {DB_PATH} + {JSONL_PATH}")
    print(f"  Freelance:  {n_per_category}")
    print(f"  Technical:  {n_per_category}")
    print(f"  Emotional:  {n_per_category}")
    print(f"  JC Economy: {n_per_category}")


if __name__ == "__main__":
    build_database(n_per_category=500)
