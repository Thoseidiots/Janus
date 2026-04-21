"""
synthetic_databases/janus_skills_db.py
=======================================
Generates synthetic skill-learning training data for Janus.

Covers:
  - Skill progression records (Beginner → Expert)
  - Learning resource evaluations
  - Skill-to-job matching examples
  - Knowledge extraction from tutorials
  - Skill decay and refresh patterns

Output: SQLite + JSONL.
"""

import json
import random
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

SEED = 77
random.seed(SEED)

DB_PATH    = Path("synthetic_databases/janus_skills.db")
JSONL_PATH = Path("synthetic_databases/janus_skills.jsonl")

SKILLS = [
    ("python",           ["functions", "classes", "async", "decorators", "type hints"]),
    ("javascript",       ["promises", "closures", "DOM", "fetch API", "modules"]),
    ("data_analysis",    ["pandas", "numpy", "matplotlib", "SQL", "statistics"]),
    ("machine_learning", ["regression", "classification", "neural networks", "evaluation", "deployment"]),
    ("writing",          ["structure", "tone", "SEO", "editing", "research"]),
    ("web_scraping",     ["requests", "BeautifulSoup", "Selenium", "rate limiting", "parsing"]),
    ("api_design",       ["REST", "authentication", "versioning", "documentation", "error handling"]),
    ("sql",              ["SELECT", "JOINs", "indexes", "transactions", "optimisation"]),
    ("docker",           ["images", "containers", "networking", "volumes", "compose"]),
    ("git",              ["branching", "merging", "rebasing", "hooks", "CI/CD"]),
]

LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

RESOURCE_TYPES = ["youtube_video", "article", "documentation", "course", "tutorial"]


def make_skill_progression(skill: str, concepts: List[str]) -> Dict[str, Any]:
    """Generate a skill progression record showing learning over time."""
    start_level = random.choice(["Beginner", "Intermediate"])
    end_level_idx = LEVELS.index(start_level) + random.randint(1, 2)
    end_level = LEVELS[min(end_level_idx, len(LEVELS) - 1)]

    days_elapsed = random.randint(7, 90)
    xp_gained = random.randint(50, 500)
    jobs_completed = random.randint(1, 20)
    success_rate = random.uniform(0.6, 0.98)

    learned = random.sample(concepts, min(len(concepts), random.randint(2, len(concepts))))

    return {
        "record_id": hashlib.md5(f"{skill}{start_level}{xp_gained}".encode()).hexdigest()[:12],
        "category": "skill_progression",
        "skill": skill,
        "start_level": start_level,
        "end_level": end_level,
        "xp_gained": xp_gained,
        "days_elapsed": days_elapsed,
        "jobs_completed": jobs_completed,
        "success_rate": round(success_rate, 3),
        "concepts_learned": learned,
        "instruction": (
            f"Janus has been working on {skill} for {days_elapsed} days, "
            f"completing {jobs_completed} jobs with a {success_rate:.0%} success rate. "
            f"What level should they be at?"
        ),
        "response": (
            f"Based on {days_elapsed} days of practice, {jobs_completed} completed jobs, "
            f"and a {success_rate:.0%} success rate, Janus has progressed from "
            f"{start_level} to {end_level} in {skill}. "
            f"Key concepts mastered: {', '.join(learned)}. "
            f"XP gained: {xp_gained}."
        ),
    }


def make_resource_evaluation(skill: str, resource_type: str) -> Dict[str, Any]:
    """Generate a learning resource evaluation record."""
    quality = random.uniform(0.4, 1.0)
    relevance = random.uniform(0.5, 1.0)
    duration = random.randint(5, 120)
    concepts_count = random.randint(2, 10)

    verdict = "recommended" if quality > 0.7 and relevance > 0.7 else "skip"

    return {
        "record_id": hashlib.md5(f"{skill}{resource_type}{quality}".encode()).hexdigest()[:12],
        "category": "resource_evaluation",
        "skill": skill,
        "resource_type": resource_type,
        "quality_score": round(quality, 3),
        "relevance_score": round(relevance, 3),
        "duration_minutes": duration,
        "concepts_count": concepts_count,
        "verdict": verdict,
        "instruction": (
            f"Evaluate this {resource_type} for learning {skill}: "
            f"quality={quality:.2f}, relevance={relevance:.2f}, "
            f"duration={duration}min, concepts={concepts_count}"
        ),
        "response": (
            f"This {resource_type} is {verdict} for {skill}. "
            f"Quality: {quality:.0%}, Relevance: {relevance:.0%}. "
            f"{'Worth the {duration} minutes — covers {concepts_count} key concepts.' if verdict == 'recommended' else 'Low quality or relevance — find a better resource.'}"
        ),
    }


def make_skill_job_match(skill: str, level: str, job_title: str, required_level: str) -> Dict[str, Any]:
    """Generate a skill-to-job matching example."""
    level_idx = LEVELS.index(level)
    required_idx = LEVELS.index(required_level)
    can_do = level_idx >= required_idx
    gap = required_idx - level_idx

    return {
        "record_id": hashlib.md5(f"{skill}{level}{job_title}".encode()).hexdigest()[:12],
        "category": "skill_job_match",
        "skill": skill,
        "current_level": level,
        "required_level": required_level,
        "job_title": job_title,
        "can_do": can_do,
        "level_gap": gap,
        "instruction": (
            f"Can Janus take this job? Job: '{job_title}' requires {required_level} {skill}. "
            f"Janus is currently {level} in {skill}."
        ),
        "response": (
            f"{'Yes' if can_do else 'No'} — Janus is {level} in {skill}, "
            f"and the job requires {required_level}. "
            f"{'This is within capability.' if can_do else f'A gap of {gap} level(s) exists. Recommend learning before attempting.'}"
            + (f" Success probability: {max(0.3, 1.0 - gap * 0.25):.0%}." if not can_do else "")
        ),
    }


def make_knowledge_extraction(skill: str, concepts: List[str]) -> Dict[str, Any]:
    """Generate a knowledge extraction example from a tutorial."""
    extracted = random.sample(concepts, min(len(concepts), random.randint(2, len(concepts))))
    noise_concepts = ["introduction", "overview", "summary", "conclusion"]

    return {
        "record_id": hashlib.md5(f"{skill}{''.join(extracted)}".encode()).hexdigest()[:12],
        "category": "knowledge_extraction",
        "skill": skill,
        "extracted_concepts": extracted,
        "instruction": (
            f"Extract the key learning concepts from this {skill} tutorial. "
            f"The tutorial covers: {', '.join(extracted + random.sample(noise_concepts, 2))}."
        ),
        "response": (
            f"Key concepts extracted from this {skill} tutorial:\n"
            + "\n".join(f"- {c}: practical skill applicable to real {skill} work" for c in extracted)
            + f"\n\nThese {len(extracted)} concepts have been added to Janus's {skill} knowledge base."
        ),
    }


# ── Database builder ──────────────────────────────────────────────────────────

def build_database(n_per_category: int = 250) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS skill_records (
            record_id   TEXT PRIMARY KEY,
            category    TEXT NOT NULL,
            skill       TEXT NOT NULL,
            instruction TEXT NOT NULL,
            response    TEXT NOT NULL,
            metadata    TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill ON skill_records(skill);
        CREATE INDEX IF NOT EXISTS idx_cat   ON skill_records(category);
    """)
    conn.commit()

    now = datetime.utcnow().isoformat()
    all_records: List[Dict] = []

    job_titles = [
        ("Write a Python script", "Intermediate"),
        ("Build a REST API", "Advanced"),
        ("Analyse sales data", "Intermediate"),
        ("Train a classifier", "Advanced"),
        ("Write SEO content", "Beginner"),
        ("Scrape product data", "Intermediate"),
        ("Design a database schema", "Advanced"),
        ("Set up CI/CD pipeline", "Expert"),
    ]

    for _ in range(n_per_category):
        skill, concepts = random.choice(SKILLS)
        all_records.append(make_skill_progression(skill, concepts))

    for _ in range(n_per_category):
        skill, _ = random.choice(SKILLS)
        rtype = random.choice(RESOURCE_TYPES)
        all_records.append(make_resource_evaluation(skill, rtype))

    for _ in range(n_per_category):
        skill, _ = random.choice(SKILLS)
        level = random.choice(LEVELS)
        job_title, required_level = random.choice(job_titles)
        all_records.append(make_skill_job_match(skill, level, job_title, required_level))

    for _ in range(n_per_category):
        skill, concepts = random.choice(SKILLS)
        all_records.append(make_knowledge_extraction(skill, concepts))

    for rec in all_records:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO skill_records
                   (record_id, category, skill, instruction, response, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec["record_id"],
                    rec["category"],
                    rec["skill"],
                    rec["instruction"],
                    rec["response"],
                    json.dumps({k: v for k, v in rec.items()
                                if k not in ("record_id", "category", "skill", "instruction", "response")}),
                    now,
                ),
            )
        except Exception as e:
            print(f"  Warning: {rec['record_id']}: {e}")

    conn.commit()
    conn.close()

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps({
                "instruction": rec["instruction"],
                "response":    rec["response"],
                "category":    rec["category"],
                "skill":       rec["skill"],
                "source":      "janus_skills_synthetic_v1",
            }) + "\n")

    print(f"Generated {len(all_records)} skill records → {DB_PATH} + {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=500)
