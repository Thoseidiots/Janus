"""
janus_reasoning_engine/communication/nlu.py
============================================
Natural Language Understanding for the Janus Reasoning Engine.

Provides:
  - NLU.parse_instruction(text) -> ParsedInstruction
  - NLU.extract_job_requirements(job_description) -> JobRequirements
  - NLU.extract_knowledge(text) -> List[str]

Uses JanusGPT when available, falls back to heuristic regex parsing.

Requirements: REQ-7.1
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParsedInstruction:
    """Structured representation of a parsed natural language instruction."""
    raw_text: str
    intent: str                          # e.g. "find_job", "learn_skill", "earn_money"
    entities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class JobRequirements:
    """Structured requirements extracted from a job description."""
    raw_text: str
    skills: List[str] = field(default_factory=list)
    budget: Optional[str] = None         # e.g. "$500", "negotiable"
    deadline: Optional[str] = None       # e.g. "2 weeks", "ASAP"
    deliverables: List[str] = field(default_factory=list)
    platform: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[tuple] = [
    (r"\b(earn|make|generate)\b.*\b(money|income|revenue|cash)\b", "earn_money"),
    (r"\b(find|search|look for|discover)\b.*\b(job|work|gig|project|client)\b", "find_job"),
    (r"\b(learn|study|acquire|pick up)\b.*\b(skill|language|framework|tool)\b", "learn_skill"),
    (r"\b(build|create|develop|write|code)\b.*\b(app|website|script|tool|bot)\b", "build_software"),
    (r"\b(apply|submit|send)\b.*\b(proposal|application|bid)\b", "apply_job"),
    (r"\b(report|update|status|progress)\b", "report_progress"),
    (r"\b(stop|pause|cancel|abort)\b", "stop_task"),
    (r"\b(help|assist|support)\b", "request_help"),
]

_BUDGET_PATTERN = re.compile(
    r"\$\s*[\d,]+(?:\.\d{2})?(?:\s*[-–]\s*\$?\s*[\d,]+(?:\.\d{2})?)?|"
    r"[\d,]+\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)",
    re.IGNORECASE,
)

_DEADLINE_PATTERN = re.compile(
    r"\b(?:within|by|before|in)\s+\d+\s+(?:day|week|month|hour)s?\b|"
    r"\bASAP\b|\burgent\b|\bimmediately\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)

_SKILL_KEYWORDS = [
    "python", "javascript", "typescript", "react", "node", "django", "flask",
    "sql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
    "aws", "azure", "gcp", "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy", "tensorflow", "pytorch",
    "html", "css", "vue", "angular", "fastapi", "graphql", "rest api",
    "photoshop", "illustrator", "figma", "blender", "unity", "unreal",
    "copywriting", "seo", "marketing", "excel", "powerpoint", "word",
    "video editing", "premiere", "after effects", "3d modeling",
]


def _heuristic_intent(text: str) -> str:
    lower = text.lower()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, lower):
            return intent
    return "general"


def _heuristic_entities(text: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    budget_match = _BUDGET_PATTERN.search(text)
    if budget_match:
        entities["budget"] = budget_match.group(0).strip()
    deadline_match = _DEADLINE_PATTERN.search(text)
    if deadline_match:
        entities["deadline"] = deadline_match.group(0).strip()
    found_skills = [s for s in _SKILL_KEYWORDS if s in text.lower()]
    if found_skills:
        entities["skills"] = found_skills
    return entities


def _heuristic_constraints(text: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    if re.search(r"\bno\s+(?:spam|scam|adult|illegal)\b", text, re.IGNORECASE):
        constraints["content_filter"] = True
    budget_match = _BUDGET_PATTERN.search(text)
    if budget_match:
        constraints["budget"] = budget_match.group(0).strip()
    return constraints


def _heuristic_skills(text: str) -> List[str]:
    return [s for s in _SKILL_KEYWORDS if s in text.lower()]


def _heuristic_deliverables(text: str) -> List[str]:
    deliverables = []
    patterns = [
        r"deliver\s+(?:a\s+)?(.+?)(?:\.|,|;|$)",
        r"provide\s+(?:a\s+)?(.+?)(?:\.|,|;|$)",
        r"create\s+(?:a\s+)?(.+?)(?:\.|,|;|$)",
        r"build\s+(?:a\s+)?(.+?)(?:\.|,|;|$)",
        r"write\s+(?:a\s+)?(.+?)(?:\.|,|;|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            item = m.group(1).strip()
            if len(item) < 80:
                deliverables.append(item)
    return deliverables[:5]


def _extract_key_facts(text: str) -> List[str]:
    """Extract key facts using simple sentence-level heuristics."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    facts = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        # Prefer sentences with concrete nouns / verbs
        if re.search(r"\b(?:is|are|was|were|can|will|must|should|use|need|require)\b", sent, re.IGNORECASE):
            facts.append(sent)
        if len(facts) >= 10:
            break
    return facts


# ---------------------------------------------------------------------------
# NLU class
# ---------------------------------------------------------------------------

class NLU:
    """
    Natural Language Understanding module.

    Uses JanusGPT when available; falls back to heuristic regex parsing.
    All external dependencies are optional and fail gracefully.
    """

    def __init__(self, gpt: Optional[Any] = None) -> None:
        """
        Args:
            gpt: Optional JanusGPT instance. If None, heuristic fallback is used.
        """
        self._gpt = gpt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_instruction(self, text: str) -> ParsedInstruction:
        """
        Parse a natural language instruction into structured form.

        Extracts intent, entities (budget, deadline, skills), and constraints.

        Args:
            text: Raw instruction text from the owner.

        Returns:
            ParsedInstruction with intent, entities, and constraints.
        """
        if not text or not text.strip():
            return ParsedInstruction(raw_text=text, intent="unknown", confidence=0.0)

        if self._gpt is not None:
            try:
                return self._parse_with_gpt(text)
            except Exception as exc:
                log.warning("JanusGPT parse_instruction failed (%s), using heuristics", exc)

        return self._parse_heuristic(text)

    def extract_job_requirements(self, job_description: str) -> JobRequirements:
        """
        Extract structured requirements from a job description.

        Args:
            job_description: Raw job posting text.

        Returns:
            JobRequirements with skills, budget, deadline, and deliverables.
        """
        if not job_description or not job_description.strip():
            return JobRequirements(raw_text=job_description)

        if self._gpt is not None:
            try:
                return self._extract_job_with_gpt(job_description)
            except Exception as exc:
                log.warning("JanusGPT extract_job_requirements failed (%s), using heuristics", exc)

        return self._extract_job_heuristic(job_description)

    def extract_knowledge(self, text: str) -> List[str]:
        """
        Extract key facts from documentation, tutorials, or other text.

        Args:
            text: Source text (tutorial, docs, blog post, etc.)

        Returns:
            List of key fact strings.
        """
        if not text or not text.strip():
            return []

        if self._gpt is not None:
            try:
                return self._extract_knowledge_with_gpt(text)
            except Exception as exc:
                log.warning("JanusGPT extract_knowledge failed (%s), using heuristics", exc)

        return _extract_key_facts(text)

    # ------------------------------------------------------------------
    # GPT-backed implementations
    # ------------------------------------------------------------------

    def _parse_with_gpt(self, text: str) -> ParsedInstruction:
        prompt = (
            f"Parse this instruction and identify: intent, entities (budget, deadline, skills), "
            f"and constraints. Instruction: {text[:500]}\nIntent:"
        )
        response = self._gpt.generate(prompt, max_new=150, temperature=0.3)
        intent = response.split("\n")[0].strip().lower().replace(" ", "_") or "general"
        # Supplement with heuristic entities since GPT output is free-form
        entities = _heuristic_entities(text)
        constraints = _heuristic_constraints(text)
        return ParsedInstruction(
            raw_text=text,
            intent=intent[:50],
            entities=entities,
            constraints=constraints,
            confidence=0.8,
        )

    def _extract_job_with_gpt(self, job_description: str) -> JobRequirements:
        prompt = (
            f"Extract from this job description: required skills, budget, deadline, deliverables.\n"
            f"Job: {job_description[:600]}\nSkills:"
        )
        response = self._gpt.generate(prompt, max_new=200, temperature=0.3)
        # Parse GPT response lines, supplement with heuristics
        skills = _heuristic_skills(job_description)
        budget_m = _BUDGET_PATTERN.search(job_description)
        deadline_m = _DEADLINE_PATTERN.search(job_description)
        deliverables = _heuristic_deliverables(job_description)
        # Try to pull extra skills from GPT response
        gpt_skills = _heuristic_skills(response)
        all_skills = list(dict.fromkeys(skills + gpt_skills))
        return JobRequirements(
            raw_text=job_description,
            skills=all_skills,
            budget=budget_m.group(0).strip() if budget_m else None,
            deadline=deadline_m.group(0).strip() if deadline_m else None,
            deliverables=deliverables,
        )

    def _extract_knowledge_with_gpt(self, text: str) -> List[str]:
        prompt = (
            f"List the key facts from this text as bullet points:\n{text[:800]}\n- "
        )
        response = self._gpt.generate(prompt, max_new=300, temperature=0.4)
        facts = []
        for line in response.split("\n"):
            line = line.lstrip("•-* ").strip()
            if len(line) > 15:
                facts.append(line)
        return facts[:10] if facts else _extract_key_facts(text)

    # ------------------------------------------------------------------
    # Heuristic fallback implementations
    # ------------------------------------------------------------------

    def _parse_heuristic(self, text: str) -> ParsedInstruction:
        intent = _heuristic_intent(text)
        entities = _heuristic_entities(text)
        constraints = _heuristic_constraints(text)
        return ParsedInstruction(
            raw_text=text,
            intent=intent,
            entities=entities,
            constraints=constraints,
            confidence=0.6,
        )

    def _extract_job_heuristic(self, job_description: str) -> JobRequirements:
        skills = _heuristic_skills(job_description)
        budget_m = _BUDGET_PATTERN.search(job_description)
        deadline_m = _DEADLINE_PATTERN.search(job_description)
        deliverables = _heuristic_deliverables(job_description)
        return JobRequirements(
            raw_text=job_description,
            skills=skills,
            budget=budget_m.group(0).strip() if budget_m else None,
            deadline=deadline_m.group(0).strip() if deadline_m else None,
            deliverables=deliverables,
        )
