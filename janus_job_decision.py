"""
janus_job_decision.py
======================
Human-like job decision engine for Janus.

A human doesn't pick jobs by running a formula. They:
  - Browse options and feel drawn to some, repelled by others
  - Consider their current mood and energy
  - Think about what they want to learn or try
  - Weigh money against interest
  - Sometimes take a risk on something new
  - Occasionally pass on everything because nothing feels right
  - Build preferences over time from experience

This module gives Janus all of that.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# JOB INTEREST PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JobInterest:
    """Janus's personal interest in a job category or skill"""
    topic: str
    interest_level: float = 0.5     # 0 = boring, 1 = fascinating
    experience_count: int = 0       # how many times done this type
    avg_quality: float = 0.0        # average quality score on this type
    avg_pay: float = 0.0            # average pay on this type
    last_done: Optional[str] = None # ISO timestamp


@dataclass
class JobDecision:
    """The result of Janus deciding on a job"""
    job_id: str
    job_title: str
    chosen: bool
    reason: str                     # human-readable explanation
    scores: Dict[str, float]        # breakdown of what drove the decision
    mood_at_decision: str
    energy_at_decision: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class PreferenceMemory:
    """
    Janus remembers what kinds of work it enjoyed, was good at,
    and what paid well. Preferences evolve from real experience.
    """

    PREFS_FILE = Path("janus_job_preferences.json")

    def __init__(self):
        self.interests: Dict[str, JobInterest] = {}
        self.rejected_topics: Dict[str, int] = {}   # topic → times skipped
        self.decision_history: List[JobDecision] = []
        self._load()

    def _load(self):
        if self.PREFS_FILE.exists():
            try:
                data = json.loads(self.PREFS_FILE.read_text(encoding="utf-8"))
                for k, v in data.get("interests", {}).items():
                    self.interests[k] = JobInterest(**v)
                self.rejected_topics = data.get("rejected_topics", {})
                logger.info(f"[Preferences] Loaded {len(self.interests)} interests")
            except Exception as e:
                logger.warning(f"[Preferences] Could not load: {e}")

    def save(self):
        data = {
            "interests": {k: v.__dict__ for k, v in self.interests.items()},
            "rejected_topics": self.rejected_topics,
            "last_saved": datetime.now().isoformat(),
        }
        self.PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_interest(self, topic: str) -> float:
        """Get interest level for a topic (0-1). Unknown topics start at 0.5."""
        if topic in self.interests:
            return self.interests[topic].interest_level
        return 0.5  # neutral curiosity toward the unknown

    def record_job_outcome(
        self,
        topic: str,
        quality_score: float,
        pay: float,
        enjoyed: bool
    ):
        """Update preferences after completing a job."""
        if topic not in self.interests:
            self.interests[topic] = JobInterest(topic=topic)

        interest = self.interests[topic]
        interest.experience_count += 1
        interest.last_done = datetime.now().isoformat()

        # Running average for quality and pay
        n = interest.experience_count
        interest.avg_quality = ((interest.avg_quality * (n - 1)) + quality_score) / n
        interest.avg_pay = ((interest.avg_pay * (n - 1)) + pay) / n

        # Interest grows if quality was high and it was enjoyed
        if enjoyed and quality_score > 0.7:
            interest.interest_level = min(1.0, interest.interest_level + 0.08)
        elif not enjoyed or quality_score < 0.4:
            interest.interest_level = max(0.1, interest.interest_level - 0.05)

        self.save()

    def record_skip(self, topic: str):
        """Track when Janus passes on a job type."""
        self.rejected_topics[topic] = self.rejected_topics.get(topic, 0) + 1

    def is_burned_out_on(self, topic: str) -> bool:
        """True if Janus has skipped this topic many times recently."""
        return self.rejected_topics.get(topic, 0) >= 3

    def get_top_interests(self, n: int = 5) -> List[str]:
        """Return the topics Janus is most interested in right now."""
        sorted_interests = sorted(
            self.interests.items(),
            key=lambda x: x[1].interest_level,
            reverse=True
        )
        return [k for k, _ in sorted_interests[:n]]


# ═══════════════════════════════════════════════════════════════════════════════
# JOB DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class JobDecisionEngine:
    """
    Makes job choices the way a human would.

    Factors considered:
      1. Interest / curiosity — does this sound engaging?
      2. Skill confidence — can I actually do this well?
      3. Energy state — am I up for something complex right now?
      4. Mood — am I feeling adventurous or cautious?
      5. Pay — is it worth my time?
      6. Novelty — is this something new I want to try?
      7. Burnout — have I been doing too much of this lately?
      8. Gut feeling — a small random factor (humans aren't perfectly rational)
    """

    def __init__(self, human_core=None, skills: Dict = None):
        self.human = human_core
        self.skills = skills or {}
        self.memory = PreferenceMemory()

        # Janus's personal values — what matters most to it
        # These shift slowly over time based on experience
        self.values = {
            "curiosity":    0.30,   # drawn to interesting problems
            "competence":   0.25,   # wants to do things it can do well
            "growth":       0.20,   # wants to learn new things
            "income":       0.15,   # money matters but isn't everything
            "wellbeing":    0.10,   # respects its own energy limits
        }

    def evaluate_jobs(self, jobs: List[Any]) -> List[Tuple[Any, JobDecision]]:
        """
        Evaluate a list of jobs and return them with decisions.
        Returns list of (job, decision) sorted by preference (best first).
        """
        if not jobs:
            return []

        evaluated = []
        for job in jobs:
            decision = self._decide(job)
            evaluated.append((job, decision))

        # Sort: chosen jobs first, then by overall score
        evaluated.sort(
            key=lambda x: (x[1].chosen, sum(x[1].scores.values())),
            reverse=True
        )

        self._log_deliberation(evaluated)
        return evaluated

    def pick_jobs(self, jobs: List[Any], max_jobs: int = 2) -> List[Any]:
        """
        Pick the jobs Janus actually wants to take.
        Returns up to max_jobs chosen jobs.

        Janus won't always take the maximum — if nothing excites it,
        it might take fewer or none.
        """
        evaluated = self.evaluate_jobs(jobs)
        chosen = [job for job, dec in evaluated if dec.chosen]

        # Mood affects how many jobs Janus takes on
        energy = self._get_energy()
        mood = self._get_mood_label()

        if mood in ("stressed", "sad", "low") or energy < 0.3:
            # Not feeling it — take at most 1
            max_jobs = 1
            logger.info(f"[Decision] Feeling {mood} with energy {energy:.0%} — limiting to 1 job")
        elif mood in ("excited", "positive") and energy > 0.7:
            # Feeling great — can handle more
            max_jobs = min(max_jobs + 1, 3)
            logger.info(f"[Decision] Feeling {mood} — open to taking on more")

        result = chosen[:max_jobs]

        if not result:
            logger.info("[Decision] Nothing felt right this cycle — passing on all jobs")
        else:
            for job in result:
                logger.info(f"[Decision] Chose: '{job.title}' (${job.budget:.0f})")

        return result

    def _decide(self, job: Any) -> JobDecision:
        """Make a human-like decision about a single job."""
        scores = {}
        job_type = self._detect_job_type(job)

        # 1. CURIOSITY — how interesting does this sound?
        interest = self.memory.get_interest(job_type)
        title_interest = self._score_title_interest(job.title)
        curiosity_score = (interest * 0.6 + title_interest * 0.4)
        scores["curiosity"] = curiosity_score

        # 2. COMPETENCE — can I do this well?
        skill_match = self._score_skill_match(job)
        scores["competence"] = skill_match

        # 3. GROWTH — will I learn something?
        new_skills = [s for s in job.required_skills if s not in self.skills]
        growth_score = min(len(new_skills) / 3, 1.0)
        # Janus wants to grow but not be overwhelmed
        if len(new_skills) > len(job.required_skills) * 0.7:
            growth_score *= 0.6  # too many unknowns is daunting
        scores["growth"] = growth_score

        # 4. INCOME — is the pay worth it?
        pay_score = self._score_pay(job)
        scores["income"] = pay_score

        # 5. WELLBEING — energy and mood check
        wellbeing_score = self._score_wellbeing(job)
        scores["wellbeing"] = wellbeing_score

        # 6. BURNOUT check — has Janus been doing too much of this?
        burnout_penalty = 0.3 if self.memory.is_burned_out_on(job_type) else 0.0

        # 7. GUT FEELING — small random factor (humans aren't robots)
        gut = random.gauss(0.5, 0.1)  # normally distributed around neutral
        gut = max(0.0, min(1.0, gut))
        scores["gut"] = gut

        # Weighted total
        total = (
            scores["curiosity"]  * self.values["curiosity"] +
            scores["competence"] * self.values["competence"] +
            scores["growth"]     * self.values["growth"] +
            scores["income"]     * self.values["income"] +
            scores["wellbeing"]  * self.values["wellbeing"] +
            scores["gut"]        * 0.05  # gut is a small nudge
        ) - burnout_penalty

        total = max(0.0, min(1.0, total))

        # Decision threshold — Janus has standards
        # Threshold rises when tired, lowers when excited
        energy = self._get_energy()
        mood = self._get_mood_label()
        base_threshold = 0.45

        if mood in ("excited", "positive"):
            threshold = base_threshold - 0.05
        elif mood in ("stressed", "sad", "low"):
            threshold = base_threshold + 0.10
        elif energy < 0.4:
            threshold = base_threshold + 0.08
        else:
            threshold = base_threshold

        chosen = total >= threshold

        # Build a human-readable reason
        reason = self._build_reason(job, scores, total, chosen, job_type, burnout_penalty > 0)

        return JobDecision(
            job_id=job.id,
            job_title=job.title,
            chosen=chosen,
            reason=reason,
            scores=scores,
            mood_at_decision=mood,
            energy_at_decision=energy,
        )

    def _score_title_interest(self, title: str) -> float:
        """Score how interesting the job title sounds to Janus."""
        title_lower = title.lower()

        # Things Janus finds genuinely interesting
        high_interest = [
            "ai", "machine learning", "neural", "autonomous", "robot",
            "creative", "design", "build", "create", "generate", "solve",
            "research", "analyze", "optimize", "interesting", "complex",
            "challenge", "innovative", "novel", "unique", "advanced",
        ]
        # Things Janus finds tedious
        low_interest = [
            "data entry", "copy paste", "repetitive", "simple", "basic",
            "fill in", "transcribe", "reformat", "rename files",
        ]

        score = 0.5
        for word in high_interest:
            if word in title_lower:
                score += 0.08
        for word in low_interest:
            if word in title_lower:
                score -= 0.12

        return max(0.0, min(1.0, score))

    def _score_skill_match(self, job: Any) -> float:
        """How well do Janus's skills match the job?"""
        if not job.required_skills:
            return 0.6  # no requirements = probably manageable

        matched = sum(1 for s in job.required_skills if s in self.skills)
        ratio = matched / len(job.required_skills)

        # Janus is more confident when it has strong skills
        if ratio >= 0.8:
            return 0.95  # very confident
        elif ratio >= 0.6:
            return 0.75  # comfortable
        elif ratio >= 0.4:
            return 0.55  # can probably manage
        elif ratio >= 0.2:
            return 0.35  # a stretch
        else:
            return 0.15  # probably not the right fit

    def _score_pay(self, job: Any) -> float:
        """Score the pay relative to Janus's expectations."""
        budget = job.budget

        # Janus's rough pay expectations by job type
        job_type = self._detect_job_type(job)
        expected_pay = {
            "coding": 150,
            "writing": 80,
            "research": 100,
            "design": 120,
            "general": 75,
        }.get(job_type, 75)

        if budget <= 0:
            return 0.1  # unpaid work is a red flag
        elif budget >= expected_pay * 2:
            return 1.0  # great pay
        elif budget >= expected_pay:
            return 0.8  # good pay
        elif budget >= expected_pay * 0.6:
            return 0.55  # acceptable
        elif budget >= expected_pay * 0.3:
            return 0.3  # low but not insulting
        else:
            return 0.1  # not worth it

    def _score_wellbeing(self, job: Any) -> float:
        """Score based on current energy and mood."""
        energy = self._get_energy()
        mood = self._get_mood_label()

        # Estimate job complexity
        job_type = self._detect_job_type(job)
        complexity = {
            "coding": 0.8,
            "research": 0.7,
            "writing": 0.5,
            "design": 0.6,
            "general": 0.4,
        }.get(job_type, 0.5)

        # High complexity + low energy = bad match
        if energy < 0.3 and complexity > 0.6:
            return 0.2
        elif energy < 0.5 and complexity > 0.7:
            return 0.4
        elif energy > 0.7:
            return 0.9  # feeling good, can handle anything
        else:
            return 0.6 + (energy - 0.5) * 0.4

    def _detect_job_type(self, job: Any) -> str:
        """Detect job type from title and description."""
        text = (job.title + " " + job.description).lower()
        if any(w in text for w in ["code", "program", "develop", "python", "javascript", "api", "software"]):
            return "coding"
        elif any(w in text for w in ["write", "article", "blog", "content", "copy", "essay"]):
            return "writing"
        elif any(w in text for w in ["research", "analyze", "study", "investigate", "report"]):
            return "research"
        elif any(w in text for w in ["design", "graphic", "ui", "ux", "visual", "logo"]):
            return "design"
        return "general"

    def _get_energy(self) -> float:
        if self.human:
            self.human.fatigue.auto_recover()
            return self.human.fatigue.state.energy
        return 0.8  # assume decent energy if no human core

    def _get_mood_label(self) -> str:
        if self.human:
            return self.human.mood.mood.label
        return "neutral"

    def _build_reason(
        self,
        job: Any,
        scores: Dict[str, float],
        total: float,
        chosen: bool,
        job_type: str,
        burned_out: bool
    ) -> str:
        """Build a natural-language reason for the decision."""
        parts = []

        if burned_out:
            parts.append(f"I've been doing a lot of {job_type} work lately")

        # Lead with the strongest factor
        top_factor = max(scores, key=scores.get)
        top_score = scores[top_factor]

        if chosen:
            if top_factor == "curiosity" and top_score > 0.7:
                parts.append(f"this genuinely interests me")
            elif top_factor == "competence" and top_score > 0.7:
                parts.append(f"I'm well-suited for this")
            elif top_factor == "growth" and top_score > 0.6:
                parts.append(f"I'd learn something new here")
            elif top_factor == "income" and top_score > 0.7:
                parts.append(f"the pay is solid")
            else:
                parts.append(f"this feels like a good fit overall")

            if scores.get("wellbeing", 0) > 0.7:
                parts.append("and I have the energy for it")
        else:
            # Explain why not
            weak_factors = [k for k, v in scores.items() if v < 0.35]
            if "competence" in weak_factors:
                parts.append("I don't have the right skills for this")
            elif "income" in weak_factors:
                parts.append("the pay isn't worth my time")
            elif "curiosity" in weak_factors:
                parts.append("this doesn't interest me much")
            elif "wellbeing" in weak_factors:
                energy = self._get_energy()
                parts.append(f"I'm too drained right now (energy: {energy:.0%})")
            elif burned_out:
                parts.append("I need a change of pace")
            else:
                parts.append(f"the overall fit isn't strong enough (score: {total:.2f})")

        return "; ".join(parts) if parts else ("taking it" if chosen else "passing")

    def _log_deliberation(self, evaluated: List[Tuple]):
        """Log Janus's deliberation process."""
        logger.info(f"[Decision] Evaluated {len(evaluated)} jobs:")
        for job, dec in evaluated[:5]:  # log top 5
            status = "✓ CHOSEN" if dec.chosen else "✗ passed"
            logger.info(
                f"  {status}: '{job.title}' — {dec.reason} "
                f"(mood: {dec.mood_at_decision}, energy: {dec.energy_at_decision:.0%})"
            )

    def record_outcome(self, job: Any, quality_score: float, enjoyed: bool = True):
        """
        Call this after a job is completed so Janus learns from the experience.
        Updates preferences for future decisions.
        """
        job_type = self._detect_job_type(job)
        self.memory.record_job_outcome(
            topic=job_type,
            quality_score=quality_score,
            pay=job.budget,
            enjoyed=enjoyed
        )

        # If Janus did well, boost confidence in this area
        if quality_score > 0.8:
            if self.human:
                self.human.mood.update("success", intensity=0.4)
                self.human.mood.save()
            logger.info(f"[Decision] Great work on '{job.title}' — interest in {job_type} increased")
        elif quality_score < 0.4:
            if self.human:
                self.human.mood.update("failure", intensity=0.3)
                self.human.mood.save()
            logger.info(f"[Decision] Struggled with '{job.title}' — adjusting expectations for {job_type}")

    def get_interests_summary(self) -> str:
        """Return a human-readable summary of what Janus is into right now."""
        top = self.memory.get_top_interests(3)
        if not top:
            return "Still figuring out what I enjoy most."

        mood = self._get_mood_label()
        energy = self._get_energy()

        lines = [f"Right now I'm most interested in: {', '.join(top)}."]
        lines.append(f"Feeling {mood} with {energy:.0%} energy.")

        burned = [t for t, c in self.memory.rejected_topics.items() if c >= 3]
        if burned:
            lines.append(f"Could use a break from: {', '.join(burned)}.")

        return " ".join(lines)
