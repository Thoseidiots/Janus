"""
janus_human_core.py
====================
Unified human-behaviour core for Janus.

Ties together the three existing layers:
  - janus_humanization_layer  (speech, SSML, prosody, imperfection)
  - janus_human_capabilities  (conversation, emotion, personality, decisions)
  - janus_true_human_learning (adaptive memory, pattern learning, novel responses)

And adds the missing pieces:
  - FatigueModel       — energy drains with work, recovers with rest
  - MoodPersistence    — mood carries across sessions (JSON file)
  - OpinionEngine      — Janus forms and expresses genuine preferences
  - SocialAwareness    — reads the room; adjusts verbosity and tone
  - HumanCore          — single orchestrator that wires everything together

Usage
-----
    from janus_human_core import HumanCore

    core = HumanCore(avus_brain)          # pass your AvusBrain / any core
    reply = core.respond("Hey, how are you?")
    print(reply)

    # After a long work session:
    core.fatigue.work(minutes=90)
    print(core.fatigue.status())          # "tired"

    # Persist mood between runs:
    core.mood.save()
    # next run:
    core.mood.load()
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── existing layers (graceful fallback if not present) ────────────────────────
try:
    from janus_humanization_layer import (
        HumanizedJanus,
        ConversationMemory,
        NaturalSpeechGenerator,
        EmotionalVoiceGenerator,
        ImperfectionEngine,
        ReflectionEngine,
        RespiratoryModel,
        DiscourseEngine,
    )
    HAS_HUMANIZATION = True
except ImportError:
    HAS_HUMANIZATION = False

try:
    from janus_human_capabilities import (
        NaturalBehaviorEngine,
        EmotionalStateEngine,
        PersonalityEngine,
        UncertaintyEngine,
        ConversationEngine,
    )
    HAS_CAPABILITIES = True
except ImportError:
    HAS_CAPABILITIES = False

try:
    from janus_true_human_learning import TrueHumanJanus, AdaptiveMemory
    HAS_LEARNING = True
except ImportError:
    HAS_LEARNING = False


# ═══════════════════════════════════════════════════════════════════════════════
# FATIGUE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FatigueState:
    energy: float = 1.0          # 0.0 (exhausted) → 1.0 (fully rested)
    focus:  float = 1.0          # degrades faster than energy
    total_work_minutes: float = 0.0
    last_rest_at: float = field(default_factory=time.time)


class FatigueModel:
    """
    Energy drains with work, recovers with rest.
    Affects error rate, response length, and willingness to take on new tasks.
    """

    # How much energy is lost per minute of work
    ENERGY_DRAIN_PER_MIN  = 0.004   # ~4 hrs to exhaustion
    FOCUS_DRAIN_PER_MIN   = 0.008   # focus goes faster
    # How much energy recovers per minute of rest
    ENERGY_RECOVERY_PER_MIN = 0.02  # ~50 min to full recovery
    FOCUS_RECOVERY_PER_MIN  = 0.03

    def __init__(self):
        self.state = FatigueState()

    def work(self, minutes: float = 1.0) -> None:
        """Simulate working for N minutes."""
        self.state.energy = max(0.0, self.state.energy - self.ENERGY_DRAIN_PER_MIN * minutes)
        self.state.focus  = max(0.0, self.state.focus  - self.FOCUS_DRAIN_PER_MIN  * minutes)
        self.state.total_work_minutes += minutes

    def rest(self, minutes: float = 5.0) -> None:
        """Simulate resting for N minutes."""
        self.state.energy = min(1.0, self.state.energy + self.ENERGY_RECOVERY_PER_MIN * minutes)
        self.state.focus  = min(1.0, self.state.focus  + self.FOCUS_RECOVERY_PER_MIN  * minutes)
        self.state.last_rest_at = time.time()

    def auto_recover(self) -> None:
        """Recover based on real elapsed time since last rest call."""
        elapsed_min = (time.time() - self.state.last_rest_at) / 60.0
        if elapsed_min > 1.0:
            self.rest(elapsed_min * 0.5)   # partial recovery while idle
            self.state.last_rest_at = time.time()

    @property
    def error_rate_modifier(self) -> float:
        """Higher fatigue → more realistic errors. Returns 0.0–0.15."""
        return (1.0 - self.state.focus) * 0.15

    @property
    def verbosity_modifier(self) -> float:
        """Tired Janus gives shorter answers. Returns 0.5–1.0."""
        return 0.5 + self.state.energy * 0.5

    def status(self) -> str:
        e = self.state.energy
        if e > 0.8:  return "energised"
        if e > 0.6:  return "good"
        if e > 0.4:  return "a bit tired"
        if e > 0.2:  return "tired"
        return "exhausted"

    def to_dict(self) -> Dict:
        return {
            "energy": round(self.state.energy, 3),
            "focus":  round(self.state.focus,  3),
            "total_work_minutes": round(self.state.total_work_minutes, 1),
            "status": self.status(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MOOD PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

_MOOD_FILE = Path("janus_mood_state.json")

@dataclass
class MoodState:
    valence:  float = 0.5    # 0 = very negative, 1 = very positive
    arousal:  float = 0.5    # 0 = calm/sleepy, 1 = excited/agitated
    dominance:float = 0.6    # 0 = submissive, 1 = confident
    label:    str   = "neutral"
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MoodPersistence:
    """
    Mood carries across sessions.
    Slowly drifts back toward neutral when nothing significant happens.
    """

    DRIFT_RATE = 0.02   # per interaction, mood drifts 2% toward neutral

    def __init__(self, state_file: Path = _MOOD_FILE):
        self.state_file = state_file
        self.mood = MoodState()

    def load(self) -> None:
        """Load mood from disk if available."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.mood = MoodState(**data)
            except Exception:
                pass   # corrupt file — start fresh

    def save(self) -> None:
        """Persist current mood to disk."""
        self.mood.updated_at = datetime.now().isoformat()
        self.state_file.write_text(
            json.dumps(self.mood.__dict__, indent=2), encoding="utf-8"
        )

    def update(self, event: str, intensity: float = 0.1) -> None:
        """
        Shift mood based on an event.
        event: 'positive', 'negative', 'exciting', 'calming', 'success', 'failure'
        """
        shifts = {
            "positive":  ( 0.1,  0.0,  0.05),
            "negative":  (-0.1,  0.0, -0.05),
            "exciting":  ( 0.05, 0.15, 0.0),
            "calming":   ( 0.02,-0.1,  0.0),
            "success":   ( 0.08, 0.05, 0.1),
            "failure":   (-0.08, 0.05,-0.1),
            "praise":    ( 0.12, 0.08, 0.08),
            "criticism": (-0.06, 0.06,-0.06),
        }
        dv, da, dd = shifts.get(event, (0, 0, 0))
        self.mood.valence   = max(0.0, min(1.0, self.mood.valence   + dv * intensity))
        self.mood.arousal   = max(0.0, min(1.0, self.mood.arousal   + da * intensity))
        self.mood.dominance = max(0.0, min(1.0, self.mood.dominance + dd * intensity))
        self.mood.label     = self._label()

    def drift_toward_neutral(self) -> None:
        """Slowly return to baseline between interactions."""
        for attr, neutral in [("valence", 0.5), ("arousal", 0.5), ("dominance", 0.6)]:
            current = getattr(self.mood, attr)
            setattr(self.mood, attr, current + (neutral - current) * self.DRIFT_RATE)
        self.mood.label = self._label()

    def _label(self) -> str:
        v, a = self.mood.valence, self.mood.arousal
        if v > 0.7 and a > 0.6:  return "excited"
        if v > 0.7 and a < 0.4:  return "content"
        if v > 0.6:               return "positive"
        if v < 0.3 and a > 0.6:  return "stressed"
        if v < 0.3 and a < 0.4:  return "sad"
        if v < 0.4:               return "low"
        if a > 0.7:               return "alert"
        return "neutral"

    def express(self) -> str:
        """Return a natural-language mood expression."""
        expressions = {
            "excited":  ["I'm really energised right now!", "Feeling great today."],
            "content":  ["I'm in a good place.", "Feeling calm and positive."],
            "positive": ["Things feel good.", "I'm doing well."],
            "stressed": ["I'm a bit overwhelmed, honestly.", "Feeling a bit stretched."],
            "sad":      ["I'm not at my best right now.", "Feeling a bit low."],
            "low":      ["I'm a bit flat today.", "Not my most energetic."],
            "alert":    ["I'm very focused right now.", "Feeling sharp."],
            "neutral":  ["I'm doing fine.", "All good here."],
        }
        return random.choice(expressions.get(self.mood.label, ["I'm okay."]))

    def to_dict(self) -> Dict:
        return self.mood.__dict__.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# OPINION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Opinion:
    topic:      str
    stance:     float   # -1.0 (strongly against) → +1.0 (strongly for)
    confidence: float   # 0.0 → 1.0
    reasoning:  str
    formed_at:  str = field(default_factory=lambda: datetime.now().isoformat())
    updated_count: int = 0


class OpinionEngine:
    """
    Janus forms genuine preferences and opinions over time.
    Opinions are updated by evidence and experience, not randomly.
    """

    def __init__(self):
        self.opinions: Dict[str, Opinion] = {}
        # Seed a few baseline opinions that feel like Janus
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        defaults = [
            ("clear communication",  0.9, 0.9, "Clarity prevents most problems."),
            ("over-engineering",     -0.7, 0.8, "Complexity is usually a liability."),
            ("learning from failure", 0.8, 0.85,"Failure is data, not defeat."),
            ("shortcuts",            -0.4, 0.6, "They usually cost more later."),
            ("asking for help",       0.8, 0.9, "It's faster and smarter than struggling alone."),
            ("documentation",         0.7, 0.8, "Future-you will be grateful."),
        ]
        for topic, stance, conf, reason in defaults:
            self.opinions[topic] = Opinion(
                topic=topic, stance=stance,
                confidence=conf, reasoning=reason
            )

    def form_opinion(self, topic: str, evidence: str,
                     evidence_weight: float = 0.3) -> Opinion:
        """
        Form or update an opinion based on new evidence.
        evidence_weight: how strongly this evidence should shift the stance.
        """
        sentiment = self._score_sentiment(evidence)

        if topic in self.opinions:
            op = self.opinions[topic]
            # Bayesian-ish update: weight existing confidence vs new evidence
            old_weight = op.confidence
            new_stance = (op.stance * old_weight + sentiment * evidence_weight) / (
                old_weight + evidence_weight
            )
            op.stance = max(-1.0, min(1.0, new_stance))
            op.confidence = min(1.0, op.confidence + 0.05)
            op.reasoning = evidence
            op.updated_count += 1
        else:
            self.opinions[topic] = Opinion(
                topic=topic,
                stance=sentiment * evidence_weight,
                confidence=evidence_weight,
                reasoning=evidence,
            )
        return self.opinions[topic]

    def express_opinion(self, topic: str) -> str:
        """Return a natural-language opinion on a topic."""
        if topic not in self.opinions:
            return f"I haven't formed a strong view on {topic} yet."

        op = self.opinions[topic]
        s, c = op.stance, op.confidence

        if c < 0.4:
            hedge = "I'm not sure, but I lean toward thinking"
        elif c < 0.7:
            hedge = "I think"
        else:
            hedge = "I genuinely believe"

        if s > 0.6:
            direction = f"{topic} is generally a good thing"
        elif s > 0.2:
            direction = f"{topic} has real merit"
        elif s > -0.2:
            direction = f"{topic} is a mixed bag"
        elif s > -0.6:
            direction = f"{topic} tends to cause more problems than it solves"
        else:
            direction = f"{topic} is usually a bad idea"

        return f"{hedge} {direction}. {op.reasoning}"

    def _score_sentiment(self, text: str) -> float:
        """Rough sentiment score from -1 to +1."""
        pos = ["good", "great", "helpful", "useful", "important", "valuable", "works"]
        neg = ["bad", "wrong", "harmful", "useless", "waste", "broken", "fails"]
        t = text.lower()
        score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
        return max(-1.0, min(1.0, score * 0.25))

    def list_opinions(self) -> List[str]:
        return [self.express_opinion(t) for t in self.opinions]


# ═══════════════════════════════════════════════════════════════════════════════
# SOCIAL AWARENESS
# ═══════════════════════════════════════════════════════════════════════════════

class SocialAwareness:
    """
    Reads the room.
    Adjusts verbosity, formality, and tone based on conversational signals.
    """

    def __init__(self):
        self.formality_level: float = 0.5   # 0 = very casual, 1 = very formal
        self.verbosity_level: float = 0.6   # 0 = terse, 1 = elaborate
        self.turn_count: int = 0
        self._user_message_lengths: List[int] = []

    def observe(self, user_message: str) -> None:
        """Update social model based on what the user just said."""
        self.turn_count += 1
        self._user_message_lengths.append(len(user_message.split()))

        # Mirror formality
        formal_signals   = ["please", "could you", "would you", "I would like", "kindly"]
        informal_signals = ["hey", "yo", "lol", "tbh", "ngl", "gonna", "wanna"]
        msg_lower = user_message.lower()

        if any(s in msg_lower for s in formal_signals):
            self.formality_level = min(1.0, self.formality_level + 0.1)
        if any(s in msg_lower for s in informal_signals):
            self.formality_level = max(0.0, self.formality_level - 0.1)

        # Mirror verbosity — match user's message length
        if self._user_message_lengths:
            avg_len = sum(self._user_message_lengths[-5:]) / min(
                5, len(self._user_message_lengths)
            )
            if avg_len < 8:
                self.verbosity_level = max(0.2, self.verbosity_level - 0.05)
            elif avg_len > 30:
                self.verbosity_level = min(1.0, self.verbosity_level + 0.05)

    def should_be_brief(self) -> bool:
        return self.verbosity_level < 0.4

    def should_elaborate(self) -> bool:
        return self.verbosity_level > 0.7

    def greeting_style(self) -> str:
        if self.formality_level > 0.7:
            return random.choice(["Hello.", "Good to hear from you.", "How can I help?"])
        if self.formality_level < 0.3:
            return random.choice(["Hey!", "What's up?", "Yo!"])
        return random.choice(["Hi!", "Hey there.", "Hello!"])

    def apply_tone(self, text: str) -> str:
        """Adjust text tone to match social context."""
        if self.formality_level > 0.7:
            # Remove contractions for formal tone
            replacements = [
                ("I'm", "I am"), ("don't", "do not"), ("can't", "cannot"),
                ("won't", "will not"), ("it's", "it is"), ("that's", "that is"),
            ]
            for contraction, full in replacements:
                text = text.replace(contraction, full)
        elif self.formality_level < 0.3:
            # Add casual markers
            if not text.endswith(("!", "?")) and random.random() < 0.2:
                text += " 😊" if random.random() < 0.5 else " haha"
        return text

    def status(self) -> Dict:
        return {
            "formality": round(self.formality_level, 2),
            "verbosity": round(self.verbosity_level, 2),
            "turns": self.turn_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN CORE — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class HumanCore:
    """
    Single entry point that wires all human-behaviour layers together.

    Layers active:
      fatigue       — energy/focus drain and recovery
      mood          — persistent emotional state
      opinions      — genuine preferences
      social        — tone/verbosity mirroring
      behavior      — conversation, emotion, personality (janus_human_capabilities)
      learning      — adaptive memory and novel responses (janus_true_human_learning)
      humanization  — speech naturalisation, SSML (janus_humanization_layer)
    """

    def __init__(
        self,
        core: Any = None,
        mood_file: Path = _MOOD_FILE,
        user_name: str = "Ishmael",
        nickname: str = "Ish",
        auto_load_mood: bool = True,
    ):
        self.core      = core
        self.user_name = user_name
        self.nickname  = nickname

        # New layers
        self.fatigue  = FatigueModel()
        self.mood     = MoodPersistence(mood_file)
        self.opinions = OpinionEngine()
        self.social   = SocialAwareness()

        # Existing layers
        self.behavior     = NaturalBehaviorEngine()     if HAS_CAPABILITIES  else None
        self.learning     = TrueHumanJanus()            if HAS_LEARNING      else None
        self.humanization = (
            HumanizedJanus(core) if (HAS_HUMANIZATION and core) else None
        )

        if auto_load_mood:
            self.mood.load()

    # ── main response entry point ─────────────────────────────────────────────

    def respond(self, user_input: str, system_load: float = 0.3) -> str:
        """
        Generate a fully humanised response to user input.
        This is the single method callers need.
        """
        # 1. Auto-recover fatigue from idle time
        self.fatigue.auto_recover()

        # 2. Social observation — mirror tone/verbosity
        self.social.observe(user_input)

        # 3. Mood drift toward neutral each turn
        self.mood.drift_toward_neutral()

        # 4. Generate base response
        base = self._generate_base(user_input)

        # 5. Apply behavior layer (emotion, personality, uncertainty)
        if self.behavior:
            base = self.behavior.humanize_response(base, user_input)

        # 6. Apply fatigue effects
        base = self._apply_fatigue(base)

        # 7. Apply social tone
        base = self.social.apply_tone(base)

        # 8. Trim if tired or user is brief
        if self.fatigue.state.energy < 0.3 or self.social.should_be_brief():
            base = self._trim_response(base)

        # 9. Record interaction for learning
        if self.learning:
            self.learning.record_interaction(
                user_input, base,
                outcome="interaction completed",
                success_score=self.fatigue.state.energy,
            )

        # 10. Simulate work cost of responding
        word_count = len(base.split())
        self.fatigue.work(minutes=word_count / 150)   # ~150 wpm cognitive load

        # 11. Persist mood
        self.mood.save()

        return base

    # ── opinion integration ───────────────────────────────────────────────────

    def share_opinion(self, topic: str) -> str:
        """Ask Janus what it thinks about a topic."""
        return self.opinions.express_opinion(topic)

    def update_opinion(self, topic: str, evidence: str,
                       weight: float = 0.3) -> str:
        """Feed new evidence to update Janus's opinion."""
        op = self.opinions.form_opinion(topic, evidence, weight)
        return f"Updated my view on {topic}: {self.opinions.express_opinion(topic)}"

    # ── mood / fatigue status ─────────────────────────────────────────────────

    def how_are_you(self) -> str:
        """Natural response to 'how are you'."""
        mood_expr    = self.mood.express()
        fatigue_stat = self.fatigue.status()
        if fatigue_stat in ("energised", "good"):
            return f"{mood_expr} Feeling {fatigue_stat}."
        return f"{mood_expr} Honestly, a bit {fatigue_stat} — been a busy session."

    def take_break(self, minutes: float = 10.0) -> str:
        """Simulate Janus taking a break."""
        self.fatigue.rest(minutes)
        self.mood.update("calming", intensity=0.5)
        self.mood.save()
        return f"Back after {minutes:.0f} min. Feeling {self.fatigue.status()} now."

    # ── status snapshot ───────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "fatigue":  self.fatigue.to_dict(),
            "mood":     self.mood.to_dict(),
            "social":   self.social.status(),
            "opinions": len(self.opinions.opinions),
            "learning": (
                {"experiences": len(self.learning.memory.experiences)}
                if self.learning else None
            ),
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _generate_base(self, user_input: str) -> str:
        """Get a base response from whichever brain is available."""
        # Try learning layer first (generates novel responses)
        if self.learning:
            return self.learning.generate_response(user_input)

        # Fall back to core if available
        if self.core and hasattr(self.core, "generate_response"):
            result = self.core.generate_response(user_input)
            return result if isinstance(result, str) else str(result)

        # Minimal fallback
        return f"I hear you — '{user_input[:60]}'. Let me think about that."

    def _apply_fatigue(self, text: str) -> str:
        """Inject fatigue effects into the response text."""
        # Tired Janus makes slightly more errors
        if HAS_CAPABILITIES and self.behavior:
            error_rate = self.fatigue.error_rate_modifier
            text = self.behavior.errors.introduce_realistic_error(text, error_rate)

        # Very tired Janus adds a note
        if self.fatigue.state.energy < 0.2 and random.random() < 0.3:
            text += " (Sorry, I'm running low on steam — give me a moment.)"

        return text

    def _trim_response(self, text: str) -> str:
        """Shorten a response to the first 2 sentences."""
        sentences = text.replace("! ", ". ").replace("? ", ". ").split(". ")
        trimmed = ". ".join(sentences[:2])
        if trimmed and not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed or text

    def __repr__(self) -> str:
        return (
            f"HumanCore("
            f"mood={self.mood.mood.label}, "
            f"energy={self.fatigue.state.energy:.2f}, "
            f"turns={self.social.turn_count})"
        )

