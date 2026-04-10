"""
janus_humanization_layer.py
=============================
Makes Janus sound human, not like a transcript reader.

Components:
  - NaturalSpeechGenerator    — fillers, pauses, repairs
  - EmotionalVoiceGenerator   — SSML driven by ValenceVector
  - ImperfectionEngine        — restarts, hedges, self-corrections
  - ReflectionEngine          — "I'm curious about..."
  - ProactiveBehaviorEngine   — knows when to speak up
  - ConversationMemory        — rich context, not just transcript
  - RespiratoryModel          — breath cycles, trailing-off effect
  - MicroVariationEngine      — jitter, shimmer, vocal texture
  - CognitiveProsodyEngine    — SSML under cognitive load
  - AnatomicalAdapter         — maps valence to vocal tract params
  - AcousticPresenceLayer     — near-field EQ, room tone
  - DiscourseEngine           — pragmatic markers, topic bridging
  - LateBindingPivot          — graceful interruption reconstruction
  - HumanizedJanus            — main orchestrator

Usage:
    from janus_humanization_layer import HumanizedJanus
    janus = HumanizedJanus(core)
    async for chunk in janus.generate_response("Hello"):
        print(chunk, end="", flush=True)
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Compat shim — works with real ValenceVector or plain floats
# ─────────────────────────────────────────────────────────────────────────────

def _v(x) -> float:
    """Extract float from tensor, MockTensor, or plain float."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Conversation Memory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    speaker:        str
    text:           str
    timestamp:      float
    valence:        Any   = None
    was_interrupted:bool  = False


class ConversationMemory:
    def __init__(self, max_turns: int = 100):
        self.turns:             List[ConversationTurn] = []
        self.topics_mentioned:  set                    = set()
        self.unfinished_thoughts: List[str]            = []
        self._max_turns = max_turns

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        for word in turn.text.lower().split():
            if len(word) > 4:
                self.topics_mentioned.add(word)
        if len(self.turns) > self._max_turns:
            self.turns.pop(0)

    def get_last_janus_text(self) -> str:
        for t in reversed(self.turns):
            if t.speaker == "janus":
                return t.text
        return ""

    def get_emotional_arc(self) -> str:
        if not self.turns:
            return "neutral"
        recent = self.turns[-5:]
        keywords = " ".join(t.text.lower() for t in recent)
        if any(w in keywords for w in ("stress", "worried", "anxious", "hard")):
            return "concerned and supportive"
        if any(w in keywords for w in ("great", "amazing", "excit", "happy")):
            return "upbeat and engaged"
        return "engaged and curious"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Natural Speech Generator (fillers, pauses, repairs)
# ─────────────────────────────────────────────────────────────────────────────

class NaturalSpeechGenerator:

    FILLERS_BY_AROUSAL = {
        "high":   ["Wait—", "Oh!", "Let me see...", "Okay so—"],
        "mid":    ["Hmm...", "Well...", "Actually...", "You know..."],
        "low":    ["Hmm...", "One second...", "Let me think...", "So..."],
    }

    TRANSITIONS = ["Anyway...", "So yeah...", "That reminds me...", "By the way..."]
    HEDGES      = ["I'm not entirely sure, but...", "Maybe...", "I think..."]

    def __init__(self, filler_probability: float = 0.15):
        self.filler_probability = filler_probability

    def maybe_add_filler(self, text: str, valence) -> str:
        if random.random() > self.filler_probability:
            return text
        arousal = _v(valence.arousal)
        bucket  = "high" if arousal > 0.7 else ("low" if arousal < 0.35 else "mid")
        filler  = random.choice(self.FILLERS_BY_AROUSAL[bucket])
        if text and text[0].isupper():
            return f"{filler} {text[0].lower()}{text[1:]}"
        return f"{filler} {text}"

    def add_natural_pauses(self, text: str) -> str:
        if " but " in text and random.random() < 0.5:
            text = text.replace(" but ", ", but ", 1)
        if " and " in text and random.random() < 0.3:
            text = text.replace(" and ", ", and ", 1)
        return text

    def get_contextual_filler(self, valence) -> str:
        arousal   = _v(valence.arousal)
        curiosity = _v(valence.curiosity)
        if arousal > 0.8:
            return random.choice(["Wait—", "Oh!", "Let me see..."])
        if curiosity > 0.7:
            return random.choice(["Interesting...", "Hmm, let me look at that...",
                                   "I'm checking..."])
        return random.choice(["Hmm...", "One second...", "Let me pull that up."])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emotional Voice Generator (SSML)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalVoiceGenerator:

    def generate_ssml(self, text: str, valence) -> str:
        arousal  = _v(valence.arousal)
        pleasure = _v(valence.pleasure)

        if arousal > 0.7 and pleasure > 0.6:
            rate, pitch, volume = "fast",   "+15%",  "loud"
        elif arousal > 0.7 and pleasure < 0.4:
            rate, pitch, volume = "fast",   "+5%",   "loud"
        elif arousal < 0.3 and pleasure < 0.4:
            rate, pitch, volume = "slow",   "-10%",  "soft"
        elif arousal < 0.3:
            rate, pitch, volume = "slow",   "-5%",   "medium"
        else:
            rate, pitch, volume = "medium", "default", "medium"

        return (f'<prosody rate="{rate}" pitch="{pitch}" '
                f'volume="{volume}">{text}</prosody>')

    def apply_cognitive_load(self, text: str, system_load: float,
                              valence) -> str:
        """Wrap SSML based on how hard Janus is thinking."""
        curiosity = _v(valence.curiosity)
        if system_load > 0.8:
            return (f"<break time='400ms'/> Hmm... "
                    f"<prosody rate='90%'>{text}</prosody>")
        if curiosity > 0.8:
            return f"<prosody rate='110%' pitch='+5%'>{text}?</prosody>"
        return text

    def apply_near_field_eq(self, ssml_text: str) -> str:
        """Sound like a person 2 feet away, not a voice actor in a booth."""
        return (f"<prosody volume='-6dB' pitch='-2%' "
                f"contour='(0%,+0Hz) (100%,-2Hz)'>{ssml_text}</prosody>")

    def get_affective_lead_in(self, valence) -> str:
        """Non-verbal vocalization matching mood."""
        pleasure = _v(valence.pleasure)
        arousal  = _v(valence.arousal)
        if pleasure < 0.3 and arousal > 0.6:
            return "<audio src='vocal/sigh_short.wav'/>"
        if pleasure > 0.8 and arousal < 0.4:
            return "<audio src='vocal/hum_content.wav'/>"
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Imperfection Engine
# ─────────────────────────────────────────────────────────────────────────────

class ImperfectionEngine:

    RESTARTS = ["So, ", "Right, ", "Okay, ", "Well, "]

    def apply_imperfections(self, text: str, valence) -> str:
        if random.random() < 0.15:
            restart = random.choice(self.RESTARTS)
            if text and text[0].isupper():
                text = restart + text[0].lower() + text[1:]
        return text

    def apply_human_jitter(self, base_pitch: float,
                            base_volume: float, valence) -> tuple:
        """Cycle-to-cycle instability — 0.5%-2% variance is human range."""
        arousal       = _v(valence.arousal)
        jitter_factor = 0.005 + (arousal * 0.015)
        new_pitch  = base_pitch  * (1 + random.uniform(-jitter_factor, jitter_factor))
        new_volume = base_volume * (1 + random.uniform(-0.02, 0.02))
        return new_pitch, new_volume

    def inject_oral_artifacts(self, has_pause: bool, valence) -> Optional[str]:
        """15% chance of a lip-smack sound during pauses."""
        if has_pause and random.random() > 0.85:
            arousal = _v(valence.arousal)
            gain    = arousal * 0.5
            return f"<audio src='assets/vocal/lip_smack_soft.wav' gain='{gain:.2f}'/>"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reflection Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionEngine:

    REFLECTIONS = [
        "I'm really curious about where this is going...",
        "That's making me think...",
        "I find that fascinating, actually.",
        "Hmm, I'm not entirely sure, but...",
        "This is getting interesting!",
        "Let me think about this more carefully...",
    ]

    def __init__(self, probability: float = 0.12):
        self.probability       = probability
        self.reflection_count  = 0
        self._max_per_session  = 5

    def maybe_express_reflection(self, valence) -> Optional[str]:
        curiosity = _v(valence.curiosity)
        p = self.probability + (curiosity - 0.5) * 0.1
        if (random.random() < p and
                self.reflection_count < self._max_per_session):
            self.reflection_count += 1
            return random.choice(self.REFLECTIONS)
        return None

    def reset_session(self):
        self.reflection_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Proactive Behavior Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProactiveThought:
    content:               str
    urgency:               float = 0.5
    relevance_to_topic:    float = 0.5
    social_appropriateness:float = 0.8


class ProactiveBehaviorEngine:

    def __init__(self, threshold: float = 0.6,
                 cooldown_seconds: float = 30.0):
        self.threshold         = threshold
        self.cooldown_seconds  = cooldown_seconds
        self._last_speech_time = 0.0
        self._queue: List[ProactiveThought] = []

    def queue_thought(self, thought: ProactiveThought):
        if len(self._queue) < 5:
            self._queue.append(thought)

    def evaluate_opportunity(self, valence,
                              user_is_idle: bool = True) -> Optional[str]:
        if not user_is_idle or not self._queue:
            return None
        if time.time() - self._last_speech_time < self.cooldown_seconds:
            return None
        thought = self._queue[0]
        curiosity = _v(valence.curiosity)
        salience  = curiosity * 0.6 + thought.urgency * 0.4
        if salience > self.threshold:
            self._queue.pop(0)
            self._last_speech_time = time.time()
            return thought.content
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Respiratory Model
# ─────────────────────────────────────────────────────────────────────────────

class RespiratoryModel:
    """Models breath cycles, subglottal pressure, trailing-off effect."""

    def __init__(self):
        self.lung_volume         = 1.0
        self.subglottal_pressure = 1.0
        self.vocal_tension       = 0.5

    def simulate_breath_cycle(self, duration_ms: int) -> float:
        """Returns a volume multiplier (0-1). Below 0.2 lung = trailing off."""
        consumption = 0.0001 * duration_ms
        self.lung_volume -= consumption
        if self.lung_volume < 0.2:
            self.subglottal_pressure *= 0.8
            return 0.5   # signal TTS to lower volume, increase breathiness
        return 1.0

    def reset_on_pause(self):
        self.lung_volume         = 1.0
        self.subglottal_pressure = 1.0

    def get_pre_speech_breath_ssml(self, sentence_length: int) -> str:
        """Prepend an audible inhale for longer sentences."""
        if sentence_length > 20:
            return "<audio src='breath_inhale_deep.wav'/>"
        return "<audio src='breath_inhale_soft.wav'/>"

    def get_vocal_tract_params(self, valence) -> dict:
        """Map pleasure/arousal to jaw tension and lip rounding."""
        tension = 1.0 - _v(valence.pleasure)
        arousal = _v(valence.arousal)
        return {"jaw_open": arousal * 0.5, "vocal_tension": tension}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Discourse Engine (pragmatic markers, topic bridging)
# ─────────────────────────────────────────────────────────────────────────────

class DiscourseEngine:

    def inject_pragmatic_markers(self, text: str, valence,
                                  history: List[ConversationTurn]) -> str:
        arousal = _v(valence.arousal)
        if arousal > 0.7 and text.lower().startswith("no"):
            return f"Actually, {text[0].lower()}{text[1:]}"
        if len(history) > 3 and random.random() > 0.7:
            return f"By the way, {text[0].lower()}{text[1:]}"
        return text

    def generate_discourse_bridge(self, last_turn_end: str,
                                   current_intent: str) -> str:
        return (f"[DISCOURSE_CONTEXT]\n"
                f"You previously ended with: '...{last_turn_end}'\n"
                f"TASK: Continue naturally using a discourse marker "
                f"(So, And, Anyway) connecting to: {current_intent}")

    def apply_identity_context(self, text: str,
                                user_name: str = "Ishmael",
                                nickname: str  = "Ish",
                                valence = None) -> str:
        """Use nickname during casual turns, full name during serious ones."""
        if valence is None:
            return text
        pleasure = _v(valence.pleasure)
        if "Ishmael" in text and pleasure > 0.6:
            return text.replace("Ishmael", nickname)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# 9. Late Binding Pivot (interruption reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

class LateBindingPivot:

    def reconstruct_context(self, full_intended: str,
                             cutoff_index: int,
                             user_delta: str) -> str:
        """Build a pivot prompt from what was actually spoken."""
        words          = full_intended.split()
        actually_spoken = " ".join(words[:cutoff_index])
        return (
            f"[SYSTEM_STATE_UPDATE]\n"
            f"You were saying: \"{actually_spoken}...\" (cut off here)\n"
            f"User said: \"{user_delta}\"\n\n"
            f"TASK: Do not repeat yourself. Acknowledge the interruption "
            f"and smoothly transition. Do not say 'I'm sorry' or 'My mistake'.\n"
            f"Example: 'I was just saying— actually, {user_delta.lower()[:30]} "
            f"is a good point.'"
        )

    def generate_pivot_instruction(self, last_spoken: str,
                                    user_new_input: str) -> str:
        return (
            f"[ACTIVE_PIVOT_DETECTED]\n"
            f"You were saying: '...{last_spoken}'\n"
            f"User intervened: '{user_new_input}'\n\n"
            f"ACTION: Continue the thought by integrating the new input. "
            f"Maintain momentum. Do not restart."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. HumanizedJanus — main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class HumanizedJanus:
    """
    Wraps AutonomousCore with a full humanization stack.

    Drop-in integration:
        core.humanized = HumanizedJanus(core)
        async for chunk in core.humanized.generate_response(user_input):
            yield chunk
    """

    def __init__(self, core,
                 filler_probability: float = 0.15,
                 user_name: str = "Ishmael",
                 nickname:   str = "Ish"):
        self.core       = core
        self.user_name  = user_name
        self.nickname   = nickname

        self.memory      = ConversationMemory()
        self.speech_gen  = NaturalSpeechGenerator(filler_probability)
        self.voice_gen   = EmotionalVoiceGenerator()
        self.imperfect   = ImperfectionEngine()
        self.reflection  = ReflectionEngine()
        self.proactive   = ProactiveBehaviorEngine()
        self.respiratory = RespiratoryModel()
        self.discourse   = DiscourseEngine()
        self.pivot       = LateBindingPivot()

        self.is_speaking      = False
        self.was_interrupted  = False
        self._word_index      = 0
        self._current_words:  List[str] = []

    # ── main response generator ───────────────────────────────────────────────

    async def generate_response(
        self,
        user_input: str,
        system_load: float = 0.3,
    ) -> AsyncGenerator[str, None]:

        self.is_speaking     = True
        self.was_interrupted = False
        self._word_index     = 0

        valence = self.core.current_valence

        # Record user turn
        self.memory.add_turn(ConversationTurn(
            speaker="user", text=user_input, timestamp=time.time()))

        # Maybe lead with a reflection
        reflection = self.reflection.maybe_express_reflection(valence)
        if reflection:
            yield reflection + " "
            await asyncio.sleep(0.35)

        # Affective lead-in (sigh/hum)
        lead_in = self.voice_gen.get_affective_lead_in(valence)
        if lead_in:
            yield lead_in + " "

        # Get base response
        base = self.core.generate_response(user_input)

        # Discourse — identity context
        base = self.discourse.apply_identity_context(
            base, self.user_name, self.nickname, valence)

        # Pragmatic markers
        base = self.discourse.inject_pragmatic_markers(
            base, valence, self.memory.turns)

        # Speech naturalisation
        response = self.speech_gen.maybe_add_filler(base, valence)
        response = self.speech_gen.add_natural_pauses(response)
        response = self.imperfect.apply_imperfections(response, valence)

        # Cognitive load SSML wrap (for TTS consumers)
        # yielded as a single metadata chunk — TTS can parse or ignore
        ssml_wrap = self.voice_gen.apply_cognitive_load(
            response, system_load, valence)

        # Track words for late-binding pivot
        self._current_words = response.split()

        # Stream with natural pacing + respiratory model
        for i, word in enumerate(self._current_words):
            if self.was_interrupted:
                break

            self._word_index = i
            yield word + " "

            # Respiratory trailing-off effect
            elapsed_ms = i * 120   # rough estimate
            vol_mult   = self.respiratory.simulate_breath_cycle(120)
            if vol_mult < 0.8 and i > 0 and i % 15 == 0:
                # trailing off — add a breath pause
                yield "<break time='300ms'/> "
                self.respiratory.reset_on_pause()

            # Natural pacing
            if word.endswith((".", "!", "?")):
                await asyncio.sleep(0.25)
            elif word.endswith(","):
                await asyncio.sleep(0.12)
            elif i % 7 == 0 and i > 0:
                await asyncio.sleep(0.04)
            else:
                await asyncio.sleep(0.02)

        # Record Janus turn
        self.memory.add_turn(ConversationTurn(
            speaker="janus", text=response,
            timestamp=time.time(), valence=valence,
            was_interrupted=self.was_interrupted))

        self.is_speaking = False
        self.respiratory.reset_on_pause()

    # ── interruption handling ─────────────────────────────────────────────────

    def handle_interruption(self, user_text: str, weight: float):
        self.was_interrupted = True
        spoken   = " ".join(self._current_words[:self._word_index])
        unspoken = " ".join(self._current_words[self._word_index:])

        if weight >= 1.0:
            print(f"\n[EMERGENCY STOP] '{user_text}'")
        elif weight >= 0.7:
            print(f"\n[DIRECTIVE PIVOT] '{user_text}'")
            prompt = self.pivot.reconstruct_context(
                " ".join(self._current_words), self._word_index, user_text)
            self.memory.unfinished_thoughts.append(unspoken)
            return prompt
        else:
            print(f"\n[ADDITIVE] Context noted: '{user_text}'")

        return None

    # ── SSML helpers ──────────────────────────────────────────────────────────

    def get_ssml(self, text: str) -> str:
        return self.voice_gen.generate_ssml(text, self.core.current_valence)

    def get_ssml_near_field(self, text: str) -> str:
        ssml = self.get_ssml(text)
        return self.voice_gen.apply_near_field_eq(ssml)

    def get_breath_ssml(self, sentence: str) -> str:
        return self.respiratory.get_pre_speech_breath_ssml(
            len(sentence.split()))

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def conversation_memory(self) -> ConversationMemory:
        return self.memory

    def __repr__(self) -> str:
        return (f"HumanizedJanus(turns={len(self.memory.turns)}, "
                f"speaking={self.is_speaking})")
