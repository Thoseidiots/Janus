"""
skill_curriculum.py
===================
Adaptive training curriculum driven by a skill tree.

Concept:
  - Skills are organized in tiers. Higher-tier skills are more valuable.
  - Skills compound: improving Reasoning lifts Planning and Code.
  - Diminishing returns: going from 0.8->0.9 on a maxed skill yields
    less benefit than going from 0.2->0.3 on a weak one.
  - Unlock gates: a Tier-2 skill only becomes trainable once its
    Tier-1 prerequisites cross a threshold.
  - The curriculum always trains the skill with the highest
    marginal value — not just the weakest.

Usage:
    from skill_curriculum import SkillTree, CurriculumSampler

    tree    = SkillTree()
    sampler = CurriculumSampler(tree)

    # After each eval step, update skill confidence scores
    tree.update("reasoning", confidence=0.45)
    tree.update("language",  confidence=0.72)

    # Get the next domain to train on
    domain = sampler.next_domain()

    # Get a weighted data mixture for this step
    weights = sampler.get_mixture_weights()

    # Visualize the skill hexagon
    tree.plot("skill_chart.png")
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Skill definition ──────────────────────────────────────────────────────────

@dataclass
class Skill:
    """
    A single trainable skill node in the skill tree.

    tier        : 1 = foundational, 2 = intermediate, 3 = advanced, 4 = elite
    value       : base value multiplier — higher tier = higher value
    prerequisites: skill names that must reach unlock_threshold before this
                   skill becomes active
    unlock_threshold: confidence level prereqs must hit to unlock this skill
    max_benefit : confidence level above which marginal gain drops sharply
                  (diminishing returns kicks in hard past this point)
    compounds   : skills that receive a passive boost when this skill improves
    """
    name:             str
    tier:             int
    value:            float
    prerequisites:    List[str]   = field(default_factory=list)
    unlock_threshold: float       = 0.4
    max_benefit:      float       = 0.85
    compounds:        List[str]   = field(default_factory=list)
    dataset_key:      str         = ""   # maps to a data generator / file

    # Runtime state
    confidence:       float       = 0.0
    trained_steps:    int         = 0
    unlocked:         bool        = False
    history:          List[float] = field(default_factory=list)

    def marginal_value(self) -> float:
        """
        How much is it worth to train this skill right now?

        Formula:
          base_value  = tier_value * (1 - confidence)   # more room = more value
          tier_bonus  = tier^1.5                         # higher tier = more valuable
          diminishing = 1 - sigmoid((confidence - max_benefit) * 12)
                        # drops sharply past max_benefit
          unlock_mult = 1.0 if unlocked else 0.0         # locked = no value

        The result is the expected training ROI for this skill at this moment.
        """
        if not self.unlocked:
            return 0.0

        room        = max(0.0, 1.0 - self.confidence)
        tier_bonus  = self.tier ** 1.5
        diminishing = 1.0 - _sigmoid((self.confidence - self.max_benefit) * 12)
        base        = self.value * room * tier_bonus * diminishing

        return base

    def update(self, confidence: float):
        self.history.append(confidence)
        self.confidence = confidence
        self.trained_steps += 1

    def is_near_cap(self) -> bool:
        return self.confidence >= self.max_benefit * 0.95

    def __repr__(self) -> str:
        lock = "✓" if self.unlocked else "✗"
        return (f"Skill({self.name} T{self.tier} "
                f"conf={self.confidence:.2f} mv={self.marginal_value():.3f} {lock})")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Skill tree ────────────────────────────────────────────────────────────────

class SkillTree:
    """
    The full skill tree for Avus.

    Tier 1 — Foundational (always unlocked)
      language, arithmetic, memory_recall, pattern_recognition

    Tier 2 — Intermediate (unlocks when T1 prereqs >= 0.4)
      reasoning, code, vision_understanding, speech_comprehension

    Tier 3 — Advanced (unlocks when T2 prereqs >= 0.5)
      planning, multimodal_fusion, self_reflection, tool_use

    Tier 4 — Elite (unlocks when T3 prereqs >= 0.65)
      autonomous_agency, meta_learning, world_modeling
    """

    DEFAULT_SKILLS = [
        # ── Tier 1: Foundational ──────────────────────────────────────────────
        Skill("language",           tier=1, value=1.0,
              prerequisites=[], unlock_threshold=0.0,
              max_benefit=0.80,
              compounds=["reasoning", "speech_comprehension", "planning"],
              dataset_key="language"),

        Skill("arithmetic",         tier=1, value=1.0,
              prerequisites=[], unlock_threshold=0.0,
              max_benefit=0.85,
              compounds=["reasoning", "code", "world_modeling"],
              dataset_key="arithmetic"),

        Skill("memory_recall",      tier=1, value=1.0,
              prerequisites=[], unlock_threshold=0.0,
              max_benefit=0.80,
              compounds=["planning", "self_reflection", "meta_learning"],
              dataset_key="memory"),

        Skill("pattern_recognition",tier=1, value=1.0,
              prerequisites=[], unlock_threshold=0.0,
              max_benefit=0.80,
              compounds=["code", "vision_understanding", "world_modeling"],
              dataset_key="pattern"),

        # ── Tier 2: Intermediate ──────────────────────────────────────────────
        Skill("reasoning",          tier=2, value=2.0,
              prerequisites=["language", "arithmetic"],
              unlock_threshold=0.4,
              max_benefit=0.82,
              compounds=["planning", "self_reflection", "meta_learning"],
              dataset_key="reasoning"),

        Skill("code",               tier=2, value=2.0,
              prerequisites=["arithmetic", "pattern_recognition"],
              unlock_threshold=0.4,
              max_benefit=0.85,
              compounds=["tool_use", "autonomous_agency"],
              dataset_key="code"),

        Skill("vision_understanding",tier=2, value=2.0,
              prerequisites=["pattern_recognition"],
              unlock_threshold=0.35,
              max_benefit=0.80,
              compounds=["multimodal_fusion", "world_modeling"],
              dataset_key="vision"),

        Skill("speech_comprehension",tier=2, value=1.8,
              prerequisites=["language"],
              unlock_threshold=0.4,
              max_benefit=0.80,
              compounds=["multimodal_fusion", "autonomous_agency"],
              dataset_key="speech"),

        # ── Tier 3: Advanced ──────────────────────────────────────────────────
        Skill("planning",           tier=3, value=3.5,
              prerequisites=["reasoning", "memory_recall"],
              unlock_threshold=0.5,
              max_benefit=0.78,
              compounds=["autonomous_agency", "meta_learning"],
              dataset_key="planning"),

        Skill("multimodal_fusion",  tier=3, value=3.0,
              prerequisites=["vision_understanding", "speech_comprehension"],
              unlock_threshold=0.5,
              max_benefit=0.75,
              compounds=["world_modeling", "autonomous_agency"],
              dataset_key="multimodal"),

        Skill("self_reflection",    tier=3, value=3.2,
              prerequisites=["reasoning", "memory_recall"],
              unlock_threshold=0.5,
              max_benefit=0.75,
              compounds=["meta_learning", "autonomous_agency"],
              dataset_key="reflection"),

        Skill("tool_use",           tier=3, value=3.0,
              prerequisites=["code", "reasoning"],
              unlock_threshold=0.5,
              max_benefit=0.80,
              compounds=["autonomous_agency"],
              dataset_key="tool_use"),

        # ── Tier 4: Elite ─────────────────────────────────────────────────────
        Skill("autonomous_agency",  tier=4, value=6.0,
              prerequisites=["planning", "tool_use", "self_reflection"],
              unlock_threshold=0.65,
              max_benefit=0.70,
              compounds=["meta_learning"],
              dataset_key="autonomy"),

        Skill("meta_learning",      tier=4, value=5.5,
              prerequisites=["self_reflection", "planning"],
              unlock_threshold=0.65,
              max_benefit=0.70,
              compounds=[],
              dataset_key="meta"),

        Skill("world_modeling",     tier=4, value=5.0,
              prerequisites=["multimodal_fusion", "reasoning"],
              unlock_threshold=0.65,
              max_benefit=0.70,
              compounds=["autonomous_agency"],
              dataset_key="world_model"),
    ]

    def __init__(self, skills: Optional[List[Skill]] = None):
        self.skills: Dict[str, Skill] = {}
        for s in (skills or self.DEFAULT_SKILLS):
            self.skills[s.name] = s
        # Tier 1 always unlocked
        for s in self.skills.values():
            if s.tier == 1:
                s.unlocked = True
        self._check_unlocks()
        self._history: List[Dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, skill_name: str, confidence: float,
               propagate_compounds: bool = True):
        """
        Update a skill's confidence score.
        Optionally propagates a compound bonus to connected skills.
        """
        if skill_name not in self.skills:
            return
        skill = self.skills[skill_name]
        old_conf = skill.confidence
        skill.update(confidence)

        # Compound propagation: improving a skill gives a small lift to
        # connected skills (simulates knowledge transfer)
        if propagate_compounds and confidence > old_conf:
            delta = (confidence - old_conf) * 0.08  # 8% compound rate
            for compound_name in skill.compounds:
                if compound_name in self.skills:
                    target = self.skills[compound_name]
                    if target.unlocked:
                        new_conf = min(1.0, target.confidence + delta)
                        target.update(new_conf)

        self._check_unlocks()
        self._log_snapshot()

    def get_marginal_values(self) -> Dict[str, float]:
        """Return marginal training value for every unlocked skill."""
        return {
            name: skill.marginal_value()
            for name, skill in self.skills.items()
            if skill.unlocked
        }

    def best_skill_to_train(self) -> str:
        """Return the skill with the highest marginal value right now."""
        mvs = self.get_marginal_values()
        if not mvs:
            return "language"
        return max(mvs, key=lambda k: mvs[k])

    def get_mixture_weights(self) -> Dict[str, float]:
        """
        Return a normalized data mixture weight per dataset_key.
        Skills near cap get very little weight.
        Locked skills get zero weight.
        """
        mvs = self.get_marginal_values()
        total = sum(mvs.values())
        if total == 0:
            return {}

        weights: Dict[str, float] = {}
        for name, mv in mvs.items():
            key = self.skills[name].dataset_key or name
            weights[key] = weights.get(key, 0.0) + mv / total

        return weights

    def skill_summary(self) -> str:
        """Human-readable summary of all skills."""
        tiers = {}
        for s in self.skills.values():
            tiers.setdefault(s.tier, []).append(s)

        lines = ["Skill Tree Status", "=" * 50]
        for tier in sorted(tiers):
            lines.append(f"\nTier {tier}:")
            for s in sorted(tiers[tier], key=lambda x: -x.confidence):
                bar = _progress_bar(s.confidence)
                lock = "UNLOCKED" if s.unlocked else "LOCKED  "
                cap  = " [NEAR CAP]" if s.is_near_cap() else ""
                mv   = s.marginal_value()
                lines.append(
                    f"  {lock} {s.name:<22} {bar} {s.confidence:.2f}"
                    f"  MV={mv:.3f}{cap}"
                )
        return "\n".join(lines)

    def save(self, path: str):
        """Save skill tree state to JSON."""
        data = {
            name: {
                "confidence":    s.confidence,
                "trained_steps": s.trained_steps,
                "unlocked":      s.unlocked,
                "history":       s.history[-50:],  # keep last 50
            }
            for name, s in self.skills.items()
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str):
        """Load skill tree state from JSON."""
        if not Path(path).exists():
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, state in data.items():
            if name in self.skills:
                s = self.skills[name]
                s.confidence    = state.get("confidence", 0.0)
                s.trained_steps = state.get("trained_steps", 0)
                s.unlocked      = state.get("unlocked", s.tier == 1)
                s.history       = state.get("history", [])
        self._check_unlocks()

    def plot(self, save_path: Optional[str] = None, show: bool = False):
        """
        Render the skill hexagon radar chart.
        Shows all unlocked skills as a radar/spider chart.
        Locked skills shown as grey outlines.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            print("[SkillTree] matplotlib not installed — skipping plot")
            return

        skills_list = list(self.skills.values())
        names  = [s.name.replace("_", "\n") for s in skills_list]
        confs  = [s.confidence for s in skills_list]
        mvs    = [s.marginal_value() for s in skills_list]
        tiers  = [s.tier for s in skills_list]
        locked = [not s.unlocked for s in skills_list]

        n = len(skills_list)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        confs_plot = confs + confs[:1]
        mvs_plot   = mvs   + mvs[:1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                        subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0d0d0d")

        tier_colors = {1: "#4fc3f7", 2: "#81c784", 3: "#ffb74d", 4: "#e57373"}

        # ── Left: Confidence radar ────────────────────────────────────────────
        ax1.set_facecolor("#1a1a2e")
        ax1.set_title("Skill Confidence", color="white", pad=20, fontsize=13)

        for i, (angle, conf, tier, lock) in enumerate(
                zip(angles[:-1], confs, tiers, locked)):
            color = "#333333" if lock else tier_colors.get(tier, "white")
            ax1.plot([angle, angle], [0, conf], color=color, linewidth=2, alpha=0.6)
            ax1.scatter([angle], [conf], color=color, s=60, zorder=5)

        ax1.plot(angles, confs_plot, color="#4fc3f7", linewidth=2, alpha=0.8)
        ax1.fill(angles, confs_plot, color="#4fc3f7", alpha=0.15)

        # Max benefit ring
        max_benefits = [s.max_benefit for s in skills_list] + [skills_list[0].max_benefit]
        ax1.plot(angles, max_benefits, color="#ff6b6b", linewidth=1,
                 linestyle="--", alpha=0.4, label="Diminishing returns")

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(names, color="white", fontsize=7)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax1.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                             color="#888888", fontsize=7)
        ax1.grid(color="#333333", linewidth=0.5)
        ax1.spines["polar"].set_color("#333333")

        # ── Right: Marginal value radar ───────────────────────────────────────
        ax2.set_facecolor("#1a1a2e")
        ax2.set_title("Marginal Training Value\n(where to train next)",
                      color="white", pad=20, fontsize=13)

        max_mv = max(mvs) if max(mvs) > 0 else 1.0
        mvs_norm = [v / max_mv for v in mvs]
        mvs_norm_plot = mvs_norm + mvs_norm[:1]

        for i, (angle, mv, tier, lock) in enumerate(
                zip(angles[:-1], mvs_norm, tiers, locked)):
            color = "#333333" if lock else tier_colors.get(tier, "white")
            ax2.plot([angle, angle], [0, mv], color=color, linewidth=2, alpha=0.6)
            ax2.scatter([angle], [mv], color=color, s=60, zorder=5)

        ax2.plot(angles, mvs_norm_plot, color="#ffb74d", linewidth=2, alpha=0.8)
        ax2.fill(angles, mvs_norm_plot, color="#ffb74d", alpha=0.15)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(names, color="white", fontsize=7)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(["25%", "50%", "75%", "100%"],
                             color="#888888", fontsize=7)
        ax2.grid(color="#333333", linewidth=0.5)
        ax2.spines["polar"].set_color("#333333")

        # ── Legend ────────────────────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=tier_colors[t], label=f"Tier {t}")
            for t in sorted(tier_colors)
        ]
        legend_patches.append(
            mpatches.Patch(color="#333333", label="Locked")
        )
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=5, facecolor="#1a1a2e", labelcolor="white",
                   fontsize=9, framealpha=0.8)

        # ── Best skill annotation ─────────────────────────────────────────────
        best = self.best_skill_to_train()
        fig.suptitle(
            f"Avus Skill Tree  |  Train next: {best.upper().replace('_', ' ')}",
            color="white", fontsize=14, y=0.98
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[SkillTree] Chart saved -> {save_path}")
        if show:
            plt.show()
        plt.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_unlocks(self):
        """Unlock skills whose prerequisites are met."""
        changed = True
        while changed:
            changed = False
            for skill in self.skills.values():
                if skill.unlocked:
                    continue
                if all(
                    self.skills[p].confidence >= skill.unlock_threshold
                    for p in skill.prerequisites
                    if p in self.skills
                ):
                    skill.unlocked = True
                    changed = True
                    print(f"[SkillTree] UNLOCKED: {skill.name} (Tier {skill.tier})")

    def _log_snapshot(self):
        self._history.append({
            "time": time.time(),
            "scores": {n: s.confidence for n, s in self.skills.items()},
            "best": self.best_skill_to_train(),
        })
        if len(self._history) > 1000:
            self._history = self._history[-500:]


# ── Curriculum sampler ────────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Drives the training data mixture based on the skill tree.

    Hysteresis:
      - LOW_THRESHOLD:  if best skill's MV drops below this, re-evaluate
      - HIGH_THRESHOLD: keep training current skill until MV drops to this

    This prevents thrashing between domains every step.
    """

    LOW_THRESHOLD  = 0.05   # re-evaluate when current skill MV drops here
    HIGH_THRESHOLD = 0.15   # switch away when MV drops below this

    def __init__(self, tree: SkillTree):
        self.tree            = tree
        self._current_domain = tree.best_skill_to_train()
        self._steps_on_domain = 0
        self._switch_log: List[Tuple[int, str, str]] = []
        self._total_steps = 0

    def next_domain(self) -> str:
        """
        Return the domain to train on this step.
        Switches when current skill's marginal value drops below HIGH_THRESHOLD
        or a higher-value skill becomes available.
        """
        self._total_steps += 1
        self._steps_on_domain += 1

        current_mv = self.tree.skills.get(
            self._current_domain,
            Skill("_", tier=1, value=0)
        ).marginal_value()

        best        = self.tree.best_skill_to_train()
        best_mv     = self.tree.skills[best].marginal_value()

        # Switch if: current skill is near cap OR a much better skill exists
        should_switch = (
            current_mv < self.HIGH_THRESHOLD or
            (best != self._current_domain and best_mv > current_mv * 1.5)
        )

        if should_switch and best != self._current_domain:
            self._switch_log.append(
                (self._total_steps, self._current_domain, best)
            )
            print(f"[Curriculum] Step {self._total_steps}: "
                  f"{self._current_domain} (MV={current_mv:.3f}) -> "
                  f"{best} (MV={best_mv:.3f})")
            self._current_domain  = best
            self._steps_on_domain = 0

        return self._current_domain

    def get_mixture_weights(self) -> Dict[str, float]:
        """
        Soft mixture: primary domain gets 60%, rest split by MV.
        This keeps some diversity while focusing on the priority skill.
        """
        weights = self.tree.get_mixture_weights()
        if not weights:
            return {"language": 1.0}

        primary_key = self.tree.skills[self._current_domain].dataset_key
        if not primary_key:
            primary_key = self._current_domain

        # Boost primary to 60%
        boosted: Dict[str, float] = {}
        for key, w in weights.items():
            if key == primary_key:
                boosted[key] = 0.60
            else:
                boosted[key] = w * 0.40 / max(
                    sum(v for k, v in weights.items() if k != primary_key), 1e-8
                )

        # Normalize
        total = sum(boosted.values())
        return {k: v / total for k, v in boosted.items()}

    def report(self) -> str:
        lines = [
            f"Curriculum Report — {self._total_steps} total steps",
            f"Current domain: {self._current_domain} "
            f"({self._steps_on_domain} steps)",
            f"Domain switches: {len(self._switch_log)}",
        ]
        if self._switch_log:
            lines.append("Last 5 switches:")
            for step, frm, to in self._switch_log[-5:]:
                lines.append(f"  step {step}: {frm} -> {to}")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ── Singleton ─────────────────────────────────────────────────────────────────

_tree:    Optional[SkillTree]        = None
_sampler: Optional[CurriculumSampler] = None

def get_skill_tree() -> SkillTree:
    global _tree
    if _tree is None:
        _tree = SkillTree()
        state_path = Path("skill_state.json")
        if state_path.exists():
            _tree.load(str(state_path))
    return _tree

def get_sampler() -> CurriculumSampler:
    global _sampler
    if _sampler is None:
        _sampler = CurriculumSampler(get_skill_tree())
    return _sampler


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    tree    = SkillTree()
    sampler = CurriculumSampler(tree)

    print(tree.skill_summary())
    print()

    # Simulate training — confidence rises gradually on trained domains
    print("Simulating 200 training steps...\n")
    for step in range(200):
        domain = sampler.next_domain()
        skill  = tree.skills.get(domain)
        if skill:
            # Simulate confidence gain: diminishing returns naturally
            gain = 0.02 * (1.0 - skill.confidence) * random.uniform(0.5, 1.5)
            tree.update(domain, min(1.0, skill.confidence + gain))

        if step % 40 == 39:
            print(f"\n--- Step {step+1} ---")
            print(tree.skill_summary())
            print()

    print("\n" + sampler.report())
    tree.plot("skill_chart.png")
    tree.save("skill_state.json")
    print("\nDone. Check skill_chart.png")
