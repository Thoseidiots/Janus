# Repository Modo

This is how Janus is built. Not a methodology borrowed from a textbook — a pattern that emerged from actually building it.

---

## The Loop

```
CONCEPT
    ↓
THEORY
    ↓
GROUNDED IMPLEMENTATION
    ↓
THEORY AGAIN
    ↓
CONCEPT AGAIN
    ↓
FINISHED PRODUCT
(something real that's better than what most companies have)
```

Each pass through the loop tightens the gap between what was imagined and what actually works.

---

## What Each Stage Means

**CONCEPT**
An instinct. A direction. Not fully formed.
"Train on weakness." "Memory should feel like something." "Skills compound."
No code yet. Just the idea.

**THEORY**
Give the concept structure. Ask: what would this actually require?
Define the variables. Find the relationships. Identify the failure modes.
This is where the idea either holds up or falls apart.

**GROUNDED IMPLEMENTATION**
Write the code. The theory meets reality.
This stage always reveals something the theory missed.
That's not failure — that's the point.

**THEORY AGAIN**
The implementation taught you something. Go back up.
Refine the model. Ask better questions.
"Does improving reasoning passively lift planning? Yes — compound propagation."
"Does a higher-tier skill immediately dominate when it unlocks? Yes — tier^1.5 multiplier."

**CONCEPT AGAIN**
Return to the original idea with new understanding.
It's usually simpler and more powerful than the first version.
"It's an RPG skill tree. Skills compound. Caps exist. The system chases highest marginal gain."

**FINISHED PRODUCT**
Something that works. Something real.
Not a perfect implementation of the original concept —
something better, because the loop refined it.

---

## Example: Skill Curriculum

| Stage | What happened |
|-------|--------------|
| Concept | "Train on weakness — focus on what the model doesn't know" |
| Theory | Marginal value formula. Diminishing returns. Hysteresis. Tier multipliers. |
| Grounded | `SkillTree`, `CurriculumSampler`, 15 skills across 4 tiers, unlock gates |
| Theory again | Higher-tier skills dominate even at low confidence. Compounds propagate passively. |
| Concept again | RPG skill tree. Skills unlock. Caps exist. Always chase highest marginal gain. |
| Finished product | Adaptive curriculum that knows what it doesn't know. Better than fixed data mixtures. |

---

## Rules

**No API keys.** Everything runs locally or on owned infrastructure.

**No borrowed models.** Avus is built from scratch. The knowledge is earned, not inherited.

**No premature scaling.** Architecture first. Training second. Scale when the foundation is solid.

**The concept is the north star.** When implementation gets complicated, return to the concept.
If the code no longer serves the concept, the code is wrong.

**Grounding is not compromise.** Making something work within real constraints
is not a lesser version of the idea. It is the idea, tested.

---

## What This Produces

Systems that are architecturally coherent because every component
was designed to serve a concept, not just to exist.

Systems that are better than the naive approach because the loop
forces you to understand why something works, not just that it works.

Systems that are genuinely novel because the loop starts from instinct,
not from copying what already exists.

---

*This document describes how Janus is built.*
*It is not a process to follow. It is a pattern to recognize.*
