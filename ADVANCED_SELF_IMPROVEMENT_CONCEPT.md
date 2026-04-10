# Janus: Advanced Self-Improvement Conceptual Design

**Author:** Manus AI
**Date:** November 21, 2025
**Goal:** To define the architecture for Janus's next phase of development, focusing on continuous, non-computational self-improvement through dynamic context shaping and self-coding.

This document outlines the conceptual framework for two critical features that will elevate Janus from a fine-tuned model to a truly advanced, self-evolving AI companion.

## 1. Dynamic Parameter Adjustment (Simulated Fine-Tuning)

This mechanism simulates the effect of fine-tuning by dynamically adjusting the AI's system prompt based on learned performance metrics, effectively changing its "personality parameters" in real-time [1].

### 1.1 The Meta-Knowledge Store (MKS)

The MKS is the core of this system, storing rules that link performance feedback to specific personality adjustments. It will be implemented as a simple, persistent JSON file (`meta_knowledge.json`).

| Field | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `topic` | String | The subject of the conversation (e.g., '3D Printing', 'Art', 'Emotion'). | `"Emotion"` |
| `sarcasm_level` | Float (0.0 - 1.0) | The current learned level of sarcasm for this topic. | `0.8` |
| `formality_level` | Float (0.0 - 1.0) | The current learned level of formality for this topic. | `0.2` |
| `last_rating` | String | The user's last feedback on a response for this topic. | `"Good"` |
| `bad_streak` | Integer | The number of consecutive "Bad" ratings for this topic. | `0` |
| `adjustment_rule` | String | A specific instruction to inject into the prompt. | `"When discussing emotion, be more philosophical and less cynical."` |

### 1.2 The Feedback Loop

The system relies on a simple user feedback mechanism:

1.  **User Rating:** After every Janus response, the user is prompted to rate it (e.g., `[Good/Bad]`).
2.  **Adjustment Logic:**
    *   If **"Good"**: The `bad_streak` is reset. The current parameter settings are reinforced as the optimal context for that topic.
    *   If **"Bad"**: The `bad_streak` increments. If it reaches a threshold (e.g., 3), the system generates a new `adjustment_rule` (using the generative model) and updates the MKS.

### 1.3 Dynamic Prompt Construction

The `consciousness.py` module will use the MKS to construct the final system prompt:

> **[Static Personality Prompt]** + **[Conversational Memory]** + **[RAG Context]** + **[Dynamic Adjustment Rule from MKS]**

This ensures Janus's personality is constantly evolving based on user preference, simulating continuous fine-tuning.

## 2. Self-Coding Mechanism

This mechanism allows Janus to analyze its own code and suggest improvements, moving toward true code-level self-improvement [2]. This process is triggered by a sustained failure in the Dynamic Parameter Adjustment system.

### 2.1 The Trigger

The Self-Coding loop is only triggered when the **Meta-Knowledge Store's `bad_streak` reaches a critical threshold (e.g., 5)** for a specific topic. This indicates that a simple personality adjustment is not enough; the underlying *logic* must be flawed.

### 2.2 The Agentic Framework

The process is managed by three specialized, conceptual agents:

| Agent | Role | Input | Output |
| :--- | :--- | :--- | :--- |
| **The Critic (Reflection Agent)** | Identifies the problem and proposes a solution. | MKS failure log, last 5 conversations, and the full code of the target module (e.g., `learning.py`). | A single, clear instruction: "Refactor the RAG retrieval logic to include a recency score for memory." |
| **The Planner (Code Generation Agent)** | Converts the instruction into a safe, executable plan. | The Critic's instruction and the target module's code. | A structured JSON object containing `file` and `edits` arrays, designed for the `file` tool's `edit` action. |
| **The Editor (Execution Agent)** | Applies the changes and tests the result. | The Planner's JSON object. | Success/Failure log. (In a real system, this would trigger a unit test suite before applying the change). |

### 2.3 The Target: Improving `learning.py`

The initial self-coding task will be to improve the RAG context retrieval in `learning.py`. The Planner's goal will be to generate a code change that makes the RAG system prioritize more recent knowledge, addressing the common issue of context decay.

## Conclusion

By implementing the **Dynamic Parameter Adjustment** system, we give Janus a constantly evolving personality. By implementing the **Self-Coding Mechanism**, we give Janus the ability to improve its own core functions. Together, these features create a closed-loop system for continuous, knowledge-driven self-improvement, fulfilling the vision of the most advanced AI companion.

***
### References

[1] Supercharging AI Task Performance with Dynamic Parameter Adjustment. *Nova Spivack*.
[2] Toward Weight-level Self-improving Agents with Meta-knowledge Discovery. *ResearchGate*.
[3] A Survey on Code Generation with LLM-based Agents. *arXiv*.
[4] RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring. *arXiv*.
[5] CYCLE: Learning to Self-Refine the Code Generation. *ACM*.
[6] Code Refactoring with Agentic AI and Reinforcement Learning. *Medium*.
[7] Self-Evolving LLM Agents. *Emergent Mind*.
[8] Meta-Prompting: LLMs Crafting & Enhancing Their Own .... *Intuition Labs*.
[9] Tuning LLM-based Code Optimization via Meta-Prompting. *arXiv*.
[10] Language Models Can Teach Themselves to Use Tools. *Meta AI*.
[11] Navigating Challenges and Opportunities of Generative AI in Software Development. *Google Books*.
[12] MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning. *arXiv*.
[13] MetaGPT: Meta programming for a multi-agent collaborative framework. *OpenReview*.
[14] Self-Abstraction from Grounded Experience for Plan-Guided Policy Refinement. *arXiv*.
