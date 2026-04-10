"""
llm_integration.py
==================
LLM code generation pipeline backed entirely by Avus (local, no API keys).

All generation routes through AvusInference — the single point of contact
for the trained model. Nothing here touches external services.

Usage:
    from llm_integration import LLMCodeGenerator, GameDevelopmentPipeline

    gen = LLMCodeGenerator()
    result = gen.generate_game_system("A player health system with regeneration")
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from avus_inference import AvusInference

logger = logging.getLogger(__name__)

# Singleton Avus instance shared across all generators
_avus: Optional[AvusInference] = None


def get_avus() -> AvusInference:
    """Return the shared AvusInference instance, loading it on first call."""
    global _avus
    if _avus is None:
        _avus = AvusInference()
        ok = _avus.load()
        if not ok:
            logger.warning("[llm_integration] Avus loaded with random weights — no checkpoint found.")
    return _avus


# ─────────────────────────────────────────────────────────────────────────────
# LLMCodeGenerator
# ─────────────────────────────────────────────────────────────────────────────

class LLMCodeGenerator:
    """
    Generates game system code using Avus.
    Drop-in replacement for the old Claude/OpenAI-backed generator.
    """

    def __init__(self):
        self.avus = get_avus()
        self.generated_code: List[Dict] = []

    def generate_game_system(
        self,
        prompt: str,
        language: str = "csharp",
        framework: str = "unity",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate a game system from a natural language description.

        Args:
            prompt: Description of the system to generate
            language: Target language (csharp, python, gdscript, etc.)
            framework: Target framework (unity, godot, pygame, etc.)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with keys: prompt, language, framework, code, success
        """
        system_prompt = self._build_system_prompt(language, framework)
        full_prompt = f"{system_prompt}\n\nTask: {prompt}\n\nCode:"

        try:
            code = self.avus.generate(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            result = {
                "prompt": prompt,
                "language": language,
                "framework": framework,
                "code": code.strip(),
                "success": True,
            }
        except Exception as e:
            logger.error(f"[LLMCodeGenerator] Generation failed: {e}")
            result = {
                "prompt": prompt,
                "language": language,
                "framework": framework,
                "code": "",
                "success": False,
                "error": str(e),
            }

        self.generated_code.append(result)
        return result

    def generate_multiple_systems(self, systems: List[Dict]) -> List[Dict]:
        """
        Generate multiple game systems sequentially.

        Args:
            systems: List of dicts with keys: prompt, language (opt), framework (opt)

        Returns:
            List of generation results
        """
        results = []
        for system in systems:
            result = self.generate_game_system(
                prompt=system["prompt"],
                language=system.get("language", "csharp"),
                framework=system.get("framework", "unity"),
            )
            results.append(result)
        return results

    def iterative_generation(
        self,
        prompt: str,
        language: str = "csharp",
        iterations: int = 3,
        temperature: float = 0.7,
    ) -> List[Dict]:
        """
        Generate multiple variations of the same system.

        Args:
            prompt: Description of the system
            language: Target language
            iterations: Number of variations to generate
            temperature: Sampling temperature

        Returns:
            List of generation results
        """
        results = []
        for i in range(iterations):
            # Slightly vary temperature per iteration for diversity
            t = temperature + (i * 0.05)
            result = self.generate_game_system(
                prompt=prompt,
                language=language,
                temperature=t,
            )
            results.append(result)
        return results

    def save_generated_code(self, code_index: int, filename: Optional[str] = None) -> str:
        """
        Save a generated code result to disk.

        Args:
            code_index: Index into self.generated_code
            filename: Optional output filename

        Returns:
            Path to saved file
        """
        if code_index >= len(self.generated_code):
            raise IndexError(f"No generated code at index {code_index}")

        result = self.generated_code[code_index]
        ext_map = {"csharp": "cs", "python": "py", "gdscript": "gd", "rust": "rs"}
        ext = ext_map.get(result["language"], "txt")

        if not filename:
            filename = f"generated_system_{code_index}.{ext}"

        path = Path(filename)
        path.write_text(result["code"])
        logger.info(f"[LLMCodeGenerator] Saved to {path}")
        return str(path)

    def get_generation_summary(self) -> Dict:
        """Return a summary of all generation attempts."""
        total = len(self.generated_code)
        successful = sum(1 for r in self.generated_code if r.get("success"))
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
        }

    def _build_system_prompt(self, language: str, framework: str) -> str:
        return (
            f"You are a game development assistant. "
            f"Generate clean, well-commented {language} code for the {framework} framework. "
            f"Output only the code, no explanations."
        )


# ─────────────────────────────────────────────────────────────────────────────
# GameDevelopmentPipeline
# ─────────────────────────────────────────────────────────────────────────────

class GameDevelopmentPipeline:
    """
    High-level pipeline for generating a full game from a description.
    Uses Avus for all generation — no external APIs.
    """

    def __init__(self):
        self.generator = LLMCodeGenerator()

    def plan_game(self, game_description: str, game_systems: List[Dict]) -> Dict:
        """
        Plan a game by ordering systems by dependency.

        Args:
            game_description: High-level description of the game
            game_systems: List of system dicts with 'name' and 'prompt' keys

        Returns:
            Plan dict with ordered systems
        """
        order = self._build_generation_order(game_systems)
        return {
            "description": game_description,
            "systems": game_systems,
            "generation_order": order,
        }

    def generate_game(self, plan: Dict, language: str = "csharp") -> Dict:
        """
        Execute a game plan and generate all systems.

        Args:
            plan: Plan dict from plan_game()
            language: Target language

        Returns:
            Dict mapping system name to generated code result
        """
        results = {}
        order = plan.get("generation_order", [s["name"] for s in plan["systems"]])
        systems_by_name = {s["name"]: s for s in plan["systems"]}

        for name in order:
            system = systems_by_name.get(name)
            if not system:
                continue
            logger.info(f"[GameDevelopmentPipeline] Generating: {name}")
            result = self.generator.generate_game_system(
                prompt=system["prompt"],
                language=language,
            )
            results[name] = result

        return results

    def _build_generation_order(self, systems: List[Dict]) -> List[str]:
        """
        Simple dependency ordering — core systems first.
        Systems with 'depends_on' key are sorted after their dependencies.
        """
        no_deps = [s["name"] for s in systems if not s.get("depends_on")]
        with_deps = [s["name"] for s in systems if s.get("depends_on")]
        return no_deps + with_deps


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = LLMCodeGenerator()

    result = gen.generate_game_system(
        "A player health system with regeneration over time",
        language="csharp",
        framework="unity",
    )

    print(f"Success: {result['success']}")
    print(f"Code preview:\n{result['code'][:300]}")
    print(f"\nSummary: {gen.get_generation_summary()}")
