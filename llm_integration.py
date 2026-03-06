"""
LLM Integration for Autonomous Code Generation
Connects prompt → LLM → Slop filter → Working code
"""

import json
import logging
import os
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("llm_integration")

# Try to import LLM providers
try:
    from anthropic import Anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False
    logger.warning("Anthropic SDK not installed - Claude not available")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI SDK not installed - GPT not available")

from slop_filter_pipeline import SlopFilterPipeline
from error_solution_searcher import ErrorSolutionSearcher


class LLMCodeGenerator:
    """
    Generate game code using LLM with grounded error fixing.
    Supports both Claude and GPT.
    """
    
    def __init__(self, provider: str = "claude", model: str = None):
        self.provider = provider.lower()
        self.model = model
        self.conversation_history = []
        self.generated_codes = []
        self.pipeline = SlopFilterPipeline()
        
        if self.provider == "claude":
            if not HAS_CLAUDE:
                raise ValueError("Anthropic SDK not installed. Install: pip install anthropic")
            self.client = Anthropic()
            self.model = model or "claude-3-5-sonnet-20241022"
        
        elif self.provider == "gpt":
            if not HAS_OPENAI:
                raise ValueError("OpenAI SDK not installed. Install: pip install openai")
            self.client = openai.OpenAI()
            self.model = model or "gpt-4"
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"LLM Code Generator initialized: {self.provider} ({self.model})")
    
    def generate_game_system(self, prompt: str, language: str = "csharp",
                            framework: str = "unity", auto_fix: bool = True,
                            test_cases: List[Dict] = None) -> Dict:
        """
        Generate a complete game system from natural language prompt.
        
        Process:
        1. Send prompt to LLM
        2. Get generated code
        3. Run through slop filter
        4. Return clean, working code
        """
        
        logger.info(f"Generating: {prompt[:60]}...")
        logger.info(f"Language: {language}, Framework: {framework}")
        
        # Step 1: Create focused prompt for LLM
        system_prompt = self._build_system_prompt(language, framework)
        
        user_message = f"""Generate complete, production-ready code for: {prompt}

Requirements:
- Fully working code with no placeholders
- Proper error handling
- Performance-optimized
- Well-commented
- Include any necessary imports/using statements
- Ready to integrate into game

Return ONLY the code, no explanation."""
        
        # Step 2: Generate code with LLM
        logger.info("[Step 1] Generating code with LLM...")
        generated_code = self._call_llm(user_message, system_prompt)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generated_code": generated_code[:500] + "..." if len(generated_code) > 500 else generated_code
        })
        
        # Step 3: Run through slop filter if requested
        if auto_fix:
            logger.info("[Step 2] Running through slop filter...")
            filter_result = self.pipeline.process_generated_code(
                generated_code,
                language=language,
                description=prompt,
                test_cases=test_cases,
                max_iterations=5
            )
            
            final_code = filter_result["code"]
            quality = filter_result["quality_score"]
            status = filter_result["status"]
            
            if status == "success":
                logger.info(f"✓ Code cleaned successfully (quality: {quality}/5)")
            else:
                logger.warning(f"⚠ Partial success (quality: {quality}/5)")
        else:
            final_code = generated_code
            quality = 2.0  # Unverified
            status = "unverified"
            filter_result = None
        
        # Step 4: Build result
        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "language": language,
            "framework": framework,
            "generated_code": generated_code,
            "final_code": final_code,
            "status": status,
            "quality_score": quality,
            "slop_filter_result": filter_result
        }
        
        self.generated_codes.append(result)
        
        return result
    
    def generate_multiple_systems(self, systems: List[Dict]) -> List[Dict]:
        """
        Generate multiple game systems in sequence.
        
        Each system dict should have:
        - prompt: what to generate
        - language: code language
        - framework: game framework
        - test_cases: optional test cases
        """
        
        logger.info(f"Generating {len(systems)} systems...")
        
        results = []
        
        for i, system in enumerate(systems, 1):
            logger.info(f"\n[{i}/{len(systems)}] {system.get('prompt', 'System')[:50]}...")
            
            result = self.generate_game_system(
                system["prompt"],
                language=system.get("language", "csharp"),
                framework=system.get("framework", "unity"),
                test_cases=system.get("test_cases"),
                auto_fix=system.get("auto_fix", True)
            )
            
            results.append(result)
        
        return results
    
    def _build_system_prompt(self, language: str, framework: str) -> str:
        """Build context-specific system prompt."""
        
        if language.lower() == "csharp" and framework.lower() == "unity":
            return """You are an expert Unity game developer writing C# code.

Guidelines:
- Use MonoBehaviour for game systems
- Follow Unity best practices
- Optimize for performance
- Use SerializeField for inspector variables
- Add proper null checks
- Include error handling
- Write clean, readable code
- Add XML documentation comments

Generate complete, working code that can be copied directly into a .cs file."""
        
        elif language.lower() == "python":
            return """You are an expert Python game developer.

Guidelines:
- Write clean, Pythonic code
- Use appropriate design patterns
- Include docstrings
- Add error handling
- Optimize for readability
- Include type hints
- Use pygame or Godot Python API if applicable

Generate complete, working code."""
        
        else:
            return f"""You are an expert {language} developer for {framework}.
Generate complete, production-ready code with proper error handling and optimization."""
    
    def _call_llm(self, user_message: str, system_prompt: str) -> str:
        """Call LLM and get response."""
        
        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.content[0].text
            
            elif self.provider == "gpt":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def iterative_generation(self, prompt: str, language: str = "csharp",
                            feedback_iterations: int = 3) -> Dict:
        """
        Generate code, get feedback, iterate.
        
        Allows for refinement based on issues.
        """
        
        logger.info(f"Starting iterative generation: {prompt[:60]}...")
        
        iteration = 0
        current_code = None
        current_quality = 0
        
        while iteration < feedback_iterations:
            iteration += 1
            logger.info(f"\nIteration {iteration}/{feedback_iterations}")
            
            # Generate or refine
            if iteration == 1:
                result = self.generate_game_system(prompt, language=language)
            else:
                # Ask LLM to improve based on quality feedback
                feedback = f"Previous quality score: {current_quality}/5. Generate improved version."
                result = self.generate_game_system(
                    f"{prompt}\n{feedback}",
                    language=language
                )
            
            current_code = result["final_code"]
            current_quality = result["quality_score"]
            
            logger.info(f"Quality: {current_quality}/5")
            
            # Stop if good enough
            if current_quality >= 4.0:
                logger.info(f"✓ Good quality reached at iteration {iteration}")
                break
        
        return {
            "prompt": prompt,
            "final_code": current_code,
            "final_quality": current_quality,
            "iterations": iteration,
            "results": self.generated_codes[-feedback_iterations:]
        }
    
    def get_generation_summary(self) -> Dict:
        """Get summary of all generations."""
        
        if not self.generated_codes:
            return {"total_generations": 0}
        
        successful = sum(1 for c in self.generated_codes if c["status"] == "success")
        avg_quality = sum(c["quality_score"] for c in self.generated_codes) / len(self.generated_codes)
        
        return {
            "total_generations": len(self.generated_codes),
            "successful": successful,
            "success_rate": successful / len(self.generated_codes),
            "avg_quality": avg_quality,
            "languages": list(set(c["language"] for c in self.generated_codes)),
            "frameworks": list(set(c["framework"] for c in self.generated_codes))
        }
    
    def save_generated_code(self, code_index: int, filename: str = None) -> str:
        """Save generated code to file."""
        
        if code_index >= len(self.generated_codes):
            raise ValueError(f"Code index {code_index} out of range")
        
        code_data = self.generated_codes[code_index]
        
        if not filename:
            prompt_short = code_data["prompt"][:30].replace(" ", "_")
            ext = ".cs" if code_data["language"].lower() == "csharp" else ".py"
            filename = f"generated_{prompt_short}{ext}"
        
        with open(filename, 'w') as f:
            f.write(code_data["final_code"])
        
        logger.info(f"Code saved to {filename}")
        return filename


class GameDevelopmentPipeline:
    """
    Full pipeline for autonomous game development.
    Coordinates code generation, fixing, and integration.
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.generator = LLMCodeGenerator(provider=llm_provider)
        self.systems = {}
        self.integration_log = []
        
        logger.info("Game Development Pipeline initialized")
    
    def plan_game(self, game_description: str, game_systems: List[Dict]) -> Dict:
        """
        Plan a game and break into generatable systems.
        
        game_systems: list of dicts with keys: name, description, dependencies
        """
        
        plan = {
            "timestamp": datetime.now().isoformat(),
            "game_description": game_description,
            "systems": game_systems,
            "generation_plan": self._build_generation_order(game_systems)
        }
        
        logger.info(f"Game plan created: {game_description}")
        logger.info(f"Systems to generate: {len(game_systems)}")
        
        return plan
    
    def _build_generation_order(self, systems: List[Dict]) -> List[str]:
        """Determine order to generate systems based on dependencies."""
        
        # Simple topological sort based on dependencies
        order = []
        remaining = {s["name"]: s for s in systems}
        
        while remaining:
            # Find system with no unsatisfied dependencies
            for name, system in remaining.items():
                deps = system.get("dependencies", [])
                if all(dep in order for dep in deps):
                    order.append(name)
                    del remaining[name]
                    break
            else:
                # Circular dependency - just add remaining
                order.extend(remaining.keys())
                break
        
        return order
    
    def generate_game(self, plan: Dict, language: str = "csharp") -> Dict:
        """
        Generate all systems in a game according to plan.
        """
        
        logger.info(f"Starting game generation...")
        
        generation_order = plan["generation_plan"]
        systems_by_name = {s["name"]: s for s in plan["systems"]}
        
        results = {}
        
        for system_name in generation_order:
            logger.info(f"\nGenerating system: {system_name}")
            
            system_info = systems_by_name[system_name]
            
            # Generate system
            result = self.generator.generate_game_system(
                system_info["description"],
                language=language,
                framework="unity"
            )
            
            results[system_name] = result
            self.systems[system_name] = result
            
            # Log integration
            self.integration_log.append({
                "timestamp": datetime.now().isoformat(),
                "system": system_name,
                "status": result["status"],
                "quality": result["quality_score"]
            })
        
        # Summary
        summary = {
            "total_systems": len(results),
            "successful": sum(1 for r in results.values() if r["status"] == "success"),
            "avg_quality": sum(r["quality_score"] for r in results.values()) / len(results),
            "systems": results
        }
        
        logger.info(f"\n✓ Game generation complete!")
        logger.info(f"  Systems: {summary['total_systems']}")
        logger.info(f"  Successful: {summary['successful']}")
        logger.info(f"  Avg quality: {summary['avg_quality']:.1f}/5")
        
        return summary


if __name__ == "__main__":
    print("LLM Code Generation with Slop Filter")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ No API keys found!")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("\nExample:")
        print('  export ANTHROPIC_API_KEY="sk-ant-..."')
        print("  python llm_integration.py")
        print("\nNote: You need either Claude or GPT API access to use this.")
        print("This is a local-first system, but LLM generation requires an API.")
    else:
        print("\n✓ API key detected")
        
        try:
            # Try to initialize
            if os.getenv("ANTHROPIC_API_KEY"):
                generator = LLMCodeGenerator(provider="claude")
                print("✓ Claude initialized")
            else:
                generator = LLMCodeGenerator(provider="gpt")
                print("✓ GPT initialized")
            
            print("\nReady for code generation!")
            print("Example usage:")
            print("""
result = generator.generate_game_system(
    "Player movement system with WASD control and jumping",
    language="csharp",
    framework="unity"
)
print(result["final_code"])
            """)
        
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
