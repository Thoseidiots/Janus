"""
Janus Local LLM Integration
Uses your own trained GPT-2 model with slop filter
NO API KEYS NEEDED - Everything runs locally
"""

import json
import logging
import os
import torch
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("janus_local_llm")

# Import your custom model
from janus_gpt import JanusGPT, load_janus_brain

from slop_filter_pipeline import SlopFilterPipeline
from error_solution_searcher import ErrorSolutionSearcher


class LocalJanusCodeGenerator:
    """
    Generate code using YOUR local Janus LLM + slop filter.
    No API keys. No cloud. Everything on your machine.
    """
    
    def __init__(self, weights_dir: str = "my-llm-project/weights", 
                 device: str = "cpu"):
        logger.info("Initializing Local Janus Code Generator...")
        
        try:
            self.model = load_janus_brain(weights_dir=weights_dir, device=device)
            self.device = device
            logger.info(f"✓ Janus LLM loaded ({self.model.config.n_layer} layers, 49M params)")
        except FileNotFoundError as e:
            logger.error(f"Could not load Janus model: {e}")
            raise
        
        self.pipeline = SlopFilterPipeline()
        self.generated_codes = []
        self.conversation_history = []
    
    def generate_code(self, prompt: str, language: str = "csharp",
                     max_length: int = 300, temperature: float = 0.7) -> str:
        """
        Generate code using local Janus LLM.
        
        Args:
            prompt: What code to generate
            language: Programming language
            max_length: Max tokens to generate
            temperature: 0.0-1.0, lower=more focused, higher=more creative
        """
        
        logger.info(f"Generating {language} code...")
        logger.info(f"Prompt: {prompt[:80]}...")
        
        # Build prompt for code generation
        if language.lower() == "csharp":
            full_prompt = f"// {language} code\n// {prompt}\n"
        else:
            full_prompt = f"# {language} code\n# {prompt}\n"
        
        try:
            # Generate with your model
            generated = self.model.generate(
                full_prompt,
                max_new=max_length,
                temperature=temperature,
                top_k=50
            )
            
            logger.info(f"Generated {len(generated)} characters")
            return generated
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_and_fix(self, prompt: str, language: str = "csharp",
                        test_cases: List[Dict] = None,
                        max_iterations: int = 5) -> Dict:
        """
        Generate code and run through slop filter automatically.
        """
        
        logger.info(f"Generating and auto-fixing: {prompt[:60]}...")
        
        # Step 1: Generate
        generated_code = self.generate_code(prompt, language)
        
        # Step 2: Run through slop filter
        result = self.pipeline.process_generated_code(
            generated_code,
            language=language,
            description=prompt,
            test_cases=test_cases,
            max_iterations=max_iterations
        )
        
        # Step 3: Log
        self.generated_codes.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generated_length": len(generated_code),
            "final_quality": result["quality_score"],
            "status": result["status"],
            "iterations": result["iterations"]
        })
        
        return result
    
    def generate_game_system(self, description: str) -> Dict:
        """
        Generate a complete game system.
        """
        
        prompt = f"""Generate complete working C# Unity game code for: {description}

Requirements:
- Fully working, no placeholders
- Proper error handling
- Optimized for performance
- Well-commented
- Include using statements
- Ready to integrate

Code:
"""
        
        logger.info(f"Generating game system: {description[:50]}...")
        
        result = self.generate_and_fix(
            prompt,
            language="csharp",
            max_iterations=5
        )
        
        return result
    
    def batch_generate(self, prompts: List[str], language: str = "csharp") -> List[Dict]:
        """Generate multiple code snippets."""
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"[{i}/{len(prompts)}] Generating...")
            result = self.generate_and_fix(prompt, language)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get generation statistics."""
        
        if not self.generated_codes:
            return {"total_generations": 0}
        
        successful = sum(1 for c in self.generated_codes if c["status"] == "success")
        avg_quality = sum(c["final_quality"] for c in self.generated_codes) / len(self.generated_codes)
        
        return {
            "total_generations": len(self.generated_codes),
            "successful": successful,
            "success_rate": successful / len(self.generated_codes),
            "avg_quality": avg_quality,
            "avg_iterations": sum(c.get("iterations", 0) for c in self.generated_codes) / len(self.generated_codes)
        }


class JanusLocalGamePipeline:
    """
    Full game development pipeline using your local Janus LLM.
    """
    
    def __init__(self, weights_dir: str = "my-llm-project/weights"):
        self.generator = LocalJanusCodeGenerator(weights_dir=weights_dir)
        self.generated_systems = {}
        logger.info("Janus Local Game Pipeline initialized")
    
    def generate_game(self, game_name: str, systems: List[Dict]) -> Dict:
        """
        Generate all systems for a game.
        
        systems: list of dicts with keys: name, description
        """
        
        logger.info(f"Generating game: {game_name}")
        logger.info(f"Systems: {len(systems)}")
        
        results = {}
        
        for system in systems:
            system_name = system["name"]
            description = system["description"]
            
            logger.info(f"\n[System] {system_name}")
            
            result = self.generator.generate_game_system(description)
            results[system_name] = result
            self.generated_systems[system_name] = result
        
        summary = {
            "game": game_name,
            "timestamp": datetime.now().isoformat(),
            "total_systems": len(results),
            "successful": sum(1 for r in results.values() if r["status"] == "success"),
            "avg_quality": sum(r["quality_score"] for r in results.values()) / len(results),
            "systems": results
        }
        
        logger.info(f"\n✓ Game generation complete!")
        logger.info(f"  Success rate: {summary['successful']}/{summary['total_systems']}")
        logger.info(f"  Avg quality: {summary['avg_quality']:.1f}/5")
        
        return summary
    
    def save_game_code(self, output_dir: str = "generated_game"):
        """Save all generated code to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for system_name, code_data in self.generated_systems.items():
            filename = f"{output_dir}/{system_name}.cs"
            with open(filename, 'w') as f:
                f.write(code_data["code"])
            logger.info(f"Saved: {filename}")


if __name__ == "__main__":
    print("="*70)
    print("JANUS LOCAL LLM + SLOP FILTER")
    print("Code generation with your own model - NO API KEYS")
    print("="*70)
    
    try:
        # Initialize
        print("\n[Step 1] Loading Janus LLM...")
        generator = LocalJanusCodeGenerator()
        
        print("\n[Step 2] Generating a simple function...")
        result = generator.generate_and_fix(
            "Calculate factorial of n",
            language="csharp"
        )
        
        print(f"\nResult:")
        print(f"  Status: {result['status']}")
        print(f"  Quality: {result['quality_score']}/5")
        print(f"  Iterations: {result['iterations']}")
        
        print(f"\nGenerated code:")
        print("=" * 70)
        print(result['final_code'][:500])
        print("..." if len(result['final_code']) > 500 else "")
        print("=" * 70)
        
        print("\nStats:")
        stats = generator.get_stats()
        print(json.dumps(stats, indent=2))
        
        print("\n✓ Local generation working!")
        print("\nTo generate a game:")
        print("""
from janus_local_llm import JanusLocalGamePipeline

pipeline = JanusLocalGamePipeline()
game = pipeline.generate_game(
    "My 2D Game",
    [
        {"name": "player", "description": "WASD movement with jumping"},
        {"name": "enemies", "description": "Simple patrol enemies"},
        {"name": "coins", "description": "Collectible coins"}
    ]
)
pipeline.save_game_code("my_game_code")
        """)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. my-llm-project/weights/janus_best.pt exists")
        print("  2. janus_training_summary.json exists")
        print("  3. janus_gpt.py is in same directory")
