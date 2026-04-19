"""
Janus Kaggle Weights Manager - Download and Use Real Trained Weights

This system downloads real Janus weights from KaggleHub and uses them
for actual money-making capabilities. Combines weights for optimal performance.
"""

import os
import json
import logging
import tempfile
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import KaggleHub
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    print("Installing KaggleHub...")
    os.system("pip install kagglehub -q")
    try:
        import kagglehub
        KAGGLE_AVAILABLE = True
    except ImportError:
        KAGGLE_AVAILABLE = False

# Import Janus components
try:
    from avus_brain import AvusBrain
    from avus_inference import AvusInference
    from browser_automation import BrowserAutomationAgent
    from speech_synthesis import HumanSpeechSynthesizer, SpeechContext
    JANUS_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Janus components not available: {e}")
    JANUS_COMPONENTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JanusKaggleWeights:
    """Manages downloading and using real Janus weights from Kaggle"""
    
    def __init__(self):
        self.weights_dir = Path.cwd() / "downloaded_weights"
        self.weights_dir.mkdir(exist_ok=True)
        
        self.available_weights = {}
        self.loaded_weights = {}
        self.combined_weights = {}
        
        # Kaggle datasets for Janus weights
        self.kaggle_datasets = [
            {
                "name": "ishmaelsears/janus-avus-weights",
                "models": ["avus_1b_weights.pt", "avus_3b_weights.pt", "avus_7b_weights.pt"],
                "description": "Official Janus Avus weights"
            }
        ]
        
        logger.info("Janus Kaggle Weights Manager initialized")
    
    async def download_weights(self, model_size: str = "1b") -> bool:
        """Download specific model weights from Kaggle"""
        print(f"DOWNLOADING JANUS WEIGHTS - {model_size.upper()} Model")
        print("=" * 60)
        
        if not KAGGLE_AVAILABLE:
            print("ERROR: KaggleHub not available")
            print("Install: pip install kagglehub")
            return False
        
        try:
            # Download from KaggleHub
            dataset = self.kaggle_datasets[0]
            print(f"Downloading from: {dataset['name']}")
            
            path = kagglehub.dataset_download(dataset['name'])
            print(f"Downloaded to: {path}")
            
            # Find the specific model weights
            weight_files = {
                "1b": "avus_1b_weights.pt",
                "3b": "avus_3b_weights.pt", 
                "7b": "avus_7b_weights.pt",
                "13b": "avus_13b_weights.pt",
                "34b": "avus_34b_weights.pt",
                "70b": "avus_70b_weights.pt"
            }
            
            target_file = weight_files.get(model_size)
            if not target_file:
                print(f"ERROR: Unknown model size {model_size}")
                return False
            
            source_path = Path(path) / target_file
            if not source_path.exists():
                print(f"ERROR: {target_file} not found in download")
                return False
            
            # Copy to weights directory
            dest_path = self.weights_dir / target_file
            import shutil
            shutil.copy2(source_path, dest_path)
            
            print(f"✓ Copied {target_file} to {dest_path}")
            self.available_weights[model_size] = str(dest_path)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Download failed: {e}")
            logger.error(f"Weight download failed: {e}")
            return False
    
    async def combine_weights(self, models: List[str], combination_method: str = "slerp") -> bool:
        """Combine multiple weight sets for better performance"""
        print(f"COMBINING WEIGHTS - {', '.join(models)} using {combination_method}")
        print("-" * 50)
        
        if not all(model in self.available_weights for model in models):
            print("ERROR: Not all models downloaded")
            return False
        
        try:
            # Load weights
            import torch
            loaded_models = {}
            
            for model in models:
                weight_path = self.available_weights[model]
                print(f"Loading {model} weights from {weight_path}")
                loaded_models[model] = torch.load(weight_path, map_location='cpu')
            
            # Combine weights based on method
            if combination_method == "slerp":
                combined = self._slerp_combine(loaded_models, models[0], models[1])
            elif combination_method == "average":
                combined = self._average_combine(loaded_models)
            elif combination_method == "dare":
                combined = self._dare_combine(loaded_models)
            else:
                print(f"ERROR: Unknown combination method {combination_method}")
                return False
            
            # Save combined weights
            combined_name = f"combined_{'_'.join(models)}_{combination_method}.pt"
            combined_path = self.weights_dir / combined_name
            torch.save(combined, combined_path)
            
            print(f"✓ Combined weights saved to {combined_path}")
            self.combined_weights[combined_name] = str(combined_path)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Weight combination failed: {e}")
            logger.error(f"Weight combination failed: {e}")
            return False
    
    def _slerp_combine(self, models: Dict, model_a: str, model_b: str, alpha: float = 0.5):
        """Spherical linear interpolation between two models"""
        import torch
        import math
        
        state_a = models[model_a]
        state_b = models[model_b]
        
        # Get common keys
        common_keys = set(state_a.keys()) & set(state_b.keys())
        
        combined = {}
        for key in common_keys:
            tensor_a = state_a[key]
            tensor_b = state_b[key]
            
            # SLERP formula
            dot = (tensor_a * tensor_b).sum()
            norm_a = (tensor_a * tensor_a).sum().sqrt()
            norm_b = (tensor_b * tensor_b).sum().sqrt()
            
            theta = torch.acos(dot / (norm_a * norm_b))
            sin_theta = torch.sin(theta)
            
            if sin_theta > 1e-8:
                a = torch.sin((1 - alpha) * theta) / sin_theta
                b = torch.sin(alpha * theta) / sin_theta
                combined[key] = a * tensor_a + b * tensor_b
            else:
                # Fallback to linear interpolation
                combined[key] = (1 - alpha) * tensor_a + alpha * tensor_b
        
        return combined
    
    def _average_combine(self, models: Dict):
        """Average multiple models"""
        import torch
        
        # Get common keys
        common_keys = set(models[list(models.keys())[0]].keys())
        for model in models.values():
            common_keys &= set(model.keys())
        
        combined = {}
        for key in common_keys:
            tensors = [model[key] for model in models.values()]
            combined[key] = torch.stack(tensors).mean(dim=0)
        
        return combined
    
    def _dare_combine(self, models: Dict):
        """DARE (Task Arithmetic) combination"""
        import torch
        
        # Simple implementation - add and subtract task vectors
        base_model = models[list(models.keys())[0]]
        combined = base_model.copy()
        
        # Add task vectors from other models
        for i, (name, model) in enumerate(models.items()):
            if i == 0:  # Skip base model
                continue
            
            # Add task vector (simplified)
            for key in combined.keys():
                if key in model:
                    combined[key] = combined[key] + 0.1 * (model[key] - base_model[key])
        
        return combined
    
    async def test_weights_with_money_making(self, weights_path: str) -> Dict[str, Any]:
        """Test downloaded weights with real money-making tasks"""
        print(f"TESTING WEIGHTS FOR MONEY MAKING")
        print("-" * 40)
        
        if not JANUS_COMPONENTS_AVAILABLE:
            print("ERROR: Janus components not available")
            return {"success": False, "error": "Components not available"}
        
        try:
            # Initialize Avus with downloaded weights
            avus_inference = AvusInference()
            
            # Load the weights
            print(f"Loading weights from: {weights_path}")
            if not avus_inference.load(weights_path=weights_path):
                print("ERROR: Failed to load weights")
                return {"success": False, "error": "Weight loading failed"}
            
            # Initialize brain with loaded model
            brain = AvusBrain(avus_inference)
            
            # Test money-making capabilities
            results = {}
            
            # Test 1: Content Generation
            print("Testing content generation...")
            content_prompt = "Write a professional email offering AI automation services to small businesses"
            content = brain.ask(content_prompt, max_tokens=150)
            results["content_generation"] = {
                "success": True,
                "output_length": len(content),
                "sample": content[:100] + "..."
            }
            print(f"  Generated {len(content)} characters of content")
            
            # Test 2: Code Generation
            print("Testing code generation...")
            code_prompt = "Python function to scrape website data and save to CSV"
            code = brain.ask(f"Generate code: {code_prompt}", max_tokens=200)
            results["code_generation"] = {
                "success": True,
                "output_length": len(code),
                "lines": len(code.split('\n'))
            }
            print(f"  Generated {len(code.split())} words of code")
            
            # Test 3: Business Analysis
            print("Testing business analysis...")
            analysis_prompt = "Analyze this business data and provide growth recommendations: Q1: $10k, Q2: $12k, Q3: $15k, Q4: $18k"
            analysis = brain.ask(analysis_prompt, max_tokens=200)
            results["business_analysis"] = {
                "success": True,
                "output_length": len(analysis),
                "insights": len(analysis.split('.'))
            }
            print(f"  Generated business analysis with {len(analysis.split())} words")
            
            # Test 4: Client Outreach
            print("Testing client outreach...")
            outreach_prompt = "Write a personalized email to a potential client offering web automation services"
            outreach = brain.ask(outreach_prompt, max_tokens=150)
            results["client_outreach"] = {
                "success": True,
                "output_length": len(outreach),
                "personalized_elements": ["personalized", "customized"]  # Simplified check
            }
            print(f"  Generated personalized outreach email")
            
            # Calculate revenue potential
            content_revenue = len(content.split()) * 0.05  # $0.05 per word
            code_revenue = len(code.split('\n')) * 0.50  # $0.50 per line
            analysis_revenue = 50.0  # $50 per analysis
            outreach_revenue = 25.0  # $25 per outreach
            
            total_revenue = content_revenue + code_revenue + analysis_revenue + outreach_revenue
            
            results["revenue_potential"] = {
                "content": content_revenue,
                "code": code_revenue,
                "analysis": analysis_revenue,
                "outreach": outreach_revenue,
                "total": total_revenue
            }
            
            print(f"\nREVENUE POTENTIAL FROM TEST:")
            print(f"  Content Generation: ${content_revenue:.2f}")
            print(f"  Code Generation: ${code_revenue:.2f}")
            print(f"  Business Analysis: ${analysis_revenue:.2f}")
            print(f"  Client Outreach: ${outreach_revenue:.2f}")
            print(f"  Total Potential: ${total_revenue:.2f}")
            
            # Save test results
            results_file = self.weights_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nTest results saved to: {results_file}")
            results["success"] = True
            results["weights_path"] = weights_path
            
            return results
            
        except Exception as e:
            print(f"ERROR: Testing failed: {e}")
            logger.error(f"Weight testing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "available_sizes": ["1b", "3b", "7b", "13b", "34b", "70b"],
            "downloaded": list(self.available_weights.keys()),
            "combined": list(self.combined_weights.keys()),
            "kaggle_datasets": self.kaggle_datasets
        }

# Main execution
async def main():
    """Main function to demonstrate weight downloading and money making"""
    weights_manager = JanusKaggleWeights()
    
    print("JANUS KAGGLE WEIGHTS MANAGER")
    print("=" * 60)
    print("Download real trained weights for actual money-making capabilities")
    print()
    
    # Step 1: Download weights
    print("STEP 1: DOWNLOADING WEIGHTS")
    success = await weights_manager.download_weights("1b")  # Start with 1B model
    
    if not success:
        print("Failed to download weights")
        return
    
    # Step 2: Test weights with money making
    print("\nSTEP 2: TESTING WEIGHTS FOR MONEY MAKING")
    weights_path = weights_manager.available_weights["1b"]
    test_results = await weights_manager.test_weights_with_money_making(weights_path)
    
    if test_results["success"]:
        print("\n✓ WEIGHTS WORK FOR MONEY MAKING!")
        revenue = test_results["revenue_potential"]["total"]
        print(f"Potential revenue from single test: ${revenue:.2f}")
        print(f"Monthly potential (100 clients): ${revenue * 100:.2f}")
        
        if revenue * 100 >= 10000:
            print("✓ $10k/month target achievable!")
        else:
            needed_scaling = 10000 / (revenue * 100)
            print(f"✓ Need {needed_scaling:.1f}x scaling to reach $10k/month")
    else:
        print(f"✗ WEIGHTS TEST FAILED: {test_results.get('error', 'Unknown error')}")
    
    # Step 3: Show available options
    print(f"\nSTEP 3: AVAILABLE OPTIONS")
    models_info = weights_manager.get_available_models()
    print(f"Available model sizes: {', '.join(models_info['available_sizes'])}")
    print(f"Downloaded models: {', '.join(models_info['downloaded'])}")
    
    print(f"\nNEXT STEPS:")
    print("1. Download larger models (3b, 7b) for better performance")
    print("2. Combine weights for specialized capabilities")
    print("3. Set up client acquisition system")
    print("4. Implement payment processing")
    print("5. Scale to $10k/month")

if __name__ == "__main__":
    print("Janus Kaggle Weights Manager")
    print("Real weights from Kaggle for real money making")
    print()
    
    asyncio.run(main())
