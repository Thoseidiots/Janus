"""
Integrated Slop Filter Pipeline
Code generation → Evaluation → Error search → Auto-fix → Iteration
"""

import json
import logging
from typing import Dict, Optional, List
from datetime import datetime

from code_quality_evaluator import CodeQualityEvaluator, CodeIterativeRefiner
from error_solution_searcher import ErrorSolutionSearcher

logger = logging.getLogger("slop_filter")


class SlopFilterPipeline:
    """
    End-to-end pipeline: generate code → fix slop → produce working output.
    """
    
    def __init__(self):
        self.evaluator = CodeQualityEvaluator()
        self.refiner = CodeIterativeRefiner(self.evaluator)
        self.searcher = ErrorSolutionSearcher()
        
        self.pipeline_runs = []
        logger.info("Slop Filter Pipeline initialized")
    
    def process_generated_code(self, generated_code: str, language: str = "python",
                              description: str = "", test_cases: List[Dict] = None,
                              max_iterations: int = 5) -> Dict:
        """
        Process AI-generated code through the slop filter.
        
        Pipeline:
        1. Evaluate code quality
        2. If broken: search for solutions
        3. Attempt fix
        4. Re-evaluate
        5. Repeat until working or max iterations
        
        Returns: cleaned code, quality report, fixes applied
        """
        
        logger.info(f"Processing code ({len(generated_code)} chars)")
        logger.info(f"Language: {language}, Description: {description}")
        
        run_id = f"run_{datetime.now().timestamp()}"
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "language": language,
            "initial_code_length": len(generated_code),
            "stages": []
        }
        
        # Stage 1: Initial evaluation
        logger.info("[Stage 1] Initial Evaluation")
        initial_eval = self.evaluator.evaluate_code(generated_code, language, test_cases)
        
        stage1 = {
            "stage": "initial_evaluation",
            "quality_score": initial_eval["quality_score"],
            "errors_found": len(initial_eval["errors"]),
            "warnings_found": len(initial_eval["warnings"])
        }
        run_data["stages"].append(stage1)
        
        # Check if already clean
        if initial_eval["quality_score"] >= 4 and not initial_eval["errors"]:
            logger.info("✓ Code is already clean - no fixes needed")
            run_data["result"] = "CLEAN"
            run_data["final_code"] = generated_code
            self.pipeline_runs.append(run_data)
            return {
                "status": "success",
                "code": generated_code,
                "quality_score": initial_eval["quality_score"],
                "iterations": 0,
                "fixes_applied": []
            }
        
        # Stage 2: Error analysis & solution search
        logger.info("[Stage 2] Error Analysis & Solution Search")
        solutions_found = []
        
        if initial_eval["errors"]:
            for error in initial_eval["errors"]:
                logger.info(f"  Analyzing error: {error['message']}")
                
                solution = self.searcher.find_solution(
                    error["message"],
                    language,
                    generated_code
                )
                
                solutions_found.append({
                    "error": error["message"],
                    "error_type": solution["error_type"],
                    "confidence": solution["confidence"],
                    "suggested_fixes": solution["solutions"][:2]
                })
                
                logger.info(f"    Found {len(solution['solutions'])} potential fixes")
        
        stage2 = {
            "stage": "solution_search",
            "errors_analyzed": len(initial_eval["errors"]),
            "solutions_found": len(solutions_found),
            "details": solutions_found
        }
        run_data["stages"].append(stage2)
        
        # Stage 3: Iterative refinement
        logger.info(f"[Stage 3] Iterative Refinement (max {max_iterations} iterations)")
        
        refine_result = self.refiner.refine_until_working(
            generated_code,
            language,
            test_cases,
            max_iterations
        )
        
        stage3 = {
            "stage": "iterative_refinement",
            "success": refine_result["success"],
            "iterations_used": refine_result["iterations"],
            "final_quality_score": refine_result["final_score"],
            "final_code_length": len(refine_result["final_code"])
        }
        run_data["stages"].append(stage3)
        
        # Generate summary
        if refine_result["success"]:
            logger.info(f"✓ Success! Fixed in {refine_result['iterations']} iterations")
            status = "success"
        else:
            logger.warning(f"Could not fully fix. Remaining errors: {len(refine_result.get('remaining_errors', []))}")
            status = "partial"
        
        final_result = {
            "status": status,
            "code": refine_result["final_code"],
            "quality_score": refine_result["final_score"],
            "iterations": refine_result["iterations"],
            "initial_quality": initial_eval["quality_score"],
            "quality_improvement": refine_result["final_score"] - initial_eval["quality_score"],
            "fixes_applied": solutions_found,
            "pipeline_summary": run_data
        }
        
        self.pipeline_runs.append(run_data)
        
        return final_result
    
    def batch_process_codes(self, codes: List[Dict]) -> List[Dict]:
        """
        Process multiple generated code snippets.
        Each dict should have: code, language, description, test_cases (optional)
        """
        
        logger.info(f"Processing batch of {len(codes)} code snippets")
        
        results = []
        for i, code_data in enumerate(codes):
            logger.info(f"\n[{i+1}/{len(codes)}] Processing...")
            
            result = self.process_generated_code(
                code_data["code"],
                code_data.get("language", "python"),
                code_data.get("description", ""),
                code_data.get("test_cases"),
                code_data.get("max_iterations", 5)
            )
            
            results.append(result)
        
        return results
    
    def get_pipeline_statistics(self) -> Dict:
        """Get statistics about pipeline performance."""
        
        if not self.pipeline_runs:
            return {"runs": 0}
        
        successful = sum(1 for r in self.pipeline_runs if r.get("result") != "CLEAN")
        avg_iterations = sum(r.get("stages", [{}])[-1].get("iterations_used", 0) 
                            for r in self.pipeline_runs) / max(len(self.pipeline_runs), 1)
        
        return {
            "total_runs": len(self.pipeline_runs),
            "successful_fixes": successful,
            "clean_on_first_try": len(self.pipeline_runs) - successful,
            "avg_iterations_needed": avg_iterations,
            "languages_processed": list(set(r["language"] for r in self.pipeline_runs)),
            "total_errors_found": sum(
                len(r.get("stages", [{}])[0].get("errors", []))
                for r in self.pipeline_runs
            ),
            "total_solutions_found": sum(
                r.get("stages", [{}])[1].get("solutions_found", 0)
                for r in self.pipeline_runs
                if len(r.get("stages", [])) > 1
            )
        }
    
    def generate_report(self, run_id: str = None) -> Dict:
        """Generate detailed report on pipeline runs."""
        
        if run_id:
            run = next((r for r in self.pipeline_runs if r["run_id"] == run_id), None)
            if not run:
                return {"error": "Run not found"}
            runs_to_report = [run]
        else:
            runs_to_report = self.pipeline_runs
        
        report = {
            "report_time": datetime.now().isoformat(),
            "runs_analyzed": len(runs_to_report),
            "statistics": self.get_pipeline_statistics(),
            "detailed_runs": runs_to_report
        }
        
        return report


if __name__ == "__main__":
    print("Slop Filter Pipeline")
    print("=" * 60)
    
    pipeline = SlopFilterPipeline()
    
    # Test 1: Clean code
    print("\n[Test 1: Clean Code]")
    clean_code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
"""
    
    result1 = pipeline.process_generated_code(
        clean_code,
        language="python",
        description="Simple addition function"
    )
    print(f"Status: {result1['status']}")
    print(f"Quality: {result1['quality_score']}/5")
    
    # Test 2: Broken code
    print("\n[Test 2: Broken Code (will attempt fix)]")
    broken_code = """
def add(a, b)
    return a + b

result = add(5, 3)
print(result)
"""
    
    result2 = pipeline.process_generated_code(
        broken_code,
        language="python",
        description="Addition function with syntax error"
    )
    print(f"Status: {result2['status']}")
    print(f"Initial quality: {result2['initial_quality']}/5")
    print(f"Final quality: {result2['quality_score']}/5")
    print(f"Iterations needed: {result2['iterations']}")
    
    # Statistics
    print("\n[Pipeline Statistics]")
    stats = pipeline.get_pipeline_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n✓ Pipeline ready for game development!")
