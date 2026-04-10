"""
Code Quality Evaluator & Autonomous Fixer
Tests generated code, identifies errors, searches for solutions, iterates until working
"""

import json
import logging
import subprocess
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger("code_quality")


class CodeQualityScore(Enum):
    """Code quality assessment."""
    EXCELLENT = 5  # Works perfectly, efficient, documented
    GOOD = 4       # Works, minor issues
    ACCEPTABLE = 3 # Works but could be better
    POOR = 2       # Works with warnings/issues
    BROKEN = 1     # Doesn't work


class CodeError:
    """Represents a code error with context."""
    
    def __init__(self, error_type: str, message: str, line: Optional[int] = None):
        self.error_type = error_type  # "syntax", "runtime", "logic", "performance"
        self.message = message
        self.line = line
        self.detected_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "type": self.error_type,
            "message": self.message,
            "line": self.line,
            "detected_at": self.detected_at
        }


class CodeQualityEvaluator:
    """
    Test and evaluate code quality.
    Identifies errors, searches for solutions, suggests fixes.
    """
    
    def __init__(self):
        self.evaluation_history = []
        self.error_solutions = {}  # Cache of error types to solutions
        logger.info("Code Quality Evaluator initialized")
    
    def evaluate_code(self, code: str, language: str = "python", 
                     test_cases: List[Dict] = None) -> Dict:
        """
        Comprehensive code evaluation.
        Returns: quality score, errors found, recommendations.
        """
        
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "quality_score": CodeQualityScore.EXCELLENT.value,
            "errors": [],
            "warnings": [],
            "performance_issues": [],
            "recommendations": []
        }
        
        logger.info(f"Evaluating {language} code ({len(code)} chars)")
        
        # Step 1: Syntax check
        syntax_errors = self._check_syntax(code, language)
        if syntax_errors:
            evaluation["errors"].extend(syntax_errors)
            evaluation["quality_score"] = min(evaluation["quality_score"], CodeQualityScore.BROKEN.value)
        
        # Step 2: Try to run/compile
        if not syntax_errors:
            runtime_result = self._try_execute(code, language)
            if not runtime_result["success"]:
                evaluation["errors"].append(runtime_result["error"])
                evaluation["quality_score"] = min(evaluation["quality_score"], CodeQualityScore.POOR.value)
            else:
                # Step 3: Run tests if provided
                if test_cases:
                    test_result = self._run_tests(code, language, test_cases)
                    if not test_result["all_passed"]:
                        evaluation["errors"].extend(test_result["failures"])
                        evaluation["quality_score"] = min(evaluation["quality_score"], CodeQualityScore.POOR.value)
        
        # Step 4: Code quality checks
        quality_issues = self._check_code_quality(code, language)
        evaluation["warnings"].extend(quality_issues["warnings"])
        
        # Step 5: Performance checks
        perf_issues = self._check_performance(code, language)
        if perf_issues:
            evaluation["performance_issues"].extend(perf_issues)
        
        # Step 6: Maintainability check
        maintainability = self._check_maintainability(code)
        if maintainability["score"] < 0.6:
            evaluation["recommendations"].append("Code lacks clear documentation")
            evaluation["recommendations"].append("Consider adding comments and docstrings")
        
        # Set final score
        if evaluation["errors"]:
            evaluation["quality_score"] = CodeQualityScore.BROKEN.value
        elif evaluation["performance_issues"]:
            evaluation["quality_score"] = CodeQualityScore.ACCEPTABLE.value
        elif evaluation["warnings"]:
            evaluation["quality_score"] = CodeQualityScore.GOOD.value
        
        self.evaluation_history.append(evaluation)
        
        logger.info(f"Evaluation complete: quality_score={evaluation['quality_score']}, errors={len(evaluation['errors'])}")
        
        return evaluation
    
    def _check_syntax(self, code: str, language: str) -> List[CodeError]:
        """Check for syntax errors."""
        errors = []
        
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(CodeError(
                    "syntax",
                    f"SyntaxError: {e.msg}",
                    e.lineno
                ))
        
        elif language in ["csharp", "cpp", "java"]:
            # For compiled languages, we'd need actual compilers
            # For now, basic regex checks
            if language == "csharp":
                if re.search(r'var\s+\w+\s*=', code):
                    pass  # Valid C# syntax
                else:
                    pass  # Would need actual compiler
        
        return errors
    
    def _try_execute(self, code: str, language: str) -> Dict:
        """Try to run code and catch runtime errors."""
        
        if language == "python":
            try:
                # Create a safe execution environment
                exec_globals = {"__builtins__": {}}
                exec(code, exec_globals)
                return {"success": True, "error": None}
            except Exception as e:
                error = CodeError("runtime", str(e))
                return {
                    "success": False,
                    "error": error.to_dict()
                }
        
        # For other languages, would need actual compilers
        return {"success": True, "error": None}
    
    def _run_tests(self, code: str, language: str, test_cases: List[Dict]) -> Dict:
        """Run provided test cases against code."""
        
        results = {
            "all_passed": True,
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for i, test in enumerate(test_cases):
            try:
                # Execute test
                exec_globals = {"__builtins__": {}}
                exec(code, exec_globals)
                
                # Call the function with inputs
                func_name = test.get("function")
                inputs = test.get("inputs", [])
                expected = test.get("expected")
                
                if func_name and func_name in exec_globals:
                    result = exec_globals[func_name](*inputs)
                    
                    if result == expected:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["all_passed"] = False
                        results["failures"].append(CodeError(
                            "logic",
                            f"Test {i}: expected {expected}, got {result}"
                        ).to_dict())
            except Exception as e:
                results["failed"] += 1
                results["all_passed"] = False
                results["failures"].append(CodeError("runtime", str(e)).to_dict())
        
        return results
    
    def _check_code_quality(self, code: str, language: str) -> Dict:
        """Check for code quality issues."""
        
        warnings = []
        
        # Universal checks
        lines = code.split('\n')
        
        # Check for very long lines
        long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            warnings.append("Lines too long (>120 chars) - reduce line length for readability")
        
        # Check for commented-out code
        commented_count = sum(1 for line in lines if line.strip().startswith('#') and len(line.strip()) > 1)
        if commented_count > len(lines) * 0.1:
            warnings.append("Too much commented-out code - clean up")
        
        # Language-specific
        if language == "python":
            # Check for unused imports
            if "import" in code and "from" in code:
                warnings.append("Review imports - ensure all are used")
            
            # Check for bare excepts
            if "except:" in code:
                warnings.append("Avoid bare except: - specify exception types")
        
        elif language == "csharp":
            # Check for null reference potential
            if "NullReferenceException" in code or ".ToString()" in code:
                warnings.append("Potential null reference issues - add null checks")
        
        return {"warnings": warnings}
    
    def _check_performance(self, code: str, language: str) -> List[str]:
        """Identify potential performance issues."""
        
        issues = []
        
        # Look for nested loops
        if code.count("for ") > 1:
            # Simple heuristic: multiple for loops might be O(n²)
            lines = code.split('\n')
            indent_levels = [len(line) - len(line.lstrip()) for line in lines]
            if max(indent_levels, default=0) > 8:  # Deep nesting
                issues.append("Potential O(n²) performance - consider optimizing nested loops")
        
        # Look for string concatenation in loops
        if "for " in code and '+=' in code and '"' in code:
            issues.append("String concatenation in loop - use list + join() for efficiency")
        
        # Look for repeated calculations
        if code.count("math.sqrt") > 3 or code.count("len(") > 3:
            issues.append("Repeated calculations detected - consider caching results")
        
        return issues
    
    def _check_maintainability(self, code: str) -> Dict:
        """Check code maintainability."""
        
        lines = code.split('\n')
        
        # Calculate simple maintainability score
        score = 1.0
        
        # Check for documentation
        doc_lines = sum(1 for line in lines if '"""' in line or "'''" in line or line.strip().startswith('#'))
        doc_ratio = doc_lines / max(len(lines), 1)
        
        # Penalize if too little documentation
        if doc_ratio < 0.1:
            score -= 0.3
        
        # Check function length
        function_lengths = []
        in_function = False
        function_lines = 0
        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                in_function = True
                function_lines = 0
            elif in_function:
                if line.strip() and not line[0].isspace():
                    function_lengths.append(function_lines)
                    in_function = False
                else:
                    function_lines += 1
        
        # Penalize if functions too long
        if function_lengths and max(function_lengths) > 50:
            score -= 0.2
        
        # Check for variable naming clarity
        bad_names = sum(1 for word in re.findall(r'\b[x-z]\b', code))
        if bad_names > 5:
            score -= 0.1
        
        return {"score": max(0.0, score)}
    
    def get_error_solutions(self, error: CodeError, language: str) -> List[Dict]:
        """
        Find solutions for an error by searching information.
        In a real system, this would search StackOverflow, docs, etc.
        """
        
        solutions = []
        error_key = f"{language}:{error.error_type}:{error.message[:50]}"
        
        # Check cache first
        if error_key in self.error_solutions:
            return self.error_solutions[error_key]
        
        # Generate solutions based on error pattern
        if "SyntaxError" in error.message:
            solutions.append({
                "source": "Python Docs",
                "category": "syntax",
                "solution": "Check Python syntax documentation - likely missing colon, indentation, or quote",
                "examples": [
                    "Missing colon after if/for/def/class",
                    "Indentation error",
                    "Mismatched parentheses or quotes"
                ]
            })
        
        elif "NullReferenceException" in error.message or "None" in error.message:
            solutions.append({
                "source": "Best Practice",
                "category": "runtime",
                "solution": "Add null/None checks before accessing properties",
                "code_pattern": "if variable is not None: ..." if language == "python" else "if (variable != null) { ... }"
            })
        
        elif "Index" in error.message:
            solutions.append({
                "source": "Array Access",
                "category": "logic",
                "solution": "Array index out of bounds - check array length before accessing",
                "debug_step": "Print array length and index being accessed"
            })
        
        elif "Type" in error.message or "isinstance" in error.message:
            solutions.append({
                "source": "Type Error",
                "category": "logic",
                "solution": "Type mismatch - ensure types match expected values",
                "debug_step": "Add type checking: print(type(variable))"
            })
        
        else:
            # Generic error handling
            solutions.append({
                "source": "Generic Debug",
                "category": "unknown",
                "solution": "Add print statements around the error line to debug",
                "debug_step": "Insert debug output to trace execution"
            })
        
        # Cache for future reference
        self.error_solutions[error_key] = solutions
        
        return solutions


class CodeIterativeRefiner:
    """
    Autonomously fix code errors through iteration.
    Up to 5 attempts max.
    """
    
    def __init__(self, evaluator: CodeQualityEvaluator):
        self.evaluator = evaluator
        self.refinement_history = []
        logger.info("Code Iterative Refiner initialized")
    
    def refine_until_working(self, code: str, language: str = "python",
                            test_cases: List[Dict] = None,
                            max_iterations: int = 5) -> Dict:
        """
        Autonomously fix code until it works.
        Returns: final code, iteration count, success status.
        """
        
        current_code = code
        iteration = 0
        
        logger.info(f"Starting refinement loop (max {max_iterations} iterations)")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Evaluate current code
            evaluation = self.evaluator.evaluate_code(current_code, language, test_cases)
            
            # Check if working
            if (evaluation["quality_score"] >= CodeQualityScore.GOOD.value and 
                not evaluation["errors"]):
                logger.info(f"✓ Code working at iteration {iteration}")
                return {
                    "success": True,
                    "final_code": current_code,
                    "iterations": iteration,
                    "final_score": evaluation["quality_score"],
                    "history": self.refinement_history
                }
            
            # If broken, get solutions
            if evaluation["errors"]:
                error = evaluation["errors"][0]
                logger.error(f"Error found: {error['message']}")
                
                solutions = self.evaluator.get_error_solutions(
                    CodeError(error["type"], error["message"]),
                    language
                )
                
                logger.info(f"Found {len(solutions)} potential solutions")
                
                # Try to apply fix (in real system, would use LLM + solutions)
                fixed_code = self._generate_fix(current_code, error, solutions, language)
                
                if fixed_code and fixed_code != current_code:
                    current_code = fixed_code
                    
                    self.refinement_history.append({
                        "iteration": iteration,
                        "error": error,
                        "solutions_found": len(solutions),
                        "fix_applied": True
                    })
                else:
                    logger.warning("Could not generate fix")
                    self.refinement_history.append({
                        "iteration": iteration,
                        "error": error,
                        "solutions_found": len(solutions),
                        "fix_applied": False
                    })
            
            # Handle warnings/performance issues
            if evaluation["warnings"] or evaluation["performance_issues"]:
                current_code = self._optimize_code(current_code, evaluation, language)
        
        logger.warning(f"Max iterations ({max_iterations}) reached without full success")
        
        return {
            "success": False,
            "final_code": current_code,
            "iterations": iteration,
            "final_score": evaluation["quality_score"],
            "remaining_errors": evaluation["errors"],
            "history": self.refinement_history
        }
    
    def _generate_fix(self, code: str, error: Dict, solutions: List[Dict], 
                     language: str) -> Optional[str]:
        """
        Generate a fix based on error and solutions found.
        In production, this would use Claude/GPT with grounded solutions.
        """
        
        # This is a simplified version - in real system, would use LLM
        # For now, return code with suggestion for where to look
        
        # Simple patterns we can fix directly
        if "SyntaxError" in error["message"]:
            # Try basic fixes
            if "EOL while scanning" in error["message"]:
                return code + "\n"  # Missing newline
            
            if "expected ':'" in error["message"]:
                # Add missing colon before block
                lines = code.split('\n')
                if error.get("line"):
                    lines[error["line"] - 1] = lines[error["line"] - 1].rstrip() + ":"
                return '\n'.join(lines)
        
        # For logic errors, we'd need LLM + solutions to generate proper fix
        logger.info("Requires LLM assistance for fix generation")
        
        return None
    
    def _optimize_code(self, code: str, evaluation: Dict, language: str) -> str:
        """Apply optimizations based on evaluation."""
        
        # Simple optimizations we can do directly
        if any("bare except" in w for w in evaluation["warnings"]):
            code = code.replace("except:", "except Exception:")
        
        if any("String concatenation" in w for w in evaluation["performance_issues"]):
            # This would require more sophisticated refactoring
            logger.info("String concatenation optimization needed - requires LLM")
        
        return code


if __name__ == "__main__":
    print("Code Quality Evaluator & Iterative Refiner")
    print("=" * 60)
    
    evaluator = CodeQualityEvaluator()
    refiner = CodeIterativeRefiner(evaluator)
    
    # Test 1: Good code
    print("\n[Test 1: Good Code]")
    good_code = """
def add(a, b):
    '''Add two numbers'''
    return a + b

result = add(2, 3)
print(result)
"""
    eval1 = evaluator.evaluate_code(good_code, "python")
    print(f"Quality score: {eval1['quality_score']}")
    print(f"Errors: {len(eval1['errors'])}")
    
    # Test 2: Broken code
    print("\n[Test 2: Broken Code]")
    broken_code = """
def add(a, b)
    return a + b

result = add(2, 3)
print(result)
"""
    eval2 = evaluator.evaluate_code(broken_code, "python")
    print(f"Quality score: {eval2['quality_score']}")
    print(f"Errors: {[e['message'] for e in eval2['errors']]}")
    
    # Test 3: Iterative refinement
    print("\n[Test 3: Iterative Refinement]")
    result = refiner.refine_until_working(broken_code, "python", max_iterations=5)
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final score: {result['final_score']}")
