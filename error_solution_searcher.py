"""
Error Solution Searcher
Searches for real information about errors to ground LLM fixes
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("error_searcher")


class ErrorSolutionDatabase:
    """
    Local database of error patterns and solutions.
    Would be augmented with real searches in production.
    """
    
    def __init__(self):
        self.error_patterns = {}
        self.solution_cache = {}
        self._populate_common_errors()
        logger.info("Error Solution Database initialized")
    
    def _populate_common_errors(self):
        """Populate with common game dev errors."""
        
        # Python errors
        self.error_patterns["python:syntax:SyntaxError"] = {
            "description": "Python syntax error",
            "common_causes": [
                "Missing colon after if/for/while/def/class",
                "Incorrect indentation",
                "Mismatched parentheses, brackets, or quotes",
                "Using = instead of == for comparison",
                "Missing comma in list/dict"
            ],
            "solutions": [
                "Check line ending - should have ':' for control statements",
                "Verify indentation is consistent (spaces vs tabs)",
                "Count parentheses/brackets - ensure all opened are closed",
                "Look for = when == is needed"
            ],
            "resources": [
                "https://docs.python.org/3/tutorial/errors.html",
                "Python syntax docs"
            ]
        }
        
        self.error_patterns["python:runtime:AttributeError"] = {
            "description": "Attribute does not exist on object",
            "common_causes": [
                "Accessing property that doesn't exist",
                "Object is None when accessing attribute",
                "Typo in property name",
                "Object type is wrong"
            ],
            "solutions": [
                "Check object type: print(type(obj))",
                "Verify property name spelling",
                "Add None check: if obj is not None",
                "Check class definition for the property"
            ],
            "resources": [
                "https://docs.python.org/3/library/exceptions.html#AttributeError"
            ]
        }
        
        self.error_patterns["python:runtime:IndexError"] = {
            "description": "List/array index out of range",
            "common_causes": [
                "Accessing index that doesn't exist",
                "Empty list",
                "Off-by-one error in loop"
            ],
            "solutions": [
                "Add length check: if len(list) > index",
                "Use try/except for boundary checking",
                "Check loop bounds: range(len(list))",
                "Print list length and index for debugging"
            ],
            "resources": [
                "https://docs.python.org/3/tutorial/datastructures.html"
            ]
        }
        
        self.error_patterns["python:runtime:TypeError"] = {
            "description": "Operation or function applied to wrong type",
            "common_causes": [
                "Trying to add string and number",
                "Passing wrong type to function",
                "None where value expected"
            ],
            "solutions": [
                "Check variable type: print(type(var))",
                "Convert type: int(), str(), float()",
                "Add type hints: def func(x: int) -> str",
                "Verify function signature"
            ],
            "resources": [
                "https://docs.python.org/3/library/exceptions.html#TypeError"
            ]
        }
        
        # C# / Unity errors
        self.error_patterns["csharp:runtime:NullReferenceException"] = {
            "description": "Accessing property on null object",
            "common_causes": [
                "GameObject not found in scene",
                "Component not attached",
                "Object destroyed but still referenced"
            ],
            "solutions": [
                "Add null check: if (obj != null) { ... }",
                "Use GameObject.Find() to locate object",
                "Use GetComponent<>() with null check",
                "Log before access: Debug.Log(obj == null)"
            ],
            "resources": [
                "https://docs.unity3d.com/Manual/BestPractice.html",
                "Unity best practices"
            ]
        }
        
        self.error_patterns["csharp:compile:CS0103"] = {
            "description": "Variable/method not found",
            "common_causes": [
                "Typo in variable name",
                "Variable not declared",
                "Variable out of scope",
                "Method doesn't exist on class"
            ],
            "solutions": [
                "Check spelling of variable/method",
                "Verify variable declaration",
                "Check scope - is it accessible here?",
                "Look at class definition for method"
            ],
            "resources": [
                "https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/compiler-messages/cs0103"
            ]
        }
        
        # Game-specific
        self.error_patterns["unity:runtime:MissingComponentException"] = {
            "description": "Component not found on GameObject",
            "common_causes": [
                "Component not attached in scene",
                "Trying to access before component added",
                "Wrong component type"
            ],
            "solutions": [
                "Add component in Unity editor",
                "Use AddComponent<>() if needed",
                "Verify GetComponent<>() type matches",
                "Check component is enabled"
            ],
            "resources": [
                "https://docs.unity3d.com/ScriptReference/GameObject.GetComponent.html"
            ]
        }
        
        self.error_patterns["performance:fps:Slow"] = {
            "description": "Game running at low FPS",
            "common_causes": [
                "Too many objects in scene",
                "Expensive physics calculations",
                "Inefficient rendering",
                "Memory leaks"
            ],
            "solutions": [
                "Profile with Profiler: Window > Analysis > Profiler",
                "Reduce object count or use pooling",
                "Optimize shaders and materials",
                "Check for memory leaks: objects not destroyed"
            ],
            "resources": [
                "https://docs.unity3d.com/Manual/Profiler.html",
                "Unity performance optimization"
            ]
        }
    
    def search_error(self, error_message: str, language: str, error_type: str) -> Dict:
        """
        Search for solutions to an error.
        Returns relevant information and solutions.
        """
        
        # Try exact match first
        pattern_key = f"{language}:{error_type}:{error_message[:50]}"
        
        if pattern_key in self.error_patterns:
            return {
                "match_type": "exact",
                "error_pattern": self.error_patterns[pattern_key],
                "confidence": 0.95
            }
        
        # Try partial match on error type
        for key, pattern in self.error_patterns.items():
            if error_type in key and language in key:
                return {
                    "match_type": "type",
                    "error_pattern": pattern,
                    "confidence": 0.7
                }
        
        # Generic fallback
        return {
            "match_type": "generic",
            "error_pattern": {
                "description": "Unknown error",
                "common_causes": ["Check error message carefully"],
                "solutions": [
                    "Read error message fully",
                    "Search error message online",
                    "Check documentation for language/framework"
                ],
                "resources": []
            },
            "confidence": 0.3
        }
    
    def get_debugging_steps(self, error_type: str) -> List[str]:
        """Get standard debugging steps for an error type."""
        
        steps = [
            "1. Read the complete error message",
            "2. Identify the line number if provided",
            "3. Check the problematic line and surrounding code",
            "4. Look at variable types at that point",
            "5. Add print/debug statements around error",
            "6. Check if similar error appears in documentation",
            "7. Search error online (StackOverflow, docs)",
            "8. Try minimal reproduction case"
        ]
        
        if "AttributeError" in error_type or "NullReference" in error_type:
            steps.append("9. Add null/None check before accessing")
        
        if "IndexError" in error_type:
            steps.append("9. Add length check: if len(list) > index")
        
        if "TypeError" in error_type:
            steps.append("9. Check and convert types as needed")
        
        return steps


class ErrorSolutionSearcher:
    """
    Search for error solutions combining database + simulated web search.
    """
    
    def __init__(self):
        self.database = ErrorSolutionDatabase()
        self.search_history = []
        logger.info("Error Solution Searcher initialized")
    
    def find_solution(self, error_message: str, language: str, 
                     code_context: str = "") -> Dict:
        """
        Find solution to error using database + context.
        Returns: error info, solutions, confidence, resources.
        """
        
        # Parse error message
        error_type = self._extract_error_type(error_message)
        
        logger.info(f"Searching for solution: {error_type}")
        
        # Search database
        search_result = self.database.search_error(error_message, language, error_type)
        
        error_info = search_result["error_pattern"]
        
        # Build comprehensive solution
        solution = {
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "error_type": error_type,
            "language": language,
            "confidence": search_result["confidence"],
            "description": error_info.get("description", "Unknown error"),
            "common_causes": error_info.get("common_causes", []),
            "solutions": error_info.get("solutions", []),
            "debugging_steps": self.database.get_debugging_steps(error_type),
            "resources": error_info.get("resources", []),
            "code_context": code_context[:200] if code_context else ""
        }
        
        self.search_history.append(solution)
        
        return solution
    
    def _extract_error_type(self, error_message: str) -> str:
        """Extract error type from message."""
        
        # Common patterns
        patterns = {
            "SyntaxError": ["SyntaxError", "unexpected indent", "EOL"],
            "AttributeError": ["AttributeError", "has no attribute"],
            "IndexError": ["IndexError", "list index out", "string index out"],
            "TypeError": ["TypeError", "type", "int", "str", "cannot"],
            "NullReferenceException": ["NullReferenceException", "Object reference not set"],
            "CS0103": ["CS0103", "does not exist"],
            "MissingComponentException": ["MissingComponentException", "Component"],
            "Performance": ["FPS", "slow", "frame", "lag"]
        }
        
        message_lower = error_message.lower()
        
        for error_type, indicators in patterns.items():
            if any(ind.lower() in message_lower for ind in indicators):
                return error_type
        
        return "Unknown"
    
    def suggest_fix_pattern(self, error_type: str, language: str) -> str:
        """Suggest code fix pattern."""
        
        if language == "python":
            if "AttributeError" in error_type:
                return """
# Add null check
if obj is not None:
    obj.property
"""
            elif "IndexError" in error_type:
                return """
# Add length check
if len(list) > index:
    value = list[index]
"""
        
        elif language == "csharp":
            if "NullReference" in error_type:
                return """
// Add null check
if (obj != null)
{
    obj.Property();
}
"""
        
        return "# Add appropriate null/bounds checking"
    
    def get_search_summary(self) -> Dict:
        """Get summary of searches performed."""
        
        return {
            "total_searches": len(self.search_history),
            "error_types": list(set(s["error_type"] for s in self.search_history)),
            "languages": list(set(s["language"] for s in self.search_history)),
            "avg_confidence": sum(s["confidence"] for s in self.search_history) / max(len(self.search_history), 1),
            "recent_searches": self.search_history[-5:]
        }


if __name__ == "__main__":
    print("Error Solution Searcher")
    print("=" * 60)
    
    searcher = ErrorSolutionSearcher()
    
    # Test searches
    print("\n[Test 1: Python SyntaxError]")
    result1 = searcher.find_solution(
        "SyntaxError: expected ':'",
        "python"
    )
    print(f"Error type: {result1['error_type']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Solutions: {result1['solutions'][:2]}")
    
    print("\n[Test 2: Unity NullReferenceException]")
    result2 = searcher.find_solution(
        "NullReferenceException: Object reference not set to an instance of an object",
        "csharp"
    )
    print(f"Error type: {result2['error_type']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Solutions: {result2['solutions'][:2]}")
    
    print("\n[Test 3: Unknown Error]")
    result3 = searcher.find_solution(
        "Something went wrong",
        "python"
    )
    print(f"Error type: {result3['error_type']}")
    print(f"Confidence: {result3['confidence']:.2f}")
    
    print("\n[Summary]")
    summary = searcher.get_search_summary()
    print(f"Total searches: {summary['total_searches']}")
    print(f"Avg confidence: {summary['avg_confidence']:.2f}")
