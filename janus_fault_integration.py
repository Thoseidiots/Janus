"""
Janus AI Fault Detection Integration

This module integrates the AI fault detection system with the existing Janus AI components
to provide real-time code quality monitoring and prevention of faulty AI-generated code.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

from ai_fault_detector import AIFaultDetector, DetectionResult, FaultSeverity
from fault_monitor import JanusFaultIntegration, RealTimeFaultMonitor

logger = logging.getLogger(__name__)

@dataclass
class JanusCodeQualityMetrics:
    """Metrics for Janus AI code quality"""
    total_code_generations: int = 0
    blocked_generations: int = 0
    average_quality_score: float = 0.0
    critical_faults_prevented: int = 0
    last_check_time: float = 0.0

class JanusAIGuard:
    """Main AI guard system for Janus"""
    
    def __init__(self):
        self.fault_integration = JanusFaultIntegration()
        self.detector = AIFaultDetector()
        self.metrics = JanusCodeQualityMetrics()
        self.quality_threshold = 70.0
        self.block_critical_faults = True
        
        # Setup integration
        self.fault_integration.setup_janus_integration()
        
        # Add custom fault callback for Janus
        self.fault_integration.monitor.add_fault_callback(self._janus_quality_callback)
        
        logger.info("Janus AI Guard initialized")
    
    def validate_ai_generation(self, content: str, content_type: str = "html", 
                             context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate AI-generated content before allowing it to be used
        
        Args:
            content: The AI-generated content
            content_type: Type of content (html, css, javascript)
            context: Additional context about the generation
            
        Returns:
            Dictionary with validation results
        """
        self.metrics.total_code_generations += 1
        self.metrics.last_check_time = time.time()
        
        # Run fault detection
        if content_type == "html":
            result = self.detector.analyze_code(content)
        elif content_type == "css":
            result = self.detector.analyze_code("", content)
        elif content_type == "javascript":
            result = self.detector.analyze_code("", "", content)
        else:
            result = self.detector.analyze_code(content, "", "")
        
        # Check for critical faults
        critical_faults = [f for f in result.fault_reports if f.severity == FaultSeverity.CRITICAL]
        
        # Update metrics
        self.metrics.average_quality_score = (
            (self.metrics.average_quality_score * (self.metrics.total_code_generations - 1) + 
             result.code_quality_score) / self.metrics.total_code_generations
        )
        
        # Determine if content should be blocked
        is_blocked = False
        block_reason = None
        
        if self.block_critical_faults and critical_faults:
            is_blocked = True
            block_reason = f"Critical faults detected: {len(critical_faults)}"
            self.metrics.blocked_generations += 1
            self.metrics.critical_faults_prevented += len(critical_faults)
        elif result.code_quality_score < self.quality_threshold:
            is_blocked = True
            block_reason = f"Quality score below threshold: {result.code_quality_score}"
            self.metrics.blocked_generations += 1
        
        # Generate suggestions
        suggestions = self._generate_suggestions(result, context)
        
        return {
            "is_allowed": not is_blocked,
            "is_blocked": is_blocked,
            "block_reason": block_reason,
            "quality_score": result.code_quality_score,
            "total_faults": len(result.fault_reports),
            "critical_faults": len(critical_faults),
            "fault_reports": [
                {
                    "category": fault.category.value,
                    "severity": fault.severity.value,
                    "title": fault.title,
                    "description": fault.description,
                    "line_number": fault.line_number,
                    "suggested_fix": fault.suggested_fix
                }
                for fault in result.fault_reports
            ],
            "suggestions": suggestions,
            "metrics": {
                "total_generations": self.metrics.total_code_generations,
                "blocked_generations": self.metrics.blocked_generations,
                "average_quality": self.metrics.average_quality_score,
                "critical_faults_prevented": self.metrics.critical_faults_prevented
            }
        }
    
    def _generate_suggestions(self, result: DetectionResult, context: Optional[Dict]) -> List[str]:
        """Generate specific suggestions based on detected faults"""
        suggestions = []
        
        # Add general recommendations from detector
        suggestions.extend(result.recommendations)
        
        # Add context-specific suggestions
        if context:
            if context.get("is_game_character", False):
                positioning_faults = [f for f in result.fault_reports 
                                   if f.category.value == "positioning"]
                if positioning_faults:
                    suggestions.append("For game characters, use viewport-relative units (vw, vh) or percentage positioning")
                    suggestions.append("Consider implementing boundary checking to keep characters within game area")
            
            if context.get("is_dialogue_system", False):
                interactivity_faults = [f for f in result.fault_reports 
                                      if f.category.value == "interactivity"]
                if interactivity_faults:
                    suggestions.append("Dialogue systems should maintain pointer-events: auto when visible")
                    suggestions.append("Use opacity transitions instead of display changes for smoother dialogue")
        
        return suggestions
    
    def _janus_quality_callback(self, session_id: str, result: DetectionResult):
        """Callback for handling Janus-specific quality issues"""
        if result.code_quality_score < 50:
            logger.warning(f"Janus AI generated very low quality code (Score: {result.code_quality_score})")
            
            # Could trigger self-correction mechanisms here
            # For example: request regeneration with different parameters
        
        # Log to holographic brain memory if available
        try:
            # This would integrate with janus_memory.py or similar
            memory_entry = {
                "timestamp": time.time(),
                "event_type": "code_quality_check",
                "quality_score": result.code_quality_score,
                "fault_count": len(result.fault_reports),
                "session_id": session_id
            }
            # Store in holographic brain memory
            logger.info(f"Quality check logged: {memory_entry}")
        except Exception as e:
            logger.debug(f"Could not log to holographic memory: {e}")
    
    def start_monitoring_session(self, project_path: str, session_name: str = "default") -> str:
        """Start a monitoring session for a project"""
        session_id = self.fault_integration.monitor.start_monitoring(project_path, session_name)
        logger.info(f"Started Janus monitoring session: {session_id}")
        return session_id
    
    def stop_monitoring_session(self, session_id: str):
        """Stop a monitoring session"""
        session = self.fault_integration.monitor.stop_monitoring(session_id)
        if session:
            logger.info(f"Stopped monitoring session {session_id}: {session.total_checks} checks, {session.faults_detected} faults")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics"""
        return {
            "total_code_generations": self.metrics.total_code_generations,
            "blocked_generations": self.metrics.blocked_generations,
            "average_quality_score": self.metrics.average_quality_score,
            "critical_faults_prevented": self.metrics.critical_faults_prevented,
            "block_rate": (self.metrics.blocked_generations / max(1, self.metrics.total_code_generations)) * 100,
            "last_check_time": self.metrics.last_check_time
        }
    
    def generate_quality_report(self) -> str:
        """Generate a comprehensive quality report"""
        metrics = self.get_quality_metrics()
        
        report = f"""
Janus AI Code Quality Report
============================

Total Code Generations: {metrics['total_code_generations']}
Blocked Generations: {metrics['blocked_generations']} ({metrics['block_rate']:.1f}%)
Average Quality Score: {metrics['average_quality_score']:.1f}/100
Critical Faults Prevented: {metrics['critical_faults_prevented']}

Quality Assessment:
"""
        
        if metrics['average_quality_score'] >= 80:
            report += "EXCELLENT - AI is generating high-quality code consistently\n"
        elif metrics['average_quality_score'] >= 70:
            report += "GOOD - AI code quality is acceptable with room for improvement\n"
        elif metrics['average_quality_score'] >= 50:
            report += "FAIR - AI code quality needs attention and refinement\n"
        else:
            report += "POOR - AI is generating low-quality code that requires intervention\n"
        
        if metrics['block_rate'] > 20:
            report += f"\nWARNING: High block rate ({metrics['block_rate']:.1f}%) indicates AI needs guidance\n"
        
        report += f"\nLast quality check: {time.ctime(metrics['last_check_time']) if metrics['last_check_time'] else 'Never'}"
        
        return report.strip()

# Integration with existing Janus systems
def integrate_with_janus_main():
    """Integration function for janus_main.py"""
    
    # Create global AI guard instance
    ai_guard = JanusAIGuard()
    
    # Example integration point for AI code generation
    def generate_and_validate_code(prompt: str, code_type: str = "html", context: Dict = None) -> Dict:
        """
        Wrapper function that generates code and validates it
        This would be called from Janus AI generation functions
        """
        # Generate code using existing Janus AI systems
        # generated_code = janus_ai.generate_code(prompt, code_type)
        
        # For demo, use placeholder
        generated_code = f"<div>Generated from: {prompt}</div>"
        
        # Validate the generated code
        validation_result = ai_guard.validate_ai_generation(generated_code, code_type, context)
        
        if validation_result["is_allowed"]:
            return {
                "success": True,
                "code": generated_code,
                "quality_score": validation_result["quality_score"],
                "suggestions": validation_result["suggestions"]
            }
        else:
            return {
                "success": False,
                "reason": validation_result["block_reason"],
                "faults": validation_result["fault_reports"],
                "suggestions": validation_result["suggestions"]
            }
    
    return ai_guard, generate_and_validate_code

# Example usage
if __name__ == "__main__":
    # Initialize the AI guard
    ai_guard = JanusAIGuard()
    
    # Test with problematic code (character off-screen)
    faulty_character_code = """
    <div class="game-character" style="position: fixed; left: -1500px; top: -800px; pointer-events: none;">
        Character sprite
    </div>
    """
    
    result = ai_guard.validate_ai_generation(
        faulty_character_code, 
        "html", 
        context={"is_game_character": True}
    )
    
    print("=== Janus AI Guard Test ===")
    print(f"Code Allowed: {result['is_allowed']}")
    print(f"Quality Score: {result['quality_score']}")
    print(f"Critical Faults: {result['critical_faults']}")
    
    if result['is_blocked']:
        print(f"Blocked Reason: {result['block_reason']}")
    
    print("\nSuggestions:")
    for suggestion in result['suggestions']:
        print(f"- {suggestion}")
    
    print("\nFault Reports:")
    for fault in result['fault_reports']:
        print(f"- [{fault['severity'].upper()}] {fault['title']}")
    
    # Generate quality report
    print("\n" + ai_guard.generate_quality_report())
    
    # Test monitoring
    session_id = ai_guard.start_monitoring_session("./test_project", "character_test")
    
    # Simulate some time passing
    time.sleep(2)
    
    ai_guard.stop_monitoring_session(session_id)
    
    print("\n=== Test Complete ===")
