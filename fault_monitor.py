"""
Real-time Fault Monitor - Integrates with Janus for continuous AI code monitoring

This module provides real-time monitoring of AI-generated code changes and
automatically detects faults as they're introduced.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ai_fault_detector import AIFaultDetector, DetectionResult, FaultSeverity, FaultCategory

logger = logging.getLogger(__name__)

@dataclass
class MonitoringSession:
    """Active monitoring session for code changes"""
    session_id: str
    start_time: float
    last_check: float
    total_checks: int
    faults_detected: int
    quality_trend: List[float] = field(default_factory=list)

class RealTimeFaultMonitor:
    """Real-time monitoring system for AI-generated code"""
    
    def __init__(self, check_interval: float = 5.0):
        self.detector = AIFaultDetector()
        self.check_interval = check_interval
        self.active_sessions: Dict[str, MonitoringSession] = {}
        self.fault_callbacks: List[Callable] = []
        self.quality_threshold = 70.0  # Alert if quality drops below this
        self.is_monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, project_path: str, session_id: Optional[str] = None) -> str:
        """Start monitoring a project for code quality"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        session = MonitoringSession(
            session_id=session_id,
            start_time=time.time(),
            last_check=0,
            total_checks=0,
            faults_detected=0
        )
        
        self.active_sessions[session_id] = session
        
        if not self.is_monitoring:
            self.is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        
        logger.info(f"Started monitoring session {session_id} for project {project_path}")
        return session_id
    
    def stop_monitoring(self, session_id: str) -> Optional[MonitoringSession]:
        """Stop monitoring a specific session"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            logger.info(f"Stopped monitoring session {session_id}")
            
            if not self.active_sessions:
                self.is_monitoring = False
            
            return session
        return None
    
    def add_fault_callback(self, callback: Callable[[str, DetectionResult], None]):
        """Add callback function to be called when faults are detected"""
        self.fault_callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring and self.active_sessions:
            current_time = time.time()
            
            for session_id, session in list(self.active_sessions.items()):
                if current_time - session.last_check >= self.check_interval:
                    self._check_session(session_id, session)
                    session.last_check = current_time
            
            time.sleep(1)
    
    def _check_session(self, session_id: str, session: MonitoringSession):
        """Check a specific monitoring session"""
        try:
            # This would integrate with file system watchers or git hooks
            # For now, we'll simulate checking recent changes
            result = self._analyze_recent_changes(session_id)
            
            if result:
                session.total_checks += 1
                session.faults_detected += len(result.fault_reports)
                session.quality_trend.append(result.code_quality_score)
                
                # Alert if quality drops below threshold
                if result.code_quality_score < self.quality_threshold:
                    self._trigger_quality_alert(session_id, result)
                
                # Call fault callbacks
                for callback in self.fault_callbacks:
                    try:
                        callback(session_id, result)
                    except Exception as e:
                        logger.error(f"Error in fault callback: {e}")
        
        except Exception as e:
            logger.error(f"Error checking session {session_id}: {e}")
    
    def _analyze_recent_changes(self, session_id: str) -> Optional[DetectionResult]:
        """Analyze recent code changes (placeholder implementation)"""
        # In a real implementation, this would:
        # 1. Check for recently modified files
        # 2. Extract HTML, CSS, JS content
        # 3. Run fault detection
        # 4. Return results
        
        # For demo purposes, return None
        return None
    
    def _trigger_quality_alert(self, session_id: str, result: DetectionResult):
        """Trigger alert for low code quality"""
        critical_faults = [f for f in result.fault_reports if f.severity == FaultSeverity.CRITICAL]
        
        alert_message = f"""
        QUALITY ALERT - Session {session_id}
        Code Quality Score: {result.code_quality_score}/100
        Critical Faults: {len(critical_faults)}
        
        Top Issues:
        """
        
        for fault in result.fault_reports[:3]:
            alert_message += f"- {fault.title}\n"
        
        logger.warning(alert_message)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a monitoring session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        return {
            'session_id': session_id,
            'duration': current_time - session.start_time,
            'total_checks': session.total_checks,
            'faults_detected': session.faults_detected,
            'average_quality': sum(session.quality_trend) / len(session.quality_trend) if session.quality_trend else 0,
            'quality_trend': session.quality_trend[-10:]  # Last 10 checks
        }

class FaultPreventionSystem:
    """Prevention system that intercepts potentially problematic AI code"""
    
    def __init__(self):
        self.detector = AIFaultDetector()
        self.blocked_patterns = {
            'positioning': [
                r'position:\s*fixed.*?(left|top|right|bottom):\s*-\d{3,}',
                r'translate[XY]\(\s*-\d{4,}',
            ],
            'interactivity': [
                r'pointer-events:\s*none.*?cursor:\s*pointer',
                r'opacity:\s*0.*?pointer-events:\s*auto',
            ],
            'security': [
                r'eval\s*\(',
                r'innerHTML\s*=.*?\+.*?user',
            ]
        }
    
    def validate_code_before_execution(self, code: str, code_type: str) -> tuple[bool, List[str]]:
        """Validate code before allowing execution"""
        issues = []
        
        # Check for blocked patterns
        for category, patterns in self.blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(f"Blocked pattern detected in {category}: {pattern}")
        
        # Run full fault detection
        if code_type == 'html':
            result = self.detector.analyze_code(code)
        elif code_type == 'css':
            result = self.detector.analyze_code("", code)
        elif code_type == 'javascript':
            result = self.detector.analyze_code("", "", code)
        else:
            result = self.detector.analyze_code(code, "", "")
        
        # Check for critical faults
        critical_faults = [f for f in result.fault_reports if f.severity == FaultSeverity.CRITICAL]
        if critical_faults:
            issues.extend([f"Critical fault: {fault.title}" for fault in critical_faults])
        
        # Block execution if critical issues found
        is_safe = len(issues) == 0
        
        return is_safe, issues
    
    def suggest_fixes(self, code: str, code_type: str) -> List[str]:
        """Suggest fixes for detected issues"""
        result = self.detector.analyze_code(code, "", "") if code_type == 'html' else \
                 self.detector.analyze_code("", code, "") if code_type == 'css' else \
                 self.detector.analyze_code("", "", code)
        
        suggestions = []
        for fault in result.fault_reports:
            if fault.suggested_fix:
                suggestions.append(f"{fault.title}: {fault.suggested_fix}")
        
        return suggestions

# Integration with existing Janus systems
class JanusFaultIntegration:
    """Integration layer for Janus AI systems"""
    
    def __init__(self):
        self.monitor = RealTimeFaultMonitor()
        self.prevention = FaultPreventionSystem()
        self.fault_history = []
        
    def setup_janus_integration(self):
        """Setup integration with Janus AI components"""
        # Add fault detection to AI output validation
        self.monitor.add_fault_callback(self._janus_fault_callback)
        
        # Setup prevention for code execution
        # This would integrate with janus_main.py or similar
        
        logger.info("Janus fault detection integration configured")
    
    def _janus_fault_callback(self, session_id: str, result: DetectionResult):
        """Callback for Janus-specific fault handling"""
        # Store fault history
        self.fault_history.append({
            'session_id': session_id,
            'timestamp': time.time(),
            'quality_score': result.code_quality_score,
            'fault_count': len(result.fault_reports),
            'critical_faults': len([f for f in result.fault_reports if f.severity == FaultSeverity.CRITICAL])
        })
        
        # Trigger Janus-specific responses
        if result.code_quality_score < 50:
            logger.critical(f"Janus AI generated low-quality code (Score: {result.code_quality_score})")
            # Could trigger self-correction mechanisms
        
        # Log to holographic brain memory if available
        try:
            # This would integrate with the holographic brain memory system
            pass
        except:
            pass
    
    def validate_ai_output(self, content: str, content_type: str) -> tuple[bool, List[str]]:
        """Validate AI-generated content before deployment"""
        return self.prevention.validate_code_before_execution(content, content_type)

# Example usage
if __name__ == "__main__":
    # Setup integration
    janus_integration = JanusFaultIntegration()
    janus_integration.setup_janus_integration()
    
    # Start monitoring
    session_id = janus_integration.monitor.start_monitoring("./project")
    
    # Example validation
    faulty_code = """
    <div style="position: fixed; left: -2000px; pointer-events: none;">
        Character
    </div>
    """
    
    is_safe, issues = janus_integration.validate_ai_output(faulty_code, "html")
    
    if not is_safe:
        print("Code blocked due to issues:")
        for issue in issues:
            print(f"- {issue}")
        
        suggestions = janus_integration.prevention.suggest_fixes(faulty_code, "html")
        print("\nSuggested fixes:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    
    # Monitor for a bit
    time.sleep(10)
    
    # Stop monitoring
    janus_integration.monitor.stop_monitoring(session_id)
