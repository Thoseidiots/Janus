"""
AI Fault Detector - Catches faulty AI-generated code before deployment

This system detects common UI/UX issues in AI-generated web code:
- Elements positioned off-screen
- Broken interactivity and clickability
- CSS positioning conflicts
- JavaScript runtime errors
- Accessibility violations
- Performance bottlenecks
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import ast
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class FaultSeverity(Enum):
    CRITICAL = "critical"  # Blocks user interaction
    HIGH = "high"         # Major UX issue
    MEDIUM = "medium"     # Noticeable problem
    LOW = "low"          # Minor issue
    INFO = "info"        # Suggestion

class FaultCategory(Enum):
    POSITIONING = "positioning"
    INTERACTIVITY = "interactivity"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    STYLING = "styling"
    JAVASCRIPT = "javascript"
    RESPONSIVE = "responsive"

@dataclass
class FaultReport:
    """Individual fault detection result"""
    category: FaultCategory
    severity: FaultSeverity
    title: str
    description: str
    line_number: Optional[int] = None
    element_selector: Optional[str] = None
    suggested_fix: Optional[str] = None
    code_snippet: Optional[str] = None

@dataclass
class DetectionResult:
    """Complete fault detection analysis"""
    total_faults: int
    faults_by_severity: Dict[FaultSeverity, int] = field(default_factory=dict)
    fault_reports: List[FaultReport] = field(default_factory=list)
    code_quality_score: float = 0.0  # 0-100
    recommendations: List[str] = field(default_factory=list)

class AIFaultDetector:
    """Main fault detection engine"""
    
    def __init__(self):
        self.fault_patterns = self._initialize_patterns()
        self.css_parser = CSSParser()
        self.html_parser = HTMLParser()
        self.js_analyzer = JavaScriptAnalyzer()
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize regex patterns for common AI code faults"""
        return {
            'positioning': [
                {
                    'pattern': r'position:\s*fixed.*?(left|top|right|bottom):\s*-\d+',
                    'severity': FaultSeverity.HIGH,
                    'title': 'Element positioned off-screen',
                    'description': 'Fixed positioning with negative values may place elements outside viewport',
                    'fix': 'Use positive values or viewport-relative positioning'
                },
                {
                    'pattern': r'translate[XY]\(\s*-\d{3,}',
                    'severity': FaultSeverity.HIGH,
                    'title': 'Element translated far off-screen',
                    'description': 'Large negative translations may move elements outside viewport',
                    'fix': 'Limit translation values to reasonable ranges'
                },
                {
                    'pattern': r'z-index:\s*(9999|99999)',
                    'severity': FaultSeverity.MEDIUM,
                    'title': 'Extremely high z-index',
                    'description': 'Very high z-index values indicate stacking context issues',
                    'fix': 'Use lower, more reasonable z-index values'
                }
            ],
            'interactivity': [
                {
                    'pattern': r'pointer-events:\s*none',
                    'severity': FaultSeverity.CRITICAL,
                    'title': 'Element made non-interactive',
                    'description': 'pointer-events: none disables all mouse interactions',
                    'fix': 'Remove this property or apply it only to decorative elements'
                },
                {
                    'pattern': r'display:\s*none.*?cursor:\s*pointer',
                    'severity': FaultSeverity.HIGH,
                    'title': 'Hidden element with pointer cursor',
                    'description': 'Element is hidden but shows pointer cursor, confusing users',
                    'fix': 'Remove cursor: pointer from hidden elements'
                },
                {
                    'pattern': r'opacity:\s*0.*?pointer-events:\s*auto',
                    'severity': FaultSeverity.HIGH,
                    'title': 'Invisible but clickable element',
                    'description': 'Fully transparent element that still captures clicks',
                    'fix': 'Add pointer-events: none to invisible interactive elements'
                }
            ],
            'accessibility': [
                {
                    'pattern': r'<(div|span)[^>]*>(?!.*?(aria-label|title|role=))',
                    'severity': FaultSeverity.MEDIUM,
                    'title': 'Interactive element lacking accessibility attributes',
                    'description': 'Interactive div/span missing aria-label, title, or role',
                    'fix': 'Add appropriate accessibility attributes'
                },
                {
                    'pattern': r'color:\s*#000000.*?background-color:\s*#000000',
                    'severity': FaultSeverity.CRITICAL,
                    'title': 'Text and background same color',
                    'description': 'Text is invisible against same-colored background',
                    'fix': 'Use contrasting colors'
                }
            ]
        }
    
    def analyze_code(self, html_content: str, css_content: str = "", js_content: str = "") -> DetectionResult:
        """Complete code analysis for faults"""
        faults = []
        
        # Analyze HTML
        html_faults = self.html_parser.analyze(html_content)
        faults.extend(html_faults)
        
        # Analyze CSS
        css_faults = self.css_parser.analyze(css_content)
        faults.extend(css_faults)
        
        # Analyze JavaScript
        js_faults = self.js_analyzer.analyze(js_content)
        faults.extend(js_faults)
        
        # Calculate quality score
        severity_weights = {
            FaultSeverity.CRITICAL: 10,
            FaultSeverity.HIGH: 5,
            FaultSeverity.MEDIUM: 2,
            FaultSeverity.LOW: 1,
            FaultSeverity.INFO: 0.5
        }
        
        total_deduction = sum(severity_weights[fault.severity] for fault in faults)
        quality_score = max(0, 100 - total_deduction)
        
        # Count faults by severity
        faults_by_severity = {}
        for fault in faults:
            faults_by_severity[fault.severity] = faults_by_severity.get(fault.severity, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(faults)
        
        return DetectionResult(
            total_faults=len(faults),
            faults_by_severity=faults_by_severity,
            fault_reports=faults,
            code_quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, faults: List[FaultReport]) -> List[str]:
        """Generate actionable recommendations based on detected faults"""
        recommendations = []
        
        critical_count = sum(1 for f in faults if f.severity == FaultSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append(f"URGENT: Fix {critical_count} critical faults that block user interaction")
        
        positioning_faults = [f for f in faults if f.category == FaultCategory.POSITIONING]
        if positioning_faults:
            recommendations.append("Review element positioning - consider using CSS Grid or Flexbox for better layout control")
        
        interactivity_faults = [f for f in faults if f.category == FaultCategory.INTERACTIVITY]
        if interactivity_faults:
            recommendations.append("Test all interactive elements to ensure they respond to user input")
        
        accessibility_faults = [f for f in faults if f.category == FaultCategory.ACCESSIBILITY]
        if accessibility_faults:
            recommendations.append("Improve accessibility by adding proper ARIA labels and semantic HTML")
        
        return recommendations

class CSSParser:
    """CSS-specific fault detection"""
    
    def analyze(self, css_content: str) -> List[FaultReport]:
        faults = []
        
        # Check for positioning issues
        positioning_issues = [
            (r'position:\s*fixed.*?(left|top|right|bottom):\s*-\d+', "Element positioned off-screen"),
            (r'translate[XY]\(\s*-\d{3,}', "Element translated far off-screen"),
            (r'position:\s*absolute.*?(left|top|right|bottom):\s*100[vw|vh]', "Element positioned outside viewport")
        ]
        
        for pattern, description in positioning_issues:
            matches = re.finditer(pattern, css_content, re.IGNORECASE)
            for match in matches:
                line_num = css_content[:match.start()].count('\n') + 1
                faults.append(FaultReport(
                    category=FaultCategory.POSITIONING,
                    severity=FaultSeverity.HIGH,
                    title=description,
                    description=f"CSS positioning issue detected: {description}",
                    line_number=line_num,
                    code_snippet=css_content[max(0, match.start()-50):match.end()+50].strip(),
                    suggested_fix="Review positioning values and ensure elements remain within viewport"
                ))
        
        # Check for interactivity issues
        interactivity_issues = [
            (r'pointer-events:\s*none', "Element made non-interactive"),
            (r'opacity:\s*0.*?pointer-events:\s*auto', "Invisible but clickable element"),
            (r'display:\s*none.*?cursor:\s*pointer', "Hidden element with pointer cursor")
        ]
        
        for pattern, description in interactivity_issues:
            matches = re.finditer(pattern, css_content, re.IGNORECASE)
            for match in matches:
                line_num = css_content[:match.start()].count('\n') + 1
                faults.append(FaultReport(
                    category=FaultCategory.INTERACTIVITY,
                    severity=FaultSeverity.CRITICAL if 'none' in description else FaultSeverity.HIGH,
                    title=description,
                    description=f"Interactivity issue detected: {description}",
                    line_number=line_num,
                    code_snippet=css_content[max(0, match.start()-50):match.end()+50].strip(),
                    suggested_fix="Review CSS properties affecting element interactivity"
                ))
        
        return faults

class HTMLParser:
    """HTML-specific fault detection"""
    
    def analyze(self, html_content: str) -> List[FaultReport]:
        faults = []
        
        # Check for accessibility issues
        accessibility_patterns = [
            (r'<(div|span)[^>]*onclick[^>]*>(?!(?:(?!</\1>).)*?(aria-label|title|role=))', "Interactive element missing accessibility attributes"),
            (r'<img[^>]*(?<!alt=)[^>]*>', "Image missing alt attribute"),
            (r'<button[^>]*>(?!(?:(?!</button>).)*?(aria-label|aria-describedby))', "Button missing accessibility label")
        ]
        
        for pattern, description in accessibility_patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                line_num = html_content[:match.start()].count('\n') + 1
                faults.append(FaultReport(
                    category=FaultCategory.ACCESSIBILITY,
                    severity=FaultSeverity.MEDIUM,
                    title=description,
                    description=f"Accessibility issue: {description}",
                    line_number=line_num,
                    code_snippet=html_content[max(0, match.start()-50):match.end()+50].strip(),
                    suggested_fix="Add appropriate ARIA attributes or use semantic HTML elements"
                ))
        
        # Check for structural issues
        structural_patterns = [
            (r'<(div|span)[^>]*tabindex[^>]*>', "Non-focusable element with tabindex"),
            (r'<script[^>]*>.*?eval\s*\(', "Use of eval() function detected"),
            (r'<a[^>]*href\s*=\s*["\']#["\']', "Empty or placeholder link")
        ]
        
        for pattern, description in structural_patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                line_num = html_content[:match.start()].count('\n') + 1
                severity = FaultSeverity.HIGH if 'eval' in description else FaultSeverity.MEDIUM
                faults.append(FaultReport(
                    category=FaultCategory.JAVASCRIPT if 'eval' in description else FaultCategory.STYLING,
                    severity=severity,
                    title=description,
                    description=f"Structural issue: {description}",
                    line_number=line_num,
                    code_snippet=html_content[max(0, match.start()-50):match.end()+50].strip(),
                    suggested_fix="Review and potentially remove or replace this pattern"
                ))
        
        return faults

class JavaScriptAnalyzer:
    """JavaScript-specific fault detection"""
    
    def analyze(self, js_content: str) -> List[FaultReport]:
        faults = []
        
        try:
            # Parse JavaScript for syntax analysis
            tree = ast.parse(js_content, mode='exec')
            
            # Check for problematic patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check for eval usage
                    if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                        line_num = node.lineno
                        faults.append(FaultReport(
                            category=FaultCategory.JAVASCRIPT,
                            severity=FaultSeverity.HIGH,
                            title="Use of eval() function",
                            description="eval() can cause security vulnerabilities and performance issues",
                            line_number=line_num,
                            suggested_fix="Replace eval() with safer alternatives like JSON.parse() or function constructors"
                        ))
                
        except SyntaxError as e:
            faults.append(FaultReport(
                category=FaultCategory.JAVASCRIPT,
                severity=FaultSeverity.CRITICAL,
                title="JavaScript syntax error",
                description=f"Syntax error in JavaScript code: {str(e)}",
                line_number=e.lineno,
                suggested_fix="Fix syntax errors before deployment"
            ))
        
        # Check for runtime issues with regex
        runtime_patterns = [
            (r'addEventListener\s*\(\s*["\']click["\'].*?preventDefault\s*\(\)', "Click event with preventDefault may block expected behavior"),
            (r'setTimeout\s*\(\s*["\'].*?["\']\s*,\s*0\s*\)', "Zero-delay timeout can cause performance issues"),
            (r'innerHTML\s*=\s*["\'].*?["\']', "Direct innerHTML assignment may cause XSS vulnerabilities")
        ]
        
        for pattern, description in runtime_patterns:
            matches = re.finditer(pattern, js_content, re.IGNORECASE)
            for match in matches:
                line_num = js_content[:match.start()].count('\n') + 1
                severity = FaultSeverity.HIGH if 'innerHTML' in description else FaultSeverity.MEDIUM
                faults.append(FaultReport(
                    category=FaultCategory.JAVASCRIPT,
                    severity=severity,
                    title=description,
                    description=f"Runtime issue: {description}",
                    line_number=line_num,
                    code_snippet=js_content[max(0, match.start()-50):match.end()+50].strip(),
                    suggested_fix="Review this pattern for potential issues"
                ))
        
        return faults

# Example usage and testing
if __name__ == "__main__":
    detector = AIFaultDetector()
    
    # Example faulty code
    faulty_html = """
    <div onclick="doSomething()" style="position: fixed; left: -1000px; pointer-events: none;">
        <img src="character.png">
    </div>
    """
    
    faulty_css = """
    .character {
        position: fixed;
        left: -1500px;
        top: -200px;
        opacity: 0;
        pointer-events: auto;
        cursor: pointer;
    }
    """
    
    faulty_js = """
    eval('alert("test")');
    element.innerHTML = user_input;
    """
    
    result = detector.analyze_code(faulty_html, faulty_css, faulty_js)
    
    print(f"Code Quality Score: {result.code_quality_score}/100")
    print(f"Total Faults: {result.total_faults}")
    print("\nDetected Issues:")
    
    for fault in result.fault_reports:
        print(f"\n[{fault.severity.value.upper()}] {fault.title}")
        print(f"  {fault.description}")
        if fault.suggested_fix:
            print(f"  Fix: {fault.suggested_fix}")
        if fault.line_number:
            print(f"  Line: {fault.line_number}")
