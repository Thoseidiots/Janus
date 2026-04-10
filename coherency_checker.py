"""
coherency_checker.py
====================
Validates synthetic training datasets for logical coherency and structural integrity.

Checks:
  1. JSON structure and validity
  2. Logical consistency (e.g., math computations)
  3. Format compliance (special tokens, structure)
  4. Parameter range validation
  5. Malformed/truncated entries

Usage:
  python coherency_checker.py <dataset_file>

  Or import and use programmatically:
    from coherency_checker import CoherencyChecker
    checker = CoherencyChecker()
    report = checker.check_file("dataset.txt")
    print(report)
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class DatasetType(Enum):
    """Detected dataset types."""
    THREED_GENERATION = "3d_generation"
    SCREEN_ACTION = "screen_action"
    LANGUAGE = "language"
    REASONING = "reasoning"
    PHASE2_RENDERING = "phase2_rendering"
    UNKNOWN = "unknown"


@dataclass
class Issue:
    """Represents a coherency issue."""
    line_num: int
    severity: str  # "error" | "warning" | "info"
    category: str  # "json" | "logic" | "format" | "range" | "structure"
    message: str
    text_preview: str = ""


@dataclass
class CheckResult:
    """Results from checking a dataset."""
    total_entries: int = 0
    valid_entries: int = 0
    issues: List[Issue] = field(default_factory=list)
    dataset_type: DatasetType = DatasetType.UNKNOWN
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, line_num: int, severity: str, category: str,
                  message: str, text_preview: str = ""):
        """Add an issue to the report."""
        self.issues.append(Issue(
            line_num=line_num,
            severity=severity,
            category=category,
            message=message,
            text_preview=text_preview[:100]
        ))

    def summary(self) -> str:
        """Generate a summary report."""
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")

        lines = [
            "="*70,
            "COHERENCY CHECK REPORT",
            "="*70,
            f"Dataset Type: {self.dataset_type.value}",
            f"Total Entries: {self.total_entries}",
            f"Valid Entries: {self.valid_entries}",
            f"Success Rate: {(self.valid_entries/max(1, self.total_entries)*100):.1f}%",
            f"",
            f"Issues Found: {len(self.issues)}",
            f"  Errors: {errors}",
            f"  Warnings: {warnings}",
            f"  Info: {len(self.issues) - errors - warnings}",
            "",
        ]

        if self.stats:
            lines.append("Statistics:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        if self.issues:
            lines.append("Issues by Category:")
            by_category = {}
            for issue in self.issues:
                by_category.setdefault(issue.category, []).append(issue)

            for category, issues in sorted(by_category.items()):
                lines.append(f"\n{category.upper()} ({len(issues)} issues):")
                for issue in issues[:10]:  # Show first 10 per category
                    lines.append(f"  Line {issue.line_num} [{issue.severity}]: {issue.message}")
                    if issue.text_preview:
                        lines.append(f"    Preview: {issue.text_preview}")

                if len(issues) > 10:
                    lines.append(f"  ... and {len(issues) - 10} more")

        lines.append("="*70)
        return "\n".join(lines)


class CoherencyChecker:
    """Checks dataset coherency across multiple formats."""

    # Special tokens used in Janus datasets
    SPECIAL_TOKENS = {
        "<|startoftext|>", "<|endoftext|>",
        "[JSON_START]", "[JSON_END]",
        "[ACT_START]", "[ACT_END]",
        "[FRAME_START]", "[FRAME_NEXT]", "[/FRAME_END]",
    }

    def __init__(self):
        self.result = CheckResult()

    def check_file(self, filepath: str) -> CheckResult:
        """Check a dataset file and return results."""
        self.result = CheckResult()
        path = Path(filepath)

        if not path.exists():
            self.result.add_issue(0, "error", "structure",
                                  f"File not found: {filepath}")
            return self.result

        # Read file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            self.result.add_issue(0, "error", "structure",
                                  f"Failed to read file: {e}")
            return self.result

        # Process each line
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            self.result.total_entries += 1
            valid = self._check_entry(line, line_num)
            if valid:
                self.result.valid_entries += 1

        # Detect dataset type based on patterns
        self._detect_dataset_type()

        return self.result

    def check_text(self, text: str) -> CheckResult:
        """Check raw text dataset."""
        self.result = CheckResult()
        lines = text.strip().split('\n')

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            self.result.total_entries += 1
            valid = self._check_entry(line, line_num)
            if valid:
                self.result.valid_entries += 1

        self._detect_dataset_type()
        return self.result

    def _check_entry(self, text: str, line_num: int) -> bool:
        """Check a single dataset entry. Returns True if valid."""
        issues_before = len(self.result.issues)

        # 1. Check for required special tokens
        if "<|startoftext|>" not in text:
            self.result.add_issue(line_num, "error", "format",
                                  "Missing <|startoftext|> token",
                                  text)

        if "<|endoftext|>" not in text:
            self.result.add_issue(line_num, "error", "format",
                                  "Missing <|endoftext|> token",
                                  text)

        # 2. Check JSON blocks if present
        self._check_json_blocks(text, line_num)

        # 3. Check specific dataset types
        if "[JSON_START]" in text:
            if "primitive" in text and "material" in text:
                self._check_3d_generation(text, line_num)
            elif "type" in text and "click" in text:
                self._check_screen_action(text, line_num)

        if "Calculate:" in text or "Step 1:" in text:
            self._check_reasoning(text, line_num)

        if any(kw in text for kw in ["Explain", "What is", "How does", "Describe"]):
            self._check_language(text, line_num)

        # 4. Check for truncation
        if text.endswith("...") or text.count('"') % 2 != 0:
            self.result.add_issue(line_num, "warning", "structure",
                                  "Entry may be truncated",
                                  text)

        # Return True if no new issues were added
        return len(self.result.issues) == issues_before

    def _check_json_blocks(self, text: str, line_num: int):
        """Validate JSON structure in text."""
        # Find all JSON blocks
        patterns = [
            (r'\[JSON_START\](.*?)\[JSON_END\]', "JSON_START/END"),
            (r'\[ACT_START\](.*?)\[ACT_END\]', "ACT_START/END"),
        ]

        for pattern, block_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    obj = json.loads(match)
                    # JSON is valid
                except json.JSONDecodeError as e:
                    self.result.add_issue(line_num, "error", "json",
                                          f"Invalid JSON in {block_type}: {e}",
                                          match)

    def _check_3d_generation(self, text: str, line_num: int):
        """Check 3D generation dataset entry."""
        try:
            match = re.search(r'\[JSON_START\](.*?)\[JSON_END\]', text)
            if not match:
                return

            obj = json.loads(match.group(1))

            # Required fields
            required = ["object", "primitive", "material", "scale", "position"]
            for field in required:
                if field not in obj:
                    self.result.add_issue(line_num, "error", "structure",
                                          f"Missing required field: {field}",
                                          text)

            # Validate scale (should be 3 floats)
            if "scale" in obj:
                if not isinstance(obj["scale"], list) or len(obj["scale"]) != 3:
                    self.result.add_issue(line_num, "error", "structure",
                                          "Scale must be array of 3 values",
                                          str(obj["scale"]))
                else:
                    for val in obj["scale"]:
                        if not (0.5 <= val <= 3.0):
                            self.result.add_issue(line_num, "warning", "range",
                                                  f"Scale value {val} outside expected range [0.5, 3.0]",
                                                  str(obj["scale"]))

            # Validate position (should be 3 floats in range -5 to 5)
            if "position" in obj:
                if not isinstance(obj["position"], list) or len(obj["position"]) != 3:
                    self.result.add_issue(line_num, "error", "structure",
                                          "Position must be array of 3 values",
                                          str(obj["position"]))
                else:
                    for val in obj["position"]:
                        if not (-5 <= val <= 5):
                            self.result.add_issue(line_num, "warning", "range",
                                                  f"Position value {val} outside expected range [-5, 5]",
                                                  str(obj["position"]))

            # Validate material properties
            if "roughness" in obj:
                r = obj["roughness"]
                if not (0.1 <= r <= 0.9):
                    self.result.add_issue(line_num, "warning", "range",
                                          f"Roughness {r} outside expected range [0.1, 0.9]")

            if "metallic" in obj:
                m = obj["metallic"]
                if not (0.0 <= m <= 1.0):
                    self.result.add_issue(line_num, "error", "range",
                                          f"Metallic {m} must be in range [0.0, 1.0]")

        except Exception as e:
            self.result.add_issue(line_num, "error", "json",
                                  f"Failed to parse 3D generation entry: {e}",
                                  text)

    def _check_screen_action(self, text: str, line_num: int):
        """Check screen action dataset entry."""
        try:
            match = re.search(r'\[ACT_START\](.*?)\[ACT_END\]', text)
            if not match:
                return

            obj = json.loads(match.group(1))

            # Required fields
            required = ["type", "x", "y", "button"]
            for field in required:
                if field not in obj:
                    self.result.add_issue(line_num, "error", "structure",
                                          f"Missing required field: {field}",
                                          text)

            # Validate coordinates (1920x1080 screen)
            if "x" in obj:
                x = obj["x"]
                if not (10 <= x <= 1910):
                    self.result.add_issue(line_num, "error", "range",
                                          f"X coordinate {x} outside valid range [10, 1910]")

            if "y" in obj:
                y = obj["y"]
                if not (10 <= y <= 1070):
                    self.result.add_issue(line_num, "error", "range",
                                          f"Y coordinate {y} outside valid range [10, 1070]")

            # Validate action type
            if "type" in obj and obj["type"] != "click":
                self.result.add_issue(line_num, "warning", "logic",
                                      f"Unexpected action type: {obj['type']}")

            # Validate button
            if "button" in obj and obj["button"] not in ["left", "right", "middle"]:
                self.result.add_issue(line_num, "warning", "logic",
                                      f"Unexpected button: {obj['button']}")

        except Exception as e:
            self.result.add_issue(line_num, "error", "json",
                                  f"Failed to parse screen action entry: {e}",
                                  text)

    def _check_reasoning(self, text: str, line_num: int):
        """Check reasoning/math dataset entry."""
        # Pattern: "Calculate: {a} {op} {b}. ... Result: {result}"
        match = re.search(r'Calculate:\s*(\d+)\s*([+\-*])\s*(\d+).*?Result:\s*(\d+)', text)
        if not match:
            self.result.add_issue(line_num, "warning", "format",
                                  "Reasoning entry doesn't match expected format",
                                  text)
            return

        a, op, b, result = match.groups()
        a, b, result = int(a), int(b), int(result)

        # Verify the math
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op == "*":
            expected = a * b
        else:
            self.result.add_issue(line_num, "error", "logic",
                                  f"Unknown operation: {op}",
                                  text)
            return

        if result != expected:
            self.result.add_issue(line_num, "error", "logic",
                                  f"Math error: {a} {op} {b} = {result}, expected {expected}",
                                  text)

    def _check_language(self, text: str, line_num: int):
        """Check language comprehension dataset entry."""
        # Basic checks for Q&A structure
        question_starters = ["Explain", "What is", "How does", "Describe",
                             "What are"]

        has_question = any(starter in text for starter in question_starters)
        if not has_question:
            self.result.add_issue(line_num, "info", "format",
                                  "Language entry may not contain a clear question",
                                  text)

        # Check if answer is too short (likely incomplete)
        content = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
        if len(content) < 50:
            self.result.add_issue(line_num, "warning", "structure",
                                  "Language entry seems too short",
                                  text)

    def _detect_dataset_type(self):
        """Detect the primary dataset type based on issue patterns."""
        if self.result.total_entries == 0:
            return

        # Count indicators
        indicators = {
            DatasetType.THREED_GENERATION: 0,
            DatasetType.SCREEN_ACTION: 0,
            DatasetType.REASONING: 0,
            DatasetType.LANGUAGE: 0,
        }

        for issue in self.result.issues:
            if "3D" in issue.message or "primitive" in issue.message.lower():
                indicators[DatasetType.THREED_GENERATION] += 1
            elif "coordinate" in issue.message.lower() or "click" in issue.message.lower():
                indicators[DatasetType.SCREEN_ACTION] += 1
            elif "math" in issue.message.lower() or "calculate" in issue.message.lower():
                indicators[DatasetType.REASONING] += 1
            elif "question" in issue.message.lower() or "language" in issue.message.lower():
                indicators[DatasetType.LANGUAGE] += 1

        if max(indicators.values()) > 0:
            self.result.dataset_type = max(indicators, key=indicators.get)

        # Add stats
        self.result.stats = {
            "3D generation indicators": indicators[DatasetType.THREED_GENERATION],
            "Screen action indicators": indicators[DatasetType.SCREEN_ACTION],
            "Reasoning indicators": indicators[DatasetType.REASONING],
            "Language indicators": indicators[DatasetType.LANGUAGE],
        }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python coherency_checker.py <dataset_file>")
        print("\nChecks synthetic datasets for logical coherency and structural integrity.")
        sys.exit(1)

    filepath = sys.argv[1]
    checker = CoherencyChecker()
    result = checker.check_file(filepath)
    print(result.summary())

    # Exit with error code if there are errors
    errors = sum(1 for i in result.issues if i.severity == "error")
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
