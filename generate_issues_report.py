"""
Extract and rank the highest-severity issues from Oxpecker scan results.
"""
from pathlib import Path
import json

try:
    from tools.universal_oxpecker.core.orchestrator import OxpeckerOrchestrator
except:
    from universal_oxpecker.core.orchestrator import OxpeckerOrchestrator

ROOT = Path(".")
orchestrator = OxpeckerOrchestrator()

# Scan key high-risk files
HIGH_RISK_PATTERNS = [
    "*.py", "*.js", "*.ts", "*.java", "*.rs", "*.go"
]

print("=" * 80)
print("UNIVERSAL OXPECKER - TOP 20 ISSUES REPORT")
print("=" * 80)
print()

all_issues = []

# Scan specific folders known to have code
folders_to_scan = [
    "tools/universal_oxpecker",
    "janus_*.py",  # files in root
]

import glob
for pattern in ["janus_*.py", "test_*.py", "janus*.py"]:
    for fpath in glob.glob(pattern):
        if not any(skip in fpath for skip in ['__pycache__', '.git', '.pyc']):
            try:
                issues = orchestrator.debug_file(fpath)
                for issue in issues:
                    all_issues.append({
                        'file': fpath,
                        'severity': issue.severity,
                        'message': issue.message,
                        'line': issue.line,
                        'tier': issue.complexity_tier or "Tier ?",
                        'suggestion': issue.fix_suggestion,
                    })
            except Exception as e:
                pass

# Sort by severity (error > warning > info) and tier
severity_order = {'error': 0, 'warning': 1, 'info': 2}
all_issues.sort(key=lambda x: (severity_order.get(x['severity'], 99), x['tier']))

# Print top 20
for idx, issue in enumerate(all_issues[:20], 1):
    print(f"{idx}. [{issue['severity'].upper()}] {issue['file']}:{issue['line'] or '?'}")
    print(f"   Tier: {issue['tier']}")
    print(f"   {issue['message']}")
    if issue['suggestion']:
        print(f"   >> Fix: {issue['suggestion'][:100]}...")
    print()

print("=" * 80)
print(f"SUMMARY: Scanned {len(set(i['file'] for i in all_issues))} files")
print(f"Total issues found: {len(all_issues)}")
print(f"  Errors: {len([i for i in all_issues if i['severity'] == 'error'])}")
print(f"  Warnings: {len([i for i in all_issues if i['severity'] == 'warning'])}")
print(f"  Info: {len([i for i in all_issues if i['severity'] == 'info'])}")
print("=" * 80)

# Export to JSON for Gumroad demo
report = {
    "title": "Universal Oxpecker - Issue Report",
    "total_files_scanned": len(set(i['file'] for i in all_issues)),
    "total_issues": len(all_issues),
    "errors": len([i for i in all_issues if i['severity'] == 'error']),
    "warnings": len([i for i in all_issues if i['severity'] == 'warning']),
    "info": len([i for i in all_issues if i['severity'] == 'info']),
    "top_20_issues": all_issues[:20],
}

with open("dist/OXPECKER_DEMO_REPORT.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to: dist/OXPECKER_DEMO_REPORT.json")
