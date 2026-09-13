# Universal Oxpecker — Quick Start Guide

Welcome to **Oxpecker**, your multi-language code debugger and auto-repair engine.

---

## Installation

### Windows (Recommended)
1. Download `oxpecker-installer.exe` from Gumroad
2. Run the installer
3. Open Command Prompt and type: `oxpecker --version`

### macOS / Linux
```bash
pip install oxpecker
oxpecker --version
```

### Activate License (Optional)
If you purchased a license:
```bash
oxpecker license activate --key YOUR_LICENSE_KEY --email your@email.com
```

If you can't afford it this month:
```bash
oxpecker license cannot-afford
```

---

## Your First Scan

### 1. Scan a Single File
```bash
oxpecker debug my_script.py
```

Output:
```
  Universal Oxpecker  v1.2.0
  Multi-language debugger, scanner & stronger safe auto-repair

  Analysing: my_script.py

  Found 2 error(s), 3 warning(s), 1 info(s)

  #1 [ERROR] (python) [Tier 1] my_script.py:15:8
     undefined variable 'x'
     >> Assign x = None before use, or check that x is in scope.

  #2 [WARNING] (python) [Tier 2] my_script.py:22:1
     unused import 'os'
     >> Remove the import if not needed.
```

### 2. Scan Your Whole Project
```bash
oxpecker scan ./myproject --workers 4
```

This will:
- Find all source files in your project
- Run 4 parallel workers (customize with `--workers N`)
- Report issues by file and severity

### 3. Auto-Repair a File
```bash
oxpecker repair my_script.py
```

Output:
```
  #1: Remove unused import 'os' [Tier 1] via ImportOptimizer / removal
     score=0.85 issue_reduction=1 tests=passed
     Repair is safe; 1 issue fixed.

  Applied patches: 1
  Rollback snapshots: 1
  Remaining issues: 1
  
  ✓ Working copy saved: my_script.py.oxpecker
  ✓ You can now review, test, and commit if satisfied.
```

### 4. Check Rollback History
```bash
oxpecker rollback my_script.py.oxpecker
```

This reverts to the previous snapshot. Add `--steps 2` to go back 2 snapshots.

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `oxpecker debug FILE` | Analyze a single file |
| `oxpecker scan DIR` | Scan a directory recursively |
| `oxpecker repair FILE` | Auto-repair a file with rollback |
| `oxpecker repair DIR --project` | Auto-repair all files in a project |
| `oxpecker rollback FILE.oxpecker` | Revert to previous repair state |
| `oxpecker license status` | Check license activation status |
| `oxpecker langs` | List supported languages |
| `oxpecker --help` | Show all commands |

---

## Tips & Tricks

### 1. Repair with Test Validation
If your project has tests:
```bash
oxpecker repair myfile.py --max-rounds 10
```
Oxpecker will only apply fixes that pass your tests.

### 2. See Ranked Candidates
```bash
oxpecker repair myfile.py --show-candidates 5
```
Shows the top 5 ranked fix suggestions before auto-applying.

### 3. Parallel Scanning
On large projects:
```bash
oxpecker scan . --workers 8 --recursive
```

### 4. Skip a Month
If you can't afford the $0.99:
```bash
oxpecker license cannot-afford
```
You'll get 30 days free access, no questions asked.

---

## Understanding the Output

### Severity Levels
- **ERROR** (red): Code will likely fail
- **WARNING** (yellow): Code may have issues
- **INFO** (cyan): Notes or suggestions

### Complexity Tiers
- **Tier 1**: Safe, low-risk fixes (style, simple bugs)
- **Tier 2**: Moderate complexity (logic fixes)
- **Tier 3**: Complex or risky (deep refactoring)

### Patch Scoring
- **score**: Quality of the proposed fix (0–1.0)
- **issue_reduction**: How many issues this fix resolves
- **tests**: Did your tests pass after the fix?

---

## Troubleshooting

### "oxpecker: command not found"
On Windows: Make sure you installed the executable and restarted Command Prompt.
On macOS/Linux: Install with `pip install oxpecker`, or add it to your PATH.

### License activation fails
1. Check your internet connection
2. Double-check the license key from your Gumroad email
3. Email support@oxpecker.dev if the issue persists

### Some repairs are marked as "pending manual review"
High-complexity or risky fixes are held for review. This is intentional safety.

---

## Getting Help

- **Documentation**: https://oxpecker.dev/docs
- **GitHub Issues**: https://github.com/thoseidiots/universal-oxpecker/issues
- **Email Support**: support@oxpecker.dev
- **Twitter**: @oxpecker_dev

---

## Next Steps

1. ✅ Scan your current project
2. ✅ Review the issues found
3. ✅ Repair a few files and test
4. ✅ Consider integrating into your CI/CD pipeline

Happy debugging! 🔍
