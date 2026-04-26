# Next Performance Optimizations for Janus Dependency Analyzer

## Current Status
- Analysis running on 3,211 apps (44 skipped from 3,255 total)
- Processing ~1-2 apps/second
- Estimated completion: 30-50 minutes
- **Problem**: Too many MEDIUM priority apps (1,707) being analyzed

## Optimization 6: More Aggressive Filtering

### Issue
Current filter only skips 44 apps. Most apps (1,707) classified as MEDIUM priority are still analyzed.

### Solution
Add more aggressive skip patterns:

```python
# Additional skip patterns to add:
skip_patterns = [
    # Windows system apps
    r".*microsoft.*", r".*windows.*", r".*edge.*", r".*webview.*",
    
    # Common non-dev apps
    r".*adobe.*", r".*acrobat.*", r".*reader.*",
    r".*office.*", r".*word.*", r".*excel.*", r".*powerpoint.*",
    r".*teams.*", r".*skype.*", r".*zoom.*", r".*slack.*",
    
    # Graphics/Design (unless specifically needed)
    r".*photoshop.*", r".*illustrator.*", r".*blender.*",
    
    # Duplicate/versioned apps
    r".*\d+\.\d+\.\d+.*",  # Skip versioned duplicates
]
```

### Expected Impact
- Reduce from 3,211 → ~800-1,000 apps
- **3-4x faster** (10-15 minutes instead of 30-50)

## Optimization 7: Deduplicate Similar Apps

### Issue
Multiple versions of same app (Python 3.9, 3.11, 3.12, 3.13, 3.14, Edge instances)

### Solution
```python
def deduplicate_apps(apps: List[Application]) -> List[Application]:
    """Keep only the latest version of each app."""
    app_groups = {}
    for app in apps:
        # Extract base name (remove version)
        base_name = re.sub(r'\s+\d+\.\d+.*', '', app.name)
        if base_name not in app_groups:
            app_groups[base_name] = []
        app_groups[base_name].append(app)
    
    # Keep latest version of each
    result = []
    for base_name, versions in app_groups.items():
        # Sort by version, keep latest
        latest = max(versions, key=lambda a: a.version or '0')
        result.append(latest)
    
    return result
```

### Expected Impact
- Reduce from 3,211 → ~1,500-2,000 apps
- **1.5-2x faster**

## Optimization 8: Early Exit on Low Confidence

### Issue
All strategies run even when high-confidence result found

### Solution
```python
def analyze_application(self, app: Application) -> List[Capability]:
    """Analyze with early exit on high confidence."""
    capabilities = []
    
    for strategy in self.strategies:
        caps = strategy.analyze(app)
        capabilities.extend(caps)
        
        # Early exit if we have high-confidence results
        high_conf = [c for c in caps if c.confidence_score > 0.8]
        if len(high_conf) >= 3:
            logger.info(f"Early exit for {app.name}: {len(high_conf)} high-confidence caps")
            break
    
    return capabilities
```

### Expected Impact
- **2-3x faster** per app
- Most useful for well-known tools (Git, Python, Docker)

## Optimization 9: Batch Analysis

### Issue
Each app analyzed independently, no batching

### Solution
```python
def analyze_batch(self, apps: List[Application]) -> Dict[str, List[Capability]]:
    """Analyze multiple apps in a batch for efficiency."""
    # Group apps by type
    by_type = {}
    for app in apps:
        app_type = self._classify_app_type(app)
        if app_type not in by_type:
            by_type[app_type] = []
        by_type[app_type].append(app)
    
    # Analyze each type with specialized strategy
    results = {}
    for app_type, type_apps in by_type.items():
        strategy = self._get_strategy_for_type(app_type)
        results.update(strategy.analyze_batch(type_apps))
    
    return results
```

### Expected Impact
- **1.5-2x faster** for similar apps
- Reduces redundant subprocess calls

## Optimization 10: Timeout Tuning

### Current
- 0.5s initial, 3s max timeout

### Proposed
```python
# Adaptive timeout based on app type
TIMEOUTS = {
    'high_priority': 5.0,   # Git, Python, Docker - worth waiting
    'medium_priority': 1.0,  # Utilities
    'unknown': 0.3,          # Everything else - fail fast
}
```

### Expected Impact
- **1.5-2x faster** for unknown apps
- Better results for known dev tools

## Combined Impact

If we implement all 5 optimizations:

### Before
- 3,211 apps × 2 seconds = **107 minutes** (1.8 hours)

### After
- Aggressive filtering: 3,211 → 1,000 apps (3.2x)
- Deduplication: 1,000 → 600 apps (1.7x)
- Early exit: 2s → 1s per app (2x)
- Batch analysis: 1s → 0.7s per app (1.4x)
- Timeout tuning: 0.7s → 0.5s per app (1.4x)

**Total: 600 apps × 0.5s = 5 minutes** ✓

## Implementation Priority

1. **Optimization 6** (Aggressive Filtering) - Biggest impact, easiest to implement
2. **Optimization 7** (Deduplication) - Easy, good impact
3. **Optimization 8** (Early Exit) - Medium difficulty, good impact
4. **Optimization 10** (Timeout Tuning) - Easy, moderate impact
5. **Optimization 9** (Batch Analysis) - Complex, moderate impact

## Next Steps

1. Let current analysis complete to get baseline
2. Implement Optimization 6 (aggressive filtering)
3. Test with filtered set
4. Implement remaining optimizations if needed
5. Document final performance

## Goal

**Target: < 5 minutes for full analysis of 3,255 apps**
