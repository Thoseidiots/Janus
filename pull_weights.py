# """
pull_weights.py

Downloads Avus weights from Kaggle on demand.
Run this once after cloning the repo -- no Git LFS needed.

Works on any platform: Windows (EliteDesk), Linux (Kaggle/Lightning), Mac.

Usage:
python pull_weights.py

Or import and call from any other script:
from pull_weights import ensure_weights
ensure_weights()
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────

# Config

# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(**file**).parent

# Kaggle dataset details

KAGGLE_OWNER   = "ishmaelsears"
KAGGLE_DATASET = "janus-avus-weights"
WEIGHTS_FILE   = "avus_1b_weights.pt"
CONFIG_FILE    = "config_avus_1b.json"

# Local paths to save to

WEIGHTS_PATH   = REPO_ROOT / WEIGHTS_FILE
CONFIG_PATH    = REPO_ROOT / CONFIG_FILE

# Expected file size in bytes (roughly 584MB -- used for sanity check)

EXPECTED_MIN_SIZE = 500_000_000

# Kaggle API endpoint

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"

# ─────────────────────────────────────────────────────────────────────────────

# Kaggle credentials

# ─────────────────────────────────────────────────────────────────────────────

def _load_kaggle_creds():
"""
Load Kaggle API credentials from:
1. Environment variables KAGGLE_USERNAME and KAGGLE_KEY
2. ~/.kaggle/kaggle.json
3. kaggle.json in repo root (gitignored)
"""
# Environment variables
username = os.environ.get("KAGGLE_USERNAME")
key      = os.environ.get("KAGGLE_KEY")
if username and key:
return username, key

```
# ~/.kaggle/kaggle.json
kaggle_dir = Path.home() / ".kaggle" / "kaggle.json"
if kaggle_dir.exists():
    with open(kaggle_dir) as f:
        creds = json.load(f)
    return creds["username"], creds["key"]

# repo root kaggle.json
local = REPO_ROOT / "kaggle.json"
if local.exists():
    with open(local) as f:
        creds = json.load(f)
    return creds["username"], creds["key"]

return None, None
```

# ─────────────────────────────────────────────────────────────────────────────

# Download helpers

# ─────────────────────────────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path,
username: str, key: str) -> bool:
"""Download a file with a progress bar. Returns True on success."""
import base64
token = base64.b64encode(f"{username}:{key}".encode()).decode()

```
req = urllib.request.Request(url)
req.add_header("Authorization", f"Basic {token}")

try:
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct  = downloaded / total * 100
                    done = int(pct / 2)
                    bar  = "█" * done + "░" * (50 - done)
                    mb   = downloaded / 1e6
                    print(f"\r  [{bar}] {pct:.1f}%  {mb:.1f}MB", end="", flush=True)

        print()  # newline after progress bar
        return True

except urllib.error.HTTPError as e:
    print(f"\n  HTTP {e.code}: {e.reason}")
    if e.code == 401:
        print("  Check your Kaggle credentials.")
    elif e.code == 403:
        print("  Make sure you have accepted the dataset terms on Kaggle.")
    return False
except Exception as e:
    print(f"\n  Download error: {e}")
    return False
```

def _kaggle_download(filename: str, dest: Path,
username: str, key: str) -> bool:
"""Download a file from the Kaggle dataset."""
url = (f"{KAGGLE_API_BASE}/datasets/{KAGGLE_OWNER}/"
f"{KAGGLE_DATASET}/versions/latest/files/{filename}")
print(f"  Downloading {filename} from Kaggle…")
return _download_with_progress(url, dest, username, key)

def _check_file(path: Path, min_size: int = 0) -> bool:
"""Check if a file exists and meets minimum size."""
return path.exists() and path.stat().st_size >= min_size

# ─────────────────────────────────────────────────────────────────────────────

# Public API

# ─────────────────────────────────────────────────────────────────────────────

def ensure_weights(force: bool = False) -> bool:
"""
Ensure avus_1b_weights.pt is present locally.
Downloads from Kaggle if missing or if force=True.

```
Returns True if weights are available, False if download failed.
"""
if not force and _check_file(WEIGHTS_PATH, EXPECTED_MIN_SIZE):
    print(f"[pull_weights] Weights already present: {WEIGHTS_PATH}")
    return True

print("[pull_weights] Weights not found locally. Downloading from Kaggle...")

username, key = _load_kaggle_creds()
if not username or not key:
    print("[pull_weights] No Kaggle credentials found.")
    print("  Options:")
    print("  1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    print("  2. Place kaggle.json in ~/.kaggle/ or repo root")
    print("  3. Download manually from:")
    print(f"     https://www.kaggle.com/datasets/{KAGGLE_OWNER}/{KAGGLE_DATASET}")
    print(f"     and place avus_1b_weights.pt in: {REPO_ROOT}")
    return False

print(f"  Kaggle user: {username}")
ok = _kaggle_download(WEIGHTS_FILE, WEIGHTS_PATH, username, key)

if ok and _check_file(WEIGHTS_PATH, EXPECTED_MIN_SIZE):
    size_mb = WEIGHTS_PATH.stat().st_size / 1e6
    print(f"[pull_weights] Weights downloaded: {size_mb:.1f}MB → {WEIGHTS_PATH}")
    return True
else:
    print("[pull_weights] Download failed or file too small. Check credentials.")
    if WEIGHTS_PATH.exists():
        WEIGHTS_PATH.unlink()  # Remove incomplete file
    return False
```

def ensure_config(force: bool = False) -> bool:
"""
Ensure config_avus_1b.json is present locally.
Downloads from Kaggle if missing.
"""
if not force and _check_file(CONFIG_PATH):
return True

```
username, key = _load_kaggle_creds()
if not username or not key:
    # Config is small -- write the default
    default = {
        "vocab_size":  50304,
        "dim":         768,
        "n_layers":    12,
        "n_heads":     12,
        "n_kv_heads":  4,
        "max_seq_len": 512
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default, f, indent=2)
    print(f"[pull_weights] Config written from defaults: {CONFIG_PATH}")
    return True

ok = _kaggle_download(CONFIG_FILE, CONFIG_PATH, username, key)
return ok and CONFIG_PATH.exists()
```

def ensure_all(force: bool = False) -> bool:
"""Download both weights and config if needed."""
config_ok  = ensure_config(force)
weights_ok = ensure_weights(force)
return config_ok and weights_ok

def setup_kaggle_credentials():
"""Interactive setup for Kaggle credentials."""
print("Kaggle Credential Setup")
print("─" * 40)
print("Go to https://www.kaggle.com/settings")
print("Scroll to ‘API’ → ‘Create New Token’")
print("This downloads kaggle.json with your username and key.")
print()

```
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
dest = kaggle_dir / "kaggle.json"

if dest.exists():
    print(f"kaggle.json already exists at {dest}")
    return

username = input("Enter your Kaggle username: ").strip()
key      = input("Enter your Kaggle API key: ").strip()

creds = {"username": username, "key": key}
with open(dest, "w") as f:
    json.dump(creds, f)

# Restrict permissions on Unix
try:
    os.chmod(dest, 0o600)
except Exception:
    pass

print(f"Credentials saved to {dest}")
```

# ─────────────────────────────────────────────────────────────────────────────

# CLI

# ─────────────────────────────────────────────────────────────────────────────

if **name** == "**main**":
import argparse

```
parser = argparse.ArgumentParser(
    description="Download Avus weights from Kaggle")
parser.add_argument("--force",  action="store_true",
                    help="Re-download even if weights already exist")
parser.add_argument("--setup",  action="store_true",
                    help="Interactive Kaggle credential setup")
parser.add_argument("--config-only", action="store_true",
                    help="Only download config, not weights")
args = parser.parse_args()

if args.setup:
    setup_kaggle_credentials()
    sys.exit(0)

if args.config_only:
    ok = ensure_config(force=args.force)
else:
    ok = ensure_all(force=args.force)

sys.exit(0 if ok else 1)
```