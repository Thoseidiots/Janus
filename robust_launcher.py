import os, sys, json, torch

REPO = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_avus_kaggle.py" in files:
        REPO = root
        break

if REPO is None:
    raise FileNotFoundError("train_avus_kaggle.py not found.")

print(f"Repo found at: {REPO}")
sys.path.insert(0, REPO)

# Patch nn.Linear and RMSNorm to be device-aware (model parallelism fix)
import torch.nn as nn
_orig_linear_forward = nn.Linear.forward
def _device_aware_linear(self, x):
    if self.weight.device != x.device:
        self.weight = nn.Parameter(self.weight.to(x.device), requires_grad=self.weight.requires_grad)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(x.device), requires_grad=self.bias.requires_grad)
    return _orig_linear_forward(self, x)
nn.Linear.forward = _device_aware_linear

# Patch avus.py RMSNorm
avus_src = open(os.path.join(REPO, "avus.py")).read()
avus_src = avus_src.replace(
    "        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n        return x * norm * self.weight",
    "        if self.weight.device != x.device:\n            self.weight = torch.nn.Parameter(self.weight.to(x.device), requires_grad=self.weight.requires_grad)\n        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n        return x * norm * self.weight"
)
with open("/kaggle/working/avus.py", "w") as f:
    f.write(avus_src)
sys.path.insert(0, "/kaggle/working")

# Set up Kaggle API (but we'll disable the problematic calls)
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("Kiro")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds = {"username": "ishmaelsears", "key": token}
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump(creds, f)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    print("Kaggle API ready")
except Exception as e:
    print(f"No 'Kiro' secret ({e})")

# Load the original script
script = open(os.path.join(REPO, "train_avus_kaggle.py")).read()

# Enable KAGGLE_MODE
script = script.replace("KAGGLE_MODE           = False", "KAGGLE_MODE           = True")

# Fix focal loss device issue
script = script.replace(
    "        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")",
    "        targets = targets.to(logits.device)\n        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")"
)

print("Loading script with monkey patches ready...")

# Execute the script to load everything into the current namespace
exec(script)

# Now monkey-patch the problematic functions directly
def patched_save_epoch_to_db(epoch, model_state, loss):
    """Database save disabled to avoid BLOB size issues. Weights are saved as files instead."""
    print(f"[db] Database save disabled - weights saved as files instead")
    # Save as file instead
    from pathlib import Path
    KAGGLE_WORKING = Path("/kaggle/working")
    MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
    fallback_path = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch}.pt"
    try:
        torch.save(model_state, fallback_path)
        size_mb = fallback_path.stat().st_size / 1e6
        print(f"[db] Saved epoch {epoch} weights to file: {fallback_path} ({size_mb:.1f} MB). Loss: {loss:.4f}")
    except Exception as e:
        print(f"[db] Failed to save epoch {epoch} weights: {e}")

def patched_auto_push_weights(version_notes="Auto-save"):
    """Save weights locally and skip Kaggle API push to avoid filesystem issues."""
    print(f"[save] Auto-save: {version_notes}")
    
    # Verify main weights file exists
    from pathlib import Path
    KAGGLE_WORKING = Path("/kaggle/working")
    MODEL_SIZE = globals().get('MODEL_SIZE', '1b')
    main_weights_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
    if main_weights_file.exists():
        size_mb = main_weights_file.stat().st_size / 1e6
        print(f"[save] Main weights ready: {main_weights_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"[save] WARNING: Main weights file not found: {main_weights_file.name}")
    
    # List all important files for manual download
    print(f"\n[save] === FILES READY FOR DOWNLOAD ===")
    for f in KAGGLE_WORKING.glob("*.pt"):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:.1f} MB")
    for f in KAGGLE_WORKING.glob("*.json"):
        print(f"  {f.name}")
    for f in KAGGLE_WORKING.glob("*.png"):
        print(f"  {f.name}")
    
    print(f"\n[save] Kaggle API push skipped due to filesystem limitations.")
    print(f"[save] Please download the files above from Kaggle output panel.")

# Apply the monkey patches
import builtins
builtins.save_epoch_to_db = patched_save_epoch_to_db
builtins.auto_push_weights = patched_auto_push_weights

# Also patch them in the current module's namespace
globals()['save_epoch_to_db'] = patched_save_epoch_to_db
globals()['auto_push_weights'] = patched_auto_push_weights

print("Monkey patches applied. Starting training...")

# Now run the training
try:
    train_avus()
    train_hbm()
    print_summary()
    auto_push_weights(version_notes=f"Avus-{globals().get('MODEL_SIZE', '1b')} epoch {globals().get('AVUS_EPOCHS', 2)}")
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
