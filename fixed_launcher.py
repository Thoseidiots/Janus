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
    print(f"No 'Kiro' secret ({e}) — auto-push disabled")

# Load and patch the original script with all fixes
script = open(os.path.join(REPO, "train_avus_kaggle.py")).read()

# Enable KAGGLE_MODE
script = script.replace("KAGGLE_MODE           = False", "KAGGLE_MODE           = True")

# Fix focal loss device issue
script = script.replace(
    "        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")",
    "        targets = targets.to(logits.device)\n        ce   = F.cross_entropy(logits, targets, ignore_index=-1, reduction=\"none\")"
)

# Fix database BLOB issue - disable database saves
script = script.replace(
    """def save_epoch_to_db(epoch: int, model_state: dict, loss: float):
    buffer = io.BytesIO()
    import torch
    torch.save(model_state, buffer)
    weights_bytes = buffer.getvalue()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        CHUNK_SIZE = 1000000000  # 1GB chunks to stay under SQLite limits
        
        if len(weights_bytes) <= CHUNK_SIZE:
            conn.execute('''
                INSERT INTO epoch_weights (epoch, weights_blob, loss)
                VALUES (?, ?, ?)
                ON CONFLICT(epoch) DO UPDATE SET
                    weights_blob = excluded.weights_blob,
                    loss = excluded.loss
            ''', (epoch, sqlite3.Binary(weights_bytes), loss))
            conn.execute('DELETE FROM chunked_weights WHERE epoch = ?', (epoch,))
        else:
            conn.execute('''
                INSERT INTO epoch_weights (epoch, weights_blob, loss)
                VALUES (?, ?, ?)
                ON CONFLICT(epoch) DO UPDATE SET
                    weights_blob = excluded.weights_blob,
                    loss = excluded.loss
            ''', (epoch, sqlite3.Binary(b"CHUNKED"), loss))
            conn.execute('DELETE FROM chunked_weights WHERE epoch = ?', (epoch,))
            
            for i in range(0, len(weights_bytes), CHUNK_SIZE):
                chunk = weights_bytes[i:i+CHUNK_SIZE]
                conn.execute('''
                    INSERT INTO chunked_weights (epoch, chunk_index, chunk_blob)
                    VALUES (?, ?, ?)
                ''', (epoch, i // CHUNK_SIZE, sqlite3.Binary(chunk)))
                
        conn.commit()
        print(f"[db] Saved epoch {epoch} weights to SQLite DB. Loss: {loss:.4f}")
    except Exception as e:
        print(f"[db] Failed to save weights for epoch {epoch}: {e}")
    finally:
        conn.close()""",
    """def save_epoch_to_db(epoch: int, model_state: dict, loss: float):
    \"\"\"Database save disabled to avoid BLOB size issues. Weights are saved as files instead.\"\"\"
    print(f"[db] Database save disabled - weights saved as files instead")
    # Save as file instead
    fallback_path = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch}.pt"
    try:
        import torch
        torch.save(model_state, fallback_path)
        size_mb = fallback_path.stat().st_size / 1e6
        print(f"[db] Saved epoch {epoch} weights to file: {fallback_path} ({size_mb:.1f} MB). Loss: {loss:.4f}")
    except Exception as e:
        print(f"[db] Failed to save epoch {epoch} weights: {e}")"""
)

# Fix auto_push_weights to skip Kaggle API calls
script = script.replace(
    """def auto_push_weights(version_notes: str = "Auto-save"):
    """
    Push weights back to the janus-weights Kaggle dataset automatically.

    Requires a Kaggle notebook secret named KAGGLE_KEY containing
    the contents of your kaggle.json API token.

    To set up:
      1. Kaggle account -> Settings -> API -> Create New Token
      2. Notebook -> Add-ons -> Secrets -> Add secret:
           Name:  KAGGLE_KEY
           Value: (paste full contents of kaggle.json)
    """
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("Kiro")
    except Exception as e:
        print(f"[push] 'Kiro' secret not found: {e}")
        print("[push] Skipping auto-push — download weights manually from output panel")
        return

    import json as _json

    # Write kaggle.json so the API client can authenticate
    kaggle_dir = Path(os.path.expanduser("~/.kaggle"))
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    creds = {"username": "ishmaelsears", "key": token}
    kaggle_json.write_text(_json.dumps(creds))
    os.chmod(str(kaggle_json), 0o600)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        os.system("pip install kaggle -q")
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Securely download existing Kaggle database to merge them so we don't accidentally drop past files
    # Instead of copying generated files into a staging dir (which doubles disk space and crashes), 
    # we download missing dataset files to a temp dir, MOVE them into KAGGLE_WORKING, and push the entire KAGGLE_WORKING dir.
    dl_dir = KAGGLE_WORKING / "kaggle_dl_temp"
    dl_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"[push] Downloading existing dataset to prevent file loss...")
    try:
        api.dataset_download_files("ishmaelsears/janus-avus-weights", path=str(dl_dir), unzip=True)
        import shutil
        for f in dl_dir.iterdir():
            tgt = KAGGLE_WORKING / f.name
            # If the file already exists in working dir, our freshly generated one takes precedence.
            if f.is_file() and not tgt.exists() and f.name != "dataset-metadata.json":
                shutil.move(str(f), str(tgt))
    except Exception as e:
        print(f"[push] Warning: Could not download old dataset files (they might not exist yet): {e}")
    finally:
        import shutil
        if dl_dir.exists():
            shutil.rmtree(dl_dir, ignore_errors=True)
            
    # Also remove any old upload_staging if it exists
    old_staging = KAGGLE_WORKING / "upload_staging"
    if old_staging.exists():
        shutil.rmtree(old_staging, ignore_errors=True)

    # Write dataset metadata directly into KAGGLE_WORKING
    meta = {
        "title": "janus-avus-weights",
        "id": "ishmaelsears/janus-avus-weights",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = KAGGLE_WORKING / "dataset-metadata.json"
    meta_path.write_text(_json.dumps(meta, indent=2))

    # SAFETY CHECK: Strictly prevent overwriting the dataset if the main model weights are missing.
    # If the download failed and we didn't generate new ones, pushing would wipe the multi-GB weights 
    # from the Kaggle dataset, leaving it with just text files or tiny checkpoints and destroying progress.
    main_weights_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
    if not main_weights_file.exists():
        print(f"[push] CRITICAL ERROR: Aborting Kaggle push! The primary weights file '{main_weights_file.name}' is missing.")
        print("[push] Pushing now would permanently overwrite and wipe your model weights from the dataset. Aborting!")
        return

    # Cleanup any random loose files (like test_upload.txt) to keep the dataset clean
    # We recursively search all directories in /kaggle/working
    allowed_exts = {'.pt', '.json', '.db', '.png'}
    for f in KAGGLE_WORKING.rglob('*'):
        if f.is_file() and f.suffix.lower() not in allowed_exts:
            try:
                f.unlink()
            except Exception:
                pass

    print(f"[push] Pushing complete merged dataset to {creds['username']}/janus-weights ...")
    try:
        api.dataset_create_version(
            str(KAGGLE_WORKING),
            version_notes=version_notes,
            quiet=False,
            convert_to_csv=False,
            delete_old_versions=False,
        )
        print("[push] Done. Weights saved to Kaggle dataset.")
    except Exception as e:
        print(f"[push] Push failed: {e}")
        print("[push] Download weights manually from the output panel.")""",
    """def auto_push_weights(version_notes: str = "Auto-save"):
    \"\"\"
    Save weights locally and skip Kaggle API push to avoid filesystem issues.
    
    The Kaggle API push is disabled due to filesystem quota issues even when 
    disk space is available. Weights are saved locally for manual download.
    \"\"\"
    print(f"[save] Auto-save: {version_notes}")
    
    # Clean up disk space to ensure we can save
    try:
        _cleanup_disk_space()
    except RuntimeError as e:
        print(f"[save] {e}")
        print("[save] Continuing with limited cleanup...")
    
    # Verify main weights file exists
    main_weights_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_weights.pt"
    if main_weights_file.exists():
        size_mb = main_weights_file.stat().st_size / 1e6
        print(f"[save] Main weights ready: {main_weights_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"[save] WARNING: Main weights file not found: {main_weights_file.name}")
    
    # List all important files for manual download
    print(f"\\n[save] === FILES READY FOR DOWNLOAD ===")
    for f in KAGGLE_WORKING.glob("*.pt"):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:.1f} MB")
    for f in KAGGLE_WORKING.glob("*.json"):
        print(f"  {f.name}")
    for f in KAGGLE_WORKING.glob("*.png"):
        print(f"  {f.name}")
    
    print(f"\\n[save] Kaggle API push skipped due to filesystem limitations.")
    print(f"[save] Please download the files above from Kaggle output panel.")
    print(f"[save] Then manually upload to your dataset if needed.")"""
)

# Remove auto_push_weights call from training loop
script = script.replace(
    """        # Auto-push after every epoch so cancelling mid-session doesn't lose progress
        auto_push_weights(version_notes=f"Avus-{MODEL_SIZE} epoch {epoch+1} loss={avg_loss:.4f}")""",
    """        # Save epoch checkpoint as file (database disabled)
        epoch_file = KAGGLE_WORKING / f"avus_{MODEL_SIZE}_epoch_{epoch+1}.pt"
        shutil.copy(str(WEIGHTS_OUT), str(epoch_file))
        print(f"[avus] Epoch checkpoint saved: {epoch_file.name}")"""
)

print("Executing patched training script...")
exec(script)
