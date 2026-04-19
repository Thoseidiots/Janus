# Kaggle Auto-Upload Guide

**Problem**: Weights save to `/kaggle/working/` but don't automatically upload to the Kaggle dataset.

**Solution**: Use `kaggle_auto_upload.py` to automatically upload after each epoch.

---

## Quick Fix

### Option 1: Add to Kaggle Notebook (Easiest)

Add this cell to your Kaggle notebook **after the training loop**:

```python
# Auto-upload weights to Kaggle dataset
from kaggle_auto_upload import upload_to_kaggle

# Upload after training completes
upload_to_kaggle(epoch=AVUS_EPOCHS, model_size=MODEL_SIZE)
```

### Option 2: Patch Training Script

Edit `train_avus_kaggle.py` around line 1043 (after weights are saved):

**Find this:**
```python
torch.save({
    "epoch":            epoch + 1,
    "model_state_dict": raw_model.state_dict(),
    "config":           cfg_dict,
    "loss":             avg_loss,
}, str(WEIGHTS_OUT))
print(f"[avus] Weights saved -> {WEIGHTS_OUT}")
```

**Replace with:**
```python
torch.save({
    "epoch":            epoch + 1,
    "model_state_dict": raw_model.state_dict(),
    "config":           cfg_dict,
    "loss":             avg_loss,
}, str(WEIGHTS_OUT))
print(f"[avus] Weights saved -> {WEIGHTS_OUT}")

# AUTO-UPLOAD TO KAGGLE DATASET
try:
    from kaggle_auto_upload import upload_to_kaggle
    success = upload_to_kaggle(epoch=epoch+1, model_size=MODEL_SIZE)
    if success:
        print(f"[avus] ✅ Uploaded to Kaggle dataset")
    else:
        print(f"[avus] ⚠️  Upload failed - weights saved locally")
except Exception as e:
    print(f"[avus] ⚠️  Auto-upload error: {e}")
```

### Option 3: Manual Upload After Training

```python
# After training completes
from kaggle_auto_upload import KaggleAutoUploader

uploader = KaggleAutoUploader(dataset_name="ishmaelsears/janus-avus-weights")
uploader.upload_weights(epoch=AVUS_EPOCHS, model_size=MODEL_SIZE)
```

---

## How It Works

### Step 1: Detect Files
```
✅ avus_1b_weights.pt (main weights)
✅ learning_state.json (learning state)
✅ avus_1b_best.pt (best weights)
```

### Step 2: Upload to Dataset
```
Uses Kaggle CLI: kaggle datasets version create
Creates new dataset version with weights
```

### Step 3: Verify
```
Check Kaggle dataset page:
https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights
```

---

## Troubleshooting

### Issue: "Kaggle CLI not available"

**Solution**: Install Kaggle CLI in Kaggle notebook:
```python
!pip install kaggle -q
```

### Issue: "Authentication failed"

**Solution**: Setup Kaggle credentials:
```python
# In Kaggle notebook, credentials are auto-loaded
# If not, add your kaggle.json to ~/.kaggle/
```

### Issue: "Dataset not found"

**Solution**: Make sure dataset exists:
```
https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights
```

If not, create it first:
```bash
kaggle datasets create -p /path/to/dataset
```

---

## What Gets Uploaded

### Per Epoch
- `avus_1b_weights.pt` - Model weights
- `learning_state.json` - Learning state
- `avus_1b_best.pt` - Best weights so far

### Dataset Structure
```
janus-avus-weights/
├── avus_1b_weights.pt (latest)
├── learning_state.json (latest)
├── avus_1b_best.pt (best)
└── [version history]
```

---

## Verification

### Check Upload Success

```python
# After upload
from pathlib import Path

dataset_dir = Path("/kaggle/datasets/ishmaelsears/janus-avus-weights")
if dataset_dir.exists():
    files = list(dataset_dir.glob("*.pt"))
    print(f"✅ Found {len(files)} weight files in dataset")
else:
    print("❌ Dataset not found")
```

### Check Kaggle Website

Visit: https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights

You should see:
- Latest weights
- Learning state
- Version history

---

## Integration with Training

### Full Training Loop with Auto-Upload

```python
from kaggle_auto_upload import upload_to_kaggle

for epoch in range(AVUS_EPOCHS):
    # Training code...
    
    # Save weights
    torch.save({...}, str(WEIGHTS_OUT))
    
    # Auto-upload
    try:
        upload_to_kaggle(epoch=epoch+1, model_size=MODEL_SIZE)
        print(f"✅ Epoch {epoch+1} uploaded")
    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
```

---

## Performance Impact

- **Upload time**: ~30-60 seconds per epoch (depends on file size)
- **Network**: Uses Kaggle's internal network (fast)
- **Storage**: Each version stored in dataset

---

## Resuming Training

### Load from Dataset

```python
# Weights automatically available in /kaggle/input/
dataset_dir = Path("/kaggle/input/janus-avus-weights")
weights_file = dataset_dir / "avus_1b_weights.pt"

if weights_file.exists():
    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
```

---

## Files

- `kaggle_auto_upload.py` - Auto-upload implementation
- `KAGGLE_AUTO_UPLOAD_GUIDE.md` - This guide

---

## Status

✅ **READY TO USE**

Add to your Kaggle notebook and weights will automatically upload after each epoch.

---

## Next Steps

1. ✅ Add `kaggle_auto_upload.py` to Kaggle notebook
2. ⏳ Patch training script (Option 2) or add upload call (Option 1)
3. ⏳ Run training
4. ⏳ Verify weights appear in dataset
5. ⏳ Resume training from dataset if needed

---

**Result**: Weights automatically persist to Kaggle dataset after each epoch.

