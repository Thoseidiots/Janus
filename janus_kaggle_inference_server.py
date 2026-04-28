"""
janus_kaggle_inference_server.py
=================================
Runs the Avus inference server on Kaggle GPU.

Paste this into a Kaggle notebook cell and run it.

Setup:
1. Add dataset: ishmaelsears/janus-avus-weights
2. Enable GPU (T4 x2)
3. Enable Internet
4. Add a Kaggle secret: NGROK_TOKEN = your ngrok authtoken
   (free account at https://dashboard.ngrok.com)
5. Run this cell — copy the URL it prints into Kaggle.env as JANUS_INFERENCE_URL
"""

import subprocess
subprocess.run(["pip", "install", "fastapi", "uvicorn", "pyngrok", "tiktoken", "-q"], check=True)

import os, sys, json, math, torch, threading, time, shutil
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

WORK_DIR    = Path("/kaggle/working")
DATASET_DIR = Path("/kaggle/input/janus-avus-weights")
device      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Copy weights from dataset ─────────────────────────────────────────────────
for fname in ["avus_1b_weights.pt", "avus_3b_weights.pt", "config_avus_1b.json"]:
    src = DATASET_DIR / fname
    dst = WORK_DIR / fname
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"Copied {fname}")

# ── Load Avus (your trained model) ────────────────────────────────────────────
# avus_inference.py and avus.py live in the dataset alongside the weights
sys.path.insert(0, str(DATASET_DIR))
sys.path.insert(0, str(WORK_DIR))

from avus_inference import AvusInference

avus = AvusInference(device=device)
ok   = avus.load()
if not ok:
    raise RuntimeError("Failed to load Avus weights — check the dataset contains avus_1b_weights.pt")

print(f"Avus ready on {device}")

# ── FastAPI server ────────────────────────────────────────────────────────────
app = FastAPI(title="Janus Avus Inference Server")

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": "avus"}

@app.post("/generate")
def generate(req: Req):
    try:
        text = avus.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
        return {"text": text, "device": device}
    except Exception as e:
        return {"text": f"ERROR: {e}", "device": device}

def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")

threading.Thread(target=_run_server, daemon=True).start()
time.sleep(3)
print("Server started on port 8080")

# ── ngrok tunnel ──────────────────────────────────────────────────────────────
from pyngrok import ngrok

# Get token from Kaggle secret or env
ngrok_token = os.environ.get("NGROK_TOKEN") or os.environ.get("NGROK_AUTHTOKEN", "")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
else:
    print("WARNING: No NGROK_TOKEN found. Add it as a Kaggle secret.")
    print("Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken")

tunnel = ngrok.connect(8080, "http")
url    = tunnel.public_url

print(f"\n{'='*55}")
print(f"  Avus Inference Server LIVE")
print(f"  URL: {url}")
print(f"{'='*55}")
print(f"\nAdd to your local Kaggle.env:")
print(f"  JANUS_INFERENCE_URL={url}")
print(f"\nKeep this notebook running to maintain the connection.")

while True:
    time.sleep(60)
    print(f"[alive] {url}")
