import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import io
import shutil
import subprocess
import json
from pathlib import Path
import logging
import kagglehub

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("model_epoch_weights.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS epoch_weights (
                epoch INTEGER PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                loss REAL NOT NULL
            )
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    finally:
        conn.close()

def save_epoch_to_db(epoch: int, model: nn.Module, loss: float):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    weights_bytes = buffer.getvalue()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            INSERT INTO epoch_weights (epoch, weights_blob, loss)
            VALUES (?, ?, ?)
            ON CONFLICT(epoch) DO UPDATE SET
                weights_blob = excluded.weights_blob,
                loss = excluded.loss
        ''', (epoch, sqlite3.Binary(weights_bytes), loss))
        conn.commit()
        logger.info(f"Saved epoch {epoch} weights to database. Loss: {loss:.4f}")
    except Exception as e:
        logger.error(f"Failed to save weights for epoch {epoch}: {e}")
    finally:
        conn.close()

def upload_db_to_kaggle(epoch: int):
    """Uploads the SQLite db to the specified kaggle dataset using Kaggle CLI."""
    upload_dir = Path("kaggle_upload_temp")
    upload_dir.mkdir(exist_ok=True)
    
    # 1. Copy the DB file to the upload directory
    if not DB_PATH.exists():
        logger.warning(f"Database {DB_PATH} does not exist, skipping upload.")
        return
    shutil.copy(DB_PATH, upload_dir / DB_PATH.name)
    
    # 2. Write Kaggle dataset-metadata.json
    metadata = {
        "title": "Janus Avus Weights",
        "id": "ishmaelsears/janus-avus-weights",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(upload_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    # 3. Call kaggle python API to upload
    logger.info(f"Pushing updated database for epoch {epoch} to Kaggle dataset...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate() # Automatically picks up KAGGLE_USERNAME and KAGGLE_KEY from env
        api.dataset_create_version(
            str(upload_dir), 
            version_notes=f"Epoch {epoch} SQLite checkpoint update", 
            dir_mode="tar"
        )
        logger.info("Successfully uploaded database to Kaggle via Python API.")
    except Exception as e:
        logger.error(f"Failed to execute Kaggle API upload: {e}")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def download_initial_weights():
    logger.info("Downloading initial dataset from Kaggle (ishmaelsears/janus-avus-weights)...")
    try:
        path = kagglehub.dataset_download("ishmaelsears/janus-avus-weights")
        logger.info(f"Dataset securely downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        return None

def train_model():
    logger.info("Initializing database...")
    init_db()
    
    dataset_path = download_initial_weights()
    if dataset_path:
        pass
    
    logger.info("Creating model...")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    dataset_size = 100
    
    inputs = torch.randn(dataset_size, 10)
    targets = torch.randint(0, 2, (dataset_size,))
    
    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Save weights to DB
        save_epoch_to_db(epoch, model, loss.item())
        
        # Upload DB to kaggle
        upload_db_to_kaggle(epoch)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
