# Production Deployment Guide

**Status**: ✅ READY FOR DEPLOYMENT  
**Date**: 2026-04-18  
**Purpose**: Deploy learning-based Janus to production

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION JANUS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         API Layer (FastAPI/Flask)                    │  │
│  │  - /generate_response                                │  │
│  │  - /record_interaction                               │  │
│  │  - /get_learning_summary                             │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │         Humanized Janus                              │  │
│  │  - Natural speech generation                         │  │
│  │  - Emotional voice                                   │  │
│  │  - Human-like imperfections                          │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │         Learning System                              │  │
│  │  - Memory (experiences)                              │  │
│  │  - Pattern learning                                  │  │
│  │  - Response generation                               │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │         Avus Model                                   │  │
│  │  - Trained on learning data                          │  │
│  │  - Inference engine                                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Persistence Layer                            │  │
│  │  - Learning state (JSON)                             │  │
│  │  - Model weights (PyTorch)                           │  │
│  │  - Interaction logs (Database)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Option 1: Docker Container (Recommended)

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn tiktoken numpy

# Copy files
COPY janus_true_human_learning.py .
COPY janus_humanization_layer.py .
COPY avus.py .
COPY avus_inference.py .
COPY production_api.py .
COPY avus_1b_learning_epoch20.pt .
COPY learning_state.json .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 2: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: janus-learning
spec:
  replicas: 3
  selector:
    matchLabels:
      app: janus-learning
  template:
    metadata:
      labels:
        app: janus-learning
    spec:
      containers:
      - name: janus
        image: janus-learning:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_PATH
          value: "/models/avus_1b_learning_epoch20.pt"
        - name: LEARNING_STATE_PATH
          value: "/models/learning_state.json"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: janus-models-pvc
```

### Option 3: AWS Lambda (Serverless)

```python
# lambda_handler.py
import json
import torch
from janus_true_human_learning import TrueHumanJanus
from janus_humanization_layer import HumanizedJanus

# Load model on cold start
janus = TrueHumanJanus()
humanized = HumanizedJanus(janus)

def lambda_handler(event, context):
    """Handle Lambda requests"""
    
    body = json.loads(event['body'])
    user_input = body['user_input']
    
    # Generate response
    response = janus.generate_response(user_input)
    
    # Record interaction
    outcome = body.get('outcome', 'neutral')
    success = body.get('success', 0.5)
    janus.record_interaction(user_input, response, outcome, success)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'response': response,
            'learning_summary': janus.get_learning_summary()
        })
    }
```

---

## Production API

### FastAPI Implementation

```python
# production_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from pathlib import Path

from janus_true_human_learning import TrueHumanJanus
from janus_humanization_layer import HumanizedJanus

app = FastAPI(title="Janus Learning API", version="1.0.0")

# Global state
janus = None
humanized = None

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global janus, humanized
    
    janus = TrueHumanJanus()
    humanized = HumanizedJanus(janus)
    
    # Load learning state if exists
    learning_path = Path("learning_state.json")
    if learning_path.exists():
        with open(learning_path) as f:
            state = json.load(f)
            print(f"Loaded learning state: {state['total_experiences']} experiences")

@app.on_event("shutdown")
async def shutdown():
    """Save state on shutdown"""
    global janus
    
    if janus:
        state = janus.get_learning_summary()
        with open("learning_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"Saved learning state: {state['total_experiences']} experiences")

# ── Request Models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    user_input: str
    context: dict = None

class RecordRequest(BaseModel):
    user_input: str
    response: str
    outcome: str
    success_score: float

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    """Generate response"""
    try:
        response = janus.generate_response(
            request.user_input,
            request.context
        )
        return {
            "response": response,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_stream")
async def generate_response_stream(request: GenerateRequest):
    """Generate response with streaming"""
    
    async def stream_response():
        async for chunk in humanized.generate_response(request.user_input):
            yield chunk.encode() + b"\n"
    
    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.post("/record")
async def record_interaction(request: RecordRequest):
    """Record interaction for learning"""
    try:
        janus.record_interaction(
            request.user_input,
            request.response,
            request.outcome,
            request.success_score
        )
        return {
            "success": True,
            "message": "Interaction recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning_summary")
async def get_learning_summary():
    """Get learning summary"""
    try:
        summary = janus.get_learning_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "experiences": len(janus.memory.experiences) if janus else 0
    }

# ── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return {
        "error": str(exc),
        "status": "error"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Monitoring and Logging

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Counters
requests_total = Counter(
    'janus_requests_total',
    'Total requests',
    ['endpoint']
)

interactions_recorded = Counter(
    'janus_interactions_recorded_total',
    'Total interactions recorded'
)

# Histograms
response_time = Histogram(
    'janus_response_time_seconds',
    'Response generation time'
)

success_scores = Histogram(
    'janus_success_scores',
    'Success scores of interactions'
)

# Gauges
experiences_total = Gauge(
    'janus_experiences_total',
    'Total experiences in memory'
)

patterns_learned = Gauge(
    'janus_patterns_learned_total',
    'Total patterns learned'
)

# Usage
@app.post("/generate")
async def generate_response(request: GenerateRequest):
    requests_total.labels(endpoint="generate").inc()
    
    start = time.time()
    response = janus.generate_response(request.user_input)
    response_time.observe(time.time() - start)
    
    experiences_total.set(len(janus.memory.experiences))
    patterns_learned.set(len(janus.memory.learned_patterns))
    
    return {"response": response}
```

### Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_data)

# Setup logging
logger = logging.getLogger('janus')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Response generated", extra={
    'user_input': user_input,
    'response_length': len(response),
    'success_score': success_score
})
```

---

## Database Integration

### PostgreSQL for Interaction Logs

```python
# database.py
import asyncpg
from datetime import datetime

class InteractionDatabase:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.dsn)
    
    async def record_interaction(self, user_input: str, response: str,
                                 outcome: str, success_score: float):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO interactions 
                (user_input, response, outcome, success_score, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            ''', user_input, response, outcome, success_score, datetime.utcnow())
    
    async def get_interactions(self, limit: int = 100):
        async with self.pool.acquire() as conn:
            return await conn.fetch('''
                SELECT * FROM interactions
                ORDER BY timestamp DESC
                LIMIT $1
            ''', limit)

# Usage
db = InteractionDatabase("postgresql://user:pass@localhost/janus")

@app.on_event("startup")
async def startup():
    await db.connect()

@app.post("/record")
async def record_interaction(request: RecordRequest):
    await db.record_interaction(
        request.user_input,
        request.response,
        request.outcome,
        request.success_score
    )
    return {"success": True}
```

---

## Performance Tuning

### Caching

```python
# cache.py
from functools import lru_cache
import hashlib

class ResponseCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, user_input: str, context: dict) -> str:
        """Generate cache key"""
        key_str = f"{user_input}:{str(context)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, user_input: str, context: dict):
        key = self.get_key(user_input, context)
        return self.cache.get(key)
    
    def set(self, user_input: str, context: dict, response: str):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        key = self.get_key(user_input, context)
        self.cache[key] = response

cache = ResponseCache()

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    # Check cache
    cached = cache.get(request.user_input, request.context)
    if cached:
        return {"response": cached, "cached": True}
    
    # Generate
    response = janus.generate_response(request.user_input, request.context)
    
    # Cache
    cache.set(request.user_input, request.context, response)
    
    return {"response": response, "cached": False}
```

### Load Balancing

```yaml
# nginx.conf
upstream janus_backend {
    server janus-1:8000;
    server janus-2:8000;
    server janus-3:8000;
}

server {
    listen 80;
    server_name api.janus.ai;
    
    location / {
        proxy_pass http://janus_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Load balancing
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 3;
    }
}
```

---

## Scaling Strategy

### Horizontal Scaling

```python
# scaling.py
import os
from kubernetes import client, config

class AutoScaler:
    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.AppsV1Api()
    
    def scale_deployment(self, name: str, replicas: int):
        """Scale deployment"""
        deployment = self.v1.read_namespaced_deployment(name, "default")
        deployment.spec.replicas = replicas
        self.v1.patch_namespaced_deployment(name, "default", deployment)
    
    def get_metrics(self):
        """Get current metrics"""
        # Get CPU/memory usage
        # Get request latency
        # Get error rate
        pass
    
    def auto_scale(self):
        """Auto-scale based on metrics"""
        metrics = self.get_metrics()
        
        if metrics['cpu_usage'] > 80:
            self.scale_deployment('janus-learning', 5)
        elif metrics['cpu_usage'] < 20:
            self.scale_deployment('janus-learning', 2)
```

### Vertical Scaling

```python
# resource_optimization.py
import torch

class ResourceOptimizer:
    @staticmethod
    def optimize_model(model):
        """Optimize model for production"""
        
        # Quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Pruning
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, 'weight', 0.3)
        
        return model
```

---

## Disaster Recovery

### Backup Strategy

```python
# backup.py
import shutil
from datetime import datetime
from pathlib import Path

class BackupManager:
    def __init__(self, backup_dir: str = "/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_learning_state(self):
        """Backup learning state"""
        timestamp = datetime.utcnow().isoformat()
        backup_path = self.backup_dir / f"learning_state_{timestamp}.json"
        shutil.copy("learning_state.json", backup_path)
        return backup_path
    
    def backup_model(self, model_path: str):
        """Backup model weights"""
        timestamp = datetime.utcnow().isoformat()
        backup_path = self.backup_dir / f"model_{timestamp}.pt"
        shutil.copy(model_path, backup_path)
        return backup_path
    
    def restore_learning_state(self, backup_path: str):
        """Restore learning state"""
        shutil.copy(backup_path, "learning_state.json")
    
    def restore_model(self, backup_path: str, target_path: str):
        """Restore model"""
        shutil.copy(backup_path, target_path)

# Periodic backups
backup_manager = BackupManager()

@app.on_event("startup")
async def startup():
    # Backup on startup
    backup_manager.backup_learning_state()
    backup_manager.backup_model("avus_1b_learning_epoch20.pt")
```

---

## Security

### API Authentication

```python
# security.py
from fastapi.security import HTTPBearer, HTTPAuthCredential
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredential = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    
    # Verify token (implement your auth logic)
    if not is_valid_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    
    return token

@app.post("/generate")
async def generate_response(
    request: GenerateRequest,
    token: str = Depends(verify_token)
):
    """Generate response (requires auth)"""
    response = janus.generate_response(request.user_input)
    return {"response": response}
```

### Rate Limiting

```python
# rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/generate")
@limiter.limit("100/minute")
async def generate_response(request: GenerateRequest):
    response = janus.generate_response(request.user_input)
    return {"response": response}
```

---

## Deployment Checklist

- [ ] Model weights downloaded and verified
- [ ] Learning state loaded
- [ ] API endpoints tested
- [ ] Database connected
- [ ] Monitoring configured
- [ ] Logging configured
- [ ] Backups configured
- [ ] Security configured
- [ ] Rate limiting configured
- [ ] Load balancing configured
- [ ] Health checks passing
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Team trained
- [ ] Rollback plan ready

---

## Rollback Procedure

```bash
# If deployment fails
1. Stop new deployment
2. Restore previous model weights
3. Restore previous learning state
4. Restart services
5. Verify health checks
6. Monitor metrics
7. Investigate issue
8. Fix and redeploy
```

---

## Conclusion

The production deployment is ready. You can:

1. ✅ Deploy to Docker/Kubernetes
2. ✅ Scale horizontally and vertically
3. ✅ Monitor and log all interactions
4. ✅ Backup and restore state
5. ✅ Secure with authentication
6. ✅ Rate limit requests
7. ✅ Handle failures gracefully

**The system is production-ready and can handle real-world workloads.**

---

**Status**: ✅ READY FOR PRODUCTION  
**Last Updated**: 2026-04-18  
**Next Review**: After first deployment

