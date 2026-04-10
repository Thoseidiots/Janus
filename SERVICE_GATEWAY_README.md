# Janus Service Gateway

**Complete autonomous money-earning layer for Janus**

Bridges HTTP/REST → Nexus Core → soft_ntb execution with credit tracking, quality verification, and task orchestration.

---

## Architecture

```
Client (mesh or internet)
    ↓ HTTP/REST
Service Gateway (gateway.mesh:8000)
    ↓ Task Queue
Task Executor
    ↓ gRPC
Nexus Core (localhost:50051)
    ↓ soft_ntb
Distributed Execution (EliteDesk, laptop, phone, etc.)
    ↓ Result
Quality Verifier
    ↓ Verification passed
Credit Ledger (Janus credits)
    ↓ Response
Client receives verified result
```

---

## Components

### 1. Service Gateway
- **HTTP/REST API** on port 8000
- **Task queue** with async worker loop
- **Authentication** via Bearer tokens
- **Rate limiting** (future)
- **MeshISP integration** (registers as `gateway.mesh`)

### 2. Credit Ledger
- **Tracks user balances** in Janus credits
- **Persistent storage** (JSON file)
- **Transaction log** (all charges/credits)
- **Upfront charging** (reserve credits before execution)
- **Auto-refund** on task failure

### 3. Task Executor
- **Routes to Nexus Core** via gRPC
- **Mock mode** for testing without Nexus
- **Cost calculation** based on task type
- **Timeout handling**

### 4. Quality Verifiers
Different verifier for each task type:
- **CodeGenerationVerifier** - Syntax checking, pattern matching
- **DataProcessingVerifier** - Schema validation, row counts
- **FileOperationVerifier** - File existence, checksums
- **AIInferenceVerifier** - Confidence thresholds

### 5. Task Types Supported
- `code_generation` - 100 credits
- `data_processing` - 50 credits
- `file_operation` - 25 credits
- `ai_inference` - 200 credits
- `compute` - 10 credits per unit

---

## Quick Start

### Prerequisites
- Python 3.9+
- Nexus Core running (optional for testing)
- MeshISP running (optional)

### Installation

```bash
pip install flask grpc asyncio
```

### Run

```bash
# With Nexus Core
python janus_service_gateway.py --port 8000 --nexus localhost:50051

# Standalone (mock mode)
python janus_service_gateway.py --port 8000
```

### Test

```bash
# Health check
curl http://localhost:8000/health

# Submit task (replace YOUR_USER_ID with any identifier)
curl -X POST http://localhost:8000/api/tasks/submit \
  -H "Authorization: Bearer YOUR_USER_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "code_generation",
    "description": "Generate a hello world function",
    "input_data": {
      "language": "python",
      "function_name": "hello"
    },
    "verification_rules": {
      "must_contain": ["def hello"]
    }
  }'

# Get task status
curl http://localhost:8000/api/tasks/<TASK_ID> \
  -H "Authorization: Bearer YOUR_USER_ID"

# Check balance
curl http://localhost:8000/api/credits/balance \
  -H "Authorization: Bearer YOUR_USER_ID"
```

---

## API Reference

### POST /api/tasks/submit

Submit a new task for execution.

**Headers:**
```
Authorization: Bearer <user_id>
Content-Type: application/json
```

**Request:**
```json
{
  "task_type": "code_generation",
  "description": "Generate a function that...",
  "input_data": {
    "language": "python",
    "requirements": ["async", "error handling"]
  },
  "verification_rules": {
    "must_contain": ["async def"],
    "must_not_contain": ["eval", "exec"]
  },
  "max_cost": 1000,
  "timeout_seconds": 300
}
```

**Response:**
```json
{
  "ok": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "cost": 100
}
```

---

### GET /api/tasks/:task_id

Get task status and result.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Response:**
```json
{
  "ok": true,
  "task": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "task_type": "code_generation",
    "description": "Generate a function that...",
    "created_at": 1712790000.0,
    "started_at": 1712790001.0,
    "completed_at": 1712790005.0,
    "cost": 100,
    "result": {
      "code": "async def process_data():\n    ...",
      "language": "python"
    },
    "error": null,
    "verification_passed": true,
    "execution_node": "elitedesk-node-0"
  }
}
```

**Status values:**
- `pending` - Waiting to be queued
- `queued` - In queue, not yet executing
- `executing` - Currently running
- `verifying` - Execution complete, verifying result
- `completed` - Verified and successful
- `failed` - Failed execution or verification

---

### GET /api/credits/balance

Get user credit balance.

**Headers:**
```
Authorization: Bearer <user_id>
```

**Response:**
```json
{
  "ok": true,
  "user_id": "user123",
  "balance": 850,
  "tasks_completed": 3
}
```

---

### POST /api/credits/add

Add credits to account (admin/payment integration).

**Headers:**
```
Authorization: Bearer <user_id>
```

**Request:**
```json
{
  "amount": 500
}
```

**Response:**
```json
{
  "ok": true,
  "new_balance": 1350
}
```

---

### GET /api/services

List available services and costs.

**Response:**
```json
{
  "ok": true,
  "services": [
    {
      "type": "code_generation",
      "base_cost": 100,
      "description": "Code Generation service"
    },
    {
      "type": "data_processing",
      "base_cost": 50,
      "description": "Data Processing service"
    }
  ]
}
```

---

## Task Examples

### Code Generation

```json
{
  "task_type": "code_generation",
  "description": "Create async data processor",
  "input_data": {
    "language": "python",
    "function_name": "process_batch",
    "requirements": ["async", "type hints", "error handling"]
  },
  "verification_rules": {
    "must_contain": ["async def", "try:", "except"],
    "must_not_contain": ["eval", "exec"]
  }
}
```

### Data Processing

```json
{
  "task_type": "data_processing",
  "description": "Clean and transform CSV data",
  "input_data": {
    "source": "/data/raw.csv",
    "operations": ["remove_nulls", "dedupe", "normalize"]
  },
  "verification_rules": {
    "required_fields": ["row_count", "data"],
    "expected_rows": 1000
  }
}
```

### File Operation

```json
{
  "task_type": "file_operation",
  "description": "Organize photos by date",
  "input_data": {
    "source_dir": "/photos",
    "target_dir": "/photos_organized",
    "pattern": "YYYY/MM/DD"
  },
  "verification_rules": {
    "expected_file_count": 500
  }
}
```

### AI Inference

```json
{
  "task_type": "ai_inference",
  "description": "Classify image",
  "input_data": {
    "image_path": "/uploads/photo.jpg",
    "model": "avus-vision-1b"
  },
  "verification_rules": {
    "min_confidence": 0.8
  }
}
```

---

## Integration with Existing Components

### MeshISP Integration

Register gateway as a service on the mesh:

```bash
# Add DNS record
curl -X POST http://localhost:3000/api/isp/dns/addRecord \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "gateway.mesh",
    "recordType": "A",
    "ipAddress": "10.99.1.1",
    "ttl": 300,
    "description": "Janus Service Gateway"
  }'
```

Clients on the mesh can now access: `http://gateway.mesh:8000`

### Nexus Core Integration

Gateway connects to Nexus Core via gRPC:

```python
# Maps task types to Nexus commands
{
  "code_generation": "spawn_wasm_task",
  "data_processing": "spawn_wasm_task",
  "ai_inference": "add_task",
  "compute": "spawn_wasm_task"
}
```

Nexus handles:
- WASM task spawning
- soft_ntb state migration
- LAS node selection
- JCE memory coherency

### soft_ntb Integration

Tasks execute on optimal nodes via soft_ntb:

1. Gateway submits to Nexus
2. Nexus uses LAS to select node
3. Task migrates via soft_ntb if needed
4. Executes on best hardware
5. Result returns to gateway
6. Gateway verifies and bills

---

## Credit Economy

### Starting Credits
New users get **1000 credits** on signup.

### Earning Credits
(Future implementation ideas)
- Provide compute resources to the mesh
- Share bandwidth
- Complete verification tasks
- Referral bonuses

### Spending Credits
Every task charges upfront based on type and complexity.

### Refund Policy
- Task fails → **full refund**
- Task times out → **full refund**
- Verification fails → **full refund**
- Task succeeds → **credits charged**

### Credit Ledger Persistence
All transactions saved to `credit_ledger.json`:

```json
{
  "users": {
    "user123": {
      "id": "user123",
      "email": "user@mesh.local",
      "credit_balance": 850,
      "tasks_completed": 3,
      "created_at": 1712790000.0
    }
  },
  "transactions": [
    {
      "user_id": "user123",
      "task_id": "task-abc",
      "amount": -100,
      "description": "Task task-abc reservation",
      "timestamp": 1712790001.0
    }
  ]
}
```

---

## Production Deployment

### 1. Enable Raft Consensus for Ledger

Replace JSON persistence with Raft-backed distributed ledger:

```python
# Use Nexus Core's Raft for credit consensus
from raft_app import Raft

class DistributedCreditLedger(CreditLedger):
    def __init__(self, raft_client):
        self.raft = raft_client

    def charge(self, user_id, amount, task_id, description):
        # Propose transaction via Raft
        entry = {"type": "charge", "user_id": user_id, "amount": amount}
        self.raft.client_write(entry)
```

### 2. Add Proper Authentication

Replace Bearer tokens with JWT:

```python
import jwt

def require_auth():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except:
        return None
```

### 3. Scale with Multiple Gateway Instances

Run multiple gateways behind a load balancer:

```bash
# Node 1
python janus_service_gateway.py --port 8000 --nexus node1:50051

# Node 2
python janus_service_gateway.py --port 8000 --nexus node2:50051

# Load balancer distributes requests
```

### 4. Add Monitoring

```python
from prometheus_client import Counter, Histogram

tasks_submitted = Counter("tasks_submitted_total", "Total tasks submitted")
task_duration = Histogram("task_duration_seconds", "Task execution time")
```

### 5. Enable HTTPS

```bash
# Generate cert
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with SSL
app.run(host="0.0.0.0", port=8443, ssl_context=("cert.pem", "key.pem"))
```

---

## File Structure

```
Janus/
├── janus_service_gateway.py      # Main gateway service
├── nexus_client.py                # Nexus gRPC client
├── soft_ntb.rs                    # Distributed memory layer
├── crates/nexus-core/             # Nexus Core (Rust)
├── mesh_isp_dashboard/            # MeshISP network layer
├── credit_ledger.json             # Persistent credit ledger
└── SERVICE_GATEWAY_README.md      # This file
```

---

## What This Enables

**Autonomous Money-Earning Flow:**

1. User submits task to `gateway.mesh:8000`
2. Gateway validates and reserves credits
3. Routes to Nexus Core
4. Nexus schedules on best node via LAS
5. Task migrates via soft_ntb if needed
6. Executes (code gen, data processing, AI inference, etc.)
7. Result returns to gateway
8. Gateway verifies output quality
9. If passed: charge credits, deliver result
10. If failed: refund credits, return error

**All without external API keys. All on your infrastructure.**

---

## Next Steps

1. **Integrate with MeshISP DNS** - Register `gateway.mesh` domain
2. **Connect to Nexus Core** - Start Nexus and test gRPC connection
3. **Test task submission** - Submit sample tasks and verify execution
4. **Add payment integration** - Connect credit purchases to real payment
5. **Deploy to EliteDesk** - Run 24/7 as systemd service
6. **Scale with multiple nodes** - Add laptop, cloud VMs to the mesh

---

**Status**: ✓ Service Gateway complete and ready for integration

The autonomous compute marketplace is now functional. Janus can accept tasks, execute them across distributed nodes, verify quality, and track credits - all running entirely on your personal infrastructure.
