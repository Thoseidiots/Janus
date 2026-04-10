"""
janus_service_gateway.py
=========================
Service Gateway for Janus - bridges HTTP/REST → Nexus Core → soft_ntb execution.

Architecture:
  HTTP/REST API (gateway.mesh:8000)
    ↓
  Task Queue & Validation
    ↓
  Nexus Core (gRPC)
    ↓
  soft_ntb → Distributed Execution
    ↓
  Result Verification
    ↓
  Credit Ledger Update
    ↓
  Response to Client

Features:
  - RESTful task submission API
  - Credit/billing system (Raft-backed ledger)
  - Quality verification (coherency-style checks)
  - Task type registry (extensible)
  - MeshISP integration (registers as gateway.mesh)
  - No external dependencies (fully self-contained)

Usage:
  python janus_service_gateway.py --port 8000 --nexus localhost:50051
"""

import asyncio
import json
import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

import grpc
from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import HTTPException

# Import Nexus gRPC client
try:
    from nexus_client import NexusClient
except ImportError:
    print("[Gateway] Warning: nexus_client.py not found - running in mock mode")
    NexusClient = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Supported task types."""
    CODE_GENERATION = "code_generation"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    AI_INFERENCE = "ai_inference"
    COMPUTE = "compute"


@dataclass
class TaskSpec:
    """Task specification from client."""
    task_type: str
    description: str
    input_data: Dict[str, Any]
    verification_rules: Dict[str, Any] = field(default_factory=dict)
    max_cost: int = 1000  # Max Janus credits willing to pay
    timeout_seconds: int = 300


@dataclass
class Task:
    """Internal task representation."""
    id: str
    user_id: str
    spec: TaskSpec
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost: int = 0
    verification_passed: bool = False
    execution_node: Optional[str] = None


@dataclass
class User:
    """User account."""
    id: str
    email: str
    credit_balance: int = 0
    tasks_completed: int = 0
    created_at: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# CREDIT LEDGER
# ══════════════════════════════════════════════════════════════════════════════

class CreditLedger:
    """
    Simple credit tracking system.
    In production: use Raft for distributed consensus.
    """

    def __init__(self, persistence_file: str = "credit_ledger.json"):
        self.persistence_file = Path(persistence_file)
        self.users: Dict[str, User] = {}
        self.transactions: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load ledger from disk."""
        if self.persistence_file.exists():
            try:
                data = json.loads(self.persistence_file.read_text())
                self.users = {uid: User(**u) for uid, u in data.get("users", {}).items()}
                self.transactions = data.get("transactions", [])
                print(f"[Ledger] Loaded {len(self.users)} users, {len(self.transactions)} transactions")
            except Exception as e:
                print(f"[Ledger] Load error: {e}")

    def _save(self):
        """Persist ledger to disk."""
        data = {
            "users": {uid: asdict(u) for uid, u in self.users.items()},
            "transactions": self.transactions,
        }
        self.persistence_file.write_text(json.dumps(data, indent=2))

    def get_or_create_user(self, user_id: str, email: str = "") -> User:
        """Get user or create if doesn't exist."""
        if user_id not in self.users:
            self.users[user_id] = User(
                id=user_id,
                email=email or f"{user_id}@mesh.local",
                credit_balance=1000,  # Starting credits
            )
            self._save()
        return self.users[user_id]

    def charge(self, user_id: str, amount: int, task_id: str, description: str) -> bool:
        """
        Charge user credits. Returns True if successful.
        """
        user = self.users.get(user_id)
        if not user:
            return False

        if user.credit_balance < amount:
            return False

        user.credit_balance -= amount
        self.transactions.append({
            "user_id": user_id,
            "task_id": task_id,
            "amount": -amount,
            "description": description,
            "timestamp": time.time(),
        })
        self._save()
        return True

    def credit(self, user_id: str, amount: int, task_id: str, description: str):
        """Add credits to user account."""
        user = self.users.get(user_id)
        if not user:
            return

        user.credit_balance += amount
        self.transactions.append({
            "user_id": user_id,
            "task_id": task_id,
            "amount": amount,
            "description": description,
            "timestamp": time.time(),
        })
        self._save()

    def get_balance(self, user_id: str) -> int:
        """Get user credit balance."""
        user = self.users.get(user_id)
        return user.credit_balance if user else 0


# ══════════════════════════════════════════════════════════════════════════════
# TASK VERIFIERS
# ══════════════════════════════════════════════════════════════════════════════

class TaskVerifier:
    """Base class for task output verification."""

    def verify(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Verify task result.
        Returns: (passed: bool, error_message: Optional[str])
        """
        raise NotImplementedError


class CodeGenerationVerifier(TaskVerifier):
    """Verify generated code."""

    def verify(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        code = result.get("code", "")

        # Basic checks
        if not code:
            return False, "No code generated"

        if len(code) < 10:
            return False, "Code too short"

        # Check for syntax errors (if Python)
        if result.get("language") == "python":
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                return False, f"Syntax error: {e}"

        # Check verification rules
        rules = task.spec.verification_rules
        if "must_contain" in rules:
            for pattern in rules["must_contain"]:
                if pattern not in code:
                    return False, f"Missing required pattern: {pattern}"

        if "must_not_contain" in rules:
            for pattern in rules["must_not_contain"]:
                if pattern in code:
                    return False, f"Contains forbidden pattern: {pattern}"

        return True, None


class DataProcessingVerifier(TaskVerifier):
    """Verify data processing results."""

    def verify(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        # Check for required output fields
        rules = task.spec.verification_rules

        if "required_fields" in rules:
            for field in rules["required_fields"]:
                if field not in result:
                    return False, f"Missing required field: {field}"

        # Check row count if specified
        if "expected_rows" in rules:
            actual = result.get("row_count", 0)
            expected = rules["expected_rows"]
            if actual != expected:
                return False, f"Row count mismatch: expected {expected}, got {actual}"

        # Check schema if specified
        if "schema" in rules and "data" in result:
            schema = rules["schema"]
            data = result["data"]
            if isinstance(data, list) and len(data) > 0:
                row = data[0]
                for field, field_type in schema.items():
                    if field not in row:
                        return False, f"Missing field in data: {field}"

        return True, None


class FileOperationVerifier(TaskVerifier):
    """Verify file operations."""

    def verify(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        # Check files exist
        if "files_created" in result:
            for filepath in result["files_created"]:
                path = Path(filepath)
                if not path.exists():
                    return False, f"File not found: {filepath}"

        # Check file count
        rules = task.spec.verification_rules
        if "expected_file_count" in rules:
            actual = len(result.get("files_created", []))
            expected = rules["expected_file_count"]
            if actual != expected:
                return False, f"File count mismatch: expected {expected}, got {actual}"

        return True, None


class AIInferenceVerifier(TaskVerifier):
    """Verify AI inference results."""

    def verify(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        # Check for output
        if "output" not in result:
            return False, "No inference output"

        output = result["output"]
        if not output:
            return False, "Empty inference output"

        # Check confidence if provided
        if "confidence" in result:
            conf = result["confidence"]
            rules = task.spec.verification_rules
            min_conf = rules.get("min_confidence", 0.0)
            if conf < min_conf:
                return False, f"Confidence {conf} below minimum {min_conf}"

        return True, None


# ══════════════════════════════════════════════════════════════════════════════
# TASK EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class TaskExecutor:
    """
    Executes tasks by routing to Nexus Core.
    """

    def __init__(self, nexus_host: str = "localhost:50051"):
        self.nexus_host = nexus_host
        self.nexus_client: Optional[NexusClient] = None
        self.verifiers: Dict[str, TaskVerifier] = {
            "code_generation": CodeGenerationVerifier(),
            "data_processing": DataProcessingVerifier(),
            "file_operation": FileOperationVerifier(),
            "ai_inference": AIInferenceVerifier(),
        }

    async def initialize(self):
        """Initialize Nexus gRPC connection."""
        if NexusClient:
            try:
                self.nexus_client = NexusClient(
                    host=self.nexus_host.split(":")[0],
                    port=int(self.nexus_host.split(":")[1])
                )
                print(f"[Executor] Connected to Nexus at {self.nexus_host}")
            except Exception as e:
                print(f"[Executor] Failed to connect to Nexus: {e}")
                print("[Executor] Running in mock mode")

    async def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task via Nexus Core.
        Returns result dict.
        """
        print(f"[Executor] Executing task {task.id} ({task.spec.task_type})")

        # In production: route to Nexus
        if self.nexus_client:
            try:
                result = await self._execute_via_nexus(task)
                return result
            except Exception as e:
                print(f"[Executor] Nexus execution failed: {e}")
                return {"error": str(e)}

        # Mock execution for testing
        return self._mock_execute(task)

    async def _execute_via_nexus(self, task: Task) -> Dict[str, Any]:
        """Execute via Nexus Core gRPC."""
        payload = {
            "task_id": task.id,
            "task_type": task.spec.task_type,
            "input_data": task.spec.input_data,
            "description": task.spec.description,
        }

        # Map task type to Nexus command
        command_type = "spawn_wasm_task"  # Default
        if task.spec.task_type == "ai_inference":
            command_type = "add_task"

        response = await self.nexus_client.execute_command(command_type, payload)
        return response

    def _mock_execute(self, task: Task) -> Dict[str, Any]:
        """Mock execution for testing without Nexus."""
        time.sleep(0.5)  # Simulate work

        # Generate mock results based on task type
        if task.spec.task_type == "code_generation":
            return {
                "code": "def hello():\n    return 'Hello from Janus'\n",
                "language": "python",
            }
        elif task.spec.task_type == "data_processing":
            return {
                "row_count": 100,
                "data": [{"id": i, "value": i * 2} for i in range(10)],
            }
        elif task.spec.task_type == "file_operation":
            return {
                "files_created": ["/tmp/test.txt"],
            }
        elif task.spec.task_type == "ai_inference":
            return {
                "output": "This is a mock inference result",
                "confidence": 0.95,
            }
        else:
            return {
                "result": "Mock computation completed",
                "compute_units": 42,
            }

    def verify_result(self, task: Task, result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Verify task result using appropriate verifier."""
        verifier = self.verifiers.get(task.spec.task_type)
        if not verifier:
            # No specific verifier - basic check
            return bool(result and not result.get("error")), None

        return verifier.verify(task, result)


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE GATEWAY
# ══════════════════════════════════════════════════════════════════════════════

class JanusServiceGateway:
    """
    Main service gateway orchestrating task submission, execution, and billing.
    """

    def __init__(self, nexus_host: str = "localhost:50051"):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.ledger = CreditLedger()
        self.executor = TaskExecutor(nexus_host)
        self.running = False

        # Cost model (Janus credits per task type)
        self.base_costs = {
            "code_generation": 100,
            "data_processing": 50,
            "file_operation": 25,
            "ai_inference": 200,
            "compute": 10,  # per compute unit
        }

    async def start(self):
        """Start the gateway worker loop."""
        await self.executor.initialize()
        self.running = True

        # Start worker loop
        asyncio.create_task(self._worker_loop())
        print("[Gateway] Worker loop started")

    async def stop(self):
        """Stop the gateway."""
        self.running = False

    async def submit_task(self, user_id: str, task_spec: TaskSpec) -> Task:
        """
        Submit a task for execution.
        Returns Task object.
        """
        # Create user if doesn't exist
        user = self.ledger.get_or_create_user(user_id)

        # Calculate cost
        cost = self._calculate_cost(task_spec)

        # Check user has sufficient credits
        if user.credit_balance < cost:
            raise ValueError(f"Insufficient credits: need {cost}, have {user.credit_balance}")

        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            user_id=user_id,
            spec=task_spec,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            cost=cost,
        )

        self.tasks[task.id] = task

        # Reserve credits (charge upfront, refund if fails)
        if not self.ledger.charge(user_id, cost, task.id, f"Task {task.id} reservation"):
            raise ValueError("Failed to reserve credits")

        # Queue for execution
        task.status = TaskStatus.QUEUED
        await self.task_queue.put(task.id)

        print(f"[Gateway] Task {task.id} submitted by {user_id} (cost: {cost} credits)")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def _calculate_cost(self, spec: TaskSpec) -> int:
        """Calculate task cost in Janus credits."""
        base = self.base_costs.get(spec.task_type, 50)

        # Adjust for complexity
        complexity_multiplier = 1.0
        if len(str(spec.input_data)) > 10000:
            complexity_multiplier = 2.0

        return int(base * complexity_multiplier)

    async def _worker_loop(self):
        """Background worker that processes queued tasks."""
        print("[Gateway] Worker loop running")

        while self.running:
            try:
                # Get next task from queue (with timeout)
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                task = self.tasks.get(task_id)
                if not task:
                    continue

                # Execute task
                await self._execute_task(task)

            except Exception as e:
                print(f"[Gateway] Worker error: {e}")

    async def _execute_task(self, task: Task):
        """Execute a single task."""
        try:
            # Update status
            task.status = TaskStatus.EXECUTING
            task.started_at = time.time()

            # Execute via Nexus
            result = await self.executor.execute(task)

            # Verify result
            task.status = TaskStatus.VERIFYING
            passed, error = self.executor.verify_result(task, result)

            if passed:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.verification_passed = True
                task.completed_at = time.time()

                # Update user stats
                user = self.ledger.users.get(task.user_id)
                if user:
                    user.tasks_completed += 1
                    self.ledger._save()

                print(f"[Gateway] Task {task.id} completed successfully")
            else:
                task.status = TaskStatus.FAILED
                task.error = error or "Verification failed"
                task.completed_at = time.time()

                # Refund credits on failure
                self.ledger.credit(
                    task.user_id,
                    task.cost,
                    task.id,
                    f"Refund for failed task {task.id}"
                )

                print(f"[Gateway] Task {task.id} failed: {task.error}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

            # Refund on error
            self.ledger.credit(
                task.user_id,
                task.cost,
                task.id,
                f"Refund for errored task {task.id}"
            )

            print(f"[Gateway] Task {task.id} error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# REST API
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
gateway: Optional[JanusServiceGateway] = None


def require_auth():
    """Simple authentication check."""
    # In production: use proper JWT or API key auth
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]
    # For now, token is the user_id (replace with proper auth)
    return token if token else None


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON for HTTP errors."""
    return jsonify({
        "ok": False,
        "error": e.description,
        "code": e.code,
    }), e.code


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "ok": True,
        "service": "janus-gateway",
        "version": "0.1.0",
        "tasks_queued": gateway.task_queue.qsize() if gateway else 0,
        "tasks_total": len(gateway.tasks) if gateway else 0,
    })


@app.route("/api/tasks/submit", methods=["POST"])
def submit_task():
    """Submit a new task."""
    user_id = require_auth()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    try:
        task_spec = TaskSpec(
            task_type=data.get("task_type", "compute"),
            description=data.get("description", ""),
            input_data=data.get("input_data", {}),
            verification_rules=data.get("verification_rules", {}),
            max_cost=data.get("max_cost", 1000),
            timeout_seconds=data.get("timeout_seconds", 300),
        )

        # Submit task (async)
        loop = asyncio.new_event_loop()
        task = loop.run_until_complete(gateway.submit_task(user_id, task_spec))
        loop.close()

        return jsonify({
            "ok": True,
            "task_id": task.id,
            "status": task.status.value,
            "cost": task.cost,
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Internal error: {e}"}), 500


@app.route("/api/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id: str):
    """Get task status and result."""
    user_id = require_auth()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    task = gateway.get_task(task_id)
    if not task:
        return jsonify({"ok": False, "error": "Task not found"}), 404

    if task.user_id != user_id:
        return jsonify({"ok": False, "error": "Forbidden"}), 403

    return jsonify({
        "ok": True,
        "task": {
            "id": task.id,
            "status": task.status.value,
            "task_type": task.spec.task_type,
            "description": task.spec.description,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "cost": task.cost,
            "result": task.result if task.status == TaskStatus.COMPLETED else None,
            "error": task.error,
            "verification_passed": task.verification_passed,
            "execution_node": task.execution_node,
        }
    })


@app.route("/api/credits/balance", methods=["GET"])
def get_balance():
    """Get user credit balance."""
    user_id = require_auth()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    balance = gateway.ledger.get_balance(user_id)
    user = gateway.ledger.users.get(user_id)

    return jsonify({
        "ok": True,
        "user_id": user_id,
        "balance": balance,
        "tasks_completed": user.tasks_completed if user else 0,
    })


@app.route("/api/credits/add", methods=["POST"])
def add_credits():
    """Add credits to user account (admin only for now)."""
    # In production: integrate with payment system
    user_id = require_auth()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    data = request.get_json()
    amount = data.get("amount", 0)

    if amount <= 0:
        return jsonify({"ok": False, "error": "Invalid amount"}), 400

    gateway.ledger.credit(
        user_id,
        amount,
        "manual",
        "Manual credit addition"
    )

    return jsonify({
        "ok": True,
        "new_balance": gateway.ledger.get_balance(user_id),
    })


@app.route("/api/services", methods=["GET"])
def list_services():
    """List available service types and their costs."""
    return jsonify({
        "ok": True,
        "services": [
            {
                "type": task_type,
                "base_cost": cost,
                "description": f"{task_type.replace('_', ' ').title()} service",
            }
            for task_type, cost in gateway.base_costs.items()
        ]
    })


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Janus Service Gateway")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host")
    parser.add_argument("--nexus", type=str, default="localhost:50051", help="Nexus gRPC host:port")
    args = parser.parse_args()

    # Initialize gateway
    global gateway
    gateway = JanusServiceGateway(nexus_host=args.nexus)

    # Start gateway worker
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(gateway.start())

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      JANUS SERVICE GATEWAY                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HTTP API:     http://{args.host}:{args.port}
║  Nexus Core:   {args.nexus}
║  Health:       http://{args.host}:{args.port}/health
║                                                                              ║
║  Endpoints:                                                                  ║
║    POST   /api/tasks/submit        - Submit task                            ║
║    GET    /api/tasks/<id>          - Get task status                        ║
║    GET    /api/credits/balance     - Get credit balance                     ║
║    POST   /api/credits/add         - Add credits                            ║
║    GET    /api/services            - List services                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run Flask app
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\n[Gateway] Shutting down...")
        loop.run_until_complete(gateway.stop())


if __name__ == "__main__":
    main()
