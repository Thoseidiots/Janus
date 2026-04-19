"""
janus_credits.py — Janus Credits (JC) economy system

JC is a compute-backed credit system. Users earn JC by contributing compute
to Janus's network and spend JC to have Janus perform tasks for them.
"""

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = "janus_credits.db"
WELCOME_BONUS = 10.0

COMPUTE_RATES = {
    "inference": 0.1,
    "training": 0.5,
    "storage": 0.01,
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create all required tables if they do not already exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                user_id     TEXT PRIMARY KEY,
                balance_jc  REAL    NOT NULL DEFAULT 0.0,
                total_earned REAL   NOT NULL DEFAULT 0.0,
                total_spent  REAL   NOT NULL DEFAULT 0.0,
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                tx_id         TEXT PRIMARY KEY,
                from_user     TEXT,
                to_user       TEXT,
                amount_jc     REAL    NOT NULL,
                tx_type       TEXT    NOT NULL,
                description   TEXT,
                compute_proof TEXT,
                timestamp     TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id      TEXT PRIMARY KEY,
                requester_id TEXT NOT NULL,
                worker_id    TEXT,
                description  TEXT NOT NULL,
                jc_reward    REAL NOT NULL,
                task_type    TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'open',
                result       TEXT,
                created_at   TEXT NOT NULL,
                completed_at TEXT
            );
        """)
        conn.commit()
    finally:
        conn.close()


_init_db()


# ---------------------------------------------------------------------------
# JCLedger
# ---------------------------------------------------------------------------

class JCLedger:
    """SQLite-backed ledger for Janus Credits (JC)."""

    def create_account(self, user_id: str) -> float:
        """
        Create a new account with a 10 JC welcome bonus.

        Returns the starting balance. Raises ValueError if the account
        already exists.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            existing = conn.execute(
                "SELECT user_id FROM accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            if existing:
                raise ValueError(f"Account '{user_id}' already exists.")

            conn.execute(
                """INSERT INTO accounts (user_id, balance_jc, total_earned, total_spent, created_at)
                   VALUES (?, ?, ?, 0.0, ?)""",
                (user_id, WELCOME_BONUS, WELCOME_BONUS, now),
            )
            tx_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO transactions
                       (tx_id, from_user, to_user, amount_jc, tx_type, description, compute_proof, timestamp)
                   VALUES (?, NULL, ?, ?, 'welcome_bonus', 'Welcome bonus', NULL, ?)""",
                (tx_id, user_id, WELCOME_BONUS, now),
            )
            conn.commit()
            return WELCOME_BONUS
        finally:
            conn.close()

    def get_balance(self, user_id: str) -> float:
        """Return the current JC balance for *user_id*."""
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT balance_jc FROM accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Account '{user_id}' not found.")
            return float(row["balance_jc"])
        finally:
            conn.close()

    def earn_jc(
        self,
        user_id: str,
        amount: float,
        compute_proof: str,
        description: str,
    ) -> None:
        """
        Credit *amount* JC to *user_id* for contributing compute.

        *compute_proof* should be a hash of the work performed.
        """
        if amount <= 0:
            raise ValueError("Earn amount must be positive.")
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT balance_jc, total_earned FROM accounts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Account '{user_id}' not found.")

            new_balance = float(row["balance_jc"]) + amount
            new_earned = float(row["total_earned"]) + amount
            conn.execute(
                "UPDATE accounts SET balance_jc = ?, total_earned = ? WHERE user_id = ?",
                (new_balance, new_earned, user_id),
            )
            tx_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO transactions
                       (tx_id, from_user, to_user, amount_jc, tx_type, description, compute_proof, timestamp)
                   VALUES (?, NULL, ?, ?, 'earn', ?, ?, ?)""",
                (tx_id, user_id, amount, description, compute_proof, now),
            )
            conn.commit()
        finally:
            conn.close()

    def spend_jc(self, user_id: str, amount: float, description: str) -> bool:
        """
        Deduct *amount* JC from *user_id*.

        Returns False if the balance is insufficient, True on success.
        """
        if amount <= 0:
            raise ValueError("Spend amount must be positive.")
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT balance_jc, total_spent FROM accounts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Account '{user_id}' not found.")

            current = float(row["balance_jc"])
            if current < amount:
                return False

            conn.execute(
                "UPDATE accounts SET balance_jc = ?, total_spent = ? WHERE user_id = ?",
                (current - amount, float(row["total_spent"]) + amount, user_id),
            )
            tx_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO transactions
                       (tx_id, from_user, to_user, amount_jc, tx_type, description, compute_proof, timestamp)
                   VALUES (?, ?, NULL, ?, 'spend', ?, NULL, ?)""",
                (tx_id, user_id, amount, description, now),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def transfer(
        self,
        from_user: str,
        to_user: str,
        amount: float,
        description: str,
    ) -> bool:
        """
        Transfer *amount* JC from *from_user* to *to_user*.

        Returns False if *from_user* has insufficient balance.
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive.")
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            sender = conn.execute(
                "SELECT balance_jc, total_spent FROM accounts WHERE user_id = ?",
                (from_user,),
            ).fetchone()
            if sender is None:
                raise ValueError(f"Account '{from_user}' not found.")

            receiver = conn.execute(
                "SELECT balance_jc, total_earned FROM accounts WHERE user_id = ?",
                (to_user,),
            ).fetchone()
            if receiver is None:
                raise ValueError(f"Account '{to_user}' not found.")

            if float(sender["balance_jc"]) < amount:
                return False

            conn.execute(
                "UPDATE accounts SET balance_jc = ?, total_spent = ? WHERE user_id = ?",
                (
                    float(sender["balance_jc"]) - amount,
                    float(sender["total_spent"]) + amount,
                    from_user,
                ),
            )
            conn.execute(
                "UPDATE accounts SET balance_jc = ?, total_earned = ? WHERE user_id = ?",
                (
                    float(receiver["balance_jc"]) + amount,
                    float(receiver["total_earned"]) + amount,
                    to_user,
                ),
            )
            tx_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO transactions
                       (tx_id, from_user, to_user, amount_jc, tx_type, description, compute_proof, timestamp)
                   VALUES (?, ?, ?, ?, 'transfer', ?, NULL, ?)""",
                (tx_id, from_user, to_user, amount, description, now),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_history(self, user_id: str, limit: int = 20) -> List[dict]:
        """Return the last *limit* transactions involving *user_id*."""
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM transactions
                   WHERE from_user = ? OR to_user = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (user_id, user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_network_stats(self) -> dict:
        """Return aggregate statistics for the entire JC network."""
        conn = _get_conn()
        try:
            total_jc = conn.execute(
                "SELECT COALESCE(SUM(balance_jc), 0.0) AS s FROM accounts"
            ).fetchone()["s"]
            total_accounts = conn.execute(
                "SELECT COUNT(*) AS c FROM accounts"
            ).fetchone()["c"]
            total_transactions = conn.execute(
                "SELECT COUNT(*) AS c FROM transactions"
            ).fetchone()["c"]
            total_compute = conn.execute(
                """SELECT COALESCE(SUM(amount_jc), 0.0) AS s
                   FROM transactions WHERE tx_type = 'earn'"""
            ).fetchone()["s"]
            return {
                "total_jc_in_circulation": float(total_jc),
                "total_accounts": int(total_accounts),
                "total_transactions": int(total_transactions),
                "total_compute_contributed": float(total_compute),
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# ComputeProofValidator
# ---------------------------------------------------------------------------

class ComputeProofValidator:
    """Generates and validates SHA-256 compute proofs for JC earning."""

    def generate_proof(
        self,
        user_id: str,
        task_type: str,
        compute_units: float,
    ) -> str:
        """
        Create a SHA-256 proof hash from user_id, task_type, compute_units,
        and the current UTC timestamp.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        raw = f"{user_id}:{task_type}:{compute_units}:{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def validate_proof(self, proof: str, user_id: str, task_type: str) -> bool:
        """
        Validate that *proof* is a well-formed 64-character hex string.

        Full cryptographic re-verification is not possible without the
        original timestamp, so this checks structural validity only.
        """
        if not isinstance(proof, str):
            return False
        if len(proof) != 64:
            return False
        try:
            int(proof, 16)
        except ValueError:
            return False
        return True

    def compute_to_jc(self, compute_units: float, task_type: str) -> float:
        """
        Convert *compute_units* to JC based on *task_type*.

        Rates: inference=0.1 JC/unit, training=0.5 JC/unit, storage=0.01 JC/unit.
        Unknown task types default to the inference rate.
        """
        rate = COMPUTE_RATES.get(task_type, COMPUTE_RATES["inference"])
        return round(compute_units * rate, 6)


# ---------------------------------------------------------------------------
# JCTaskMarket
# ---------------------------------------------------------------------------

class JCTaskMarket:
    """Peer-to-peer task market backed by JC escrow."""

    def __init__(self, ledger: JCLedger) -> None:
        """Initialise the market with a shared *ledger* instance."""
        self.ledger = ledger

    def post_task(
        self,
        requester_id: str,
        task_description: str,
        jc_reward: float,
        task_type: str,
    ) -> str:
        """
        Post a new task and hold *jc_reward* JC in escrow.

        Returns the new task_id. Raises ValueError if the requester has
        insufficient balance.
        """
        if jc_reward <= 0:
            raise ValueError("JC reward must be positive.")

        ok = self.ledger.spend_jc(
            requester_id,
            jc_reward,
            f"Escrow for task: {task_description[:80]}",
        )
        if not ok:
            raise ValueError(
                f"Insufficient balance to post task (need {jc_reward} JC)."
            )

        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO tasks
                       (task_id, requester_id, worker_id, description, jc_reward,
                        task_type, status, result, created_at, completed_at)
                   VALUES (?, ?, NULL, ?, ?, ?, 'open', NULL, ?, NULL)""",
                (task_id, requester_id, task_description, jc_reward, task_type, now),
            )
            conn.commit()
        finally:
            conn.close()

        return task_id

    def claim_task(self, worker_id: str, task_id: str) -> bool:
        """
        Assign *task_id* to *worker_id*.

        Returns False if the task does not exist or is not open.
        """
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] != "open":
                return False

            conn.execute(
                "UPDATE tasks SET worker_id = ?, status = 'claimed' WHERE task_id = ?",
                (worker_id, task_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def complete_task(self, task_id: str, result: str) -> bool:
        """
        Mark *task_id* as completed, store *result*, and release escrow to worker.

        Returns False if the task is not in 'claimed' status.
        """
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT status, worker_id, jc_reward, description FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None or row["status"] != "claimed":
                return False

            worker_id = row["worker_id"]
            jc_reward = float(row["jc_reward"])
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                "UPDATE tasks SET status = 'completed', result = ?, completed_at = ? WHERE task_id = ?",
                (result, now, task_id),
            )
            conn.commit()
        finally:
            conn.close()

        # Release escrow: credit the worker outside the tasks transaction
        self.ledger.earn_jc(
            worker_id,
            jc_reward,
            compute_proof="escrow_release",
            description=f"Task reward for task_id={task_id}",
        )
        return True

    def get_open_tasks(self) -> List[dict]:
        """Return all tasks with status 'open'."""
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = 'open' ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_task_status(self, task_id: str) -> dict:
        """Return the full record for *task_id*, or an empty dict if not found."""
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class EarnRequest(BaseModel):
    """Request body for POST /jc/earn."""
    user_id: str
    compute_units: float
    task_type: str


class SpendRequest(BaseModel):
    """Request body for POST /jc/spend."""
    user_id: str
    amount: float
    description: str


class PostTaskRequest(BaseModel):
    """Request body for POST /jc/tasks."""
    requester_id: str
    task_description: str
    jc_reward: float
    task_type: str


class CompleteTaskRequest(BaseModel):
    """Request body for POST /jc/tasks/{task_id}/complete."""
    result: str


class ClaimTaskRequest(BaseModel):
    """Request body for POST /jc/tasks/{task_id}/claim."""
    worker_id: str


# ---------------------------------------------------------------------------
# JCFastAPI
# ---------------------------------------------------------------------------

ledger = JCLedger()
validator = ComputeProofValidator()
market = JCTaskMarket(ledger)

app = FastAPI(title="Janus Credits API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/jc/balance/{user_id}", summary="Get JC balance")
def get_balance(user_id: str) -> dict:
    """Return the current JC balance for *user_id*, creating the account if needed."""
    try:
        balance = ledger.get_balance(user_id)
    except ValueError:
        # Auto-create account on first access
        balance = ledger.create_account(user_id)
    return {"user_id": user_id, "balance_jc": balance}


@app.post("/jc/earn", summary="Earn JC for compute contribution")
def earn_jc(req: EarnRequest) -> dict:
    """Earn JC by contributing compute units of the specified task type."""
    try:
        # Ensure account exists
        try:
            ledger.get_balance(req.user_id)
        except ValueError:
            ledger.create_account(req.user_id)

        proof = validator.generate_proof(req.user_id, req.task_type, req.compute_units)
        amount = validator.compute_to_jc(req.compute_units, req.task_type)
        ledger.earn_jc(
            req.user_id,
            amount,
            compute_proof=proof,
            description=f"Compute contribution: {req.task_type} x {req.compute_units} units",
        )
        return {
            "user_id": req.user_id,
            "earned_jc": amount,
            "compute_proof": proof,
            "new_balance": ledger.get_balance(req.user_id),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jc/spend", summary="Spend JC")
def spend_jc(req: SpendRequest) -> dict:
    """Deduct JC from the user's account for a described purpose."""
    try:
        ok = ledger.spend_jc(req.user_id, req.amount, req.description)
        if not ok:
            raise HTTPException(status_code=402, detail="Insufficient JC balance.")
        return {
            "user_id": req.user_id,
            "spent_jc": req.amount,
            "new_balance": ledger.get_balance(req.user_id),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jc/history/{user_id}", summary="Transaction history")
def get_history(user_id: str, limit: int = 20) -> dict:
    """Return the last *limit* transactions for *user_id*."""
    try:
        history = ledger.get_history(user_id, limit)
        return {"user_id": user_id, "transactions": history}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jc/network", summary="Network statistics")
def network_stats() -> dict:
    """Return aggregate statistics for the entire JC network."""
    try:
        return ledger.get_network_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/jc/tasks", summary="Post a new task")
def post_task(req: PostTaskRequest) -> dict:
    """Post a task to the market; JC reward is held in escrow."""
    try:
        task_id = market.post_task(
            req.requester_id,
            req.task_description,
            req.jc_reward,
            req.task_type,
        )
        return {"task_id": task_id, "status": "open"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jc/tasks", summary="List open tasks")
def list_open_tasks() -> dict:
    """Return all currently open tasks."""
    try:
        tasks = market.get_open_tasks()
        return {"tasks": tasks}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/jc/tasks/{task_id}/claim", summary="Claim a task")
def claim_task(task_id: str, req: ClaimTaskRequest) -> dict:
    """Assign the task to the requesting worker."""
    try:
        ok = market.claim_task(req.worker_id, task_id)
        if not ok:
            raise HTTPException(
                status_code=404, detail="Task not found or not available."
            )
        return {"task_id": task_id, "worker_id": req.worker_id, "status": "claimed"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jc/tasks/{task_id}/complete", summary="Complete a task")
def complete_task(task_id: str, req: CompleteTaskRequest) -> dict:
    """Mark the task as completed and release the JC reward to the worker."""
    try:
        ok = market.complete_task(task_id, req.result)
        if not ok:
            raise HTTPException(
                status_code=404, detail="Task not found or not in claimed state."
            )
        return {"task_id": task_id, "status": "completed"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("janus_credits:app", host="0.0.0.0", port=8004, reload=False)
