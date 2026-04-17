"""
janus_credits.py
=================
Janus Credits — compute-backed value token system.

Users contribute compute (CPU/GPU cycles) to the Janus network.
In return, they receive Janus Credits that appreciate over time
as the network grows and compute becomes scarcer.

Like Bitcoin, but the "mining" is providing actual useful compute
for AI services rather than proof-of-work.

Key mechanics:
    - Credits issued for compute contributions (metered by task-hours)
    - Bitcoin-style halving: issuance rate halves every epoch
    - Value accrues through scarcity (decreasing issuance) + demand (growing usage)
    - All state persisted to JSON ledger on disk
    - No real money involved — pure value token

Usage:
    from janus_credits import JanusCreditEngine
    engine = JanusCreditEngine()

    # Register a compute provider
    provider = engine.register_provider("node-alpha", gpu=True, cores=8)

    # Record compute contribution
    contribution = engine.record_contribution("node-alpha", task_hours=2.5, tasks_completed=12)

    # Mint credits based on contribution
    minted = engine.mint_for_contribution(contribution)
    print(f"Earned {minted.amount:.2f} JC (value: ${minted.estimated_value:.4f} each)")

    # Check balance
    balance = engine.get_balance("node-alpha")
    print(f"Balance: {balance:.2f} JC")
"""

from __future__ import annotations

import json
import math
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("janus.credits")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Maximum total supply of Janus Credits that will ever exist
MAX_SUPPLY = 21_000_000.0

# Initial issuance rate: credits per compute-hour
INITIAL_RATE_PER_HOUR = 50.0

# Halving interval: after this many credits are minted, rate halves
HALVING_INTERVAL = 500_000.0

# Minimum issuance rate (stops halving below this)
MIN_ISSUANCE_RATE = 0.001

# GPU bonus multiplier (GPU compute earns more than CPU)
GPU_MULTIPLIER = 3.0

# Reliability bonus: extra % for high-uptime providers
RELIABILITY_TIERS = {
    0.99: 1.25,   # 99%+ uptime → 25% bonus
    0.95: 1.15,   # 95%+ uptime → 15% bonus
    0.90: 1.05,   # 90%+ uptime →  5% bonus
}

# Starting "value" of 1 JC in abstract value units
# This isn't dollars — it's an internal measure that grows with the network
INITIAL_VALUE = 0.001


# ══════════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComputeProvider:
    """A node contributing compute to the Janus network."""
    node_id: str
    display_name: str = ""
    gpu: bool = False
    cores: int = 1
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    total_task_hours: float = 0.0
    total_tasks_completed: int = 0
    total_credits_earned: float = 0.0
    uptime_samples: int = 0
    uptime_positive: int = 0
    active: bool = True

    @property
    def uptime_ratio(self) -> float:
        if self.uptime_samples == 0:
            return 0.0
        return self.uptime_positive / self.uptime_samples

    @property
    def reliability_bonus(self) -> float:
        ratio = self.uptime_ratio
        for threshold, bonus in sorted(RELIABILITY_TIERS.items(), reverse=True):
            if ratio >= threshold:
                return bonus
        return 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uptime_ratio"] = round(self.uptime_ratio, 4)
        d["reliability_bonus"] = self.reliability_bonus
        return d

    @staticmethod
    def from_dict(d: dict) -> "ComputeProvider":
        # Remove computed fields
        d.pop("uptime_ratio", None)
        d.pop("reliability_bonus", None)
        return ComputeProvider(**d)


@dataclass
class ComputeContribution:
    """A record of compute contributed by a provider."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    node_id: str = ""
    task_hours: float = 0.0
    tasks_completed: int = 0
    gpu_hours: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CreditMintEvent:
    """Record of credits being created and assigned."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    node_id: str = ""
    contribution_id: str = ""
    amount: float = 0.0
    issuance_rate: float = 0.0
    halving_epoch: int = 0
    total_supply_at_mint: float = 0.0
    estimated_value: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CreditTransaction:
    """A credit transfer or spend event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    from_id: str = ""        # "" = minted (new credits)
    to_id: str = ""
    amount: float = 0.0
    tx_type: str = "mint"    # mint, spend, transfer, burn
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  VALUE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ValueModel:
    """
    Tracks the value of Janus Credits over time.

    Value increases based on:
      1. Scarcity: as issuance rate drops (halving), new credits are harder to earn
      2. Network growth: more providers = more demand for credits
      3. Usage: credits being spent on services creates demand

    This is NOT a financial instrument — it's an internal value metric.
    """

    def __init__(self):
        self.snapshots: List[dict] = []

    def calculate_value(
        self,
        total_supply: float,
        total_providers: int,
        total_tasks_processed: int,
        current_epoch: int,
    ) -> float:
        """
        Calculate current credit value.

        Formula:
            value = base_value * scarcity_factor * network_factor * usage_factor

        scarcity_factor:  increases with each halving (2^epoch)
        network_factor:   sqrt of provider count (network effect)
        usage_factor:     log of total tasks (utility drives value)
        """
        # Scarcity: each halving doubles the value floor
        scarcity = 2.0 ** current_epoch

        # Network effect: value grows with sqrt of network size
        network = max(1.0, math.sqrt(total_providers))

        # Usage: logarithmic growth from actual utility
        usage = max(1.0, math.log2(max(1, total_tasks_processed)))

        # Supply pressure: value increases as we approach max supply
        supply_ratio = total_supply / MAX_SUPPLY if MAX_SUPPLY > 0 else 0
        supply_pressure = 1.0 + supply_ratio * 2.0  # up to 3x at max supply

        value = INITIAL_VALUE * scarcity * network * usage * supply_pressure

        # Record snapshot
        self.snapshots.append({
            "timestamp": time.time(),
            "value": round(value, 8),
            "total_supply": round(total_supply, 2),
            "providers": total_providers,
            "tasks": total_tasks_processed,
            "epoch": current_epoch,
            "scarcity": round(scarcity, 2),
            "network": round(network, 4),
            "usage": round(usage, 4),
        })

        # Keep only last 1000 snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]

        return value

    def value_history(self, n: int = 50) -> List[dict]:
        return self.snapshots[-n:]


# ══════════════════════════════════════════════════════════════════════════════
#  HALVING SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

class HalvingSchedule:
    """
    Bitcoin-style halving for credit issuance.

    Every HALVING_INTERVAL credits minted, the issuance rate halves.
    This ensures scarcity — early contributors earn more.
    """

    def current_rate(self, total_minted: float) -> float:
        """Get current issuance rate based on total credits ever minted."""
        epoch = self.current_epoch(total_minted)
        rate = INITIAL_RATE_PER_HOUR / (2.0 ** epoch)
        return max(rate, MIN_ISSUANCE_RATE)

    def current_epoch(self, total_minted: float) -> int:
        """Which halving epoch are we in?"""
        if total_minted <= 0:
            return 0
        return int(total_minted / HALVING_INTERVAL)

    def next_halving_at(self, total_minted: float) -> float:
        """How many more credits until the next halving?"""
        epoch = self.current_epoch(total_minted)
        next_threshold = (epoch + 1) * HALVING_INTERVAL
        return next_threshold - total_minted

    def schedule_summary(self, total_minted: float) -> dict:
        epoch = self.current_epoch(total_minted)
        rate = self.current_rate(total_minted)
        return {
            "current_epoch": epoch,
            "current_rate_per_hour": round(rate, 6),
            "initial_rate": INITIAL_RATE_PER_HOUR,
            "total_minted": round(total_minted, 2),
            "next_halving_at": round(self.next_halving_at(total_minted), 2),
            "next_halving_total": (epoch + 1) * HALVING_INTERVAL,
            "max_supply": MAX_SUPPLY,
            "percent_minted": round((total_minted / MAX_SUPPLY) * 100, 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CREDIT LEDGER
# ══════════════════════════════════════════════════════════════════════════════

class CreditLedger:
    """
    Persistent ledger tracking all credit balances and transactions.

    Every operation is recorded in an append-only transaction log.
    Balances are derived from the transaction history.
    """

    def __init__(self, ledger_path: str = "janus_credit_ledger.json"):
        self.path = Path(ledger_path)
        self.providers: Dict[str, ComputeProvider] = {}
        self.transactions: List[CreditTransaction] = []
        self.mint_events: List[CreditMintEvent] = []
        self.contributions: List[ComputeContribution] = []
        self.total_minted: float = 0.0
        self.total_burned: float = 0.0
        self._load()

    # ── Balance queries ───────────────────────────────────────────────────────

    def get_balance(self, node_id: str) -> float:
        """Calculate current balance from transaction history."""
        balance = 0.0
        for tx in self.transactions:
            if tx.to_id == node_id:
                balance += tx.amount
            if tx.from_id == node_id:
                balance -= tx.amount
        return round(balance, 6)

    def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero balances."""
        balances: Dict[str, float] = {}
        for tx in self.transactions:
            if tx.to_id:
                balances[tx.to_id] = balances.get(tx.to_id, 0.0) + tx.amount
            if tx.from_id:
                balances[tx.from_id] = balances.get(tx.from_id, 0.0) - tx.amount
        return {k: round(v, 6) for k, v in balances.items() if abs(v) > 0.000001}

    # ── Mutations ─────────────────────────────────────────────────────────────

    def record_mint(self, to_id: str, amount: float, mint_event: CreditMintEvent):
        """Record newly minted credits."""
        tx = CreditTransaction(
            from_id="",  # minted from nothing
            to_id=to_id,
            amount=amount,
            tx_type="mint",
            description=f"Minted for contribution {mint_event.contribution_id}",
        )
        self.transactions.append(tx)
        self.mint_events.append(mint_event)
        self.total_minted += amount
        self._save()

    def record_spend(self, from_id: str, amount: float, description: str) -> bool:
        """Spend credits on a service. Returns False if insufficient balance."""
        balance = self.get_balance(from_id)
        if balance < amount:
            return False

        tx = CreditTransaction(
            from_id=from_id,
            to_id="janus_treasury",
            amount=amount,
            tx_type="spend",
            description=description,
        )
        self.transactions.append(tx)
        self._save()
        return True

    def record_transfer(self, from_id: str, to_id: str, amount: float,
                        description: str = "") -> bool:
        """Transfer credits between users. Returns False if insufficient."""
        balance = self.get_balance(from_id)
        if balance < amount:
            return False

        tx = CreditTransaction(
            from_id=from_id,
            to_id=to_id,
            amount=amount,
            tx_type="transfer",
            description=description or f"Transfer {from_id} -> {to_id}",
        )
        self.transactions.append(tx)
        self._save()
        return True

    def record_contribution(self, contribution: ComputeContribution):
        """Log a compute contribution."""
        self.contributions.append(contribution)
        # Update provider stats
        provider = self.providers.get(contribution.node_id)
        if provider:
            provider.total_task_hours += contribution.task_hours
            provider.total_tasks_completed += contribution.tasks_completed
            provider.last_heartbeat = time.time()
        self._save()

    # ── Provider management ───────────────────────────────────────────────────

    def register_provider(self, node_id: str, display_name: str = "",
                          gpu: bool = False, cores: int = 1) -> ComputeProvider:
        """Register a new compute provider."""
        if node_id in self.providers:
            return self.providers[node_id]

        provider = ComputeProvider(
            node_id=node_id,
            display_name=display_name or node_id,
            gpu=gpu,
            cores=cores,
        )
        self.providers[node_id] = provider
        self._save()
        logger.info(f"Registered provider: {node_id} (GPU={gpu}, cores={cores})")
        return provider

    def heartbeat(self, node_id: str) -> bool:
        """Record a heartbeat from a provider (for uptime tracking)."""
        provider = self.providers.get(node_id)
        if not provider:
            return False
        provider.last_heartbeat = time.time()
        provider.uptime_samples += 1
        provider.uptime_positive += 1
        provider.active = True
        self._save()
        return True

    def mark_offline(self, node_id: str):
        """Mark a provider as offline (missed heartbeat)."""
        provider = self.providers.get(node_id)
        if provider:
            provider.uptime_samples += 1
            # uptime_positive NOT incremented
            provider.active = False
            self._save()

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_supply(self) -> float:
        """Current circulating supply."""
        return self.total_minted - self.total_burned

    @property
    def total_providers(self) -> int:
        return len(self.providers)

    @property
    def active_providers(self) -> int:
        return sum(1 for p in self.providers.values() if p.active)

    @property
    def total_tasks_processed(self) -> int:
        return sum(c.tasks_completed for c in self.contributions)

    @property
    def total_compute_hours(self) -> float:
        return sum(c.task_hours for c in self.contributions)

    def network_stats(self) -> dict:
        return {
            "total_providers": self.total_providers,
            "active_providers": self.active_providers,
            "total_supply": round(self.total_supply, 2),
            "total_minted": round(self.total_minted, 2),
            "total_tasks_processed": self.total_tasks_processed,
            "total_compute_hours": round(self.total_compute_hours, 2),
            "transactions": len(self.transactions),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        data = {
            "saved_at": datetime.now().isoformat(),
            "total_minted": self.total_minted,
            "total_burned": self.total_burned,
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "transactions": [tx.to_dict() for tx in self.transactions[-5000:]],
            "mint_events": [me.to_dict() for me in self.mint_events[-1000:]],
            "contributions": [c.to_dict() for c in self.contributions[-2000:]],
        }
        self.path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.total_minted = data.get("total_minted", 0.0)
            self.total_burned = data.get("total_burned", 0.0)

            for nid, pd in data.get("providers", {}).items():
                self.providers[nid] = ComputeProvider.from_dict(pd)

            for td in data.get("transactions", []):
                self.transactions.append(CreditTransaction(**td))

            for md in data.get("mint_events", []):
                self.mint_events.append(CreditMintEvent(**md))

            for cd in data.get("contributions", []):
                self.contributions.append(ComputeContribution(**cd))

            logger.info(
                f"Loaded ledger: {self.total_providers} providers, "
                f"{self.total_minted:.2f} JC minted, "
                f"{len(self.transactions)} transactions"
            )
        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  JANUS CREDIT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class JanusCreditEngine:
    """
    Top-level API for the Janus Credits system.

    Handles:
      - Provider registration
      - Compute contribution recording
      - Credit minting with halving schedule
      - Value tracking
      - Balance queries
      - Spending and transfers
    """

    def __init__(self, ledger_path: str = "janus_credit_ledger.json"):
        self.ledger = CreditLedger(ledger_path)
        self.halving = HalvingSchedule()
        self.value_model = ValueModel()

    # ── Provider management ───────────────────────────────────────────────────

    def register_provider(self, node_id: str, display_name: str = "",
                          gpu: bool = False, cores: int = 1) -> ComputeProvider:
        """Register a new compute provider on the network."""
        return self.ledger.register_provider(node_id, display_name, gpu, cores)

    def heartbeat(self, node_id: str) -> bool:
        """Record provider heartbeat."""
        return self.ledger.heartbeat(node_id)

    # ── Compute contribution & minting ────────────────────────────────────────

    def record_contribution(
        self,
        node_id: str,
        task_hours: float,
        tasks_completed: int = 0,
        gpu_hours: float = 0.0,
    ) -> ComputeContribution:
        """Record compute contributed by a provider."""
        contribution = ComputeContribution(
            node_id=node_id,
            task_hours=task_hours,
            tasks_completed=tasks_completed,
            gpu_hours=gpu_hours,
        )
        self.ledger.record_contribution(contribution)
        return contribution

    def mint_for_contribution(self, contribution: ComputeContribution) -> CreditMintEvent:
        """
        Mint new credits based on a compute contribution.

        Amount = (cpu_hours * rate) + (gpu_hours * rate * GPU_MULTIPLIER)
        Adjusted by halving schedule and reliability bonus.
        """
        # Check supply cap
        if self.ledger.total_minted >= MAX_SUPPLY:
            logger.warning("Max supply reached — no new credits can be minted")
            return CreditMintEvent(
                node_id=contribution.node_id,
                contribution_id=contribution.id,
                amount=0,
            )

        # Current issuance rate
        rate = self.halving.current_rate(self.ledger.total_minted)
        epoch = self.halving.current_epoch(self.ledger.total_minted)

        # Calculate base amount
        cpu_credits = contribution.task_hours * rate
        gpu_credits = contribution.gpu_hours * rate * GPU_MULTIPLIER
        base_amount = cpu_credits + gpu_credits

        # Apply reliability bonus
        provider = self.ledger.providers.get(contribution.node_id)
        if provider:
            base_amount *= provider.reliability_bonus

        # Cap at remaining supply
        remaining = MAX_SUPPLY - self.ledger.total_minted
        amount = min(base_amount, remaining)
        amount = round(amount, 6)

        # Calculate current value
        current_value = self.current_value()

        # Create mint event
        mint_event = CreditMintEvent(
            node_id=contribution.node_id,
            contribution_id=contribution.id,
            amount=amount,
            issuance_rate=rate,
            halving_epoch=epoch,
            total_supply_at_mint=self.ledger.total_supply,
            estimated_value=current_value,
        )

        # Record in ledger
        self.ledger.record_mint(contribution.node_id, amount, mint_event)

        # Update provider stats
        if provider:
            provider.total_credits_earned += amount

        logger.info(
            f"Minted {amount:.4f} JC for {contribution.node_id} "
            f"(rate={rate:.4f}/hr, epoch={epoch}, value={current_value:.6f})"
        )

        return mint_event

    # ── Spending & transfers ──────────────────────────────────────────────────

    def spend(self, node_id: str, amount: float, service_description: str) -> bool:
        """Spend credits on a Janus service."""
        return self.ledger.record_spend(node_id, amount, service_description)

    def transfer(self, from_id: str, to_id: str, amount: float,
                 description: str = "") -> bool:
        """Transfer credits between providers."""
        return self.ledger.record_transfer(from_id, to_id, amount, description)

    # ── Value & balance queries ───────────────────────────────────────────────

    def get_balance(self, node_id: str) -> float:
        """Get credit balance for a provider."""
        return self.ledger.get_balance(node_id)

    def current_value(self) -> float:
        """Get current value of 1 Janus Credit."""
        return self.value_model.calculate_value(
            total_supply=self.ledger.total_supply,
            total_providers=self.ledger.total_providers,
            total_tasks_processed=self.ledger.total_tasks_processed,
            current_epoch=self.halving.current_epoch(self.ledger.total_minted),
        )

    def holdings_value(self, node_id: str) -> dict:
        """Get the total value of a provider's holdings."""
        balance = self.get_balance(node_id)
        value_per_credit = self.current_value()
        return {
            "node_id": node_id,
            "balance": round(balance, 6),
            "value_per_credit": round(value_per_credit, 8),
            "total_value": round(balance * value_per_credit, 6),
            "timestamp": datetime.now().isoformat(),
        }

    # ── Network status ────────────────────────────────────────────────────────

    def network_status(self) -> dict:
        """Full network status report."""
        value = self.current_value()
        schedule = self.halving.schedule_summary(self.ledger.total_minted)
        stats = self.ledger.network_stats()

        return {
            "network": stats,
            "credit_value": round(value, 8),
            "total_network_value": round(self.ledger.total_supply * value, 4),
            "halving": schedule,
            "max_supply": MAX_SUPPLY,
        }

    def leaderboard(self, n: int = 10) -> List[dict]:
        """Top credit holders."""
        balances = self.ledger.get_all_balances()
        value = self.current_value()
        sorted_holders = sorted(balances.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "rank": i + 1,
                "node_id": nid,
                "balance": round(bal, 4),
                "value": round(bal * value, 6),
                "provider": self.ledger.providers.get(nid, ComputeProvider(nid)).to_dict()
                           if nid in self.ledger.providers else None,
            }
            for i, (nid, bal) in enumerate(sorted_holders[:n])
            if nid != "janus_treasury"
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """Run a simulation demonstrating the credit system."""
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  JANUS CREDITS - Compute-for-Value Demo")
    print("=" * 60)

    engine = JanusCreditEngine(ledger_path="janus_credit_demo_ledger.json")

    # Register providers
    print("\n[1] Registering compute providers...")
    providers = [
        ("node-alpha", True, 8),    # GPU node, 8 cores
        ("node-beta", False, 16),    # CPU-only, 16 cores
        ("node-gamma", True, 4),     # GPU node, 4 cores
        ("node-delta", False, 4),    # small CPU node
    ]
    for name, gpu, cores in providers:
        p = engine.register_provider(name, gpu=gpu, cores=cores)
        print(f"  Registered: {name} (GPU={gpu}, cores={cores})")

    # Simulate heartbeats for uptime
    print("\n[2] Simulating uptime heartbeats...")
    for _ in range(20):
        for name, _, _ in providers:
            engine.heartbeat(name)
    # node-delta misses some
    engine.ledger.mark_offline("node-delta")
    engine.ledger.mark_offline("node-delta")
    engine.ledger.mark_offline("node-delta")

    # Simulate compute contributions across multiple rounds
    print("\n[3] Simulating compute contributions (5 rounds)...")
    for round_num in range(1, 6):
        print(f"\n  -- Round {round_num} --")
        contributions = [
            ("node-alpha", 4.0, 20, 3.0),   # 4 task-hrs, 3 GPU-hrs
            ("node-beta",  6.0, 30, 0.0),    # 6 task-hrs, no GPU
            ("node-gamma", 2.0, 10, 1.5),    # 2 task-hrs, 1.5 GPU-hrs
            ("node-delta", 1.0, 5,  0.0),    # small contribution
        ]

        for name, hours, tasks, gpu_hrs in contributions:
            c = engine.record_contribution(name, hours, tasks, gpu_hrs)
            mint = engine.mint_for_contribution(c)
            print(f"    {name}: +{mint.amount:.2f} JC "
                  f"(rate={mint.issuance_rate:.2f}/hr, "
                  f"value={mint.estimated_value:.6f})")

    # Show balances
    print("\n[4] Current balances:")
    for name, _, _ in providers:
        h = engine.holdings_value(name)
        print(f"    {name}: {h['balance']:.2f} JC "
              f"(total value: {h['total_value']:.4f})")

    # Show network status
    print("\n[5] Network status:")
    status = engine.network_status()
    print(f"    Providers:       {status['network']['total_providers']}")
    print(f"    Active:          {status['network']['active_providers']}")
    print(f"    Total supply:    {status['network']['total_supply']:.2f} JC")
    print(f"    Credit value:    {status['credit_value']:.8f}")
    print(f"    Network value:   {status['total_network_value']:.4f}")
    print(f"    Halving epoch:   {status['halving']['current_epoch']}")
    print(f"    Current rate:    {status['halving']['current_rate_per_hour']:.4f} JC/hr")
    print(f"    Next halving in: {status['halving']['next_halving_at']:.2f} JC")
    print(f"    % minted:        {status['halving']['percent_minted']:.4f}%")

    # Show leaderboard
    print("\n[6] Leaderboard:")
    for entry in engine.leaderboard():
        print(f"    #{entry['rank']} {entry['node_id']}: "
              f"{entry['balance']:.2f} JC ({entry['value']:.6f} value)")

    # Simulate a spend
    print("\n[7] Spending credits...")
    if engine.spend("node-alpha", 50.0, "AI inference task"):
        new_bal = engine.get_balance("node-alpha")
        print(f"    node-alpha spent 50 JC on AI inference. "
              f"New balance: {new_bal:.2f} JC")

    # Simulate a transfer
    print("\n[8] Transferring credits...")
    if engine.transfer("node-beta", "node-delta", 100.0, "Helping out"):
        print(f"    node-beta -> node-delta: 100 JC")
        print(f"    node-beta balance: {engine.get_balance('node-beta'):.2f} JC")
        print(f"    node-delta balance: {engine.get_balance('node-delta'):.2f} JC")

    # Value history
    print("\n[9] Value trend (last 5 snapshots):")
    for snap in engine.value_model.value_history(5):
        print(f"    Supply={snap['total_supply']:.0f}, "
              f"Providers={snap['providers']}, "
              f"Value={snap['value']:.8f}")

    print("\n" + "=" * 60)
    print("  Demo complete. Ledger saved to janus_credit_demo_ledger.json")
    print("=" * 60)

    # Cleanup demo ledger
    cleanup = input("\nDelete demo ledger? [y/N] ").strip().lower()
    if cleanup == "y":
        Path("janus_credit_demo_ledger.json").unlink(missing_ok=True)
        print("Deleted.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        print("Janus Credits Engine")
        print("  --demo    Run interactive demo simulation")
        print("\nUsage as library:")
        print("  from janus_credits import JanusCreditEngine")
        print("  engine = JanusCreditEngine()")
        print("  engine.register_provider('my-node', gpu=True)")
