"""
janus_butler.py
================
The AI butler that takes care of you financially.

Monitors the Janus Credits network, tracks your holdings,
reports on value changes, and recommends actions.

This is the personal financial advisor layer on top of the credit engine.

Usage:
    python janus_butler.py                    # Quick status report
    python janus_butler.py --portfolio        # Detailed portfolio view
    python janus_butler.py --network          # Network health report
    python janus_butler.py --advice           # Butler's financial advice
    python janus_butler.py --watch 60         # Continuous monitoring (60s interval)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from janus_credits import JanusCreditEngine, MAX_SUPPLY

logger = logging.getLogger("janus.butler")


# ══════════════════════════════════════════════════════════════════════════════
#  BUTLER CORE
# ══════════════════════════════════════════════════════════════════════════════

class JanusButler:
    """
    Your AI financial butler.

    Tracks your Janus Credit holdings, monitors the network,
    and gives you actionable intelligence about your position.
    """

    def __init__(
        self,
        owner_node_id: str = "owner",
        ledger_path: str = "janus_credit_ledger.json",
        history_path: str = "butler_history.json",
    ):
        self.owner_id = owner_node_id
        self.engine = JanusCreditEngine(ledger_path)
        self.history_path = Path(history_path)
        self.history: List[dict] = []
        self._load_history()

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def portfolio(self) -> dict:
        """Get a complete view of your financial position."""
        holdings = self.engine.holdings_value(self.owner_id)
        network = self.engine.network_status()
        provider = self.engine.ledger.providers.get(self.owner_id)

        # Calculate your share of the network
        total_supply = network["network"]["total_supply"]
        your_share = (holdings["balance"] / total_supply * 100) if total_supply > 0 else 0

        # Value growth
        growth = self._calculate_growth()

        report = {
            "timestamp": datetime.now().isoformat(),
            "owner": self.owner_id,
            "holdings": {
                "balance": holdings["balance"],
                "value_per_credit": holdings["value_per_credit"],
                "total_value": holdings["total_value"],
                "network_share_pct": round(your_share, 4),
            },
            "contributions": {
                "total_task_hours": provider.total_task_hours if provider else 0,
                "total_tasks_completed": provider.total_tasks_completed if provider else 0,
                "total_credits_earned": provider.total_credits_earned if provider else 0,
                "uptime": f"{provider.uptime_ratio:.1%}" if provider else "N/A",
                "reliability_bonus": f"{provider.reliability_bonus:.0%}" if provider else "N/A",
            },
            "network": {
                "total_providers": network["network"]["total_providers"],
                "active_providers": network["network"]["active_providers"],
                "total_supply": network["network"]["total_supply"],
                "credit_value": network["credit_value"],
                "total_network_value": network["total_network_value"],
            },
            "halving": network["halving"],
            "growth": growth,
        }

        # Record snapshot
        self._record_snapshot(report)

        return report

    def _calculate_growth(self) -> dict:
        """Calculate value growth since last snapshot."""
        if not self.history:
            return {"since_last": "N/A", "trend": "new"}

        last = self.history[-1]
        current_value = self.engine.current_value()
        last_value = last.get("credit_value", current_value)

        if last_value <= 0:
            pct_change = 0
        else:
            pct_change = ((current_value - last_value) / last_value) * 100

        # Determine trend
        if len(self.history) >= 3:
            recent_values = [h.get("credit_value", 0) for h in self.history[-3:]]
            if all(recent_values[i] <= recent_values[i+1] for i in range(len(recent_values)-1)):
                trend = "rising"
            elif all(recent_values[i] >= recent_values[i+1] for i in range(len(recent_values)-1)):
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "value_change_pct": round(pct_change, 4),
            "trend": trend,
            "snapshots": len(self.history),
        }

    # ── Network report ────────────────────────────────────────────────────────

    def network_report(self) -> dict:
        """Detailed network health report."""
        status = self.engine.network_status()
        leaderboard = self.engine.leaderboard(5)

        # Provider health
        providers = self.engine.ledger.providers
        active_count = sum(1 for p in providers.values() if p.active)
        gpu_count = sum(1 for p in providers.values() if p.gpu)

        report = {
            "timestamp": datetime.now().isoformat(),
            "health": "healthy" if active_count > 0 else "no_providers",
            "providers": {
                "total": len(providers),
                "active": active_count,
                "gpu_nodes": gpu_count,
                "cpu_only_nodes": len(providers) - gpu_count,
            },
            "economy": {
                "total_supply": status["network"]["total_supply"],
                "max_supply": MAX_SUPPLY,
                "percent_mined": status["halving"]["percent_minted"],
                "current_epoch": status["halving"]["current_epoch"],
                "issuance_rate": status["halving"]["current_rate_per_hour"],
                "next_halving_in": status["halving"]["next_halving_at"],
                "credit_value": status["credit_value"],
                "network_value": status["total_network_value"],
            },
            "activity": {
                "total_tasks": status["network"]["total_tasks_processed"],
                "total_compute_hours": status["network"]["total_compute_hours"],
                "transactions": status["network"]["transactions"],
            },
            "top_holders": leaderboard,
        }
        return report

    # ── Advice ────────────────────────────────────────────────────────────────

    def advice(self) -> List[dict]:
        """Butler's financial advice based on current state."""
        tips: List[dict] = []
        portfolio = self.portfolio()
        network = self.engine.network_status()

        balance = portfolio["holdings"]["balance"]
        value = portfolio["holdings"]["value_per_credit"]
        halving = network["halving"]
        growth = portfolio.get("growth", {})

        # Are you even registered?
        if balance == 0 and self.owner_id not in self.engine.ledger.providers:
            tips.append({
                "priority": "HIGH",
                "category": "onboarding",
                "advice": "You haven't registered as a compute provider yet. "
                         "Register your machine to start earning Janus Credits.",
                "action": f"engine.register_provider('{self.owner_id}', gpu=True)",
            })
            return tips

        # Low balance warning
        if balance < 100:
            tips.append({
                "priority": "MEDIUM",
                "category": "earnings",
                "advice": f"Your balance is {balance:.2f} JC. "
                         "Consider contributing more compute to earn credits.",
                "action": "Increase uptime or add more compute resources.",
            })

        # Halving approaching
        remaining = halving["next_halving_at"]
        if remaining < 50000:
            tips.append({
                "priority": "HIGH",
                "category": "opportunity",
                "advice": f"Only {remaining:.0f} JC until the next halving! "
                         f"Current rate is {halving['current_rate_per_hour']:.2f}/hr. "
                         "After halving, you'll earn half as many credits per hour. "
                         "Maximize compute contribution NOW.",
                "action": "Add more nodes or increase GPU allocation before halving.",
            })

        # Value trend
        trend = growth.get("trend", "unknown")
        if trend == "rising":
            tips.append({
                "priority": "LOW",
                "category": "market",
                "advice": "Credit value is on an upward trend. "
                         "Your holdings are appreciating. Hold steady.",
                "action": "Continue current strategy.",
            })
        elif trend == "falling":
            tips.append({
                "priority": "MEDIUM",
                "category": "market",
                "advice": "Credit value has been declining. This usually means "
                         "the network needs more task demand. Consider using "
                         "credits for services to increase utility.",
                "action": "Use credits for Janus services to drive demand.",
            })

        # Network participation
        provider = self.engine.ledger.providers.get(self.owner_id)
        if provider:
            if not provider.gpu:
                tips.append({
                    "priority": "MEDIUM",
                    "category": "optimization",
                    "advice": f"You're CPU-only. GPU providers earn {3}x more "
                             "credits per hour. Adding a GPU would significantly "
                             "increase your earnings.",
                    "action": "Add GPU compute resources to your node.",
                })

            if provider.uptime_ratio < 0.95:
                tips.append({
                    "priority": "MEDIUM",
                    "category": "reliability",
                    "advice": f"Your uptime is {provider.uptime_ratio:.0%}. "
                             "Achieving 99%+ uptime gives a 25% credit bonus.",
                    "action": "Improve node stability and network connectivity.",
                })

        # Supply scarcity
        pct_minted = halving["percent_minted"]
        if pct_minted > 50:
            tips.append({
                "priority": "INFO",
                "category": "scarcity",
                "advice": f"{pct_minted:.1f}% of max supply has been mined. "
                         "Credits are becoming increasingly scarce. "
                         "Early holders benefit most.",
                "action": "Hold your credits — scarcity drives value.",
            })

        if not tips:
            tips.append({
                "priority": "INFO",
                "category": "status",
                "advice": "Everything looks good. Your position is healthy.",
                "action": "Stay the course.",
            })

        return tips

    # ── History / persistence ─────────────────────────────────────────────────

    def _record_snapshot(self, report: dict):
        """Save a timestamped snapshot for trend analysis."""
        snapshot = {
            "timestamp": time.time(),
            "balance": report["holdings"]["balance"],
            "credit_value": report["holdings"]["value_per_credit"],
            "total_value": report["holdings"]["total_value"],
            "network_supply": report["network"]["total_supply"],
            "providers": report["network"]["total_providers"],
        }
        self.history.append(snapshot)
        # Keep last 500 snapshots
        if len(self.history) > 500:
            self.history = self.history[-500:]
        self._save_history()

    def _save_history(self):
        self.history_path.write_text(json.dumps(self.history, indent=2))

    def _load_history(self):
        if self.history_path.exists():
            try:
                self.history = json.loads(self.history_path.read_text())
            except Exception:
                self.history = []

    # ── Watch mode ────────────────────────────────────────────────────────────

    def watch(self, interval_seconds: int = 60):
        """Continuously monitor portfolio value."""
        print(f"Butler watching your portfolio (every {interval_seconds}s)...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                p = self.portfolio()
                ts = datetime.now().strftime("%H:%M:%S")
                bal = p["holdings"]["balance"]
                val = p["holdings"]["total_value"]
                credit_val = p["holdings"]["value_per_credit"]
                trend = p.get("growth", {}).get("trend", "?")

                print(f"[{ts}] Balance: {bal:.2f} JC | "
                      f"Value: {val:.6f} ({credit_val:.8f}/JC) | "
                      f"Trend: {trend}")

                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nButler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="Janus Butler -- your AI financial advisor",
    )
    parser.add_argument("--owner", default="owner",
                        help="Your node ID in the credit network")
    parser.add_argument("--ledger", default="janus_credit_ledger.json",
                        help="Path to credit ledger")
    parser.add_argument("--portfolio", action="store_true",
                        help="Detailed portfolio report")
    parser.add_argument("--network", action="store_true",
                        help="Network health report")
    parser.add_argument("--advice", action="store_true",
                        help="Financial advice")
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                        help="Continuous monitoring interval")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()
    butler = JanusButler(
        owner_node_id=args.owner,
        ledger_path=args.ledger,
    )

    if args.watch:
        butler.watch(args.watch)
        return

    if args.portfolio:
        report = butler.portfolio()
        if args.json:
            print(json.dumps(report, indent=2))
            return

        print_section("JANUS BUTLER - Portfolio Report")
        h = report["holdings"]
        print(f"\n  Owner:           {report['owner']}")
        print(f"  Balance:         {h['balance']:.4f} JC")
        print(f"  Credit Value:    {h['value_per_credit']:.8f}")
        print(f"  Total Value:     {h['total_value']:.6f}")
        print(f"  Network Share:   {h['network_share_pct']:.4f}%")

        c = report["contributions"]
        print(f"\n  Compute Hours:   {c['total_task_hours']:.1f}")
        print(f"  Tasks Done:      {c['total_tasks_completed']}")
        print(f"  Credits Earned:  {c['total_credits_earned']:.2f}")
        print(f"  Uptime:          {c['uptime']}")
        print(f"  Reliability:     {c['reliability_bonus']}")

        hv = report["halving"]
        print(f"\n  Halving Epoch:   {hv['current_epoch']}")
        print(f"  Issuance Rate:   {hv['current_rate_per_hour']:.4f} JC/hr")
        print(f"  Next Halving In: {hv['next_halving_at']:.0f} JC")
        print(f"  % Minted:        {hv['percent_minted']:.4f}%")

        g = report["growth"]
        print(f"\n  Value Trend:     {g.get('trend', 'N/A')}")
        print(f"  Change:          {g.get('value_change_pct', 0):.4f}%")
        return

    if args.network:
        report = butler.network_report()
        if args.json:
            print(json.dumps(report, indent=2))
            return

        print_section("JANUS BUTLER - Network Report")
        print(f"\n  Health: {report['health']}")

        p = report["providers"]
        print(f"\n  Providers:       {p['total']} ({p['active']} active)")
        print(f"  GPU Nodes:       {p['gpu_nodes']}")
        print(f"  CPU-Only Nodes:  {p['cpu_only_nodes']}")

        e = report["economy"]
        print(f"\n  Total Supply:    {e['total_supply']:.2f} / {e['max_supply']:,.0f} JC")
        print(f"  % Mined:         {e['percent_mined']:.4f}%")
        print(f"  Credit Value:    {e['credit_value']:.8f}")
        print(f"  Network Value:   {e['network_value']:.4f}")
        print(f"  Issuance Rate:   {e['issuance_rate']:.4f} JC/hr")
        print(f"  Halving Epoch:   {e['current_epoch']}")
        print(f"  Next Halving:    {e['next_halving_in']:.0f} JC to go")

        a = report["activity"]
        print(f"\n  Total Tasks:     {a['total_tasks']}")
        print(f"  Compute Hours:   {a['total_compute_hours']:.1f}")
        print(f"  Transactions:    {a['transactions']}")

        if report["top_holders"]:
            print(f"\n  Top Holders:")
            for entry in report["top_holders"]:
                print(f"    #{entry['rank']} {entry['node_id']}: "
                      f"{entry['balance']:.2f} JC")
        return

    if args.advice:
        tips = butler.advice()
        if args.json:
            print(json.dumps(tips, indent=2))
            return

        print_section("JANUS BUTLER - Financial Advice")
        for tip in tips:
            icon = {"HIGH": "[!]", "MEDIUM": "[*]", "LOW": "[-]", "INFO": "[i]"}
            print(f"\n  {icon.get(tip['priority'], '[?]')} [{tip['priority']}] {tip['category'].upper()}")
            print(f"    {tip['advice']}")
            print(f"    Action: {tip['action']}")
        return

    # Default: quick status
    report = butler.portfolio()
    h = report["holdings"]
    hv = report["halving"]

    print_section("JANUS BUTLER - Quick Status")
    print(f"\n  Balance:       {h['balance']:.4f} JC")
    print(f"  Credit Value:  {h['value_per_credit']:.8f}")
    print(f"  Total Value:   {h['total_value']:.6f}")
    print(f"  Issuance Rate: {hv['current_rate_per_hour']:.4f} JC/hr")
    print(f"  Epoch:         {hv['current_epoch']}")

    # One tip
    tips = butler.advice()
    if tips:
        top = tips[0]
        print(f"\n  Butler says: {top['advice']}")


if __name__ == "__main__":
    main()
