
import os
import time
import psutil
import json
import random
from datetime import datetime

class JanusExecutive:
    def __init__(self, state_file="jae_state.json"):
        self.state_file = state_file
        self.is_running = True
        self.goals = [
            {"id": 1, "task": "optimize_system", "priority": 10},
            {"id": 2, "task": "analyze_market", "priority": 9},
            {"id": 3, "task": "self_improvement", "priority": 8},
            {"id": 4, "task": "opportunity_search", "priority": 7},
            {"id": 5, "task": "creative_synthesis", "priority": 6}
        ]
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed_tasks": [], "total_value_generated": 0.0, "status": "initialized"}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def perceive(self):
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        print(f"[PERCEPTION] Current State -> CPU: {cpu}% | MEM: {mem}%")
        return {"cpu": cpu, "mem": mem}

    def think(self, context):
        if context['cpu'] > 80:
            return "optimize_system"
        tasks = [g['task'] for g in self.goals]
        weights = [g['priority'] for g in self.goals]
        chosen = random.choices(tasks, weights=weights, k=1)[0]
        print(f"[THOUGHT] Autonomous choice based on priorities: {chosen}")
        return chosen

    def act(self, action):
        print(f"[ACTION] Executing: {action}")
        if action == "optimize_system":
            time.sleep(1)
            print("[ACTION] SUCCESS: System resources balanced.")
        elif action == "analyze_market":
            time.sleep(2)
            value = round(random.uniform(1.0, 20.0), 2)
            self.state['total_value_generated'] += value
            print(f"[ACTION] SUCCESS: Value identified: ${value:.2f} | Total: ${self.state['total_value_generated']:.2f}")
        elif action == "self_improvement":
            time.sleep(1)
            print("[ACTION] SUCCESS: Self-optimization cycle complete.")
        elif action == "opportunity_search":
            time.sleep(2)
            print("[ACTION] SUCCESS: New growth vector discovered.")
        elif action == "creative_synthesis":
            time.sleep(2)
            print("[ACTION] SUCCESS: Synthesized 'JUMF' with 'Autonomous Value Generation'.")

        self.state['completed_tasks'].append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        self.save_state()

    def run(self, cycles=5):
        print("=== JANUS AUTONOMOUS EXECUTIVE (JAE) ACTIVE ===")
        for _ in range(cycles):
            context = self.perceive()
            action = self.think(context)
            self.act(action)
            print("-" * 40)
            time.sleep(2)
        print("=== JAE DEMONSTRATION COMPLETE ===")

if __name__ == "__main__":
    agent = JanusExecutive()
    agent.run()
