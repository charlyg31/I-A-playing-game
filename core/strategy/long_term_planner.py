
import json
import os

class LongTermPlanner:
    def __init__(self, storage_path="long_term_memory.json"):
        self.storage_path = storage_path
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def record_outcome(self, context, action, result):
        key = f"{context}:{action}"
        if key not in self.memory:
            self.memory[key] = {"tries": 0, "successes": 0}
        self.memory[key]["tries"] += 1
        if result:
            self.memory[key]["successes"] += 1
        self.save_memory()

    def get_effectiveness(self, context, action):
        key = f"{context}:{action}"
        data = self.memory.get(key, {"tries": 0, "successes": 0})
        if data["tries"] == 0:
            return 0.0
        return data["successes"] / data["tries"]

    def suggest_best_action(self, context, actions):
        scored = [(action, self.get_effectiveness(context, action)) for action in actions]
        sorted_actions = sorted(scored, key=lambda x: -x[1])
        return sorted_actions[0][0] if sorted_actions else None
