
class FusionResultEvaluator:
    def __init__(self):
        self.result_stats = {}  # result_card: {"atk": avg, "def": avg, "success": count}

    def observe_fusion_result(self, result_card, atk, defense):
        if result_card not in self.result_stats:
            self.result_stats[result_card] = {"atk": 0, "def": 0, "success": 0}

        stats = self.result_stats[result_card]
        stats["atk"] = (stats["atk"] * stats["success"] + atk) / (stats["success"] + 1)
        stats["def"] = (stats["def"] * stats["success"] + defense) / (stats["success"] + 1)
        stats["success"] += 1

    def evaluate_card(self, card_name):
        stats = self.result_stats.get(card_name)
        if not stats:
            return "unknown"
        if stats["atk"] >= 2000 or stats["def"] >= 2000:
            return "strong"
        if stats["atk"] >= 1500:
            return "moderate"
        return "weak"

    def get_stats(self, card_name):
        return self.result_stats.get(card_name, "unknown")


import json, os

def save_memory(self, filepath="fusionresultevaluator_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"result_stats": self.result_stats}, f, indent=2)

def load_memory(self, filepath="fusionresultevaluator_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.result_stats = data.get("result_stats", self.result_stats)
