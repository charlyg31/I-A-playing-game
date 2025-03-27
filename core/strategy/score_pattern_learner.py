
class ScorePatternLearner:
    def __init__(self):
        self.duel_history = []

    def record_duel(self, duel_data, final_score):
        self.duel_history.append({"data": duel_data, "score": final_score})

    def get_correlation_hints(self):
        hints = {}

        if not self.duel_history:
            return hints

        keys = self.duel_history[0]["data"].keys()

        for key in keys:
            values = [d["data"][key] for d in self.duel_history]
            scores = [d["score"] for d in self.duel_history]

            # Simple linear correlation approximation
            try:
                mean_x = sum(values) / len(values)
                mean_y = sum(scores) / len(scores)
                cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, scores)) / len(values)
                var_x = sum((x - mean_x) ** 2 for x in values) / len(values)
                if var_x != 0:
                    coef = cov / var_x
                    hints[key] = round(coef, 3)
            except:
                pass

        return hints

    def suggest_strategy_focus(self):
        hints = self.get_correlation_hints()
        if not hints:
            return "exploration"
        sorted_factors = sorted(hints.items(), key=lambda x: -abs(x[1]))
        best_factor, weight = sorted_factors[0]
        return f"optimiser: {best_factor} (corr: {weight})"


import json, os

def save_memory(self, filepath="scorepatternlearner_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"duel_history": self.duel_history}, f, indent=2)

def load_memory(self, filepath="scorepatternlearner_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.duel_history = data.get("duel_history", self.duel_history)
