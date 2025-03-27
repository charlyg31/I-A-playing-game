from collections import defaultdict

class ComboLearner:
    def __init__(self):
        self.combo_success = defaultdict(int)
        self.combo_uses = defaultdict(int)

    def record_combo(self, combo, success):
        key = tuple(combo)
        self.combo_uses[key] += 1
        if success:
            self.combo_success[key] += 1

    def get_best_combos(self):
        results = []
        for combo in self.combo_uses:
            success = self.combo_success[combo]
            total = self.combo_uses[combo]
            ratio = success / total if total else 0
            results.append((combo, ratio, total))
        return sorted(results, key=lambda x: x[1], reverse=True)
