class VictoryPredictor:
    def __init__(self):
        self.history = []

    def log_duel_result(self, state, result):
        self.history.append((state, result))

    def estimate_chance(self, current_state):
        if not self.history:
            return 0.5  # 50% par défaut
        similar = [r for s, r in self.history if s == current_state]
        if not similar:
            return 0.5
        return similar.count("win") / len(similar)
