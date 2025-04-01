
class FusionLearner:
    def __init__(self):
        self.known_fusions = {}

    def observe_fusion(self, card1, card2, result):
        key = tuple(sorted([card1, card2]))
        self.known_fusions[key] = result

    def predict_fusion(self, card1, card2):
        key = tuple(sorted([card1, card2]))
        return self.known_fusions.get(key)

    def has_fusion(self, card1, card2):
        key = tuple(sorted([card1, card2]))
        return key in self.known_fusions