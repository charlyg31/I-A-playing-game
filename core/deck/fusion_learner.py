
class FusionLearner:
    def __init__(self):
        self.successful_fusions = {}  # (card1, card2): result
        self.failed_fusions = set()   # (card1, card2)

    def observe_fusion_attempt(self, card1, card2, result_card=None):
        pair = tuple(sorted([card1, card2]))
        if result_card:
            self.successful_fusions[pair] = result_card
        else:
            self.failed_fusions.add(pair)

    def has_seen(self, card1, card2):
        pair = tuple(sorted([card1, card2]))
        return pair in self.successful_fusions or pair in self.failed_fusions

    def get_result(self, card1, card2):
        pair = tuple(sorted([card1, card2]))
        return self.successful_fusions.get(pair)

    def is_known_failure(self, card1, card2):
        pair = tuple(sorted([card1, card2]))
        return pair in self.failed_fusions

    def suggest_fusions(self, hand_cards):
        suggestions = []
        for i in range(len(hand_cards)):
            for j in range(i+1, len(hand_cards)):
                pair = tuple(sorted([hand_cards[i], hand_cards[j]]))
                if pair in self.successful_fusions:
                    suggestions.append((pair, self.successful_fusions[pair]))
        return suggestions


import json, os

def save_memory(self, filepath="fusionlearner_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"successful_fusions": self.successful_fusions}, f, indent=2)

def load_memory(self, filepath="fusionlearner_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.successful_fusions = data.get("successful_fusions", self.successful_fusions)
