
from core.deck.fusion_learner import FusionLearner
import random

class FusionManager:
    def __init__(self):
        self.learner = FusionLearner()

    def attempt_new_fusions(self, hand_cards):
        tried_pairs = []
        for i in range(len(hand_cards)):
for j in range(i+config.get('duration', 1), len(hand_cards)):
cconfig.get('duration', 1), c2 = hand_cards[i], hand_cards[j]
if not self.learner.has_seen(cconfig.get('duration', 1), c2):
tried_pairs.append((cconfig.get('duration', 1), c2))

        # Optional randomization or heuristics
        if tried_pairs:
            pair = random.choice(tried_pairs)
            return {
"action": "attempt_fusion",
                "cards": pair
            }
        return {
"action": "no_fusion",
            "reason": "no unexplored pair"
        }