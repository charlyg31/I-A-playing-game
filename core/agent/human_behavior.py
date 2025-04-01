
import time
import random
import numpy as np

class HumanLikeBehavior:
    def __init__(self):
        self.action_memory = []

    def pause_for_observation(self):
        time.sleep(random.uniform(0.1, 0.3))

    def decision_delay(self):
        return random.uniform(0.2, 0.5)

    def compute_visual_change(self, features):
        return features.get("change_score", random.uniform(0.0, 1.0))

    def novelty_score(self, features):
        return features.get("novelty", random.uniform(0.0, 1.0))

    def compute_reward(self, delta_image, novelty):
        return (0.7 * delta_image + 0.3 * novelty)

    def update_action_memory(self, button_id):
        self.action_memory.append(button_id)
        if len(self.action_memory) > 50:
            self.action_memory.pop(0)