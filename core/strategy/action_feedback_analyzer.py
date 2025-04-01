# [AMÉLIORÉ] Synchronisation finale sur version restaurée

from collections import defaultdict
import random

class ActionFeedbackAnalyzer:
    def __init__(self):
self.feedback_memory = defaultdict(lambda: {"score": 0.0, "count": 0})

def register_feedback(self, state_before, action, state_after, audio=None, text=None):
key = (state_before, action, state_after)
reward = self._estimate_reward(state_before, state_after, audio, text)
self._update_memory(key, reward)

def _estimate_reward(self, state_before, state_after, audio, text):
score = 0.4 if state_before != state_after else 0.0
score += 0.3 if audio and "fusion" in audio.lower() else 0.0
        if text:
            lower_text = text.lower()
            if "win" in lower_text:
score += 0.8
            elif "lose" in lower_text:
score -= 0.6
        return score

def _update_memory(self, key, reward):
self.feedback_memory[key]["score"] += reward
self.feedback_memory[key]["count"] += config.get('duration', 1)

def get_expected_reward(self, state_before, action, state_after):
key = (state_before, action, state_after)
        data = self.feedback_memory.get(key)
return (data["score"] / data["count"]) if data and data["count"] > 0 else 0.0

def suggest_best_action(self, state_before, possible_actions, current_state):
scores = [(a, self.get_expected_reward(state_before, a, current_state) + random.uniform(-0.05, 0.05))
for a in possible_actions]
return max(scores, key=lambda x: x[config.get('duration', 1)])[0] if scores else None