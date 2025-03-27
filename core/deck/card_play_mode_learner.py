
class CardPlayModeLearner:
    def __init__(self):
        self.observations = {}  # card_name: {'face_up': [effects], 'face_down': [effects]}

    def observe_play(self, card_name, mode, effect_observed):
        if card_name not in self.observations:
            self.observations[card_name] = {'face_up': [], 'face_down': []}
        self.observations[card_name][mode].append(effect_observed)

    def get_mode_profile(self, card_name):
        data = self.observations.get(card_name, {'face_up': [], 'face_down': []})
        return {
            "face_up_effect_rate": self._success_rate(data["face_up"]),
            "face_down_effect_rate": self._success_rate(data["face_down"]),
            "recommended": self._recommend_mode(data)
        }

    def _success_rate(self, effects):
        if not effects:
            return 0.0
        return sum(1 for e in effects if e) / len(effects)

    def _recommend_mode(self, data):
        up_rate = self._success_rate(data["face_up"])
        down_rate = self._success_rate(data["face_down"])
        if up_rate > down_rate:
            return "face_up"
        elif down_rate > up_rate:
            return "face_down"
        return "unclear"


import json, os

def save_memory(self, filepath="cardplaymodelearner_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"observations": self.observations}, f, indent=2)

def load_memory(self, filepath="cardplaymodelearner_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.observations = data.get("observations", self.observations)
