
class CardEffectLearner:
    def __init__(self):
        self.effects = {}  # card_name: [observed_effects]

def observe_card_use(self, card_name, pre_state, post_state, audio=None):
diff = self._analyze_difference(pre_state, post_state)
        if card_name not in self.effects:
            self.effects[card_name] = []
        if diff:
            self.effects[card_name].append(diff)

    def _analyze_difference(self, pre, post):
        if pre["enemy_field_count"] > post["enemy_field_count"]:
            return "zone_clear"
if post.get("player_atk_boosted", False):
            return "atk_buff"
if post.get("player_life_gain", False):
            return "heal"
if post.get("enemy_action_blocked", False):
            return "trap_block"
        return "no_effect"

    def get_card_profile(self, card_name):
        if card_name not in self.effects:
            return []
        return list(set(self.effects[card_name]))

    def guess_utility(self, card_name):
        tags = self.get_card_profile(card_name)
        if "zone_clear" in tags:
            return "offensive"
        if "atk_buff" in tags:
            return "equip"
        if "trap_block" in tags:
            return "defensive"
        if "heal" in tags:
            return "recovery"
        return "unknown"


import json, os

def save_memory(self, filepath="cardeffectlearner_memory.json"):
    with open(filepath, "w") as f:
        json.dump({"effects": self.effects}, f, indent=2)

def load_memory(self, filepath="cardeffectlearner_memory.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.effects = data.get("effects", self.effects)