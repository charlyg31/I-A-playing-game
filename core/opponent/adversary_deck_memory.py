from collections import defaultdict

class AdversaryDeckMemory:
    def __init__(self):
        self.opponents = defaultdict(lambda: {"cards": set(), "style": None})

    def log_card_for_opponent(self, opponent_name, card_name):
        self.opponents[opponent_name]["cards"].add(card_name)

    def get_known_deck(self, opponent_name):
        return list(self.opponents[opponent_name]["cards"])

    def estimate_style(self, opponent_name):
        cards = self.opponents[opponent_name]["cards"]
        if not cards:
            return "inconnu"
        attack_keywords = ["dragon", "flame", "thunder", "blade", "red"]
        defense_keywords = ["shield", "wall", "elf", "mist", "mirror"]
        atk = sum(1 for c in cards if any(k in c.lower() for k in attack_keywords))
        defn = sum(1 for c in cards if any(k in c.lower() for k in defense_keywords))
        if atk > defn:
            return "agressif"
        elif defn > atk:
            return "défensif"
        return "mixte"
