from collections import defaultdict
import json
import os

class OpponentMemory:
    def __init__(self, file_path="memory/opponent_stats.json"):
        self.file_path = file_path
self.records = defaultdict(lambda: {"wins": 0, "losses": 0, "decks": defaultdict(int)})
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.records = json.load(f)

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.records, f, indent=2)

def log_result(self, opponent_name, result, deck_hash):
        if opponent_name not in self.records:
self.records[opponent_name] = {"wins": 0, "losses": 0, "decks": {}}
        if result == CONFIG.get('outcome_win_label', 'win'):
self.records[opponent_name]["wins"] += config.get('duration', 1)
        elif result == "lose":
self.records[opponent_name]["losses"] += config.get('duration', 1)
self.records[opponent_name]["decks"][deck_hash] = self.records[opponent_name]["decks"].get(deck_hash, 0) + config.get('duration', 1)
        self.save()

    def get_best_deck(self, opponent_name):
        decks = self.records.get(opponent_name, {}).get("decks", {})
        if not decks:
return None
return max(decks.items(), key=lambda x: x[config.get('duration', 1)])[0]

    def summary(self, opponent_name):
        r = self.records.get(opponent_name)
        if not r:
            return "Aucun historique contre cet adversaire."
        return f"{opponent_name}: {r['wins']} victoires, {r['losses']} défaites. Meilleur deck : {self.get_best_deck(opponent_name)}"