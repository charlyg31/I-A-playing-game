import json
from datetime import datetime

class DeckMemory:
    def __init__(self, file_path="memory/decks.json"):
        self.file_path = file_path
        self.memory = []
        self.load()

    def load(self):
        try:
            with open(self.file_path, "r") as f:
                self.memory = json.load(f)
        except:
            self.memory = []

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def log_result(self, deck, result):
        self.memory.append({
            "deck": deck,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_best_decks(self):
        wins = [d for d in self.memory if d["result"] == "win"]
        return sorted(wins, key=lambda x: x["deck"])
