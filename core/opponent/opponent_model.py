
class OpponentModel:
    def __init__(self):
        self.deck_memory = set()
        self.actions = []
        self.strategy = None

    def observe_card_played(self, card_name):
        self.deck_memory.add(card_name)
        self.actions.append(f"played:{card_name}")
        self._infer_strategy()

    def observe_behavior(self, behavior):
        self.actions.append(f"behavior:{behavior}")
        self._infer_strategy()

    def _infer_strategy(self):
        if "played:Blue-Eyes White Dragon" in self.actions:
            self.strategy = "power"
        elif "played:Magic Cylinder" in self.actions:
            self.strategy = "defensive"
        elif len(self.deck_memory) > 15:
            self.strategy = "balanced"

    def predict_strategy(self):
        return self.strategy or "unknown"

    def get_deck_memory(self):
        return list(self.deck_memory)

    def reset(self):
        self.deck_memory.clear()
        self.actions.clear()
        self.strategy = None
