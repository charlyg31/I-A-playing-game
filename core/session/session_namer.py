import random

class SessionNamer:
    def __init__(self):
        self.themes = ["Deck", "AI", "Mind", "Nova", "Strat", "Shadow", "Yami", "Core", "Bot"]

    def generate_name(self):
        return random.choice(self.themes) + random.choice(self.themes)