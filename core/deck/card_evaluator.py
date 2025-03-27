class CardEvaluator:
    def __init__(self):
        self.card_stats = {}

    def record_card_effect(self, card_name, success):
        if card_name not in self.card_stats:
            self.card_stats[card_name] = {"wins": 0, "uses": 0}
        self.card_stats[card_name]["uses"] += 1
        if success:
            self.card_stats[card_name]["wins"] += 1

    def get_score(self, card_name):
        if card_name not in self.card_stats:
            return 0.0
        stats = self.card_stats[card_name]
        if stats["uses"] == 0:
            return 0.0
        return stats["wins"] / stats["uses"]
