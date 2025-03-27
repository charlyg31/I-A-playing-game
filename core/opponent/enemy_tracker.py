class EnemyTracker:
    def __init__(self):
        self.card_appearance = {}

    def observe_enemy_card(self, card_name):
        if card_name not in self.card_appearance:
            self.card_appearance[card_name] = 1
        else:
            self.card_appearance[card_name] += 1

    def get_frequent_threats(self, threshold=2):
        return [card for card, count in self.card_appearance.items() if count >= threshold]
