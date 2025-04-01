
import random

class IntelligentDeckManager:
    def __init__(self):
        self.deck = []
        self.card_stats = {}

    def build_initial_deck(self, base_cards, size=40):
        self.deck = random.choices(base_cards, k=size)
        for card in self.deck:
            self.card_stats.setdefault(card, {'wins': 0, 'losses': 0})
        return self.deck

    def update_deck_feedback(self, used_cards, won):
        for card in used_cards:
            if card not in self.card_stats:
                self.card_stats[card] = {'wins': 0, 'losses': 0}
            if won:
                self.card_stats[card]['wins'] += 1
            else:
                self.card_stats[card]['losses'] += 1

    def get_deck(self):
        return self.deck

    def optimize_deck(self):
        for i, card in enumerate(self.deck):
            stats = self.card_stats.get(card, {'wins': 0, 'losses': 0})
            total = stats['wins'] + stats['losses']
            if total > 3 and stats['losses'] > stats['wins']:
                alternative = random.choice(list(self.card_stats.keys()))
                self.deck[i] = alternative

    def get_card_stats(self):
        return self.card_stats