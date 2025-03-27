
import random

class DeckPlanner:
    def __init__(self):
        self.card_stats = {}  # nom_carte: {'played': 0, 'won': 0}
        self.current_deck = []

    def record_card_usage(self, card_name, won=False):
        if card_name not in self.card_stats:
            self.card_stats[card_name] = {'played': 0, 'won': 0}
        self.card_stats[card_name]['played'] += 1
        if won:
            self.card_stats[card_name]['won'] += 1

    def get_card_score(self, card_name):
        stats = self.card_stats.get(card_name, {'played': 0, 'won': 0})
        if stats['played'] == 0:
            return 0.0
        return stats['won'] / stats['played']

    def suggest_improvements(self, available_cards, max_deck_size=40):
        scored_cards = [(card, self.get_card_score(card)) for card in available_cards]
        sorted_cards = sorted(scored_cards, key=lambda x: -x[1])
        self.current_deck = [card for card, _ in sorted_cards[:max_deck_size]]
        return self.current_deck

    def get_current_deck(self):
        return self.current_deck

    def reset_stats(self):
        self.card_stats.clear()
