import random

class DeckMutator:
    def __init__(self, all_cards):
        self.all_cards = all_cards

    def mutate_deck(self, base_deck, mutation_rate=0.1):
        new_deck = base_deck[:]
        for i in range(len(new_deck)):
            if random.random() < mutation_rate:
                new_deck[i] = random.choice(self.all_cards)
        return new_deck
