
class ThoughtGenerator:
    def __init__(self, game_state):
        self.game_state = game_state  # L'état actuel du jeu, utilisé pour générer des pensées

    def generate_thoughts(self):
        """ Génère des pensées ou décisions basées sur l'état actuel du jeu. """
        if 'menu' in self.game_state:
            return 'Deciding action in menu'
        elif 'duel' in self.game_state:
            return 'Deciding action in duel'
        return 'No context available for thought generation'

    def evaluate_decision(self, decision):
        """ Évalue si une décision est pertinente dans le contexte actuel. """
        if decision in ['move', 'attack', 'defend']:
            return True
        return False

    def update_thoughts(self, new_state):
        """ Met à jour les pensées avec un nouvel état de jeu. """
        self.game_state = new_state
