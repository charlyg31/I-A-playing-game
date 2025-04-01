
class ContextAwareness:
    def __init__(self, game_state):
        self.game_state = game_state  # L'état actuel du jeu, à analyser pour ajuster les actions

    def update_game_state(self, new_state):
        """ Met à jour l'état du jeu en fonction des informations reçues. """
        self.game_state = new_state

    def analyze_context(self):
        """ Analyse le contexte actuel (par exemple, menu, phase de jeu) et prend des décisions. """
        if 'menu' in self.game_state:
            return 'in_menu'
        elif 'duel' in self.game_state:
            return 'in_duel'
        return 'unknown'

    def adjust_action_based_on_context(self):
        """ Ajuste l'action en fonction du contexte du jeu. """
        context = self.analyze_context()
        if context == 'in_menu':
            return 'Navigate through menu'
        elif context == 'in_duel':
            return 'Perform action in duel'
        return 'Wait for context'
