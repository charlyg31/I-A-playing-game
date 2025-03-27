from collections import defaultdict, deque

class AdversaryPredictor:
    def __init__(self):
        self.patterns = defaultdict(list)
        self.history = defaultdict(lambda: deque(maxlen=3))

    def record_action(self, opponent_name, game_state_text, action):
        key = tuple(game_state_text[-3:])  # dernier contexte visuel ou texte
        self.patterns[(opponent_name, key)].append(action)
        self.history[opponent_name].append((key, action))

    def predict_next_action(self, opponent_name, current_state_text):
        key = tuple(current_state_text[-3:])
        actions = self.patterns.get((opponent_name, key), [])
        if not actions:
            return "Aucune prédiction disponible."
        from collections import Counter
        most_common = Counter(actions).most_common(1)[0]
        return f"Action probable de {opponent_name} dans ce contexte : {most_common[0]}"
