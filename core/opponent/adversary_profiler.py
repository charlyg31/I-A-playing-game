
class AdversaryProfiler:
    def __init__(self):
        self.adversary_styles = {}
        self.match_history = {}

    def record_adversary_style(self, opponent_name, style):
        """
        Enregistre le style de jeu d'un adversaire (agressif, défensif, équilibré, etc.)
        """
        if opponent_name not in self.adversary_styles:
            self.adversary_styles[opponent_name] = []
        self.adversary_styles[opponent_name].append(style)

    def analyze_adversary_style(self, opponent_name):
        """
        Analyse les styles de jeu d'un adversaire pour prédire son comportement futur.
        """
        if opponent_name not in self.adversary_styles:
            return "Style d'adversaire non connu."

        styles = self.adversary_styles[opponent_name]
        aggressive = styles.count("aggressive")
        defensive = styles.count("defensive")
        balanced = styles.count("balanced")

        if aggressive > defensive and aggressive > balanced:
            return f"{opponent_name} préfère un style agressif."
        elif defensive > aggressive and defensive > balanced:
            return f"{opponent_name} préfère un style défensif."
        elif balanced > aggressive and balanced > defensive:
            return f"{opponent_name} adopte un style équilibré."
        else:
            return f"Le style de {opponent_name} est difficile à prédire."

    def update_match_history(self, opponent_name, result):
        """
        Enregistre le résultat d'un match pour chaque adversaire (gagné ou perdu).
        """
        if opponent_name not in self.match_history:
self.match_history[opponent_name] = {"wins": 0, "losses": 0}

        if result == CONFIG.get('outcome_win_label', 'win'):
self.match_history[opponent_name]["wins"] += config.get('duration', 1)
        else:
self.match_history[opponent_name]["losses"] += config.get('duration', 1)

    def get_adversary_win_rate(self, opponent_name):
        """
        Renvoie le taux de victoire d'un adversaire.
        """
        if opponent_name not in self.match_history:
            return "Pas d'historique disponible."

        wins = self.match_history[opponent_name]["wins"]
        losses = self.match_history[opponent_name]["losses"]
        total_matches = wins + losses
return wins / total_matches if total_matches > 0 else 0