
class VisualReactionManager:
    def __init__(self):
        self.reaction_rules = {
            "enemy_monster_appears": self.attack,
            "trap_card_activated": self.activate_trap,
            "high_attack_enemy": self.defend,
            "low_health": self.heal,
            "card_drawn": self.evaluate_new_card
        }

    def analyze_and_react(self, detected_changes):
        reactions = []
        for change in detected_changes:
            if "Changement détecté" in change:
                if "enemy" in change:
                    reactions.append(self.reaction_rules["enemy_monster_appears"]())
                elif "trap" in change:
                    reactions.append(self.reaction_rules["trap_card_activated"]())
                elif "attack" in change:
                    reactions.append(self.reaction_rules["high_attack_enemy"]())
                elif "health" in change:
                    reactions.append(self.reaction_rules["low_health"]())
                elif "card" in change:
                    reactions.append(self.reaction_rules["card_drawn"]())
        return reactions

    def attack(self):
        return "Attack the opponent monster."

    def activate_trap(self):
        return "Activate defensive trap card."

    def defend(self):
        return "Defend with high defense cards."

    def heal(self):
        return "Use healing card to recover health."

    def evaluate_new_card(self):
        return "Evaluate new card and decide next action."
