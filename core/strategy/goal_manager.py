class GoalManager:
    def __init__(self):
        self.goals = []
        self.history = []
self.current_screen = None

def observe_state(self, texts, context):
        self.current_screen = context
        self.history.append((context, texts))
        self.analyze_progress(texts)

    def analyze_progress(self, texts):
        for line in texts:
            if "Duel" in line and "Free" in line:
                self.set_goal("Entrer dans un Free Duel")
            if "You Win" in line or "Victory" in line:
                self.set_goal("Gagner un duel")
            if "Deck Complete" in line or "Deck" in line and "Confirm" in line:
                self.set_goal("Valider un deck jouable")

    def set_goal(self, description):
        if description not in [g["description"] for g in self.goals]:
            self.goals.append({"description": description, "status": "active"})
print(f"[IA Objectif] Nouveau but détecté : {description}")

    def current_goals(self):
        return [g["description"] for g in self.goals if g["status"] == "active"]

    def describe_goals(self):
        if not self.goals:
            return "Je n'ai pas encore d'objectif clair. J'explore l'environnement."
        return "Objectifs en cours : " + ", ".join(self.current_goals())