
class AutoRewardEngine:
    def __init__(self):
        self.last_lp_player = 8000
        self.last_lp_enemy = 8000

    def compute_reward(self, state):
        reward = 0
        lp_player = state.get("lp_player", 8000)
        lp_enemy = state.get("lp_enemy", 8000)

        if lp_enemy < self.last_lp_enemy:
            reward += (self.last_lp_enemy - lp_enemy) * 0.01
        if lp_player < self.last_lp_player:
            reward -= (self.last_lp_player - lp_player) * 0.01

        if lp_enemy <= 0:
            reward += 10
        elif lp_player <= 0:
            reward -= 10

        self.last_lp_player = lp_player
        self.last_lp_enemy = lp_enemy
        return reward

# === Ajout automatique ===

# Amélioration du système de récompense
if fusion_echouee:
    reward -= 0.5  # Pénalité pour mauvaise fusion
if victoire:
    reward += 5  # Bonus pour victoire
if combo_utilise:
    reward += 2  # Bonus pour combo efficace
print(f"[INFO] Reward Calculé : {reward}")
