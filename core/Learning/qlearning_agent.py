
import json
import random
import os

class QLearningAgent:
    def __init__(self, actions, q_table_file="q_table.json", alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.actions = actions  # Liste des actions possibles
        self.q_table_file = q_table_file
        self.alpha = alpha      # Taux d'apprentissage
        self.gamma = gamma      # Facteur de rÃ©duction
        self.epsilon = epsilon  # Taux d'exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.load_q_table()

    def get_q(self, state, action):
        return self.q_table.get(str(state), {}).get(action, 0.0)

    def choose_action(self, state):
        state_str = str(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.get_best_action(state_str)

    def get_best_action(self, state_str):
        q_values = self.q_table.get(state_str, {})
        if not q_values:
            return random.choice(self.actions)
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)
        if state_str not in self.q_table:
            self.q_table[state_str] = {}
        if action not in self.q_table[state_str]:
            self.q_table[state_str][action] = 0.0

        max_q_next = max(self.q_table.get(next_state_str, {}).values(), default=0.0)
        old_q = self.q_table[state_str][action]
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q_table[state_str][action] = new_q

        # RÃ©duction epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save_q_table()

    def save_q_table(self):
        try:
            with open(self.q_table_file, 'w') as f:
                json.dump(self.q_table, f, indent=2)
        except Exception as e:
            print(f"[QLearningAgent] Erreur de sauvegarde: {e}")

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'r') as f:
                    self.q_table = json.load(f)
            except Exception as e:
                print(f"[QLearningAgent] Erreur de chargement: {e}")
