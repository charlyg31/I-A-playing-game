import pyvjoy
import time
import json
import hashlib
import random

class GameController:
    def __init__(self):
        self.j = pyvjoy.VJoyDevice(1)

    def press_button(self, button_id, duration=0.2):
        self.j.set_button(button_id, 1)
        time.sleep(duration)
        self.j.set_button(button_id, 0)

class AgentLibre:
    def __init__(self, text_scanner):
        self.text_scanner = text_scanner
        self.controller = GameController()
        self.running = False
        self.paused = False
        self.turn = 0
        self.last_thought = "Agent initialisé"
        self.last_action = "Aucune."
        self.q_table = {}
        self.epsilon = 0.2
        self.alpha = 0.6
        self.gamma = 0.9
        self.last_image = self.text_scanner.image
        self.init_logging()
        self.load_q_table()

    def save_q_table(self, path='q_table.json'):
        with open(path, 'w') as f:
            json.dump({f'{k[0]}_{k[1]}': v for k, v in self.q_table.items()}, f)

    def load_q_table(self, path='q_table.json'):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.q_table = {(k.split('_')[0], int(k.split('_')[1])): v for k, v in data.items()}
        except FileNotFoundError:
            self.q_table = {}

    def init_logging(self):
        self.logs = []
        self.session_id = time.strftime("%Y%m%d-%H%M%S")

    def log_step(self, state_before, action, reward, state_after):
        self.logs.append({
            "turn": self.turn,
            "state_before": state_before,
            "action": action,
            "reward": reward,
            "state_after": state_after
        })

    def save_logs(self):
        import os
        log_path = f"logs/session_{self.session_id}.json"
        os.makedirs("logs", exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def start(self):
        self.running = True
        self.paused = False
        self.loop()

    def state_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def choose_action(self, state_text):
        state_id = self.state_hash(state_text)
        if random.random() < self.epsilon:
            return random.randint(1, 14)
        q_values = [self.q_table.get((state_id, a), 0) for a in range(1, 15)]
        max_value = max(q_values)
        best_actions = [a for a, v in zip(range(1, 15), q_values) if v == max_value]
        return random.choice(best_actions)

    def learn(self, old_text, action, reward, new_text):
        old_state = self.state_hash(old_text)
        new_state = self.state_hash(new_text)
        old_q = self.q_table.get((old_state, action), 0)
        future_q = max([self.q_table.get((new_state, a), 0) for a in range(1, 15)])
        updated_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[(old_state, action)] = updated_q

    def get_current_text(self):
        img = self.text_scanner.capture_image()
        self.last_image = img
        text = self.text_scanner.ocr(img).strip().lower()
        return text

    def gpt_style_comment(self, texte):
        from core.vision.deck_detector import deck_status_from_image
        deck_info = deck_status_from_image(self.last_image)
        texte = texte.lower()

        if "your deck isn't ready" in texte:
            return f"Le jeu m’indique que le deck n’est pas prêt. [{deck_info}]"
        elif "push start" in texte:
            return "Il semble que je sois sur l'écran d'accueil. Je vais essayer de lancer le jeu."
        elif "you'll find me" in texte or "worthy opponent" in texte:
            return "Un adversaire me met au défi. J'accepte avec détermination."
        elif "duel" in texte or "battle" in texte or "phase" in texte:
            return "Le duel commence. Je dois analyser la situation attentivement."
        elif "you win" in texte:
            return "Victoire ! Ma stratégie a porté ses fruits."
        elif "you lose" in texte:
            return "J'ai perdu... Il me faudra réfléchir à une meilleure tactique."
        elif "life points" in texte:
            return "Les points de vie sont affichés. Le duel bat son plein."
        elif "pass" in texte:
            return "Un choix s'offre à moi. Duel ou Pass ? Je dois faire le bon."

        return f"Je vois à l’écran : '{texte[:50]}...'. Je cherche à comprendre ce que cela implique. [{deck_info}]"

    # Nouvelle méthode 'pause' pour mettre en pause l'IA
    def pause(self):
        self.paused = True
        self.last_thought = "Pause activée. J'observe le jeu."

    # Nouvelle méthode 'stop' pour arrêter l'IA
    def stop(self):
        self.running = False
        self.save_logs()
        self.save_q_table()
        self.last_thought = "IA arrêtée."
        self.last_action = "Aucune."

    def loop(self):
        from threading import Thread

        def run():
            while self.running:
                try:
                    if self.paused:
                        self.last_thought = "En pause. J’observe le jeu."
                        time.sleep(2)
                        continue

                    self.turn += 1
                    state_before = self.get_current_text()
                    action = self.choose_action(state_before)
                    self.controller.press_button(action)
                    time.sleep(1.5)
                    state_after = self.get_current_text()

                    reward = 1.0 if state_before != state_after else -0.1
                    self.learn(state_before, action, reward, state_after)
                    self.log_step(state_before, action, reward, state_after)

                    self.last_thought = f"[TOUR {self.turn}] " + self.gpt_style_comment(state_after)
                    self.last_action = f"[TOUR {self.turn}] Bouton {action} pressé."
                    print(self.last_thought)
                    print(self.last_action)

                except Exception as e:
                    self.last_thought = f"[ERREUR] {str(e)}"
                    self.last_action = "Erreur lors de l'exécution."
                    print("[EXCEPTION]", e)

        Thread(target=run, daemon=True).start()