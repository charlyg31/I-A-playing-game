
import json
import os
import matplotlib.pyplot as plt

def load_latest_log():
    logs_path = "logs"
    if not os.path.exists(logs_path):
        print("Aucun dossier de logs trouvé.")
        return []

    files = sorted([f for f in os.listdir(logs_path) if f.endswith(".json")])
    if not files:
        print("Aucun fichier de log trouvé.")
        return []

    with open(os.path.join(logs_path, files[-1]), 'r') as f:
        data = json.load(f)
    return data

def plot_rewards(log_data):
    rewards = [step["reward"] for step in log_data]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Récompense par action")
    plt.xlabel("Tour")
    plt.ylabel("Récompense")
    plt.title("Performance de l'IA")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data = load_latest_log()
    if data:
        plot_rewards(data)
