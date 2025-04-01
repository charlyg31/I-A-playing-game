
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def charger_dernier_log():
    dossier_logs = "logs"
    if not os.path.exists(dossier_logs):
        print("Aucun dossier 'logs' trouvé.")
        return []

    fichiers = sorted([f for f in os.listdir(dossier_logs) if f.endswith(".json")])
    if not fichiers:
        print("Aucun fichier de log trouvé.")
        return []

    chemin = os.path.join(dossier_logs, fichiers[-1])
    with open(chemin, "r") as f:
        return json.load(f)

def generer_heatmap(data):
    # Compter le nombre d'utilisations de chaque bouton (1-14)
    freqs = [0] * 14
    for step in data:
        action = step.get("action")
        if isinstance(action, int) and 1 <= action <= 14:
            freqs[action - 1] += 1

    plt.figure(figsize=(10, 1.5))
    plt.imshow([freqs], cmap="hot", aspect="auto")
    plt.colorbar(label="Utilisation")
    plt.xticks(np.arange(14), [f"B{i+1}" for i in range(14)], rotation=45)
    plt.yticks([])
    plt.title("Fréquence d'utilisation des boutons")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = charger_dernier_log()
    if data:
        generer_heatmap(data)
