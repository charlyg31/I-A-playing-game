
import tkinter as tk
from tkinter import ttk

class IADashboard:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.label_title = ttk.Label(self.frame, text="🧠 Dashboard IA", font=("Arial", 14, "bold"))
        self.label_title.pack(pady=10)

        self.label_eps = ttk.Label(self.frame, text="Épsilon (exploration): ...")
        self.label_eps.pack(anchor=tk.W)

        self.label_last_reward = ttk.Label(self.frame, text="Dernière récompense: ...")
        self.label_last_reward.pack(anchor=tk.W)

        self.label_action = ttk.Label(self.frame, text="Dernière action: ...")
        self.label_action.pack(anchor=tk.W)

        self.label_victoire = ttk.Label(self.frame, text="Victoires: 0 | Défaites: 0")
        self.label_victoire.pack(anchor=tk.W)

    def update(self, epsilon, reward, action, wins, losses):
        self.label_eps.config(text=f"Épsilon (exploration): {epsilon:.2f}")
        self.label_last_reward.config(text=f"Dernière récompense: {reward:.2f}")
        self.label_action.config(text=f"Dernière action: {action}")
        self.label_victoire.config(text=f"Victoires: {wins} | Défaites: {losses}")