import tkinter as tk

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agent.agent_libre import AgentLibre
from tkinter import messagebox
from core.vision.text_scanner import TextScanner
from PIL import ImageTk, Image
from io import BytesIO
import os

# Dépendance principale : AgentLibre
# (il est injecté ailleurs dans le fichier projet)

class PureSensorialLauncher:
    def __init__(self):
        self.text_scanner = TextScanner()
        self.agent = AgentLibre(self.text_scanner)
        self.text_output = self.agent.get_current_text()

    def start(self):
        print("Lancement du système...")
        text_output = self.text_output
        self.launch_gui(text_output)

    def launch_gui(self, text_output):
        root = tk.Tk()
        root.title("IA Yu-Gi-Oh! Forbidden Memories")

        tk.Label(root, text="Texte détecté :", font=("Arial", 12, "bold")).pack()
        self.text_box = tk.Text(root, wrap=tk.WORD, height=6, width=60)
        self.text_box.pack(pady=5)
        self.text_box.insert(tk.END, text_output)
        self.text_box.config(state=tk.DISABLED)

        tk.Label(root, text="Réflexion de l'IA :", font=("Arial", 12, "bold")).pack()
        self.thought_text = tk.Text(root, wrap=tk.WORD, height=6, width=60)
        self.thought_text.pack(pady=5)
        self.thought_text.insert(tk.END, self.agent.last_thought + "\n" + self.agent.last_action)
        self.thought_text.config(state=tk.DISABLED)

        self.timeline_label = tk.Label(root, text="Historique :", font=("Arial", 12, "bold"))
        self.timeline_label.pack()
        self.timeline_box = tk.Text(root, wrap=tk.WORD, height=8, width=60)
        self.timeline_box.pack(pady=5)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="▶ Démarrer", command=self.agent.start).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="⏸ Pause", command=self.agent.pause).grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="🟥 Stop", command=self.agent.stop).grid(row=0, column=2, padx=5)

        def open_heatmap():
            import subprocess
            subprocess.Popen(["python", "visualiser_heatmap.py"])

        tk.Button(control_frame, text="📊 Heatmap", command=open_heatmap).grid(row=0, column=3, padx=5)

        def update_gui():
            self.text_box.config(state=tk.NORMAL)
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, self.agent.get_current_text())
            self.text_box.config(state=tk.DISABLED)

            self.thought_text.config(state=tk.NORMAL)
            self.thought_text.delete("1.0", tk.END)
            self.thought_text.insert(tk.END, self.agent.last_thought + "\n" + self.agent.last_action)
            self.thought_text.config(state=tk.DISABLED)

            self.timeline_box.delete("1.0", tk.END)
            for log in self.agent.logs[-5:]:
                self.timeline_box.insert(tk.END, f"[T{log['turn']}] A{log['action']} | R={log['reward']:.2f}\n")

            state = self.agent.state_hash(self.agent.get_current_text())
            q_vals = [self.agent.q_table.get((state, a), 0) for a in range(1, 15)]
            self.timeline_box.insert(tk.END, "\nQ-values:\n")
            for i, val in enumerate(q_vals, 1):
                self.timeline_box.insert(tk.END, f"B{i}: {val:.2f}\n")

            root.after(2000, update_gui)

        update_gui()
        root.mainloop()

# Lanceur principal
if __name__ == "__main__":
    launcher = PureSensorialLauncher()
    launcher.start()