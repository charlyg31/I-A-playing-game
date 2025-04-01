
import tkinter as tk

class MemoryInspector:
    def __init__(self, root):
        self.window = tk.Toplevel(root)
        self.window.title("📚 Mémoire Visuelle de l'IA")
        self.window.geometry("300x400")

        self.title_label = tk.Label(self.window, text="Boutons testés sur cet écran", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=10)

        self.text_box = tk.Text(self.window, height=20, width=35)
        self.text_box.pack(padx=10, pady=5)

    def update_memory(self, button_scores):
        self.text_box.delete("1.0", tk.END)
        if not button_scores:
            self.text_box.insert(tk.END, "Aucun bouton connu pour cet écran.")
        else:
            for bid, score in sorted(button_scores.items(), key=lambda x: -x[1]):
                self.text_box.insert(tk.END, f"B{bid} : Score {score:.2f}\n")