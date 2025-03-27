from core.audio.epsxe_audio_listener import EPSXESoundListener

import tkinter as tk
from threading import Thread
import time
import traceback

class IALauncher:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Contrôle IA Dueliste")
        self.running = False
        self.paused = False
        self.audio_listener = EPSXESoundListener()

        self.label_status = tk.Label(self.window, text="État : En attente", font=("Arial", 14))
        self.label_status.pack(pady=10)

        self.start_button = tk.Button(self.window, text="Lancer l'IA", width=20, command=self.start_ia)
        self.start_button.pack(pady=5)

        self.pause_button = tk.Button(self.window, text="Pause / Reprendre", width=20, command=self.toggle_pause)
        self.pause_button.pack(pady=5)

        self.stop_button = tk.Button(self.window, text="Arrêter", width=20, command=self.stop_ia)
        self.stop_button.pack(pady=5)

        self.output_text = tk.Text(self.window, height=12, width=60)
        self.output_text.pack(pady=10)

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_ia(self):
        if not self.running:
            self.running = True
            self.paused = False
        self.audio_listener = EPSXESoundListener()
            self.label_status.config(text="État : En fonctionnement")
            Thread(target=self.run_ia_loop, daemon=True).start()

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused
            status = "En pause" if self.paused else "En fonctionnement"
            self.label_status.config(text=f"État : {status}")

    def stop_ia(self):
        if self.running:
            self.running = False
            self.label_status.config(text="État : Arrêté")

    def run_ia_loop(self):
        self.log("L'IA commence l'analyse...")
        while self.running:
            if not self.paused:
                try:
                    try:
    self.log("[IA] Observation de l'écran et analyse en cours...")
    if self.audio_listener.is_audio_active():
        self.log("[AUDIO] Son détecté dans ePSXe.")
    else:
        self.log("[AUDIO] Silence ou faible activité audio.")
    time.sleep(2)
                    # Ici : appel aux modules réels d'IA (vision, prédiction, etc.)
                    # ex: result = vision.analyze_full_frame(...)
                    time.sleep(2)
                except Exception as e:
                    error_text = traceback.format_exc()
                    self.log("[ERREUR] Exception détectée :\n" + error_text)
                    self.running = False
                    self.label_status.config(text="État : Erreur")
        self.log("L'IA a été arrêtée.")

    def log(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def on_close(self):
        self.running = False
        self.window.destroy()

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    launcher = IALauncher()
    launcher.run()
