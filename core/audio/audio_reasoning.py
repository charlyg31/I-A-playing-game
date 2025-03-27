from core.strategy.ia_state_manager import IAStateManager

import numpy as np
import sounddevice as sd
import scipy.signal
import time

class AudioReasoner:
    def __init__(
        self.state_manager = IAStateManager()
self, samplerate=44100, duration=1):
        self.samplerate = samplerate
        self.duration = duration
        self.last_audio = None

    def listen(self):
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float32')
        sd.wait()
        self.last_audio = audio.flatten()
        return self.last_audio

    def detect_peak_energy(self, audio_data):
        energy = np.sum(audio_data ** 2)
        return energy > 0.01  # seuil empirique

    def detect_audio_event(self):
        audio_data = self.listen()
        if self.detect_peak_energy(audio_data):
            self.state_manager.register_audio("audio_detected")
        return "Son détecté (pic d'énergie)"
        else:
            return "Silence ou bruit faible"
