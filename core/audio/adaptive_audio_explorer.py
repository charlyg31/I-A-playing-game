
import numpy as np
import sounddevice as sd

class AdaptiveAudioExplorer:
    def __init__(self, samplerate=44100, duration=1, slices=3):
        self.samplerate = samplerate
        self.duration = duration
        self.slices = slices

    def listen_and_slice(self):
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        segment_length = len(audio) // self.slices
        active_segments = []

        for i in range(self.slices):
            segment = audio[i * segment_length:(i + 1) * segment_length]
            energy = np.sum(segment ** 2)
            if energy > 0.01:
                active_segments.append((i, float(energy)))

        return active_segments if active_segments else [{"message": "Pas d'activité audio significative"}]
