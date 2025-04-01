
import numpy as np
import sounddevice as sd
import threading

class SoundAnalyzer:
    def __init__(self, sample_rate=44100, chunk_size=2048):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.current_volume = 0.0

    def start(self):
        self.running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stream_loop(self):
        def callback(indata, frames, time, status):
            volume_norm = np.linalg.norm(indata) * 10
            self.current_volume = volume_norm

        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate, blocksize=self.chunk_size):
            while self.running:
                sd.sleep(100)

    def stop(self):
        self.running = False

    def get_volume(self):
        return self.current_volume