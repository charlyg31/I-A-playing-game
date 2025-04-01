
import numpy as np

class StateBuilder:
    def __init__(self):
        self.state_size = 4  # Exemple: volume, last_button, brightness_avg, white_zone_ratio

    def get_state_size(self):
        return self.state_size

    def encode(self, observation):
        # Convertir observation dict -> vecteur normalisé
        return np.array([
            observation.get("volume", 0.0),
            observation.get("last_button", 0.0) / CONFIG.get('button_range', 14),
            observation.get("brightness_avg", 0.0),
            observation.get("white_zone_ratio", 0.0)
        ], dtype=np.float32)