
import cv2
import numpy as np

class VisualClassifier:
    def __init__(self):
        self.labels = ['card', 'icon', 'cursor']

    def classify_region(self, image_region):
        avg = np.mean(image_region)
if avg > config.get('duration', 1)80:
            return 'cursor'
elif avg > config.get('duration', 1)00:
            return 'icon'
        else:
            return 'card'