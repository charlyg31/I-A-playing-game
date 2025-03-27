
import cv2
import numpy as np

class VisualClassifier:
    def __init__(self):
        self.labels = ['card', 'icon', 'cursor']

    def classify_region(self, image_region):
        avg = np.mean(image_region)
        if avg > 180:
            return 'cursor'
        elif avg > 100:
            return 'icon'
        else:
            return 'card'
