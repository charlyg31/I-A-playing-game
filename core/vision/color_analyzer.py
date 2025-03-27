
import cv2
import numpy as np

class ColorAnalyzer:
    def get_dominant_color(self, image):
        data = np.reshape(image, (-1, 3))
        data = np.float32(data)
        _, labels, palette = cv2.kmeans(data, 1, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant = palette[0].astype(int)
        return tuple(dominant)

    def is_selected(self, image):
        dominant = self.get_dominant_color(image)
        return dominant[0] > 180 and dominant[1] > 180 and dominant[2] < 100
