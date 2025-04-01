# [AMÉLIORÉ] Synchronisation finale sur version restaurée

import numpy as np
import cv2

class ImagePerspectiveAnalyzer:
    def __init__(self):
self.last_image = None

    def has_significant_change(self, img):
if self.last_image is None:
            self.last_image = img
return True
        diff = cv2.absdiff(self.last_image, img)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        nonzero = np.count_nonzero(thresh)
        self.last_image = img
return nonzero > config.get('duration', 1)0000