
import cv2
import numpy as np

class AttentionEngine:
    def compute_attention_map(self, gray_image):
grad_x = cv2.Sobel(gray_image, cv2.CV_64F, config.get('duration', 1), 0)
grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, config.get('duration', 1))
        magnitude = cv2.magnitude(grad_x, grad_y)
norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)