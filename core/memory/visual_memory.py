
import numpy as np
import hashlib
import cv2
from collections import defaultdict

class VisualMemory:
    def __init__(self):
        self.known_images = {}
        self.button_scores = defaultdict(lambda: defaultdict(float))  # image_hash -> button_id -> score

    def _hash_frame(self, gray_image):
        resized = cv2.resize(gray_image, (32, 32))
        return hashlib.md5(resized.tobytes()).hexdigest()

    def record_observation(self, gray_image, button_id, reward):
        image_id = self._hash_frame(gray_image)
        self.known_images[image_id] = gray_image
        self.button_scores[image_id][button_id] += reward

    def get_best_buttons(self, gray_image):
        image_id = self._hash_frame(gray_image)
        if image_id in self.button_scores:
            return sorted(self.button_scores[image_id].items(), key=lambda x: x[1], reverse=True)
        return []

    def has_seen(self, gray_image):
        image_id = self._hash_frame(gray_image)
        return image_id in self.known_images

    def get_button_score(self, gray_image, button_id):
        image_id = self._hash_frame(gray_image)
        return self.button_scores[image_id][button_id]