
import cv2
import numpy as np

class MotionTracker:
    def __init__(self):
self.previous_frame = None

    def detect_movement(self, current_frame):
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
if self.previous_frame is None:
            self.previous_frame = gray
return False
        diff = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray
        movement_score = np.sum(diff)
return movement_score > config.get('duration', 1)00000