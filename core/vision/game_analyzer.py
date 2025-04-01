
import cv2
import numpy as np

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import ImageGrab

class GameAnalyzer:
    def capture_screen(self):
        screen = ImageGrab.grab()
        return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    def detect_end_game_state(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).upper()
        win_detected = "YOU WIN" in text
        lose_detected = "YOU LOSE" in text
        return win_detected, lose_detected