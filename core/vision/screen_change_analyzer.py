
import cv2
import numpy as np

class ScreenChangeAnalyzer:
    def __init__(self):
        self.frames = []

    def observe_frame(self, frame):
        self.frames.append(frame)

    def analyze_changes(self):
        if len(self.frames) < 2:
            return {"message": "Pas assez de frames pour analyser les changements"}

        last_frame = self.frames[-2]
        current_frame = self.frames[-1]
        
        changes = []

        gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(gray_last, gray_current)
        _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                changes.append(f"Changement détecté à {cv2.boundingRect(contour)}")

        if not changes:
            changes.append("Aucun changement visuel détecté")

        return {"changes": changes}
