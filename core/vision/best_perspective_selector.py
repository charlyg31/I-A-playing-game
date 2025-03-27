
import pytesseract
import cv2

class BestPerspectiveSelector:
    def __init__(self, mode="text"):
        self.mode = mode  # "text", "visual", ou "hybrid"

    def select_best(self, perspectives):
        scores = {}

        for name, image in perspectives.items():
            if self.mode == "text":
                score = self._ocr_score(image)
            elif self.mode == "visual":
                score = self._contour_score(image)
            elif self.mode == "hybrid":
                score = self._ocr_score(image) + self._contour_score(image)
            else:
                score = 0
            scores[name] = score

        best_view = max(scores, key=scores.get)
        return perspectives[best_view], best_view

    def get_all_ranked(self, perspectives):
        ranked = []

        for name, image in perspectives.items():
            score = self._ocr_score(image) + self._contour_score(image)
            ranked.append((score, name, image))

        ranked.sort(reverse=True)
        return ranked

    def _ocr_score(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            text = pytesseract.image_to_string(gray)
            return len(text.strip())
        except Exception:
            return 0

    def _contour_score(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edged = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len(contours)
        except Exception:
            return 0
