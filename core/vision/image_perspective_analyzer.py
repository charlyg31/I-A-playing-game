
import cv2
import numpy as np

class ImagePerspectiveAnalyzer:
    def __init__(self):
        pass

    def generate_perspectives(self, image):
        perspectives = {
            "original": image,
            "grayscale": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            "threshold": self.apply_threshold(image),
            "clahe": self.apply_clahe(image),
            "sharpened": self.apply_sharpen(image),
            "edges": self.apply_canny(image),
            "adaptive_thresh": self.apply_adaptive_threshold(image),
            "inverted": self.apply_invert(image),
            "dilated": self.apply_dilation(image),
            "eroded": self.apply_erosion(image),
            "contour_highlight": self.highlight_contours(image)
        }
        return perspectives

    def apply_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh

    def apply_adaptive_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

    def apply_clahe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)

    def apply_sharpen(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def apply_canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    def apply_invert(self, image):
        return cv2.bitwise_not(image)

    def apply_dilation(self, image):
        kernel = np.ones((2,2), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.dilate(gray, kernel, iterations=1)

    def apply_erosion(self, image):
        kernel = np.ones((2,2), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.erode(gray, kernel, iterations=1)

    def highlight_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = image.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
        return output
