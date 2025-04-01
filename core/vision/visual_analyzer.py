
import cv2
import numpy as np

class VisualAnalyzer:
    def __init__(self):
        self.last_features = {}

    def process(self, frame):
        features = {}

        resized = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features["gray"] = gray

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        features["edges"] = sobel_mag

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        features["blur"] = blur

        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
        features["binary"] = th

        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
        features["hist_bgr"] = (hist_b, hist_g, hist_r)

        self.last_features = features
        return features

    def extract_useful_info(self, features):
        info = {}
        gray = features.get("gray")
        edges = features.get("edges")
        binary = features.get("binary")

        if gray is not None:
            info["brightness_avg"] = np.mean(gray)

        if edges is not None:
            info["edge_intensity"] = np.mean(edges)

        if binary is not None:
            white_zone_ratio = np.sum(binary > 128) / (binary.shape[0] * binary.shape[1])
            info["white_zone_ratio"] = white_zone_ratio

        return info