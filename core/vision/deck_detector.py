import cv2
import numpy as np
from PIL import Image

# Convertit l'image PIL en image OpenCV binaire nette pour l'OCR
def preprocess_image(pil_img):
    img = np.array(pil_img.convert('L'))  # grayscale
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img

# Compte les symboles/icônes dans la zone de droite du deck (visuellement)
def count_deck_icons(pil_img):
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape
    deck_zone = img[:, int(w * 0.85):]  # on prend le bord droit

    gray = cv2.cvtColor(deck_zone, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if 10 < cw < 50 and 10 < ch < 50:  # filtre des rectangles valides
            count += 1
    return count

# Renvoie un texte expliquant si le deck est prêt ou non
def deck_status_from_image(pil_img):
    icons = count_deck_icons(pil_img)
    if icons < 40:
        return f"Deck incomplet ({icons} cartes détectées)"
    return f"Deck prêt ({icons} cartes détectées)"