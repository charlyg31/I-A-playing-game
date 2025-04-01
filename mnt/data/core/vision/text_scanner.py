
import pytesseract
import cv2

class TextScanner:
    def __init__(self, image):
        self.image = image  # L'image à analyser pour le texte

    def scan_text(self):
        """ Utilise OCR (Tesseract) pour extraire le texte de l'image. """ 
        text = pytesseract.image_to_string(self.image)
        return text

    def get_text_from_region(self, x, y, w, h):
        """ Extrait du texte d'une région spécifique de l'image. """ 
        region = self.image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(region)
        return text

    def analyze_text_for_action(self, text):
        """ Analyse le texte pour déterminer l'action à effectuer. """ 
        if 'confirm' in text.lower():
            return 'Press Confirm'
        elif 'cancel' in text.lower():
            return 'Press Cancel'
        return 'No action identified'
