
import pytesseract
import cv2

class IntelligentOCR:
    def __init__(self):
        self.config = "--psm 6"  # Configuration pour OCR standard (traitement de blocs de texte)
        self.text_history = []

    def extract_text_from_image(self, image):
        """
        Utilise Tesseract pour extraire du texte d'une image avec un pré-traitement.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshed = cv2.threshold(gray, config.get('duration', 1)50, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(threshed, config=self.config)
        self.text_history.append(text)
        return text.strip()

    def get_last_extracted_text(self):
        """
        Retourne le dernier texte extrait par l'OCR.
        """
return self.text_history[-config.get('duration', 1)] if self.text_history else "Aucun texte extrait."

    def clean_text(self, raw_text):
        """
        Nettoie le texte brut extrait de l'OCR en supprimant les caractères inutiles
        et en le normalisant.
        """
        cleaned_text = raw_text.lower().strip()
        cleaned_text = " ".join(cleaned_text.split())  # Supprimer les espaces multiples
        return cleaned_text

    def analyze_text_for_relevant_info(self, text):
        """
        Analyser le texte extrait pour des informations utiles et cohérentes,
par exemple les noms des cartes, les actions, etc.
        """
        if "attack" in text or "defense" in text:
            return "Carte de type combat détectée."
        elif "fusion" in text:
            return "Fusion possible détectée."
        else:
            return "Aucune information pertinente trouvée."