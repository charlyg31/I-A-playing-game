import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from core.vision.window_capture import WindowCapture

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_contrast(image_pil):
    enhancer = ImageEnhance.Contrast(image_pil)
    return enhancer.enhance(2.0)

class TextScanner:
    def __init__(self):
        self.wincap = WindowCapture("ePSXe - Enhanced PSX emulator")
        screenshot = self.wincap.get_screenshot()
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(screenshot_rgb)

    def scan_text(self):
        # Prétraitement de l’image avant OCR
        processed_image = enhance_contrast(self.image)
        text = pytesseract.image_to_string(processed_image)
        return text


    def capture_image(self):
        screenshot = self.wincap.get_screenshot()
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return Image.fromarray(screenshot_rgb)

    def ocr(self, image):
        processed_image = enhance_contrast(image)
        return pytesseract.image_to_string(processed_image)
    

    def detect_cursor(self, text):
        lines = text.split('\n')
        for line in lines:
            if '>' in line or line.strip().startswith('*'):
                return True
        return False
