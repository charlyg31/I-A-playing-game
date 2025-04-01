class NameInputDetector:
    def __init__(self):
        self.keywords = ["name", "enter your name", "nom du", "type your name"]

    def detect_name_screen(self, ocr_texts):
        for line in ocr_texts:
            if any(k in line.lower() for k in self.keywords):
return True
return False