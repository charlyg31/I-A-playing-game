
import cv2
import os

class TemplateMatcher:
    def __init__(self, template_dir='core/vision/templates'):
        self.templates = {}
        for fname in os.listdir(template_dir):
            path = os.path.join(template_dir, fname)
            if fname.endswith('.png') or fname.endswith('.jpg'):
                self.templates[fname] = cv2.imread(path, 0)

    def match_templates(self, gray_image, threshold=0.8):
        matches = []
        for name, template in self.templates.items():
            res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            loc = zip(*((res >= threshold).nonzero()[::-1]))
            for pt in loc:
                matches.append((name, pt))
        return matches
