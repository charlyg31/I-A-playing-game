from core.strategy.ia_state_manager import IAStateManager
import pygetwindow as gw
import pyautogui
import cv2
import numpy as np

class VisionSystem:
    def __init__(
        self.state_manager = IAStateManager()
self):
        self.card_database = {}
        self.feature_extractor = cv2.ORB_create()

    def load_database(self):
        pass

    def recognize_card(self, card_image):
        return "unknown_card"

    def capture_screen(self):
        try:
            epsxe_window = gw.getWindowsWithTitle('ePSXe')[0]
            left, top, width, height = epsxe_window.left, epsxe_window.top, epsxe_window.width, epsxe_window.height
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return image
        except IndexError:
            print("[ERREUR] Fenêtre ePSXe non trouvée.")
            return None

    def capture_card_image(self, screenshot, position):
        x, y, w, h = position
        card_image = screenshot[y:y+h, x:x+w]
        return card_image


def explore_frame(self, frame, grid_size=(3, 3)):
    h, w = frame.shape[:2]
    zone_h, zone_w = h // grid_size[0], w // grid_size[1]
    found_zones = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y1, y2 = i * zone_h, (i + 1) * zone_h
            x1, x2 = j * zone_w, (j + 1) * zone_w
            zone = frame[y1:y2, x1:x2]

            attention = self.attention_engine.compute_attention_map(cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY))
            dom_color = self.color_analyzer.get_dominant_color(zone)
            visual_type = self.classifier.classify_region(zone)

            if np.mean(attention) > 40 or visual_type != "card":  # heuristique
                found_zones.append({
                    "row": i, "col": j,
                    "attention": float(np.mean(attention)),
                    "color": dom_color,
                    "type": visual_type
                })

    if found_zones:
        selected = sorted(found_zones, key=lambda z: -z["attention"])[0]
        self.state_manager.register_focus((selected["row"], selected["col"]))
        self.state_manager.register_visual(selected["type"])
        return selected
    else:
        return {"message": "Aucune zone pertinente détectée"}
