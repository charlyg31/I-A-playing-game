from PIL import ImageGrab
import numpy as np
import win32gui

class WindowCapture:
    def __init__(self, window_name):
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        if not self.hwnd:
            raise Exception(f'Fenêtre non trouvée : {self.window_name}')

        # Obtenir les dimensions de la fenêtre
        rect = win32gui.GetWindowRect(self.hwnd)
        self.left, self.top, self.right, self.bottom = rect

    def get_screenshot(self):
        # Capture d'écran dans la zone de la fenêtre
        img = ImageGrab.grab(bbox=(self.left, self.top, self.right, self.bottom))
        return np.array(img)
