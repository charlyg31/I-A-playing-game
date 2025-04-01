
import pyvjoy
import time

class VJoyController:
    def __init__(self):
        self.joystick = pyvjoy.VJoyDevice(1)

    def press_button(self, button_id, duration=0.1):
        self.joystick.set_button(button_id, 1)
        time.sleep(duration)
        self.joystick.set_button(button_id, 0)

    def hold_button(self, button_id):
        self.joystick.set_button(button_id, 1)

    def release_button(self, button_id):
        self.joystick.set_button(button_id, 0)
