
import pyvjoy
import time

# Initialisation du joystick virtuel (ID 1)
joystick = pyvjoy.VJoyDevice(1)

def press_button(button_id, duration=0.1):
    """Appuie puis relâche un bouton donné par son ID (entier)."""
    if isinstance(button_id, int):
        joystick.set_button(button_id, 1)
        time.sleep(duration)
        joystick.set_button(button_id, 0)

def hold_button(button_id):
    """Maintient un bouton pressé (par ID)."""
    if isinstance(button_id, int):
        joystick.set_button(button_id, 1)

def release_button(button_id):
    """Relâche un bouton pressé (par ID)."""
    if isinstance(button_id, int):
        joystick.set_button(button_id, 0)