
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, AudioSessionManager2, ISimpleAudioVolume
from pycaw.utils import AudioUtilities as AudioUtils
from pycaw.constants import CLSID_MMDeviceEnumerator, eRender
import sounddevice as sd

class EPSXESoundListener:
    def __init__(self, target_name="epsxe"):
        self.target_name = target_name.lower()
        self.session = self._get_epsxe_session()

    def _get_epsxe_session(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            if session.Process and self.target_name in session.Process.name().lower():
                return session
        return None

    def is_audio_active(self):
        # Very basic: check if volume is being output by the app
        if self.session is None:
            self.session = self._get_epsxe_session()
            if self.session is None:
                return False
        volume = self.session._ctl.QueryInterface(ISimpleAudioVolume)
        return volume.GetMasterVolume() > 0.01  # simplistic check

    def listen_loop(self, duration=10):
        print(f"[Audio] Écoute du son de {self.target_name.upper()} pendant {duration} secondes...")
        t_end = time.time() + duration
        while time.time() < t_end:
            active = self.is_audio_active()
            if active:
                print("[Audio] Son détecté dans ePSXe !")
            else:
                print("[Audio] Silence (aucun signal actif de ePSXe)")
            time.sleep(1)
