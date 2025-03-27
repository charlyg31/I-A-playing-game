
from core.discovery.autodiscover_manager import AutoDiscoverManager

class MechanicObserver:
    def __init__(self):
        self.auto_discover = AutoDiscoverManager()

    def observe_unknown_mechanic(self, context, result, name_hint="mecanique_inconnue"):
        print("[Observer] Nouvelle mécanique détectée. Enregistrement...")
        self.auto_discover.discover_mechanic(name_hint, context, result)
