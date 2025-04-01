# [AMÉLIORÉ] Synchronisation finale sur version restaurée

class VisualFocusPlanner:
    def __init__(self):
        self.context_zones = {
            "menu": "top",
            "duel": "center",
            "deck": "bottom",
            "fusion": "center",
            "result": "center"
        }

    def determine_focus_area(self, context):
        return self.context_zones.get(context, "center")