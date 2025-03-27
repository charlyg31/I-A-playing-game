
class VisualFocusPlanner:
    def determine_focus_area(self, context):
        if context == "menu":
            return "top"
        elif context == "duel":
            return "center"
        elif context == "deck":
            return "bottom"
        return "center"
