
class IAStateManager:
    def __init__(self):
        self.context = "unknown"
        self.last_texts = []
        self.last_audio_event = None
        self.last_visual_element = None
        self.current_focus = None
        self.motion_detected = False
        self.template_matches = []

    def update_context(self, context):
        self.context = context

    def register_texts(self, texts):
        self.last_texts = texts

    def register_audio(self, event):
        self.last_audio_event = event

    def register_visual(self, element):
        self.last_visual_element = element

    def register_focus(self, zone):
        self.current_focus = zone

    def register_motion(self, motion_flag):
        self.motion_detected = motion_flag

    def register_templates(self, matches):
        self.template_matches = matches

    def summary(self):
        return {
            "context": self.context,
            "audio_event": self.last_audio_event,
            "visual_element": self.last_visual_element,
            "focus_area": self.current_focus,
            "motion": self.motion_detected,
            "templates": self.template_matches,
            "texts": self.last_texts
        }


def decide_next_action(self):
    if self.last_visual_element == "cursor" and self.motion_detected:
        return "suivre le curseur"
    if self.last_audio_event == "audio_detected":
        return "vérifier retour audio"
    if self.context == "menu" and self.current_focus == "top":
        return "descendre vers une option"
    if self.context == "duel" and self.last_visual_element == "icon":
        return "jouer une carte ou interagir"
    return "explorer davantage"
