
class ActionFeedbackAnalyzer:
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.last_audio_event = None

    def record_action(self, action, visual_snapshot, audio_event):
        self.last_action = action
        self.last_state = visual_snapshot
        self.last_audio_event = audio_event

    def analyze_feedback(self, new_visual_snapshot, new_audio_event):
        feedback = {
            "action": self.last_action,
            "visual_change": new_visual_snapshot != self.last_state,
            "audio_reaction": new_audio_event != self.last_audio_event,
            "success": False
        }

        if feedback["visual_change"] or feedback["audio_reaction"]:
            feedback["success"] = True

        return feedback
