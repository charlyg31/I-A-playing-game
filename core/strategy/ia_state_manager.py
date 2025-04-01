# [AMÉLIORÉ] Synchronisation finale sur version restaurée

import hashlib
import numpy as np
from collections import deque

class IAStateManager:
    def __init__(self):
self.context_history = deque(maxlen=30)
self.state_signatures = {}
self.last_state = None

def observe_context(self, vision_data=None, audio_label=None, text=None):
        components = [vision_data or "", audio_label or "", text or ""]
        combined = "|".join(components).lower()
        signature = hashlib.md5(combined.encode()).hexdigest()
        self.context_history.append(signature)
        return signature

def register_context_as_state(self, signature, label):
self.state_signatures[signature] = label

def get_current_state(self):
        if not self.context_history:
            return "unknown"
sig = self.context_history[-config.get('duration', 1)]
return self.state_signatures.get(sig, "unlabeled_state")

def state_changed(self):
current = self.get_current_state()
changed = current != self.last_state
self.last_state = current
        return changed

def get_state_similarity_score(self, signature):
        matches = sum(sig == signature for sig in self.context_history)
return matches / len(self.context_history) if self.context_history else 0