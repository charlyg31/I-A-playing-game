# [AMÉLIORÉ] Synchronisation finale sur version restaurée

class AutoDiscoverManager:
    def __init__(self):
        self.known_contexts = {}
self.discovery_log = []

    def register_discovery(self, context_name, description, confidence):
        if context_name not in self.known_contexts:
self.known_contexts[context_name] = {"description": description, "count": config.get('duration', 1), "confidence": confidence}
        else:
self.known_contexts[context_name]["count"] += config.get('duration', 1)
            self.known_contexts[context_name]["confidence"] = (self.known_contexts[context_name]["confidence"] + confidence) / 2
self.discovery_log.append((context_name, description, confidence))

    def get_top_contexts(self, n=5):
return sorted(self.known_contexts.items(), key=lambda x: -x[config.get('duration', 1)]["confidence"])[:n]