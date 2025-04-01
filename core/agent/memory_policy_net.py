
class MemoryPolicyNet:
    def __init__(self, memory_data):
        self.memory_data = memory_data  # Les données de mémoire pour la prise de décision

    def evaluate_policy(self, state):
        """ Évalue la politique de mémoire pour un état donné. """
        if state in self.memory_data:
            return 'Policy Applied: Action based on memory data'
        return 'No relevant memory found'

    def update_policy(self, new_memory_data):
        """ Met à jour la politique de mémoire avec de nouvelles données. """
        self.memory_data = new_memory_data
        return 'Memory policy updated'

    def summarize_policy(self):
        """ Résume la politique de mémoire actuelle. """
        return f'Memory policy based on {len(self.memory_data)} states.'


# === Ajout automatique ===

# Ajout après calcul de l'action
print(f"[INFO] PolicyNet - État : {state} | Action choisie : {chosen_action}")
