
class ClusterActionMemory:
    def __init__(self):
        self.cluster_actions = {}  # Dictionnaire pour stocker les actions par cluster

    def store_action_in_cluster(self, cluster_id, action):
        """ Stocke une action dans un cluster donné. """
        if cluster_id not in self.cluster_actions:
            self.cluster_actions[cluster_id] = []
        self.cluster_actions[cluster_id].append(action)

    def retrieve_actions_for_cluster(self, cluster_id):
        """ Récupère toutes les actions pour un cluster spécifique. """
        return self.cluster_actions.get(cluster_id, [])

    def clear_actions(self, cluster_id):
        """ Efface les actions associées à un cluster. """
        if cluster_id in self.cluster_actions:
            del self.cluster_actions[cluster_id]

    def get_cluster_action_summary(self, cluster_id):
        """ Résume les actions prises dans un cluster. """
        actions = self.retrieve_actions_for_cluster(cluster_id)
        if actions:
            return f"Cluster {cluster_id}: {len(actions)} actions performed"
        return f"Cluster {cluster_id}: No actions performed yet"
