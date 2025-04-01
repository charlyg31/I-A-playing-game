
class ClusterTextLinker:
    def __init__(self):
        self.cluster_text_links = {}  # Dictionnaire pour lier des textes aux clusters

    def link_text_to_cluster(self, cluster_id, text):
        """ Lier un texte spécifique à un cluster. """
        if cluster_id not in self.cluster_text_links:
            self.cluster_text_links[cluster_id] = []
        self.cluster_text_links[cluster_id].append(text)

    def retrieve_linked_text(self, cluster_id):
        """ Récupérer tous les textes liés à un cluster. """
        return self.cluster_text_links.get(cluster_id, [])

    def update_linked_text(self, cluster_id, old_text, new_text):
        """ Mettre à jour un texte lié à un cluster. """
        if cluster_id in self.cluster_text_links:
            if old_text in self.cluster_text_links[cluster_id]:
                index = self.cluster_text_links[cluster_id].index(old_text)
                self.cluster_text_links[cluster_id][index] = new_text
                return True
        return False

    def visualize_text_links(self):
        """ Afficher les relations entre les clusters et les textes. """
        for cluster_id, texts in self.cluster_text_links.items():
            print(f"Cluster {cluster_id}: Linked texts: {texts}")
