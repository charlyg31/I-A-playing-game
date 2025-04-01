# [AMÉLIORÉ] Synchronisation finale sur version restaurée

from sklearn.cluster import KMeans
import numpy as np

class ConceptLearner:
    def __init__(self):
        self.feature_vectors = []
        self.labels = []

    def observe(self, features):
        self.feature_vectors.append(features)

    def cluster_concepts(self, n_clusters=5):
        if len(self.feature_vectors) >= n_clusters:
            model = KMeans(n_clusters=n_clusters, n_init="auto").fit(self.feature_vectors)
            self.labels = model.labels_
            return model
return None