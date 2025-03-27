
import json
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np

class ConceptLearner:
    def __init__(self, memory_file="concept_memory.json"):
        self.memory_file = memory_file
        self.clusters = {}
        self.data = []
        self.labels = []
        self.label_map = {}
        self.load()

    def add_example(self, feature_vector, label=None):
        self.data.append(feature_vector)
        self.labels.append(label or "unknown")

    def cluster_concepts(self, n_clusters=3):
        if not self.data:
            return

        X = np.array(self.data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        y_kmeans = kmeans.fit_predict(X)

        self.clusters = defaultdict(list)
        self.label_map = {}
        for i, cluster_id in enumerate(y_kmeans):
            self.clusters[cluster_id].append(self.data[i])
            self.label_map[cluster_id] = self.labels[i]

        self.save()

    def get_concept_for_vector(self, feature_vector):
        if not self.clusters:
            return "unknown"

        # Compare to cluster centroids
        centroids = [np.mean(self.clusters[c], axis=0) for c in self.clusters]
        distances = [np.linalg.norm(np.array(feature_vector) - centroid) for centroid in centroids]
        closest_cluster = int(np.argmin(distances))
        return self.label_map.get(closest_cluster, f"concept_{closest_cluster}")

    def save(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump({
                    "data": self.data,
                    "labels": self.labels
                }, f, indent=2)
        except Exception as e:
            print(f"[ConceptLearner] Error saving: {e}")

    def load(self):
        try:
            with open(self.memory_file, "r") as f:
                memory = json.load(f)
                self.data = memory.get("data", [])
                self.labels = memory.get("labels", [])
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[ConceptLearner] Error loading: {e}")
