import json
import numpy as np
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from handposeutils.data.data_reader import DataReader
from handposeutils.embeddings.vector import get_fused_pose_embedding


def load_handposes(base_dirs):
    data = []
    for gesture_name, folder in base_dirs.items():
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r") as f:
                    pose_data = json.load(f)

                pose = DataReader.convert_json_to_HandPose(pose_data)
                pose.normalize()
                embedding = get_fused_pose_embedding(pose)

                data.append({
                    "name": pose_data["pose"]["name"],
                    "gesture_class": gesture_name,
                    "embedding": embedding
                })
    return data


# Run clustering
def run_clustering(embeddings):
    results = {}
    # K-Means
    km = KMeans(n_clusters=3, random_state=42)
    results["kmeans"] = km.fit_predict(embeddings)

    # Hierarchical
    hc = AgglomerativeClustering(n_clusters=3)
    results["hierarchical"] = hc.fit_predict(embeddings)

    # DBSCAN
    db = DBSCAN(eps=2.5, min_samples=3)
    results["dbscan"] = db.fit_predict(embeddings)

    return results

# Visualize clusters
def visualize(embeddings, labels, title, names):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure()
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10")
    for i, name in enumerate(names):
        plt.text(reduced[i,0], reduced[i,1], name, fontsize=6)
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":
    base_dirs = {
        "peace": "poses/split_peace_frames",
        "rock": "poses/split_rock_frames",
        "thumbs_up": "poses/split_thumbs_up_frames"
    }

    poses = load_handposes(base_dirs)
    embeddings = np.array([p["embedding"] for p in poses])
    names = [p["name"] for p in poses]

    clustering_results = run_clustering(embeddings)

    for method, labels in clustering_results.items():
        visualize(embeddings, labels, f"{method} clustering", names)
