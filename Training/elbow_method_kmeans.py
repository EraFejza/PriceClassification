import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans
from kneed import KneeLocator

def plot_elbow_method(file_path, max_clusters=10):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :-1]
    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(x)
        inertia.append(kmeans.inertia_)

    # Find the elbow point (optimal k)
    kneedle = KneeLocator(range(1, max_clusters + 1), inertia, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow

    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal Number of Clusters')

    # Mark the optimal k with a red 'x'
    if optimal_k is not None:
        plt.plot(optimal_k, inertia[optimal_k - 1], 'rx', markersize=12)

    plt.show()

if __name__ == "__main__":
    plot_elbow_method("../Normalized_Datasets/Train/raw_z_score_scaled.csv")