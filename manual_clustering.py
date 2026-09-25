import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C3BCF", "#E8604C", "#2e7d4f"]

# centroides iniciales, elegidos a mano según los 3 perfiles de apps que esperamos
INITIAL_CENTROIDS = np.array([
    [15.0, 4.5],   # app liviana bien valorada
    [300.0, 4.5],  # app pesada bien valorada
    [250.0, 2.8],  # app pesada mal valorada
])


class ManualKMeans:
    # simulación de K-Means hecha a mano, paso a paso (sin usar KMeans de sklearn)

    def __init__(self, n_iterations=3):
        self.n_iterations = n_iterations

    def make_plot(self, points, labels, centroids, title):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        if labels is None:
            ax.scatter(points[:, 0], points[:, 1], alpha=0.6, color="#999999", label="Apps")
        else:
            for k in range(len(centroids)):
                cluster_points = points[labels == k]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           alpha=0.6, color=COLORS[k], label=f"Cluster {k + 1}")

        ax.scatter(centroids[:, 0], centroids[:, 1],
                   color="black", marker="X", s=180, linewidths=2,
                   edgecolors="white", label="Centroids", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Size (MB)")
        ax.set_ylabel("Rating")
        ax.legend(fontsize=8)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return encoded

    def calculate_variance(self, points, labels, centroids):
        # promedio de qué tan lejos quedó cada punto de su centroide
        total = 0.0
        for k in range(len(centroids)):
            cluster_points = points[labels == k]
            if len(cluster_points) == 0:
                continue
            total = total + np.sum((cluster_points - centroids[k]) ** 2)
        return total / len(points)

    def run(self):
        df = pd.read_csv("data/apps_data_manual.csv")
        points = df[["size_mb", "rating"]].to_numpy()
        names = df["app_name"].to_numpy()

        centroids = INITIAL_CENTROIDS.copy()
        initial_plot = self.make_plot(points, None, centroids, "Initial Data and Centroids")

        iterations = []
        variances = []

        for i in range(1, self.n_iterations + 1):

            # paso 1: distancia de cada app a cada uno de los 3 centroides
            distances = np.zeros((len(points), len(centroids)))
            for row in range(len(points)):
                for k in range(len(centroids)):
                    distances[row, k] = np.sqrt(np.sum((points[row] - centroids[k]) ** 2))

            # paso 2: cada app se asigna al centroide más cercano
            labels = np.argmin(distances, axis=1)

            table_rows = []
            for row in range(len(points)):
                table_rows.append({
                    "app_name": names[row],
                    "size_mb": points[row][0],
                    "rating": points[row][1],
                    "dist_C1": round(distances[row][0], 2),
                    "dist_C2": round(distances[row][1], 2),
                    "dist_C3": round(distances[row][2], 2),
                    "assigned_cluster": labels[row] + 1,
                })

            variance = self.calculate_variance(points, labels, centroids)
            variances.append(round(variance, 2))

            # paso 3: cada centroide se mueve al promedio de sus apps asignadas
            new_centroids = np.zeros_like(centroids)
            for k in range(len(centroids)):
                cluster_points = points[labels == k]
                new_centroids[k] = cluster_points.mean(axis=0)

            plot = self.make_plot(points, labels, new_centroids,
                                  f"Iteration {i}: Clusters and Updated Centroids")

            iterations.append({
                "number": i,
                "table_sample": table_rows[:10],
                "total_rows": len(table_rows),
                "centroids": new_centroids.round(2).tolist(),
                "variance": round(variance, 2),
                "plot": plot,
            })

            centroids = new_centroids

        cluster_sizes = []
        for k in range(len(centroids)):
            cluster_sizes.append(int((labels == k).sum()))

        return {
            "initial_centroids": INITIAL_CENTROIDS.tolist(),
            "initial_plot": initial_plot,
            "iterations": iterations,
            "variances": variances,
            "final_centroids": centroids.round(2).tolist(),
            "cluster_sizes": cluster_sizes,
            "record_count": len(df),
        }