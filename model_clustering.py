import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURE_NAMES = ["size_mb", "rating"]
CLUSTER_COLORS = ["#4C3BCF", "#E8604C", "#2e7d4f"]


class AppsClusteringModel:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def get_data(self):
        # dataset de apps que registramos en Teams
        df = pd.read_csv("data/apps_data.csv")
        return df

    def implement_clustering(self):
        df = self.get_data()

        x = df[FEATURE_NAMES].values

        # escalamos porque size_mb y rating no manejan la misma escala
        x_scaled = self.scaler.fit_transform(x)

        labels = self.model.fit_predict(x_scaled)
        df["cluster"] = labels

        # silhouette score nos dice qué tan bien separados quedaron los clusters
        score = silhouette_score(x_scaled, labels)

        # devolvemos los centroides a la escala original (MB y rating reales)
        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)

        summary = []
        for k in range(self.n_clusters):
            cluster_df = df[df["cluster"] == k]
            summary.append({
                "cluster": k,
                "count": len(cluster_df),
                "avg_size_mb": round(cluster_df["size_mb"].mean(), 1),
                "avg_rating": round(cluster_df["rating"].mean(), 2),
                "centroid_size_mb": round(centroids[k][0], 1),
                "centroid_rating": round(centroids[k][1], 2),
            })

        return {
            "df": df,
            "summary": summary,
            "silhouette": score,
        }

    def generate_plot(self, df):
        # gráfico de dispersión con un color por cluster, más los centroides
        fig, ax = plt.subplots(figsize=(7, 5))

        for k in range(self.n_clusters):
            cluster_df = df[df["cluster"] == k]
            ax.scatter(cluster_df["size_mb"], cluster_df["rating"],
                       alpha=0.5, color=CLUSTER_COLORS[k], label=f"Cluster {k + 1}")

        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   color="black", marker="X", s=200, linewidths=2,
                   edgecolors="white", label="Centroids", zorder=5)

        ax.set_title("Mobile App Clusters: Size vs. Rating")
        ax.set_xlabel("Size (MB)")
        ax.set_ylabel("Rating")
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return encoded