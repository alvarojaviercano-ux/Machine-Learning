import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Etiquetas legibles para la variable objetivo (0/1/2)
LEVEL_NAMES = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
FEATURE_NAMES = ["distancia_km", "tiempo_min", "altimetria_m", "pulsaciones_prom", "cadencia_rpm"]

# Cargar los datos
df_knn = pd.read_csv("data/dataset_ciclismo.csv")

x = df_knn[FEATURE_NAMES]
y = df_knn["nivel"]

# División train/test (80/20), estratificada para mantener proporción de clases
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# KNN es sensible a la escala de las variables (distancia_km vs altimetria_m
# tienen rangos muy distintos), por eso se escalan antes de entrenar.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)

# Métricas calculadas una sola vez al iniciar la app (no en cada request)
y_pred = model.predict(x_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred, average="macro")
knn_recall = recall_score(y_test, y_pred, average="macro")
knn_f1 = f1_score(y_test, y_pred, average="macro")
knn_conf_matrix = confusion_matrix(y_test, y_pred)


def predict_cyclist_level(distancia_km, tiempo_min, altimetria_m, pulsaciones_prom, cadencia_rpm):
    """Recibe las 5 variables y devuelve (nivel_predicho, probabilidades por clase)."""
    input_df = pd.DataFrame([{
        "distancia_km": distancia_km,
        "tiempo_min": tiempo_min,
        "altimetria_m": altimetria_m,
        "pulsaciones_prom": pulsaciones_prom,
        "cadencia_rpm": cadencia_rpm,
    }])
    input_scaled = scaler.transform(input_df)

    predicted_level = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]

    return predicted_level, probabilities


def generate_knn_plot():
    """Genera un scatter plot de los ciclistas agrupados por nivel (clase)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {0: "#E8604C", 1: "#F2B134", 2: "#4C3BCF"}

    for level, name in LEVEL_NAMES.items():
        subset = df_knn[df_knn["nivel"] == level]
        ax.scatter(subset["distancia_km"], subset["altimetria_m"],
                   alpha=0.6, color=colors[level], label=name)

    ax.set_title("Cyclist Level Classification: Distance vs. Elevation Gain")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation gain (m)")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return encoded