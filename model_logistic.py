import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

FEATURE_NAMES = [
    "edad",
    "ingreso_mensual",
    "visitas_web_mes",
    "tiempo_sitio_min",
    "compras_previas",
    "descuento_usado",
]

# dataset que nos dio el profe
df_logistic = pd.read_csv("data/dataset_regresion_logistica.csv")

x = df_logistic[FEATURE_NAMES]
y = df_logistic["target"]

# 80% para entrenar, 20% para probar
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# escalo porque ingreso_mensual y edad tienen escalas muy distintas
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# calculo las métricas una sola vez al arrancar la app
y_pred = model.predict(x_test_scaled)

logistic_accuracy = accuracy_score(y_test, y_pred)
logistic_precision = precision_score(y_test, y_pred)
logistic_recall = recall_score(y_test, y_pred)
logistic_f1 = f1_score(y_test, y_pred)
logistic_conf_matrix = confusion_matrix(y_test, y_pred)


def predict_purchase(edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado):
    # toma los datos del formulario y devuelve si compra o no, más la probabilidad
    input_df = pd.DataFrame([{
        "edad": edad,
        "ingreso_mensual": ingreso_mensual,
        "visitas_web_mes": visitas_web_mes,
        "tiempo_sitio_min": tiempo_sitio_min,
        "compras_previas": compras_previas,
        "descuento_usado": descuento_usado,
    }])
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probability = model.predict_proba(input_scaled)[0][1]  # esta es la probabilidad de que compre

    return prediction, probability


def generate_logistic_plot():
    # arma el gráfico de dispersión separando compradores y no compradores
    fig, ax = plt.subplots(figsize=(7, 5))

    no_compra = df_logistic[df_logistic["target"] == 0]
    si_compra = df_logistic[df_logistic["target"] == 1]

    ax.scatter(no_compra["tiempo_sitio_min"], no_compra["ingreso_mensual"],
               alpha=0.6, color="#E8604C", label="No Purchase (0)")
    ax.scatter(si_compra["tiempo_sitio_min"], si_compra["ingreso_mensual"],
               alpha=0.6, color="#4C3BCF", label="Purchase (1)")

    ax.set_title("Customer Purchase Classification: Time on Site vs. Monthly Income")
    ax.set_xlabel("Time on site (min)")
    ax.set_ylabel("Monthly income")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return encoded