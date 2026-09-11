import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

from model import df, model, calculate_price
from model_knn import (
    df_knn,
    knn_accuracy,
    knn_precision,
    knn_recall,
    knn_f1,
    knn_conf_matrix,
    predict_cyclist_level,
    generate_knn_plot,
    LEVEL_NAMES,
)
from model_logistic import (
    df_logistic,
    logistic_accuracy,
    logistic_precision,
    logistic_recall,
    logistic_f1,
    logistic_conf_matrix,
    predict_purchase,
    generate_logistic_plot,
)

app = Flask(__name__)


def generate_plot(predicted_point=None):
    """Builds the scatter plot + regression line, returns it as a base64 PNG string."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df['square_meters'], df['price'],
               alpha=0.4, color='#4C3BCF', label='Housing data')

    x_line = pd.DataFrame({
        'square_meters': [df['square_meters'].min(), df['square_meters'].max()]
    })
    y_line = model.predict(x_line)
    ax.plot(x_line['square_meters'], y_line, color='#E8604C', linewidth=2.5, label='Regression line')

    if predicted_point:
        ax.scatter([predicted_point[0]], [predicted_point[1]],
                   color='#2e7d4f', s=140, zorder=5, marker='*', label='Your prediction')

    ax.set_title('House Price vs. Square Meters')
    ax.set_xlabel('Square Meters (m²)')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded


USE_CASES = {
    1: {
        "name": "Medical Image Diagnosis",
        "context": "Hospitals need to detect diseases (such as tumors) early and accurately from medical images like X-rays or MRIs, where manual review is slow and prone to human error.",
        "data": "Thousands of labeled medical images, each tagged by specialists as 'healthy' or 'disease detected'.",
        "target": "Classify a new image as healthy or as showing signs of a specific disease.",
        "ml_type": "Supervised Learning (Classification)",
        "benefit": "Faster, more consistent diagnoses that support doctors and can catch cases that might otherwise be missed."
    },
    2: {
        "name": "Retail Customer Segmentation",
        "context": "A retail store wants to understand its customers better to design targeted marketing campaigns, but has no predefined customer categories.",
        "data": "Purchase history, spending amount, frequency of visits, and product categories bought by each customer.",
        "target": "Discover natural groups of customers with similar buying behavior.",
        "ml_type": "Unsupervised Learning (Clustering)",
        "benefit": "More effective, personalized marketing and better inventory decisions based on real customer segments."
    },
    3: {
        "name": "Banking Fraud Detection",
        "context": "Banks need to detect fraudulent transactions in real time, among millions of legitimate ones, to protect customers' accounts.",
        "data": "Historical transaction records labeled as 'fraudulent' or 'legitimate', including amount, location, and time.",
        "target": "Classify a new transaction as fraudulent or legitimate as it happens.",
        "ml_type": "Supervised Learning (Classification)",
        "benefit": "Faster fraud detection that reduces financial losses and protects customers, with minimal manual review."
    },
    4: {
        "name": "Self-Driving Car Navigation",
        "context": "An autonomous vehicle must learn to make safe driving decisions (steering, braking, accelerating) in a constantly changing environment.",
        "data": "Continuous sensor data (camera, radar, lidar) describing the car's surroundings, along with rewards or penalties based on driving outcomes.",
        "target": "Discover the sequence of actions that maximizes safety and efficiency while driving.",
        "ml_type": "Reinforcement Learning",
        "benefit": "A system that improves its driving policy over time through experience, adapting to new situations."
    },
}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/concepts')
def concepts():
    return render_template('concepts.html')


@app.route('/types')
def types():
    return render_template('types.html')


@app.route('/usecase/<int:n>')
def usecase(n):
    case = USE_CASES.get(n)
    if not case:
        return "Use case not found", 404
    return render_template('usecase.html', case=case, n=n)


@app.route('/regression/concepts')
def regression_concepts():
    return render_template('regression_concepts.html')


@app.route('/regression/application', methods=['GET', 'POST'])
def regression_application():
    prediction = None
    error = None
    input_value = None

    if request.method == 'POST':
        input_value = request.form.get('square_meters', '').strip()

        if not input_value:
            error = "Please enter a value."
        else:
            try:
                sqm = float(input_value)
                if sqm <= 0:
                    error = "Please enter a positive number."
                else:
                    predicted_price = calculate_price(sqm)
                    prediction = round(predicted_price, 2)
            except ValueError:
                error = "Please enter a valid numeric value."

    chart_point = (float(input_value), prediction) if (prediction is not None and input_value) else None
    chart_base64 = generate_plot(predicted_point=chart_point)

    return render_template(
        'regression_application.html',
        chart_base64=chart_base64,
        prediction=prediction,
        error=error,
        input_value=input_value,
        record_count=len(df),
        coef=round(model.coef_[0][0], 2),
        intercept=round(model.intercept_[0], 2),
    )


# ============================================================
# LOGISTIC REGRESSION — Actividad 2
# ============================================================

@app.route('/logistic/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')


@app.route('/logistic/application', methods=['GET', 'POST'])
def logistic_application():
    prediction_label = None
    probability = None
    error = None
    form_values = {}

    fields = ['edad', 'ingreso_mensual', 'visitas_web_mes', 'tiempo_sitio_min', 'compras_previas', 'descuento_usado']

    if request.method == 'POST':
        form_values = {f: request.form.get(f, '').strip() for f in fields}

        if not all(form_values.values()):
            error = "Please fill in all fields."
        else:
            try:
                edad = float(form_values['edad'])
                ingreso_mensual = float(form_values['ingreso_mensual'])
                visitas_web_mes = float(form_values['visitas_web_mes'])
                tiempo_sitio_min = float(form_values['tiempo_sitio_min'])
                compras_previas = float(form_values['compras_previas'])
                descuento_usado = float(form_values['descuento_usado'])

                if any(v < 0 for v in [edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas]):
                    error = "Please enter non-negative numbers."
                elif descuento_usado not in (0, 1):
                    error = "Discount code must be Yes or No."
                else:
                    pred, prob = predict_purchase(
                        edad, ingreso_mensual, visitas_web_mes,
                        tiempo_sitio_min, compras_previas, descuento_usado
                    )
                    prediction_label = "Will buy" if pred == 1 else "Will not buy"
                    probability = round(prob * 100, 1)
            except ValueError:
                error = "Please enter valid numeric values."

    return render_template(
        'logistic_application.html',
        chart_base64=generate_logistic_plot(),
        prediction_label=prediction_label,
        probability=probability,
        error=error,
        form_values=form_values,
        record_count=len(df_logistic),
    )


@app.route('/logistic/metrics')
def logistic_metrics():
    return render_template(
        'logistic_metrics.html',
        accuracy=round(logistic_accuracy * 100, 2),
        precision=round(logistic_precision * 100, 2),
        recall=round(logistic_recall * 100, 2),
        f1=round(logistic_f1 * 100, 2),
        conf_matrix=logistic_conf_matrix.tolist(),
    )


# ============================================================
# K-NEAREST NEIGHBORS — Actividad 2 (Modelo Asignado)
# ============================================================

@app.route('/knn/concepts')
def knn_concepts():
    return render_template('knn_concepts.html')


@app.route('/knn/application', methods=['GET', 'POST'])
def knn_application():
    prediction_label = None
    probabilities = None
    error = None
    form_values = {}

    fields = ['distancia_km', 'tiempo_min', 'altimetria_m', 'pulsaciones_prom', 'cadencia_rpm']

    if request.method == 'POST':
        form_values = {f: request.form.get(f, '').strip() for f in fields}

        if not all(form_values.values()):
            error = "Please fill in all fields."
        else:
            try:
                values = {f: float(v) for f, v in form_values.items()}
                if any(v <= 0 for v in values.values()):
                    error = "Please enter positive numbers."
                else:
                    predicted_level, probs = predict_cyclist_level(**values)
                    prediction_label = LEVEL_NAMES[predicted_level]
                    probabilities = [
                        {"label": LEVEL_NAMES[i], "value": round(p * 100, 1)}
                        for i, p in enumerate(probs)
                    ]
            except ValueError:
                error = "Please enter valid numeric values."

    return render_template(
        'knn_application.html',
        chart_base64=generate_knn_plot(),
        prediction_label=prediction_label,
        probabilities=probabilities,
        error=error,
        form_values=form_values,
        record_count=len(df_knn),
    )


@app.route('/knn/metrics')
def knn_metrics():
    return render_template(
        'knn_metrics.html',
        accuracy=round(knn_accuracy * 100, 2),
        precision=round(knn_precision * 100, 2),
        recall=round(knn_recall * 100, 2),
        f1=round(knn_f1 * 100, 2),
        conf_matrix=knn_conf_matrix.tolist(),
        level_names=list(LEVEL_NAMES.values()),
    )


if __name__ == '__main__':
    app.run(debug=True)