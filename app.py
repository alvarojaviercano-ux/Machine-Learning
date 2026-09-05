import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from flask import Flask, render_template, request

from model import df, model, calculate_price

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


if __name__ == '__main__':
    app.run(debug=True)
    