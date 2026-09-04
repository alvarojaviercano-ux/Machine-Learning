from flask import Flask, render_template

app = Flask(__name__)


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


@app.route('/regression/application')
def regression_application():
    return render_template('coming_soon.html', page_name="Linear Regression - Application")


if __name__ == '__main__':
    app.run(debug=True)