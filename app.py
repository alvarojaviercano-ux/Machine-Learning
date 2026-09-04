from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/concepts')
def concepts():
    return render_template('concepts.html', page_name="Concepts")


@app.route('/types')
def types():
    return render_template('coming_soon.html', page_name="Types of Machine Learning")


@app.route('/usecase/<int:n>')
def usecase(n):
    return render_template('coming_soon.html', page_name=f"Use Case {n}")


@app.route('/regression/concepts')
def regression_concepts():
    return render_template('coming_soon.html', page_name="Linear Regression - Concepts")


@app.route('/regression/application')
def regression_application():
    return render_template('coming_soon.html', page_name="Linear Regression - Application")


if __name__ == '__main__':
    app.run(debug=True)  