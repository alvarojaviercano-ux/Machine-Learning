from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return "hello world"

@app.route("/template")
def template():
    return render_template("index.html")