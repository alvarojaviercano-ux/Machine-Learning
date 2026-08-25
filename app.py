from flask import Flask, render_template


app = Flask(__name__)

@app.route('/')
def home():
    
    return "HELLO WORLD"


@app.route('/template')
def pagina():
    return render_template('index.html')

