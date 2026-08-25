from flask import Flask, render_template

<<<<<<< HEAD
app = Flask(__name__)

@app.route("/")
def home():
    return "hello world"

@app.route("/template")
def template():
    return render_template("index.html")
=======

app = Flask(__name__)

@app.route('/')
def home():
    
    return "HELLO WORLD"


@app.route('/template')
def pagina():
    return render_template('index.html')

>>>>>>> d0c20246067752f3ea1f4eccbdfbe2162038addb
