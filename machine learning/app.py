from flask import Flask, request, render_template
from HousePrice import calculatePrice, NumRecords, IndependentVar, DependentVar


app = Flask(__name__)

# Informacion del dataset que se muestra en la pagina de Application
dataset_info = {
    "num_records": NumRecords,
    "independent_var": IndependentVar,
    "dependent_var": DependentVar,
    "independent_unit": "square meters (m²)",
    "dependent_unit": "Colombian Pesos (COP)",
    "source": "Generated dataset for academic purposes (simulated housing data)"
}

# Esta es la pagina principal cuando entro al sitio 
@app.route('/')
def home():
    return render_template('home.html')


# Pagina que explica que es el Machine Learning
@app.route('/ml/concepts')
def ml_concepts():
    return render_template('MLConcepts.html')


# Pagina que explica los diferentes tipos de Machine Learning
@app.route('/ml/types')
def ml_types():
    return render_template('MlTypes.html')


# Pagina del primer caso de uso 
@app.route('/ml/usecase1')
def use_case1():
    return render_template('UseCase1.html')


# Pagina del segundo caso de uso 
@app.route('/ml/usecase2')
def use_case2():
    return render_template('UseCase2.html')


# Pagina del tercer caso de uso 
@app.route('/ml/usecase3')
def use_case3():
    return render_template('UseCase3.html')


# Pagina del cuarto caso de uso 
@app.route('/ml/usecase4')
def use_case4():
    return render_template('UseCase4.html')


# Pagina que explica la Regresión Lineal
@app.route('/supervised/linear-regression/concepts')
def lr_concepts():
    return render_template('LrConcepts.html')


# Pagina donde se muestra el formulario para predecir el precio de una casa
@app.route('/supervised/linear-regression/application')
def lr_application():
    return render_template('LrApplication.html', result=None, error=None, dataset_info=dataset_info)


# Esta es la que se activa cuando alguien  le da clic a calculate price
@app.route('/predecir', methods=['POST'])
def predecir():
    result = None  # aca vamos a guardar el precio calculado si todo sale bien
    error = None   # aca vamos a guardar un mensaje de error si sale mal

    try:
        # Tomamos el numero que la persona escribio en el formulario
        area = float(request.form['area'])

        # Revisamos que el número tenga sentido antes de usarlo
        if area <= 0:
            error = "El área debe ser mayor que 0."
        elif area > 1000:
            error = "El área ingresada es demasiado grande."
        else:
            result = calculatePrice(area)

    except ValueError:
        # Esto pasa si la persona escribió letras
        error = "Ingrese un numero válido."

    # Mostramos de nuevo la misma Pagina pero ahora con el resultado o el error
    return render_template('LrApplication.html', result=result, error=error, dataset_info=dataset_info)


# Esto hace que la aplicación se prenda y quede corriendo cuando ejecutas el archivo
if __name__ == '__main__':
    app.run(debug=True)