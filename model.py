import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar los datos
df = pd.read_csv("data/housing_data.csv")

x = df[["square_meters"]]
y = df[["price"]]

model = LinearRegression()
model.fit(x, y)


def calculate_price(square_meters):
    result = model.predict([[square_meters]])[0]
    return result[0]