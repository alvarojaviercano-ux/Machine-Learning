import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_records = 550

square_meters = np.random.uniform(30, 300, n_records)

# Precio base: ~ $650 por metro cuadrado, más ruido aleatorio realista
price = square_meters * 650 + np.random.normal(0, 8000, n_records)
price = np.clip(price, 15000, None)

df = pd.DataFrame({
    "square_meters": np.round(square_meters, 1),
    "price": np.round(price, 0)
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/housing_data.csv", index=False)

print(f"Dataset generado: {len(df)} registros en data/housing_data.csv")