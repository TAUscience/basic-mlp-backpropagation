import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Para saltar las primeras 22 líneas que no contienen datos 
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Unión de las 2 filas para un único registro, cada registro tiene 2 filas.
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Creación de un arreglo con las características (X) y la etiqueta (y)
X = np.array(data)
y = np.array(target)

# División del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalizar las etiquetas
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # Normaliza y convierte a 1D
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()  # Normaliza y convierte a 1D

# Guardar los datos normalizados en arreglos
train_data = np.column_stack((X_train_scaled, y_train))
test_data = np.column_stack((X_test_scaled, y_test))

"""
# Gráfica de la relación entre el número de habitaciones (RM) y el valor medio de la vivienda (MEDV)
plt.figure(figsize=(8,6))
sns.scatterplot(x=train_data[:, 5], y=train_data[:, -1])  
plt.title("Relación entre número de habitaciones (RM) y valor de la vivienda (MEDV)")
plt.xlabel("Número de habitaciones (RM)")
plt.ylabel("Valor medio de la vivienda (MEDV)")
plt.show()

# Gráfica de la relación entre CRIM (tasa de criminalidad) y MEDV
plt.figure(figsize=(8,6))
sns.scatterplot(x=train_data[:, 0], y=train_data[:, -1])  
plt.title("Relación entre tasa de criminalidad (CRIM) y valor de la vivienda (MEDV)")
plt.xlabel("Tasa de criminalidad (CRIM)")
plt.ylabel("Valor medio de la vivienda (MEDV)")
plt.show()
"""