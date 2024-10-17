import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Para saltar las primera 22 lineas que no contienen datos 
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Unión de las 2 filas par un único registro, cada registro tiene 2 filas.
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]
X = pd.DataFrame(data, columns=column_names)

# Crear un DataFrame con el valor medio de la vivienda (MEDV)
y = pd.DataFrame(target, columns=["MEDV"])

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar 
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

df_train = pd.DataFrame(X_train, columns=column_names)
df_train["MEDV"] = y_train.values

# Relación entre el número de habitaciones (RM) y el valor medio de la vivienda (MEDV)
plt.figure(figsize=(8,6))
sns.scatterplot(x="RM", y="MEDV", data=df_train)
plt.title("Relación entre número de habitaciones (RM) y valor de la vivienda (MEDV)")
plt.show()

# Relación entre CRIM (tasa de criminalidad) y MEDV
plt.figure(figsize=(8,6))
sns.scatterplot(x="CRIM", y="MEDV", data=df_train)
plt.title("Relación entre tasa de criminalidad (CRIM) y valor de la vivienda (MEDV)")
plt.show()
