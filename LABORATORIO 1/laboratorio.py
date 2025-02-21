# 1. Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 2. Cargar el dataset
# Asegúrate de cambiar 'path_to_your_dataset.csv' por la ruta correcta de tu archivo CSV.
data = pd.read_csv('California_Houses.csv')
#
# print(data.columns)

# 3. Seleccionar características (features) y variable objetivo (target)
# Aquí seleccionamos 11 columnas numéricas como características y 'Median_House_Value' como la variable objetivo.
features = data[['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 
                     'Population', 'Households', 'Latitude', 'Longitude', 
                     'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego']]  # Columnas seleccionadas
target = data['Median_House_Value']  # Variable objetivo

# Normalizar las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo usando el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio (MSE): {mse}')

# Visualizar los resultados (valores reales vs. predichos)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Comparación de Valores Reales y Predichos')
plt.show()