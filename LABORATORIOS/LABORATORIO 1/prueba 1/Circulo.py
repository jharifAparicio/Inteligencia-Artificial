import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Verificar si el archivo existe
ruta_archivo = 'California_Houses.csv'  # Asegúrate de que el nombre sea exacto
if not os.path.exists(ruta_archivo):
    print(f"El archivo no existe en la ruta: {os.path.abspath(ruta_archivo)}")
else:
    print(f"El archivo existe. Cargando datos...")
    data = pd.read_csv(ruta_archivo)

    # Imprimir los nombres de las columnas para verificar
    print("Columnas disponibles en el dataset:")
    print(data.columns)

    # Seleccionar características (features) y variable objetivo (target)
    features = data[['Median_Income', 'Median_Age']]  # Usamos solo dos características para simplificar
    target = data['Median_House_Value']  # Variable objetivo

    # Normalizar las características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Agregar una columna de unos para el término de sesgo (bias)
    X_b = np.c_[np.ones((len(features_scaled), 1)), features_scaled]

    # Función de costo (Error Cuadrático Medio)
    def calcular_costo(X, y, theta):
        m = len(y)
        predicciones = X.dot(theta)
        error = predicciones - y
        costo = (1 / (2 * m)) * np.sum(error ** 2)
        return costo

    # Crear una cuadrícula de valores para theta0 y theta1
    theta0_vals = np.linspace(-10, 10, 100)  # Rango de theta0
    theta1_vals = np.linspace(-10, 10, 100)  # Rango de theta1
    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

    # Calcular el costo para cada combinación de theta0 y theta1
    costos = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = np.array([theta0_vals[i], theta1_vals[j], 0])  # theta2 se fija en 0
            costos[i, j] = calcular_costo(X_b, target, theta)

    # Graficar el contorno
    plt.figure(figsize=(10, 8))
    contour = plt.contour(theta0_grid, theta1_grid, costos.T, levels=np.logspace(-2, 3, 20), cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)  # Etiquetas en las curvas de nivel
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')
    plt.title('Contorno de la Función de Costo')
    plt.colorbar(label='Costo (J)')
    plt.show()