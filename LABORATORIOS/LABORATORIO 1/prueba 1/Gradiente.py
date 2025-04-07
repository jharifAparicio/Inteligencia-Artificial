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

    # Graficar la regresión lineal (valores reales vs. predichos)
    plt.scatter(y_test, y_pred, alpha=0.5, label='Datos de prueba')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea de regresión')
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Regresión Lineal: Valores Reales vs. Predichos')
    plt.legend()
    plt.show()

    # Mostrar los coeficientes del modelo
    print("Coeficientes del modelo:", model.coef_)
    print("Intercepto del modelo:", model.intercept_)

    # Visualización del descenso de gradiente (opcional)
    # Para esto, necesitamos implementar el descenso de gradiente manualmente.
    # Aquí te muestro un ejemplo simplificado.

    # Función de costo (Error Cuadrático Medio)
    def calcular_costo(X, y, theta):
        m = len(y)
        predicciones = X.dot(theta)
        error = predicciones - y
        costo = (1 / (2 * m)) * np.sum(error ** 2)
        return costo  

    # Descenso de gradiente
    def descenso_gradiente(X, y, theta, alpha, iteraciones):
        m = len(y)
        costos = []
        for i in range(iteraciones):
            predicciones = X.dot(theta)
            error = predicciones - y
            gradiente = (1 / m) * X.T.dot(error)
            theta -= alpha * gradiente
            costos.append(calcular_costo(X, y, theta))
        return theta, costos

    # Preparar los datos para el descenso de gradiente
    X_b = np.c_[np.ones((len(X_train), 1)), X_train]  # Agregar una columna de unos para el término de sesgo (bias)
    theta_inicial = np.random.randn(X_b.shape[1])  # Inicializar theta aleatoriamente
    alpha = 0.01  # Tasa de aprendizaje
    iteraciones = 500  # Número de iteraciones

    # Ejecutar el descenso de gradiente
    theta_optimo, costos = descenso_gradiente(X_b, y_train, theta_inicial, alpha, iteraciones)

    # Graficar el descenso de gradiente (costo vs. iteraciones)
    plt.plot(range(iteraciones), costos)
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo (MSE)')
    plt.title('Descenso de Gradiente: Costo vs. Iteraciones')
    plt.show()