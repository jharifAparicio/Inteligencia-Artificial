# %%
# Calculo cientifico y vectorial para python
import numpy as np

# Librerias para graficar
from matplotlib import pyplot

import joblib

# %% [markdown]
# # Cargamos los datos  en python

# %%
# Función para cargar datos densos desde un archivo .data
def load_dense_data(file_path, num_features):
    with open(file_path, 'r') as f:
        data = [list(map(int, line.strip().split())) for line in f]
    
    # Crear una matriz densa inicializada con ceros
    dense_matrix = np.zeros((len(data), num_features), dtype=int)
    
    # Rellenar la matriz con 1 en las posiciones de las características no nulas
    for i, row in enumerate(data):
        dense_matrix[i, [col - 1 for col in row]] = 1  # Restamos 1 para ajustar a 0-indexing
    
    return dense_matrix

# Número total de características (según la documentación del conjunto de datos)
num_features = 100000

# Cargar datos de entrenamiento y validación como matrices densas
X_train = load_dense_data("data/dorothea_train.data", num_features)
X_valid = load_dense_data("data/dorothea_valid.data", num_features)

# Cargar etiquetas
Y_train = np.loadtxt("data/dorothea_train.labels")
Y_valid = np.loadtxt("data/dorothea_valid.labels")

# Convertir etiquetas de -1/0 y mantener 1 para que sea compatible con el modelo
def convert_labels(y):
    return np.where(y == -1, 0, y)

Y_train = convert_labels(Y_train)
Y_valid = convert_labels(Y_valid)

X = X_train
Y = Y_train

# Verificar las dimensiones
print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de Y_train: {Y_train.shape}")
print(f"Dimensiones de X_valid: {X_valid.shape}")
print(f"Dimensiones de Y_valid: {Y_valid.shape}")

# Mostrar las primeras filas de X_train y Y_train
print(f"\nPrimeras 5 filas de X_train:\n{X_train[:5]}")
print(f"Primeras 5 etiquetas de Y_train:\n{Y_train[:5]}")

# %% [markdown]
# en este caso no nesecitamos normalizar porque los datos son 0 o 1, no existen valores intermedios que nos llegarian a complicar el modelo.

# %%
# Verificar el rango de cada característica
print("Valor mínimo de X_train:", X_train.min())
print("Valor máximo de X_train:", X_train.max())

# %% [markdown]
# # clasificación binario o regresion logistica

# %% [markdown]
# <a id="section1"></a>
# ### 1.2 Implementacion
# 
# #### 1.2.1 Fución Sigmoidea
# 
# La hipotesis para la regresión logistica se define como:
# 
# $$ h_\theta(x) = g(\theta^T x)$$
# 
# donde la función $g$ is la función sigmoidea. La función sigmoidea se define como:
# 
# $g(z) = \frac{1}{1+e^{-z}}.$
# 
# Los resultados que debe generar la funcion sigmoidea para valores positivos amplios de `x`, deben ser cercanos a 1, mientras que para valores negativos grandes, la sigmoide debe generar valores cercanos 0. La evaluacion de `sigmoid(0)` debe dar un resultado exacto de 0.5. Esta funcion tambien debe poder trabajar con vectores y matrices.

# %%
def sigmoid(z):
    # Calcula la sigmoide de una entrada z
    # convierte la intrada a un arreglo numpy
    z = np.array(z)

    g = np.zeros(z.shape)

    g = 1 / (1 + np.exp(-z))

    return g

# %%
entrada = np.array([1, 0, 1])
salida = sigmoid(entrada)
print("Entrada:", entrada)
print("Salida:", salida)

# %%
def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    epsilon = 1e-8  # Para evitar división por cero
    X_norm = (X - mu) / (sigma + epsilon)

    return X_norm, mu, sigma

# %%
X_norm, mu, sigma = featureNormalize(X)

print(X_norm, mu, sigma)

# %% [markdown]
# Función de Costo y Gradiente

# %% [markdown]
# Se implementa la funcion cost y gradient, para la regresión logistica. Antes de continuar es importante agregar el termino de intercepcion a X.

# %%
# Configurar la matriz adecuadamente, y agregar una columna de unos que corresponde al termino de intercepción.
m, n = X.shape
# Agraga el termino de intercepción a A
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# %%
def calcularCosto(theta, X, y):
    # Inicializar algunos valores utiles
    m = y.size  # numero de ejemplos de entrenamiento
    epsilon = 1e-5  # Pequeño valor para evitar log(0)
    J = 0
    h = sigmoid(X.dot(theta.T))
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h + epsilon)))

    return J

# %%
def descensoGradiente(theta, X, y, alpha, num_iters):
    # Inicializa algunos valores
    m = y.shape[0] # numero de ejemplos de entrenamiento

    # realiza una copia de theta, el cual será acutalizada por el descenso por el gradiente
    theta = theta.copy()
    J_history = []

    for i in range(num_iters):
        h = sigmoid(X.dot(theta.T))
        theta = theta - (alpha / m) * (h - y).dot(X)

        J_history.append(calcularCosto(theta, X, y))
        print(f"Iteración {i + 1}/{num_iters}, Costo: {J_history[-1]}")
    return theta, J_history

# %%
# Elegir algun valor para alpha (probar varias alternativas)
alpha = 0.001
num_iters = 5500

# inicializa theta y ejecuta el descenso por el gradiente
theta = np.zeros(num_features + 1)
theta, J_history = descensoGradiente(theta, X, Y, alpha, num_iters)

# Grafica la convergencia del costo
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')

# Muestra los resultados del descenso por el gradiente
print('theta calculado por el descenso por el gradiente: {:s}'.format(str(theta)))

# %%
print(X_valid)
print(theta)

# %%
X_test_norm, mu, sigma = featureNormalize(X_valid)
m, n = X_test_norm.shape
X_test_norm = np.concatenate([np.ones((m, 1)), X_test_norm], axis=1)
aprueba = sigmoid(np.dot(X_test_norm, theta))   # Se debe cambiar esto
print(aprueba)

# %%
# Cargar el modelo desde el archivo
theta = joblib.load('modelo_clasificacion.pkl')

print("Theta cargado:", theta)

# %%
print(X_test_norm)
print(theta)
predict = sigmoid(np.dot(X_test_norm[50], theta))   # Se debe cambiar esto
print(predict)
print(Y_valid[50])

# %%
# guardamos el modelo entrenado( clasificacion [theta])
joblib.dump(theta, 'modelo_clasificacion.pkl')


