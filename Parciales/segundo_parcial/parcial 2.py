# %% [markdown]
# # Introducción
# El objetivo es aplicar un método de aprendizaje no supervisado al dataset de páginas web maliciosas y benignas para identificar automáticamente grupos (clusters) sin usar las etiquetas originales. Esto ayuda a descubrir patrones o clasificaciones naturales en los datos.
# 
# Usaremos K-Means, una técnica de clustering que agrupa los datos en k clusters basados en similitud. En este caso, esperamos dos clusters: uno para páginas maliciosas y otro para benignas.

# %% [markdown]
# # Paso 1: Carga del dataset

# %% [markdown]
# Explicación:
# Se importa pandas para manipular datos. Se carga el archivo CSV y se revisan las primeras filas y el tipo de datos para conocer qué columnas tiene el dataset.

# %%
import pandas as pd

# Carga el archivo CSV al entorno de Colab y luego usa la ruta
df = pd.read_csv('weatherAUS.csv')

# Mostrar las primeras filas para entender la estructura
print(df.head())
print(df.info())

# %%
# imprimir los nombres de las columnas
print("Columnas en el DataFrame:")
print(df.columns.tolist())

# %%
# y: etiqueta (RainTomorrow)
y = df['RainTomorrow']

# X: todas las demás columnas menos la etiqueta y la fecha
X = df.drop(columns=['RainTomorrow', 'Date'])

# %% [markdown]
# # Paso 2: Preprocesamiento
# Convertimos url en la longitud, una variable numérica que resume la URL.
# 
# Para IP, contamos cuántos puntos hay, es una forma simple de convertirla a número.
# 
# Eliminamos columnas no numéricas residuales.
# 
# Aplicamos PCA solo a variables numéricas.
# 
# Esto permite mantener parte de la información original sin eliminar toda la columna.
# 
# Y mostramos la distribucion de los datos

# %%
# limpiar los datos eliminando filas con valores faltantes
X_clean = X.dropna().copy()
y_clean = y.loc[X_clean.index]  # Aseguramos que y tenga el mismo índice

# %%
from sklearn.preprocessing import LabelEncoder

X_encoded = X_clean.copy()
cat_cols = X_encoded.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.7)
plt.title('Distribución de datos tras PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()


# %% [markdown]
# # buscamos la mejor k
# En clustering no supervisado, no hay etiquetas para validar directamente.
# 
# El número de clusters 
# 𝑘
# k no siempre es evidente.
# 
# Para elegir 
# 𝑘
# k óptimo, se usa el Silhouette Score:
# 
# Mide qué tan bien separado y compacto está cada cluster.
# 
# Va de -1 a 1.
# 
# Valores cercanos a 1 indican clusters bien definidos.
# 
# Se prueba KMeans con varios valores de 
# 𝑘
# k.
# 
# Se calcula el Silhouette para cada 
# 𝑘
# k.
# 
# Se selecciona el 
# 𝑘
# k que maximiza el Silhouette Score.

# %%
from sklearn.preprocessing import OneHotEncoder

# Usar sparse_output en lugar de sparse para mayor compatibilidad
encoder = OneHotEncoder(sparse_output=False)
X_clean_encoded = encoder.fit_transform(X_clean.select_dtypes(include=['object']))

# Si hay variables numéricas, concatenar:
import numpy as np
X_num = X_clean.select_dtypes(exclude=['object']).to_numpy()
X_clean_encoded = np.hstack([X_num, X_clean_encoded])

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

inertia = []
silhouette = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_clean_encoded)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_clean_encoded, labels))
    print(f'k={k}, Inercia={kmeans.inertia_}, Silhouette={silhouette[-1]}')

best_k = K[np.argmax(silhouette)]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertia, 'bx-')
plt.axvline(best_k, linestyle='--', color='grey', label=f'Mejor k = {best_k}')
plt.xlabel('Número de clusters k')
plt.ylabel('Inercia (Suma de cuadrados)')
plt.title('Método del codo')
plt.legend()

plt.subplot(1,2,2)
plt.plot(K, silhouette, 'rx-')
plt.axvline(best_k, linestyle='--', color='grey', label=f'Mejor k = {best_k}')
plt.xlabel('Número de clusters k')
plt.ylabel('Silhouette Score')
plt.title('Índice de Silhouette')
plt.legend()

plt.show()


# %%
# Ajustar KMeans con mejor k
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
labels_best = kmeans_best.fit_predict(X_clean_encoded)

# Reducir a 2D con PCA para visualizar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean_encoded)

# Graficar clusters
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_best, cmap='tab10', s=10)
plt.title(f'Visualización clusters con PCA y k={best_k}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score

# Convertir y_clean a formato numérico (necesario para las métricas) y eliminar valores NaN
y_clean_encoded = y_clean.map({'No': 0, 'Yes': 1})
y_clean_encoded = y_clean_encoded.fillna(-1)  # Reemplazar NaN con -1 o algún otro valor

# Asegurarnos de que trabajamos con las mismas filas en los clusters y las etiquetas
mask = ~y_clean_encoded.isna()
y_clean_encoded = y_clean_encoded.astype(int)

# Calcular métricas entre las etiquetas reales y las predicciones de KMeans
ari = adjusted_rand_score(y_clean_encoded, labels_best)
nmi = normalized_mutual_info_score(y_clean_encoded, labels_best)
homogeneity = homogeneity_score(y_clean_encoded, labels_best)
completeness = completeness_score(y_clean_encoded, labels_best)
v_measure = v_measure_score(y_clean_encoded, labels_best)

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-measure: {v_measure:.3f}")


# %%
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
import matplotlib.pyplot as plt

# Asumiendo que tienes X_clean_encoded (datos numéricos listos para cluster)
# y y_true (etiquetas reales)

# Define número de clusters (k) según el mejor encontrado previamente
k = best_k # ejemplo

# Entrenar GMM
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X_clean_encoded)

# Evaluación interna
sil_score = silhouette_score(X_clean_encoded, gmm_labels)

# Evaluación externa (si tienes etiquetas reales)
ari = adjusted_rand_score(y_clean_encoded, gmm_labels)
nmi = normalized_mutual_info_score(y_clean_encoded, gmm_labels)
homogeneity = homogeneity_score(y_clean_encoded, gmm_labels)
completeness = completeness_score(y_clean_encoded, gmm_labels)
v_measure = v_measure_score(y_clean_encoded, gmm_labels)

print(f'Silhouette Score: {sil_score:.3f}')
print(f'Adjusted Rand Index (ARI): {ari:.3f}')
print(f'Normalized Mutual Information (NMI): {nmi:.3f}')
print(f'Homogeneity: {homogeneity:.3f}')
print(f'Completeness: {completeness:.3f}')
print(f'V-measure: {v_measure:.3f}')

# Visualización 2D (con PCA si X tiene más de 2 dimensiones)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean_encoded)

plt.scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap='viridis', s=30)
plt.title('Clusters según Gaussian Mixture Models')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# %%
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Parámetros DBSCAN (ajustar según datos)
eps = 15
min_samples = 10

# Entrenar DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_clean_encoded)

# Filtrar ruido (-1)
mask = dbscan_labels != -1

# Evaluación interna sólo para los puntos no ruido
sil_score = silhouette_score(X_clean_encoded[mask], dbscan_labels[mask]) if sum(mask) > 1 else -1

# Evaluación externa (ignorar ruido)
ari = adjusted_rand_score(y_clean_encoded[mask], dbscan_labels[mask])
nmi = normalized_mutual_info_score(y_clean_encoded[mask], dbscan_labels[mask])
homogeneity = homogeneity_score(y_clean_encoded[mask], dbscan_labels[mask])
completeness = completeness_score(y_clean_encoded[mask], dbscan_labels[mask])
v_measure = v_measure_score(y_clean_encoded[mask], dbscan_labels[mask])

print(f'Silhouette Score (sin ruido): {sil_score:.3f}')
print(f'Adjusted Rand Index (ARI): {ari:.3f}')
print(f'Normalized Mutual Information (NMI): {nmi:.3f}')
print(f'Homogeneity: {homogeneity:.3f}')
print(f'Completeness: {completeness:.3f}')
print(f'V-measure: {v_measure:.3f}')

# Visualización 2D con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean_encoded)

plt.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels, cmap='plasma', s=30)
plt.title('Clusters según DBSCAN')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


