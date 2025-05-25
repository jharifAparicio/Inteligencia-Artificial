# %% [markdown]
# # ML - Aprendizaje No Supervisado

# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from PIL import Image

# Descargar el dataset
lfw = fetch_lfw_people(min_faces_per_person=0, resize=0.7)

# Crear carpeta para guardar imágenes
folder = "lfw_images"
os.makedirs(folder, exist_ok=True)

# Guardar imágenes como archivos PNG
for i, img_array in enumerate(lfw.images):
    img = Image.fromarray((img_array * 255).astype(np.uint8))  # escala a 0-255
    img.save(os.path.join(folder, f"img_{i}.png"))

# mensaje de éxito
print(f"Imágenes guardadas en la carpeta: {folder}")

# %%
# mostremos el tamaño de una muestra de las imágenes
sample_images = lfw.images[:5]
print("Tamaño de las primeras 5 imágenes:")
for i, img in enumerate(sample_images):
    print(f"Imagen {i}: {img.shape}")
# cantidad de características por imagen
num_features = lfw.images.shape[1] * lfw.images.shape[2]
print(f"Número de características por imagen: {num_features}")

# %%
# 2. Crear CSV a partir de las imágenes guardadas (aplanadas)
data = []
for filename in sorted(os.listdir(folder)):
    filepath = os.path.join(folder, filename)
    img = Image.open(filepath).convert("L")  # Convertir a escala de grises
    img_array = np.array(img).flatten()      # Aplanar imagen
    data.append(img_array)

data = np.array(data)

# Guardar CSV sin etiquetas
df = pd.DataFrame(data)
df.to_csv("lfw_no_labels.csv", index=False)

print(f"CSV generado con shape {df.shape}")

# %%
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar CSV sin etiquetas
df = pd.read_csv("lfw_no_labels.csv")

# Aplicar PCA a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.values)

# Graficar
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Datos LFW proyectados en 2D con PCA")
plt.grid(True)
plt.show()


# %%
# cargamos el csv en una X
X = pd.read_csv("lfw_no_labels.csv").values
# y mostramos el shape
print(f"Shape de X: {X.shape}")

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para reducción de dimensionalidad
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Rango de k para clustering
K = range(2, 15)

inertia = []
silhouette_scores = []
kmeans_per_k = {}

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    kmeans_per_k[k] = kmeans  # Guardar modelo
    inertia.append(kmeans.inertia_)
    labels = kmeans.labels_
    sil_score = silhouette_score(X_pca, labels)
    silhouette_scores.append(sil_score)

# Detección automática del mejor k
second_derivative = np.diff(inertia, 2)  # segunda derivada discreta
best_k_elbow = K[np.argmin(second_derivative) + 1]  # +1 por desfase
best_k_silhouette = K[np.argmax(silhouette_scores)]

print(f"Mejor k según método del codo (aprox.): {best_k_elbow}")
print(f"Mejor k según Silhouette Score: {best_k_silhouette}")

# Graficar método del codo y silhouette score
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertia, 'bo-', label='Inertia')
plt.axvline(x=best_k_elbow, color='green', linestyle='--', label=f'Mejor k={best_k_elbow}')
plt.xlabel('Número de clusters k')
plt.ylabel('Inertia (Suma de distancias al cuadrado)')
plt.title('Método del Codo')
plt.legend()

plt.subplot(1,2,2)
plt.plot(K, silhouette_scores, 'ro-', label='Silhouette Score')
plt.axvline(x=best_k_silhouette, color='green', linestyle='--', label=f'Mejor k={best_k_silhouette}')
plt.xlabel('Número de clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Visualizar clustering para todos los k, resaltando los mejores
plt.figure(figsize=(20, 12))

for idx, k in enumerate(K, start=1):
    labels = kmeans_per_k[k].labels_
    plt.subplot(4, 4, idx)  # 4x4 = 16 espacios para 13 gráficos
    edge_color = 'red' if (k == best_k_elbow or k == best_k_silhouette) else 'none'
    lw = 2 if (k == best_k_elbow or k == best_k_silhouette) else 0

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', s=10, alpha=0.7,
                          edgecolor=edge_color, linewidth=lw)
    plt.title(f'k={k}' + (f' ← Mejor' if k == best_k_elbow or k == best_k_silhouette else ''))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

k = 2
kmeans = kmeans_per_k[k]
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

n_rep = 5
fig, axes = plt.subplots(k, n_rep, figsize=(20, 8))
fig.suptitle(f'Imágenes representativas para k={k}', fontsize=20)

for cluster_id in range(k):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_points = X_pca[cluster_indices]

    distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
    closest_indices = cluster_indices[np.argsort(distances)[:n_rep]]

    for i, idx_img in enumerate(closest_indices):
        ax = axes[cluster_id, i]
        img = X[idx_img].reshape(87, 65)  # Tamaño imagen LFW (87x65)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_id}', fontsize=12)
        if i == 0:
            ax.set_ylabel(f'Cluster {cluster_id}', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# # para 7 clusters

# %%
import matplotlib.pyplot as plt
import numpy as np

k = 7
kmeans = kmeans_per_k[k]
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

n_rep = 5
fig, axes = plt.subplots(k, n_rep, figsize=(50, 24))
fig.suptitle(f'Imágenes representativas para k={k}', fontsize=20)

for cluster_id in range(k):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_points = X_pca[cluster_indices]

    distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
    closest_indices = cluster_indices[np.argsort(distances)[:n_rep]]

    for i, idx_img in enumerate(closest_indices):
        ax = axes[cluster_id, i]
        img = X[idx_img].reshape(87, 65)  # Tamaño imagen LFW (87x65)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_id}', fontsize=12)
        if i == 0:
            ax.set_ylabel(f'Cluster {cluster_id}', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()



