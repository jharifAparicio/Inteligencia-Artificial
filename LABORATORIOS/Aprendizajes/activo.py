# %% [markdown]
# # Aprendizaje activo

# %% [markdown]
# instalamos librerias

# %%
!pip install torch torchvision matplotlib scikit-learn

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% [markdown]
# descargar el dataset

# %%
# 1. Descargar CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# %% [markdown]
# convertir a array

# %%
class_map = {
    0: 'avioneta',
    1: 'automóvil',
    2: 'pájaro',
    3: 'gato',
    4: 'ciervo',
    5: 'perro',
    6: 'rana',
    7: 'caballo',
    8: 'barco',
    9: 'camión'
}


# %%
# 2. Convertir a arrays
X_train = torch.stack([img for img, _ in trainset]).numpy()

# %%
# 3. Preprocesar: normalizar y aplanar
X_train_flat = X_train[:1000].reshape(1000, -1)

# %%
#4 Aplanar imágenes para LogisticRegression
N, C, H, W = X_train.shape
X_train_flat = X_train.reshape(N, -1)

# %%
# Diccionario de clases para mostrar nombre al usuario
class_map = {
    0: 'avioneta', 1: 'automóvil', 2: 'pájaro', 3: 'gato', 4: 'ciervo',
    5: 'perro', 6: 'rana', 7: 'caballo', 8: 'barco', 9: 'camión'
}

#5 Seleccionar 20 imágenes al azar para etiquetar manualmente
np.random.seed(42)
initial_idxs = np.random.choice(N, size=20, replace=False)

X_labeled = X_train_flat[initial_idxs]

# 6. Etiquetar manualmente con input()
y_labeled = []
print("Escribe la etiqueta correcta para cada imagen (usa el número de la clase):")
for i, idx in enumerate(initial_idxs):
    img = X_train[idx].transpose(1, 2, 0)  # C,H,W a H,W,C
    plt.imshow(img)
    plt.title(f"Imagen {i+1}")
    plt.axis('off')
    plt.show()

    print("Opciones de etiquetas:")
    for key, val in class_map.items():
        print(f"{key}: {val}")
    
    while True:
        try:
            label = int(input(f"Etiqueta para imagen {i+1} (0-avioneta,1-auto,...,9-camión): "))
            if label in class_map:
                y_labeled.append(label)
                break
            else:
                print("Etiqueta inválida. Intenta de nuevo.")
        except:
            print("Entrada inválida. Intenta con un número entre 0 y 9.")

y_labeled = np.array(y_labeled)

# %%
# 7. Entrenar modelo con etiquetas manuales
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_labeled, y_labeled)

print("Modelo entrenado con tus etiquetas manuales.")

# %% [markdown]
# # repetir las siguientes 2 celdas una y otravez hasta estar seguros que funciona

# %%
# 8. Calcular incertidumbre en el resto de imágenes (sin etiquetar)
unlabeled_mask = np.ones(N, dtype=bool)
unlabeled_mask[initial_idxs] = False
X_unlabeled = X_train_flat[unlabeled_mask]

probas = model.predict_proba(X_unlabeled)
uncertainty = 1 - probas.max(axis=1)

top_uncertain_idxs = np.argsort(uncertainty)[-10:]
print("Índices de las 10 imágenes con mayor incertidumbre (sin etiquetar):", top_uncertain_idxs)

# %%
# 9. Obtener las imágenes con mayor incertidumbre
num_to_label = 10
top_uncertain_idxs = np.argsort(uncertainty)[-num_to_label:]

# 10. Mostrar imágenes con mayor incertidumbre para etiquetar
print("Por favor etiqueta estas imágenes con mayor incertidumbre:")
new_labels = []
for i, idx in enumerate(top_uncertain_idxs):
    img = X_unlabeled[idx].reshape(C, H, W).transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Imagen incierta {i+1}")
    plt.axis('off')
    plt.show()

    print("Opciones de etiquetas:")
    for key, val in class_map.items():
        print(f"{key}: {val}")

    while True:
        try:
            label = int(input(f"Etiqueta para imagen incierta {i+1}: "))
            if label in class_map:
                new_labels.append(label)
                break
            else:
                print("Etiqueta inválida, intenta de nuevo.")
        except:
            print("Entrada inválida, ingresa un número entre 0 y 9.")

new_labels = np.array(new_labels)

# 11. Actualizar conjunto de etiquetas
# Convierte índices relativos en X_unlabeled a índices reales en X_train
real_indices = np.where(unlabeled_mask)[0][top_uncertain_idxs]

# 12. Actualizar arrays de datos etiquetados
X_labeled = np.vstack([X_labeled, X_train_flat[real_indices]])
y_labeled = np.concatenate([y_labeled, new_labels])

# 13. Reentrenar el modelo con las etiquetas nuevas
model.fit(X_labeled, y_labeled)
print("Modelo reentrenado con nuevas etiquetas.")


# %% [markdown]
#     'avioneta': 0,
#     'automóvil': 1,
#     'pájaro': 2,
#     'gato': 3,
#     'ciervo': 4,
#     'perro': 5,
#     'rana': 6,
#     'caballo': 7,
#     'barco': 8,
#     'camión': 9

# %%
import random
import matplotlib.pyplot as plt

# Seleccionar 10 imágenes aleatorias del conjunto etiquetado manualmente
indices = random.sample(range(len(X_labeled)), 10)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(indices):
    img = X_labeled[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # Si CIFAR-10: 3 canales, 32x32
    true_label = class_map[y_labeled[idx]]
    pred_label = class_map[model.predict(X_labeled[idx].reshape(1, -1))[0]]

    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Manual: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.show()


# %%
import random
import matplotlib.pyplot as plt

indices = random.sample(range(len(X_train_flat)), 10)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(indices):
    img = X_train[idx].transpose(1, 2, 0)  # CIFAR-10: convertir (3,32,32) a (32,32,3)
    pred_label = class_map[model.predict(X_train_flat[idx].reshape(1, -1))[0]]

    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Predicción: {pred_label}")
    plt.axis('off')
plt.show()



