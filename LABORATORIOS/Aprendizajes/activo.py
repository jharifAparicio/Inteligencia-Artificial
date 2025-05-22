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
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# %% [markdown]
# convertir a array

# %%
class_map = {
    'avioneta': 0,
    'automóvil': 1,
    'pájaro': 2,
    'gato': 3,
    'ciervo': 4,
    'perro': 5,
    'rana': 6,
    'caballo': 7,
    'barco': 8,
    'camión': 9
}

# %%
# 2. Convertir a arrays
X_train = torch.stack([img for img, _ in trainset]).numpy()
y_train = torch.tensor([label for _, label in trainset]).numpy()
X_test = torch.stack([img for img, _ in testset]).numpy()
y_test = torch.tensor([label for _, label in testset]).numpy()

# %%
# 3. Preprocesar: normalizar y aplanar
X_train_flat = X_train[:1000].reshape(1000, -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# %%
# 4. Entrenar modelo base
log_reg = LogisticRegression(max_iter=1000, multi_class='ovr')
# el fit que es el que entrena el modelo
log_reg.fit(X_train_flat, y_train[:1000])

# %%
# 5. Calcular incertidumbre (confianza)
probas = log_reg.predict_proba(X_train_flat)
pred_labels = np.argmax(probas, axis=1)
confidences = np.max(probas, axis=1)

# %%
# 6. Elegir k muestras menos confiables mayor incertidumbre
muestras_inciertas = 15
sorted_ixs = np.argsort(confidences)
selected_ixs = sorted_ixs[:muestras_inciertas]

# %%
# 7. Mostrar imágenes
classes = trainset.classes
plt.figure(figsize=(20, 5))
for i, idx in enumerate(selected_ixs):
    img = X_train[idx].transpose(1, 2, 0)
    plt.subplot(1, muestras_inciertas, i + 1)
    plt.imshow(img)
    plt.title(f"Idx {idx}")
    plt.axis('off')
plt.tight_layout()
plt.show()

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
manual_labels = []
for i, idx in enumerate(selected_ixs):
    label_text = input("Ingrese etiqueta en numero: ")
    manual_labels.append(int(label_text))

# %%
# mostrar las etiquetas
print("Etiquetas manuales:", manual_labels)

# %%
# 9. Reentrenar con etiquetas corregidas
y_train_corrected = y_train[:1000].copy()
for i, idx in enumerate(selected_ixs):
    y_train_corrected[idx] = manual_labels[i]

log_reg2 = LogisticRegression(max_iter=1000, multi_class='ovr')
log_reg2.fit(X_train_flat, y_train_corrected)

# %%
# 10. Evaluar
y_pred1 = log_reg.predict(X_test_flat)
y_pred2 = log_reg2.predict(X_test_flat)

print("Precisión original:", accuracy_score(y_test, y_pred1))
print("Precisión tras corrección:", accuracy_score(y_test, y_pred2))


