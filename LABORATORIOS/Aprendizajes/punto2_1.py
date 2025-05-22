# %%
#%% [markdown]
# # Aprendizaje Semi-Supervisado en CIFAR-10 con PyTorch + KMeans
#
# Pipeline completo:
# 1. Configuración e imports
# 2. Carga y partición de datos (labeled / unlabeled)
# 3. Visualización de ejemplos
# 4. Extracción de features con ResNet18 pretrained
# 5. Método del codo & Silhouette para elegir k
# 6. Clustering + pseudo-etiquetado
# 7. Definición y entrenamiento del clasificador
# 8. Evaluación y ejemplo de predicciones

#%% 1) Imports y configuración
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

# Ajustes generales
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print(f"Dispositivo: {device}")

# %%
#%% 2) Carga y preprocesamiento de CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

data_root = './data'
train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

# Partición: 5 000 imágenes etiquetadas, el resto sin etiquetar
num_labeled = 5000
indices     = np.random.permutation(len(train_set))
labeled_idx = indices[:num_labeled]
unlabeled_idx = indices[num_labeled:]

labeled_set   = Subset(train_set, labeled_idx)
unlabeled_set = Subset(train_set, unlabeled_idx)

batch_size = 128
labeled_loader   = DataLoader(labeled_set, batch_size=batch_size, shuffle=True,  num_workers=0)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader      = DataLoader(test_set,      batch_size=batch_size, shuffle=False, num_workers=0)

# %%
#%% 3) Visualización de ejemplos
classes = train_set.classes

def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.figure(figsize=(6,3))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')

# Mostrar 8 de las etiquetadas
dataiter = iter(labeled_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:8], nrow=8))
print("Clases:", [classes[l] for l in labels[:8]])


# %%
#%% 4) Extracción de features con ResNet18
from torchvision.models import resnet18

feature_extractor = resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.to(device).eval()

def extract_features(loader):
    feats = []
    for imgs, _ in tqdm(loader, desc="Extrayendo features"):
        imgs = imgs.to(device)
        with torch.no_grad():
            out = feature_extractor(imgs).view(imgs.size(0), -1)
        feats.append(out.cpu().numpy())
    return np.vstack(feats)

features_unlab = extract_features(unlabeled_loader)
print("Shape features:", features_unlab.shape)


# %%
#%% 5) Método del codo y Silhouette para k
inertia, sil_scores = [], []
K_range = range(2,50)

for k in tqdm(K_range, desc="Buscando k óptimo"):
    km = KMeans(n_clusters=k, random_state=0).fit(features_unlab)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(features_unlab, km.labels_))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, 'o-')
plt.title("Método del codo")
plt.xlabel("k"); plt.ylabel("Inercia")
plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, 'o-')
plt.title("Silhouette Score")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.tight_layout()
plt.show()

# %%
#%% 6) Clustering final y pseudo-etiquetado
k_opt = 10  # ajusta según tus gráficas
kmeans = KMeans(n_clusters=k_opt, random_state=0).fit(features_unlab)
pseudo_labels = kmeans.labels_

class PseudoLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, subset, pseudo_labels):
        self.subset = subset
        self.pseudo_labels = pseudo_labels
    def __getitem__(self, i):
        img, _ = self.subset[i]
        return img, self.pseudo_labels[i]
    def __len__(self):
        return len(self.subset)

pseudo_set = PseudoLabeledDataset(unlabeled_set, pseudo_labels)
semi_supervised_set = ConcatDataset([labeled_set, pseudo_set])
semi_loader = DataLoader(semi_supervised_set, batch_size=batch_size, shuffle=True, num_workers=0)


# %%
#%% 7) Definición y entrenamiento del modelo
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

model     = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    print(f"\n↪️ Iniciando epoch {epoch+1}/{epochs}")
    for imgs, labels in tqdm(semi_loader, desc="Entrenando batches"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
    avg_loss = running_loss / len(semi_loader.dataset)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# %%
#%% 8) Evaluación en test y visualización
model.eval()
correct = total = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

acc = correct/total
print(f"\n✅ Precisión en test: {acc*100:.2f}%")

# Mostrar algunas predicciones
imgs, labels = next(iter(test_loader))
preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
imshow(torchvision.utils.make_grid(imgs[:8], nrow=8))
print("Verdaderas: ", [classes[l] for l in labels[:8]])
print("Predichas:", [classes[p] for p in preds[:8]])

# %%
#%% 5.1) Determinación automática de la k más representativa

# Encontrar la k con el mayor silhouette score
best_k_idx = np.argmax(sil_scores)
k_optimal = K_range[best_k_idx]
best_silhouette = sil_scores[best_k_idx]

print(f"ANÁLISIS DE K ÓPTIMA:")
print(f"K más representativa: {k_optimal}")
print(f"Mejor Silhouette Score: {best_silhouette:.4f}")
print(f"Posición en el rango evaluado: {best_k_idx+1} de {len(K_range)}")

# Mostrar los top 5 mejores k
top_indices = np.argsort(sil_scores)[-5:][::-1]  # Top 5 en orden descendente
print(f"\n TOP 5 MEJORES K:")
for i, idx in enumerate(top_indices, 1):
    k_val = K_range[idx]
    sil_val = sil_scores[idx]
    print(f"  {i}. k={k_val:2d} → Silhouette = {sil_val:.4f}")

# Visualización mejorada destacando la k óptima
plt.figure(figsize=(15, 5))

# Gráfico del codo con k óptima marcada
plt.subplot(1, 3, 1)
plt.plot(K_range, inertia, 'o-', color='skyblue', markersize=4)
plt.axvline(k_optimal, color='red', linestyle='--', alpha=0.7, label=f'k óptima = {k_optimal}')
plt.title("Método del Codo")
plt.xlabel("k")
plt.ylabel("Inercia")
plt.legend()
plt.grid(True, alpha=0.3)

# Silhouette score con k óptima resaltada
plt.subplot(1, 3, 2)
plt.plot(K_range, sil_scores, 'o-', color='lightgreen', markersize=4)
plt.scatter(k_optimal, best_silhouette, color='red', s=100, zorder=5, label=f'Mejor: k={k_optimal}')
plt.axvline(k_optimal, color='red', linestyle='--', alpha=0.7)
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.legend()
plt.grid(True, alpha=0.3)

# Comparación de métricas normalizadas
plt.subplot(1, 3, 3)
# Normalizar las métricas para compararlas
norm_inertia = [(max(inertia) - x) / (max(inertia) - min(inertia)) for x in inertia]  # Invertir para que mayor sea mejor
norm_sil = [(x - min(sil_scores)) / (max(sil_scores) - min(sil_scores)) for x in sil_scores]

plt.plot(K_range, norm_inertia, 'o-', label='Inercia (normalizada)', alpha=0.7)
plt.plot(K_range, norm_sil, 'o-', label='Silhouette (normalizada)', alpha=0.7)
plt.axvline(k_optimal, color='red', linestyle='--', alpha=0.7, label=f'k óptima = {k_optimal}')
plt.title("Comparación de Métricas")
plt.xlabel("k")
plt.ylabel("Valor Normalizado")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Actualizar k_opt para el resto del código
k_opt = k_optimal
print(f"\n Variable k_opt actualizada a: {k_opt}")

# Información adicional sobre la elección
print(f"\n JUSTIFICACIÓN:")
print(f"   • El silhouette score mide qué tan bien separados están los clusters")
print(f"   • Valores cercanos a 1 indican clusters muy bien definidos")
print(f"   • k={k_optimal} maximiza la cohesión interna y separación entre clusters")
print(f"   • Esta k proporciona el mejor balance para el pseudo-etiquetado")


