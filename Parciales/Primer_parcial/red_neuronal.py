# %% [markdown]
# # 1. cargamos los datos

# %%
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix

# %%
def load_sparse_data(file_path, num_features):
    """
    Carga datos dispersos desde un archivo .data y los convierte en una matriz dispersa CSR.
    
    Parámetros:
    - file_path: Ruta al archivo .data.
    - num_features: Número total de características en el dataset.
    
    Retorna:
    - Una matriz dispersa CSR que representa los datos.
    """
    data = []
    indices = []
    indptr = [0]
    
    with open(file_path, 'r') as f:
        for line in f:
            # Dividir la línea en índices
            feature_indices = line.strip().split()
            
            # Convertir índices a enteros y ajustar a base 0
            for idx in feature_indices:
                indices.append(int(idx) - 1)  # Índices son 1-based, convertir a 0-based
                data.append(1.0)  # Todos los valores son implícitamente 1
            
            indptr.append(len(data))
    
    # Crear la matriz dispersa CSR
    sparse_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, num_features))
    return sparse_matrix

# Número total de características en el dataset Dorothea
num_features = 100000  # Este valor debe ajustarse según el dataset

# Cargar datos de entrenamiento
X_train = load_sparse_data('data/dorothea_train.data', num_features)
y_train = np.loadtxt('data/dorothea_train.labels')  # Etiquetas de entrenamiento

# Cargar datos de validación
X_valid = load_sparse_data('data/dorothea_valid.data', num_features)
y_valid = np.loadtxt('data/dorothea_valid.labels')  # Etiquetas de validación

# Cargar datos de prueba
X_test = load_sparse_data('data/dorothea_test.data', num_features)

# Verificar formas
print(f"Entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Validación: {X_valid.shape}, {y_valid.shape}")
print(f"Prueba: {X_test.shape}")

# %% [markdown]
# # 2. Normalización de las características

# %%
from sklearn.preprocessing import MaxAbsScaler

# Crear el escalador
scaler = MaxAbsScaler()

# Ajustar el escalador con los datos de entrenamiento y transformar todos los conjuntos
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Verificar que los datos sigan siendo dispersos
print(f"Entrenamiento (disperso): {X_train_scaled.shape}")
print(f"Validación (disperso): {X_valid_scaled.shape}")
print(f"Prueba (disperso): {X_test_scaled.shape}")

y_train = np.where(y_train == -1, 0, y_train)  # Convertir -1 a 0
y_valid = np.where(y_valid == -1, 0, y_valid)  # Convertir -1 a 0

# %% [markdown]
# # 3. Convertir en tensores de pytorch

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convertir matrices dispersas a tensores densos
X_train_tensor = torch.tensor(X_train_scaled.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_valid_tensor = torch.tensor(X_valid_scaled.toarray(), dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled.toarray(), dtype=torch.float32)

# Crear datasets y dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Verificar formas de los tensores
print(f"Entrenamiento (tensor): {X_train_tensor.shape}, {y_train_tensor.shape}")
print(f"Validación (tensor): {X_valid_tensor.shape}, {y_valid_tensor.shape}")
print(f"Prueba (tensor): {X_test_tensor.shape}")

# %% [markdown]
# # 4. Definimos el modelo en pythorch

# %%
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # Capa oculta 1
        self.relu = nn.ReLU()         # Función de activación     
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(32, 16)         # Capa oculta 2
        self.fc3 = nn.Linear(16, 1)           # Capa de salida
        self.sigmoid = nn.Sigmoid()           # Activación para clasificación binaria

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Inicializar el modelo
input_size = X_train_scaled.shape[1]  # Número de características
model = NeuralNet(input_size)

# Definir función de pérdida y optimizador
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
# Modificar el optimizador para incluir weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# Verificar la arquitectura del modelo
print(model)

# %%
# 3. Verificar GPU y mover el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo seleccionado: {device}")

# limpiar_memoria()  # Libera memoria de la GPU antes de mover el modelo
model = model.to(device)  # Mueve el modelo a GPU/CPU
print("¡Modelo enviado a", device, "!")

# %%
import time
import psutil
import GPUtil

def verificar_temperatura():
    """
    Verifica temperaturas de CPU/GPU:
    1. Si >80°C: Pausa el entrenamiento
    2. Espera hasta que ambas <=60°C
    3. Continúa el entrenamiento
    """
    while True:
        # Obtener temperaturas actuales
        cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
        gpu_temp = GPUtil.getGPUs()[0].temperature

        print(f"Monitor: CPU={cpu_temp}°C | GPU={gpu_temp}°C")

        # Si está dentro del rango seguro (<=80°C), continuar
        if cpu_temp <= 90 and gpu_temp <= 90:
            return

        # Si excede 80°C, esperar hasta <=60°C
        print(f"⚠️ Pausado: Temperaturas altas (CPU={cpu_temp}°C, GPU={gpu_temp}°C)")
        while True:
            time.sleep(10)  # Esperar 10 segundos
            cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
            gpu_temp = GPUtil.getGPUs()[0].temperature
            print(f"Esperando... CPU={cpu_temp}°C | GPU={gpu_temp}°C")

            # Verificar si bajó a 60°C o menos
            if cpu_temp <= 70 and gpu_temp <= 70:
                print("✅ Temperaturas seguras. Reanudando entrenamiento...")
                return

# %% [markdown]
# # 5. Entrenamiento del modelo

# %%
import matplotlib.pyplot as plt

# Entrenamiento
epochs = 12
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(epochs):
    # Modo entrenamiento
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Mover a GPU/CPU
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Modo evaluación (validación)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mover a GPU/CPU
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = running_loss / len(valid_loader)
    valid_accuracy = correct / total
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    verificar_temperatura()  # Verificar temperatura antes de continuar
    time.sleep(5)  # Esperar 5 segundos entre épocas

# Gráficas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='orange')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
# guardamos el modelo
torch.save(model.state_dict(), 'modelo_dorothea.pth')

# %%
#cargar el modelo
model.load_state_dict(torch.load('modelo_dorothea.pth'))

# %%
model.eval()
predictions = []

with torch.no_grad():
    for inputs in test_loader:
        # Mover los inputs al mismo dispositivo que el modelo (GPU/CPU)
        inputs_device = inputs[0].to(device)
        outputs = model(inputs_device).squeeze()
        # Mover de vuelta a CPU para operaciones con numpy
        predicted = (outputs > 0.5).float().cpu()  # Predicciones en [0, 1]
        predictions.extend(predicted.numpy())

# Convertir predicciones: 0 -> -1, 1 -> 1
predictions = [-1 if pred == 0 else 1 for pred in predictions]

# Guardar predicciones en un archivo
np.savetxt('dorothea_predictions.txt', predictions, fmt='%d')
print("Predicciones guardadas en 'dorothea_predictions.txt'")

# %%



