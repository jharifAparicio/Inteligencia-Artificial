import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import time
from time import sleep
from IPython.display import clear_output
import psutil
import GPUtil
import gc
from PIL import Image
import torchvision.transforms as transforms

dim = 100
bach_size=64

def limpiar_memoria():
    """
    Libera memoria CUDA y realiza recolección de basura.

    - Vacía la caché de la GPU no utilizada con torch.cuda.empty_cache().
    - Ejecuta gc.collect() para la recolección de basura en Python.
    - Si CUDA está disponible, muestra la memoria utilizada y reservada.
    """
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        print(f"Memoria liberada. Disponible: {torch.cuda.memory_allocated()/1024**2:.2f} MB en uso")
        print(f"Memoria total reservada: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

# Llamar a la función para liberar la memoria
limpiar_memoria()

class FruitMLP(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),  # Convierte la imagen (C x H x W) en un vector 1D

            nn.Linear((dim**2)*3, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(2048, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 169)  # Capa de salida
        )

    def forward(self, x):
        return self.layers(x)

# 2. Instanciar el modelo
model = FruitMLP(dim=dim)
print("Modelo creado!")
print(model)

# 3. Verificar GPU y mover el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo seleccionado: {device}")

# limpiar_memoria()  # Libera memoria de la GPU antes de mover el modelo
model = model.to(device)  # Mueve el modelo a GPU/CPU
print("¡Modelo enviado a", device, "!")


# Transformación TEMPORAL para cálculo (sin normalización)
temp_transform = transforms.Compose([
    transforms.Resize((dim, dim)),
    transforms.ToTensor(),
])

# Cargar el conjunto de datos de entrenamiento en la GPU
train_dataset_temp = datasets.ImageFolder(
    root='../fruits-360/Training',
    transform=temp_transform
)

train_loader_temp = torch.utils.data.DataLoader( # dataLolder -> es un objeto iterable
    train_dataset_temp,
    batch_size=bach_size,  # Tamaño del lote
    shuffle=False,   # Barajar los datos en False
)
# Cálculo de la media y desviación estándar
mean = torch.zeros(3).to(device)  # Inicializa el tensor de media
std = torch.zeros(3).to(device)   # Inicializa el tensor de desviación estándar
total_pixels = 0  # Inicializa el contador de píxeles

# Iterar sobre el conjunto de datos de entrenamiento
# para calcular la media y desviación estándar
# (sin normalización)
for images, _ in train_loader_temp:
    images = images.to(device)  # Manda batch a GPU
    batch_pixels = images.size(0) * images.size(2) * images.size(3)  # batch * altura * ancho
    images = images.view(images.size(0), 3, -1)  # [batch, canales, pixels]
    
    mean += images.mean(2).sum(0)  # Suma medias por canal
    std += images.std(2).sum(0)    # Suma std por canal
    total_pixels += batch_pixels

mean /= len(train_loader_temp.dataset)
std /= len(train_loader_temp.dataset)

print("Media (Train):", mean.cpu().tolist()) # media 
print("Std (Train):", std.cpu().tolist()) # desviación estándar / standard deviation

# 1. Extraer todas las etiquetas del DataLoader
all_labels = []

for _, labels in train_loader_temp:
    all_labels.extend(labels.tolist())  # Convertir a lista y agregar

# 2. Contar muestras por clase
class_counts = torch.tensor(list(Counter(all_labels).values()), dtype=torch.float32)
print("Muestras por clase:", class_counts)



# Distribución de clases
plt.bar(range(len(class_counts)), class_counts.cpu().numpy())
plt.title("Muestras por clase")
plt.xlabel("Clase")
plt.ylabel("Número de muestras")
plt.show()

# valores calculados
# Media = mean
# Std = std

Media = [0.6726435422897339, 0.5792443752288818, 0.508468508720398]
Std = [0.26989850401878357, 0.32609033584594727, 0.3682645261287689]

#Media = [0.6726435422897339, 0.5792443752288818, 0.508468508720398]
#Std = [0.26989850401878357, 0.32609033584594727, 0.3682645261287689]

# Normalización de los datos
transform = transforms.Compose([
    transforms.Resize((dim, dim)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Media, std=Std)  # Normalización crítica
])

# Cargar el conjunto de datos de entrenamiento y Test
train_dataset = datasets.ImageFolder( # el ImageFolder es un objeto iterable
    # Cargar el conjunto de datos de entrenamiento
    root= "../fruits-360/Training",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="../fruits-360/Test",
    transform=transform
)

# Creamos DataLoader para cargar los datos / que es el dataLoader -> 
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=bach_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=bach_size,
    shuffle=False,
)

# Verificación
print(f"Clases: {len(train_dataset.classes)}")
print(f"Imágenes en entrenamiento: {len(train_dataset)}")
print(f"Imágenes en test: {len(test_dataset)}")

print(f"Clases: {train_dataset.classes}")
print("indice de clases:", train_dataset.class_to_idx)

print(f"clases de test: {test_dataset.classes}")
print("indice de clases:", test_dataset.class_to_idx)

# El DataLoader en PyTorch es una utilidad fundamental que se encarga de cargar, organizar y procesar tus datos de manera eficiente durante el entrenamiento de tu red neuronal.

def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))  # Convertir a HWC
    img = img * std.cpu().numpy() + mean.cpu().numpy()  # Desnormalizar
    img = np.clip(img, 0, 1)                     # Asegurar rango [0,1]
    plt.imshow(img)
    plt.axis('off')

# Mostrar primera imagen del batch
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(images[0])
print(f"Etiqueta: {train_dataset.classes[labels[0]]}")

# Mostrar primera imagen del batch
dataiter = iter(test_loader)
images, labels = next(dataiter)
imshow(images[0])
print(f"Etiqueta: {train_dataset.classes[labels[0]]}")

for images, labels in train_loader:
    print("labels en entrenamiento", labels)
    break

# 1. Optimizador: SGD (Descenso de Gradiente Estocástico)
# Explicación breve para el curso:
# El SGD (Stochastic Gradient Descent) es uno de los optimizadores más básicos en Deep Learning. Su función es ajustar los pesos de la red neuronal mediante el cálculo del gradiente (derivada) de la función de pérdida.
# 
# Ventajas:
# 
# Simple y fácil de entender.
# 
# Funciona bien en muchos problemas clásicos.
# 
# Permfine ajustar manualmente la tasa de aprendizaje (learning rate).
# 
# Hiperparámetros clave:
# 
# lr (Learning Rate): Controla el tamaño de los pasos en la actualización de los pesos (demasiado alto → inestabilidad; demasiado bajo → lento).
# 
# momentum (opcional): Ayuda a evitar mínimos locales y acelera la convergencia.
# 
# 2. Función de Pérdida (Loss Function)
# La más básica y necesaria: CrossEntropyLoss
# 
# ¿Qué hace?
# 
# Mide qué tan lejos están las predicciones de la red (logits) de las etiquetas reales.
# 
# Combina LogSoftmax + NLLLoss (Negative Log Likelihood Loss) en una sola función.
# 
# Ideal para problemas de clasificación multiclase (como tu caso con 170 clases).
# 
# 

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9, nesterov=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
print("¡Optimizador y función de pérdida creados!")
print("optimizador", optimizer)
print("función de pérdida", criterion)
print("Configuración del scheduler:")
print(f"  mode: {scheduler.mode}")
print(f"  patience: {scheduler.patience}")
print(f"  factor: {scheduler.factor}")
print(f"  threshold: {scheduler.threshold}")
print(f"  cooldown: {scheduler.cooldown}")
print(f"  min_lr: {scheduler.min_lrs}")
print(f"  verbose: {scheduler.verbose}")

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
        if cpu_temp <= 95 and gpu_temp <= 95:
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

# # funcion de entrenamiento

def entrenar_epoca(modelo, cargador, optimizador, criterio, dispositivo):
    modelo.train()
    perdida_total = 0.0
    correctas = 0
    total = 0
    
    with tqdm(cargador, unit="batch", desc="Entrenamiento") as barra:
        for imagenes, etiquetas in barra:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            
            optimizador.zero_grad()
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida.backward()
            optimizador.step()
            
            perdida_total += perdida.item()
            _, predichas = torch.max(salidas, 1)
            correctas += (predichas == etiquetas).sum().item()
            total += etiquetas.size(0)
            
            barra.set_postfix(
                perdida=perdida.item(),
                precision=f"{100 * correctas / total:.2f}%"
            )
    
    # calcular el porcentaje de error
    error = 100 * (1 - correctas / total)
    return perdida_total / len(cargador), 100 * correctas / total, error

# # funcion de validacion

def evaluar(modelo, cargador, criterio, dispositivo):
    modelo.eval()  # Modo evaluación
    perdida_total = 0.0
    correctas = 0
    total = 0

    with torch.no_grad():  # Desactiva gradientes
        for imagenes, etiquetas in tqdm(cargador, desc="Prueba"):
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            
            # Métricas
            perdida_total += perdida.item()
            _, predichas = torch.max(salidas, 1)
            correctas += (predichas == etiquetas).sum().item()
            total += etiquetas.size(0)

    error_porcentaje = (1 - correctas / total) * 100
    
    return perdida_total / len(cargador), 100 * correctas / total, error_porcentaje


def graficar_resultados(perdidas_ent, precision_ent, perdidas_test, precision_test, epoca, optimizer):
    # Obtener la tasa de aprendizaje actual del optimizador
    lr_actual = optimizer.param_groups[0]['lr']
    ult_perd_entr = perdidas_ent[-1]
    ult_perd_test = perdidas_test[-1]
    ult_precision_entr = precision_ent[-1]
    ult_precision_test = precision_test[-1]

    # Crear la figura
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(perdidas_ent, label='Entrenamiento')
    plt.plot(perdidas_test, label='Prueba')
    plt.title(f'Pérdida por Época \nTrain: {ult_perd_entr:.4f} | Test: {ult_perd_test:.4f}',pad=20,fontsize=10)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.xticks(range(0, len(perdidas_ent)))
    plt.legend()
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(precision_ent, label='Entrenamiento')
    plt.plot(precision_test, label='Prueba')
    plt.title(f'precision por Época \nTrain: {ult_precision_entr:.4f}% | Test: {ult_precision_test:.4f}%',pad=20,fontsize=10)
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.xticks(range(0, len(precision_ent)))
    plt.legend()
    
    # Añadir el valor de la tasa de aprendizaje en el gráfico
    plt.figtext(0.5, 0.97, f'Tasa de Aprendizaje: {lr_actual:.7f}', ha='center', va='top', fontsize=10, color='blue')

    plt.tight_layout()

    # Guardar imagen
    plt.savefig(f"./graficos de entrenamiento/grafica de progreso epoca {epoca + 1}.png", dpi=300, bbox_inches='tight')

    # Mostrar la gráfica
    plt.show()

# Función para convertir segundos a hh:mm:ss
def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

perdidas_entrenamiento = []
precisiones_entrenamiento = []
perdidas_prueba = []
precisiones_prueba = []

# # --- 5. Bucle de Entrenamiento ---
# Guardar el tiempo de inicio total
start_time_total = time.time()

epocas = 100  # Número de épocas
for epoca in range(epocas):
    torch.cuda.empty_cache()  # Limpia memoria antes de cada época
    print(f"VRAM usada: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    verificar_temperatura()  # Verificar temperatura antes de cada época
    print(f"\nÉpoca {epoca + 1}/{epocas}")
    
    # Entrenamiento
    perdida_ent, precision_ent, train_error = entrenar_epoca(
        model, train_loader, optimizer, criterion, device
    )
    
    perdidas_entrenamiento.append(perdida_ent)
    precisiones_entrenamiento.append(precision_ent)

    # 2. Evaluación en Test (opcional: cada X épocas para ahorrar tiempo)
    if (epoca + 1) % 1 == 0:  # Evalúa cada época
        perdida_test, precision_test, test_error = evaluar(
            model, test_loader, criterion, device
        )
        perdidas_prueba.append(perdida_test)
        precisiones_prueba.append(precision_test)

    clear_output(wait=True)
    lr_actual = optimizer.param_groups[0]['lr']

    # 3. Mostrar resumen
    print(f"\nResumen Época {epoca + 1}:")
    print(f"  Pérdida: Train = {perdida_ent:.4f} | Test = {perdida_test:.4f}")
    print(f"  Precisión: Train = {precision_ent:.2f}% | Test = {precision_test:.2f}%")
    print(f"  Error: entrenamiento: {train_error:.4f}% | error test: {test_error:.4f}%")
    print(f"valor de lr en la presente epoca: {lr_actual:.7f}")
    print("-" * 50)

    graficar_resultados(
        perdidas_entrenamiento, precisiones_entrenamiento,
        perdidas_prueba, precisiones_prueba, epoca, optimizer
    )
    
    if(precision_test > 90 and test_error < 4.5):
        print("Precisión de entrenamiento superior al 95%")
        break

    scheduler.step(precision_test) # se envia la precision_test al scheduler para que ajuste la tasa de aprendizaje

torch.save(model.state_dict(), "modelo_frutas_final_20_epocs.pth")
print("Modelo guardado como 'modelo_frutas_final_20_epocs.pth'")

perdida_final, precision_final, error = evaluar(model, test_loader, criterion, device)
print(f"\n--- Resultado Final en Test ---")
print(f"  Pérdida: {perdida_final:.4f}")
print(f"  Precisión: {precision_final:.2f}%")
print(f"porcentaje de error: {error:.2f}%")

# Tiempo total de entrenamiento
end_time_total = time.time()
total_duration = end_time_total - start_time_total
total_duration_hms = seconds_to_hms(total_duration)
print(f"\nDuración total del entrenamiento: {total_duration_hms}")

# sleep(10)  # Esperar 10 segundos antes de apagar

# os.system("shutdown -h now")

# # Graficar resultados

graficar_resultados(
    perdidas_entrenamiento, precisiones_entrenamiento,
    perdidas_prueba, precisiones_prueba
)

# # Guardar modelo

torch.save(model.state_dict(), "modelo_frutas_final.pth")
print("Modelo guardado como 'modelo_frutas_final.pth'")

# # Evaluación FINAL en Test

perdida_final, precision_final, error = evaluar(model, test_loader, criterion, device)
print(f"\n--- Resultado Final en Test ---")
print(f"  Pérdida: {perdida_final:.4f}")
print(f"  Precisión: {precision_final:.2f}%")
print(f"porcentaje de error: {error:.2f}%")

# creamos las etiquetas
etiquetas = train_dataset.classes  # Etiquetas de las clases
print("Etiquetas:", etiquetas)

# # cargar y probar el modelo con una imagen cualquiera
# 1️⃣ Cargar el modelo entrenado

# Cargar el modelo
modelo_cargado = FruitMLP() # Crear una nueva instancia del modelo
modelo_cargado.load_state_dict(torch.load("modelo_frutas_final.pth")) # Cargar los pesos
modelo_cargado.eval()  # Poner en modo evaluación

modelo_cargado.to(device)  # Mover el modelo al mismo dispositivo que la imagen

# Transformaciones para la imagen (ajustar según tu modelo)
transformaciones = transforms.Compose([
    transforms.Resize((dim , dim)),  # Ajustar tamaño
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean= Media, std= Std)  # Normalizar
])

# Cargar imagen Apple hit 1
imagen = Image.open("C:/Users/OMEN/Downloads/26_100.jpg")  # Reemplaza con tu imagen
# imagen = Image.open("./Avocado 1.jpg")  # Reemplaza con tu imagen
imagen_mostrar = imagen
imagen = transformaciones(imagen)
imagen = imagen.unsqueeze(0)  # Agregar dimensión batch
imagen = imagen.to(device)  # Mover a GPU/CPU
# 3️⃣ Realizar la predicción
# Pasar la imagen al modelo
with torch.no_grad():  # Desactivar gradientes para inferencia
    salida = modelo_cargado(imagen)
    prediccion = torch.argmax(salida, dim=1).item()  # Obtener la clase con mayor probabilidad

if prediccion < len(etiquetas):
    print(f"La imagen fue clasificada como la clase: {etiquetas[prediccion]}")
    print(f"Probabilidad: {torch.softmax(salida, dim=1)[0][prediccion].item() * 100:.2f}%")
    print(f"Error de clasificación: {(1 - torch.softmax(salida, dim=1)[0][prediccion].item()) * 100:.2f}%")
else:
    print("La imagen fue clasificada como 'desconocido'")
plt.imshow(imagen_mostrar)  # Mostrar la imagen original
plt.axis('off')  # Quitar los ejes
plt.show()