import csv

# Leer datos desde un archivo CSV
def leer_datos_csv(nombre_archivo):
    data = []
    with open(nombre_archivo, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        next(lector)  # Saltar la cabecera
        for fila in lector:
            edad, calorias = map(int, fila)
            data.append((edad, calorias))
    return data

data = leer_datos_csv('./predicciones/calorias.csv')

print("Datos leídos del archivo:")
for edad, calorias in data:
    print(f"Edad: {edad}, Calorías consumidas: {calorias}")

# Predicción simple basada en el promedio de edades similares
def predecir_calorias(edad, data):
    edades_similares = [cal for e, cal in data if abs(e - edad) <= 5]
    return sum(edades_similares) // len(edades_similares) if edades_similares else sum(c[1] for c in data) // len(data)

# Hacer predicciones
test_edades = [20, 25, 30, 40, 50, 55]
for edad in test_edades:
    pred = predecir_calorias(edad, data)
    print(f"Edad: {edad}, Predicción de calorías consumidas: {pred}")