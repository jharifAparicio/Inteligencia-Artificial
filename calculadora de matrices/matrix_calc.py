import numpy as np

def generar_matriz(n):
    return np.random.randint(0, 10, (n, n))

def multiplicar_matrices(matriz1, matriz2):
    return np.dot(matriz1, matriz2)

def main():
    n = int(input("Ingrese el tamaño de la matriz (N): "))
    matriz1 = generar_matriz(n)
    matriz2 = generar_matriz(n)
    
    print("\nMatriz 1:")
    print(matriz1)
    print("\nMatriz 2:")
    print(matriz2)
    
    resultado = multiplicar_matrices(matriz1, matriz2)
    
    print("\nMatriz Resultado de la Multiplicación:")
    print(resultado)

if __name__ == "__main__":
    main()