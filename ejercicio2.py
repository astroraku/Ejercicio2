import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd

def cargar_dataset():
    ruta_archivo = input("Ingrese la ruta del archivo del dataset: ")
    try:
        # Cargar el conjunto de datos desde el archivo
        dataset = pd.read_csv(ruta_archivo)
        return dataset
    except FileNotFoundError:
        print("¡Archivo no encontrado! Por favor, verifique la ruta e intente nuevamente.")
        return None

def generar_particiones(dataset, num_particiones, porcentaje_entrenamiento):
    particiones = []

    for i in range(num_particiones):
        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], 
                                                            test_size=1 - porcentaje_entrenamiento, 
                                                            random_state=i)
        
        # Crear y entrenar el perceptrón
        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)
        
        # Realizar predicciones en el conjunto de prueba
        y_pred = perceptron.predict(X_test)
        
        # Calcular la precisión y almacenar la partición
        accuracy = accuracy_score(y_test, y_pred)
        particion = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'accuracy': accuracy}
        particiones.append(particion)

    return particiones

# Ejemplo de uso
dataset = cargar_dataset()

if dataset is not None:
    # Parámetros
    num_particiones = int(input("Ingrese la cantidad de particiones: "))
    porcentaje_entrenamiento = float(input("Ingrese el porcentaje de patrones de entrenamiento (0.0 a 1.0): "))

    # Generar particiones
    particiones = generar_particiones(dataset, num_particiones, porcentaje_entrenamiento)

    # Mostrar resultados
    for i, particion in enumerate(particiones):
        print(f'Partición {i + 1}:')
        print(f'Precisión: {particion["accuracy"]:.2f}')
        print(f'Tamaño conjunto de entrenamiento: {len(particion["X_train"])}')
        print(f'Tamaño conjunto de prueba: {len(particion["X_test"])}\n')






