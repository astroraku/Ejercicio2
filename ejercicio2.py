import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd

def cargar_conjunto_datos():
    ruta_archivo = input("Ingrese la ruta del archivo del conjunto de datos: ")
    try:
        conjunto_datos = pd.read_csv(ruta_archivo)
        return conjunto_datos
    except FileNotFoundError:
        print("¡Archivo no encontrado! Por favor, verifique la ruta e intente nuevamente.")
        return None

def generar_particiones(conjunto_datos, num_particiones, porcentaje_entrenamiento):
    particiones = []

    for i in range(num_particiones):
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            conjunto_datos.iloc[:, :-1], conjunto_datos.iloc[:, -1],
            test_size=1 - porcentaje_entrenamiento, 
            random_state=i
        )
        
        perceptron = Perceptron()
        perceptron.fit(X_entrenamiento, y_entrenamiento)
        
        y_prediccion = perceptron.predict(X_prueba)
        
        precision = accuracy_score(y_prueba, y_prediccion)
        particion = {'X_entrenamiento': X_entrenamiento, 'X_prueba': X_prueba, 
                     'y_entrenamiento': y_entrenamiento, 'y_prueba': y_prueba, 'precision': precision}
        particiones.append(particion)

    return particiones

conjunto_datos = cargar_conjunto_datos()

if conjunto_datos is not None:
    num_particiones = int(input("Ingrese la cantidad de particiones: "))
    porcentaje_entrenamiento = float(input("Ingrese el porcentaje de patrones de entrenamiento (0.0 a 1.0): "))

    particiones = generar_particiones(conjunto_datos, num_particiones, porcentaje_entrenamiento)

    for i, particion in enumerate(particiones):
        print(f'Partición {i + 1}:')
        print(f'Precisión: {particion["precision"]:.2f}')
        print(f'Tamaño conjunto de entrenamiento: {len(particion["X_entrenamiento"])}')
        print(f'Tamaño conjunto de prueba: {len(particion["X_prueba"])}\n')







