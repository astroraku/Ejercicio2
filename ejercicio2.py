import numpy as np
import matplotlib.pyplot as plt

def entrenar_perceptron_simple(entradas, objetivos, tasa_aprendizaje, max_epocas, umbral_convergencia):
    # Función de entrenamiento del perceptrón simple
    tamano_entrada = entradas.shape[1]
    pesos = np.random.rand(tamano_entrada)
    epocas = 0

    while epocas < max_epocas:
        salidas = np.dot(entradas, pesos)
        errores = objetivos - np.where(salidas > 0, 1, 0)

        if np.sum(np.abs(errores)) < umbral_convergencia:
            print(f"Convergencia alcanzada en la época {epocas + 1}.")
            break

        pesos += tasa_aprendizaje * np.dot(errores, entradas)
        epocas += 1

    return pesos

def predecir_perceptron_simple(entradas, pesos):
    # Función de predicción del perceptrón simple
    return np.where(np.dot(entradas, pesos) > 0, 1, 0)

def cargar_datos(ruta_archivo):
    # Función para cargar datos desde un archivo CSV
    datos = np.loadtxt(ruta_archivo, delimiter=',')
    entradas = datos[:, :-1]
    objetivos = datos[:, -1]
    return entradas, objetivos

def crear_particiones_entrenamiento_prueba(entradas, objetivos, porcentaje_entrenamiento):
    # Función para crear particiones de entrenamiento y prueba
    num_muestras = len(objetivos)
    num_muestras_entrenamiento = int(num_muestras * porcentaje_entrenamiento)
    
    indices = np.arange(num_muestras)
    np.random.shuffle(indices)
    
    indices_entrenamiento = indices[:num_muestras_entrenamiento]
    indices_prueba = indices[num_muestras_entrenamiento:]
    
    entradas_entrenamiento, objetivos_entrenamiento = entradas[indices_entrenamiento], objetivos[indices_entrenamiento]
    entradas_prueba, objetivos_prueba = entradas[indices_prueba], objetivos[indices_prueba]
    
    return entradas_entrenamiento, objetivos_entrenamiento, entradas_prueba, objetivos_prueba

def calcular_precision(predicciones, objetivos):
    # Función para calcular la precisión
    return np.sum(predicciones == objetivos) / len(objetivos)

def main():
    # Solicitar el nombre del archivo de datos
    nombre_archivo = input("Ingrese el nombre del archivo de datos (con extensión, por ejemplo: dataset.csv): ")
    ruta_archivo = nombre_archivo

    try:
        # Cargar el conjunto de datos
        entradas, objetivos = cargar_datos(ruta_archivo)
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {nombre_archivo}. Verifique el nombre y la ubicación del archivo.")
        return
    except Exception as e:
        print(f"Error al cargar el archivo {nombre_archivo}: {e}")
        return

    # Solicitar la cantidad de particiones
    num_particiones = int(input("Ingrese la cantidad de particiones a generar: "))

    # Solicitar el porcentaje de patrones de entrenamiento
    porcentaje_entrenamiento = float(input("Ingrese el porcentaje de patrones de entrenamiento (por ejemplo, 80 para 80%): ")) / 100

    # Parámetros de entrenamiento del perceptrón
    tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje para el perceptrón: "))
    max_epocas = int(input("Ingrese el número máximo de épocas de entrenamiento para el perceptrón: "))
    umbral_convergencia = float(input("Ingrese el umbral de convergencia para el perceptrón: "))

    precisiones = []

    for particion in range(num_particiones):
        # Crear particiones de entrenamiento y prueba
        entradas_entrenamiento, objetivos_entrenamiento, entradas_prueba, objetivos_prueba = crear_particiones_entrenamiento_prueba(
            entradas, objetivos, porcentaje_entrenamiento
        )

        # Entrenar el perceptrón
        pesos = entrenar_perceptron_simple(
            entradas_entrenamiento, objetivos_entrenamiento, tasa_aprendizaje, max_epocas, umbral_convergencia
        )

        # Realizar predicciones en la partición de prueba
        predicciones = predecir_perceptron_simple(entradas_prueba, pesos)

        # Calcular y almacenar la precisión
        precision = calcular_precision(predicciones, objetivos_prueba)
        precisiones.append(precision)

        # Mostrar información sobre la partición actual
        print(f"\nPartición {particion + 1}:")
        print(f"Muestras de entrenamiento: {len(objetivos_entrenamiento)}")
        print(f"Muestras de prueba: {len(objetivos_prueba)}")
        print(f"Precisión del perceptrón en datos de prueba: {precision * 100:.2f}%")

    # Mostrar la gráfica de barras de precisión
    plt.bar(range(1, num_particiones + 1), precisiones, color='blue', alpha=0.7)
    plt.title('Precisión del Perceptrón en Múltiples Particiones')
    plt.xlabel('Partición')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)  # Ajustar el rango del eje y entre 0 y 1
    plt.show()

if __name__ == "__main__":
    main()



