import numpy as np
import matplotlib.pyplot as plt


def graficar_error_tiempo(error_entrenamiento, error_prueba):
    time = np.arange(len(error_entrenamiento))  # Generar un arreglo de tiempo basado en el tamaño del error de entrenamiento

    # Crear la gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(time, error_entrenamiento, label='Error de Entrenamiento', color='blue')
    plt.plot(time, error_prueba, label='Error de Prueba', color='orange')
    
    plt.title('Gráfica de Error a lo largo del Tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.show()