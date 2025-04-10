import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generar una imagen de 60x60 con los colores de la bandera de Paraguay (Rojo, Blanco y Azul)
def generar_bandera_paraguay():
    # Crear una imagen de 60x60 con 3 canales (RGB)
    bandera = np.zeros((60, 60, 3), dtype=np.uint8)

    # Dividir la bandera en tres franjas horizontales
    bandera[:20, :] = [0, 0, 255]  # Rojo (BGR)
    bandera[20:40, :] = [255, 255, 255]  # Blanco (BGR)
    bandera[40:, :] = [255, 0, 0]  # Azul (BGR)

    return bandera

# 2. Generar una imagen de 60x60 con colores aleatorios y realizar suma/resta con la bandera
def operaciones_con_imagenes(bandera):
    # Generar una imagen aleatoria
    imagen_aleatoria = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)

    # Suma de imágenes (evitando desbordamiento con cv2.add)
    suma = cv2.add(bandera, imagen_aleatoria)

    # Resta de imágenes (evitando desbordamiento con cv2.subtract)
    resta = cv2.subtract(bandera, imagen_aleatoria)

    return imagen_aleatoria, suma, resta

# 3. Multiplicar la imagen por un escalar usando una tabla de búsqueda (LUT)
def multiplicar_por_escalar(imagen, escalar):
    # Crear una tabla de búsqueda (LUT)
    lut = np.arange(256, dtype=np.uint8) * escalar
    lut = np.clip(lut, 0, 255)  # Evitar desbordamiento

    # Aplicar la LUT a la imagen
    resultado = cv2.LUT(imagen, lut)

    return resultado

# 4. Mostrar las imágenes una a la vez
def mostrar_imagenes_una_a_la_vez(bandera, imagen_aleatoria, suma, resta, multiplicada):
    # Mostrar la bandera de Paraguay
    cv2.imshow("Bandera de Paraguay", bandera)
    cv2.waitKey(0)  # Esperar a que se presione una tecla
    cv2.destroyAllWindows()  # Cerrar la ventana

    # Mostrar la imagen aleatoria
    cv2.imshow("Imagen Aleatoria", imagen_aleatoria)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mostrar la suma de las imágenes
    cv2.imshow("Suma", suma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mostrar la resta de las imágenes
    cv2.imshow("Resta", resta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mostrar la imagen multiplicada por el escalar
    cv2.imshow("Multiplicada por Escalar", multiplicada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. Función para ecualizar una imagen en escala de grises
def ecualizar_imagen(imagen):
    # Convertir la imagen a escala de grises si es a color
    if len(imagen.shape) == 3:  # Si la imagen es a color
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen

    # Ecualizar la imagen en escala de grises
    imagen_ecualizada = cv2.equalizeHist(imagen_gris)

    return imagen_gris, imagen_ecualizada

# 6. Función para mostrar el histograma de una imagen
def mostrar_histograma(imagen, titulo):
    plt.hist(imagen.ravel(), 256, [0, 256])
    plt.title(titulo)
    plt.show()

# Programa principal
if __name__ == "__main__":
    # 1. Generar la bandera de Paraguay
    bandera = generar_bandera_paraguay()

    # 2. Generar imagen aleatoria y realizar suma/resta
    imagen_aleatoria, suma, resta = operaciones_con_imagenes(bandera)

    # 3. Pedir un escalar al usuario y multiplicar la imagen
    escalar = float(input("Ingrese un valor escalar para multiplicar la imagen: "))
    multiplicada = multiplicar_por_escalar(bandera, escalar)

    # 4. Mostrar las imágenes una a la vez
    mostrar_imagenes_una_a_la_vez(bandera, imagen_aleatoria, suma, resta, multiplicada)

    # 5. Ecualizar la imagen de la bandera
    bandera_gris, bandera_ecualizada = ecualizar_imagen(bandera)

    # Mostrar el histograma de la imagen en escala de grises antes de la ecualización
    mostrar_histograma(bandera_gris, "Histograma antes de la ecualización")

    # Mostrar el histograma de la imagen ecualizada
    mostrar_histograma(bandera_ecualizada, "Histograma después de la ecualización")

    # Mostrar la imagen en escala de grises y la ecualizada
    cv2.imshow("Bandera en Escala de Grises", bandera_gris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Bandera Ecualizada", bandera_ecualizada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
