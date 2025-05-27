# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:32:14 2024

@author: jvazquez
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation

def apply_vertical_erosion(image_path, line_size):
    """
    Aplica erosión binaria con un elemento estructurante lineal vertical.
    
    :param image_path: Ruta de la imagen de entrada.
    :param line_size: Tamaño del elemento estructurante lineal vertical.
    :return: Imagen erosionada.
    """
    # Leer la imagen en escala de grises
    image = cv2.imread(image_path, 0)

    # Convertir la imagen a binaria
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Crear el elemento estructurante lineal vertical
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_size))
    
    # El código structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_size)) de OpenCV crea un elemento 
    # estructurante rectangular, donde (1, line_size) especifica el tamaño del rectángulo. Aquí (1, line_size) significa 
    # que el rectángulo tiene un ancho de 1 y una altura de line_size.
    # En OpenCV, el tamaño del elemento estructurante se especifica como (ancho, alto), lo que puede ser diferente de cómo 
    # se especifica en NumPy, donde las dimensiones se suelen especificar como (alto, ancho).

    # Aplicar la erosión binaria
    eroded_image = cv2.erode(binary_image, structuring_element, iterations=1)

    return binary_image, eroded_image



# Ruta de la imagen de entrada
image_path = 'Texto.png'

# Tamaño del elemento estructurante lineal vertical
line_size = 50 # Puedes ajustar este valor

# Aplicar la erosión
original_image, eroded_image = apply_vertical_erosion(image_path, line_size)

# Guardar el resultado
cv2.imwrite('Eroded_Text.png', eroded_image)

# Mostrar las imágenes
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Imagen Binaria Original')
plt.imshow(original_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f'Erosión con línea vertical de tamaño {line_size}')
plt.imshow(eroded_image, cmap='gray')

plt.show()


# # Generar un array de una línea vertical de tamaño 1x5
# vertical_line_numpy = np.ones((5, 1), dtype=np.uint8)

# print("Elemento estructurante con NumPy:")
# print(vertical_line_numpy)


# def morphological_reconstruction(marker, mask):
#     """
#     Realiza la reconstrucción morfológica binaria.
    
#     :param marker: Imagen binaria que sirve como marcador.
#     :param mask: Imagen binaria que sirve como máscara.
#     :return: Imagen reconstruida.
#     """
#     # Verificar que las dimensiones de marker y mask sean las mismas
#     if marker.shape != mask.shape:
#         raise ValueError("Las dimensiones de marker y mask deben ser iguales.")
    
#     # Inicializar la imagen reconstruida
#     reconstructed = marker.copy()
    
#     while True:
#         previous = reconstructed.copy()
#         # Aplicar la dilatación a la imagen reconstruida
#         dilated = binary_dilation(reconstructed, structure=np.ones((3, 3)))
#         # Realizar la intersección con la máscara usando AND lógico
#         reconstructed = dilated & mask
#         # Comprobar si hay cambios
#         if np.array_equal(reconstructed, previous):
#             break
    
#     return reconstructed

# # Llamar a la función de reconstrucción morfológica
# reconstructed = morphological_reconstruction( eroded_image, original_image)

# # Convertir la imagen reconstruida a formato 8-bit para visualización
# reconstructed = (reconstructed * 255).astype(np.uint8)

# # Mostrar las imágenes
# cv2.imshow('Reconstrucción', reconstructed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
