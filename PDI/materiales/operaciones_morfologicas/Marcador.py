# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:15:40 2024

@author: jvazquez
"""

import numpy as np
import cv2

def generate_marker(mask, structuring_element_size):
    """
    Genera la imagen del marcador aplicando una erosión binaria a la imagen de la máscara.
    
    :param mask: Imagen binaria que sirve como máscara.
    :param structuring_element_size: Tamaño del elemento estructurante para la erosión.
    :return: Imagen del marcador.
    """
    # Crear el elemento estructurante
    structuring_element = np.ones((structuring_element_size, structuring_element_size), np.uint8)
    
    # Aplicar una erosión binaria a la imagen de la máscara
    marker = cv2.erode(mask, structuring_element, iterations=1)
    
    return marker

# Leer la imagen de la máscara en escala de grises
mask = cv2.imread('Mask.png', 0)

# Convertir la imagen a binaria
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Generar la imagen del marcador aplicando erosión con un elemento estructurante grande
structuring_element_size = 11  # Ajusta este tamaño según sea necesario
marker = generate_marker(mask_bin, structuring_element_size)

# Guardar y mostrar las imágenes
cv2.imwrite('Marker_generated.png', marker)

cv2.imshow('Máscara', mask_bin)
cv2.imshow('Marcador', marker)
cv2.waitKey(0)
cv2.destroyAllWindows()