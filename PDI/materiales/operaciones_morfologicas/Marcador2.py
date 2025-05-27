# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:18:41 2024

@author: jvazquez
"""

import numpy as np
import cv2

def generate_marker(mask, iterations):
    """
    Genera la imagen del marcador aplicando erosiones binarias a la imagen de la máscara.
    
    :param mask: Imagen binaria que sirve como máscara.
    :param iterations: Número de iteraciones de erosión a aplicar.
    :return: Imagen del marcador.
    """
    # Aplicar erosión binaria a la imagen de la máscara
    marker = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=iterations)
    return marker

# Leer la imagen de la máscara en escala de grises
mask = cv2.imread('Mask.png', 0)

# Convertir la imagen a binaria
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Generar la imagen del marcador aplicando erosiones
iterations = 15  # Puedes ajustar el número de iteraciones según sea necesario
marker = generate_marker(mask_bin, iterations)

# Guardar y mostrar las imágenes
cv2.imwrite('Marker_generated.png', marker)

cv2.imshow('Máscara', mask_bin)
cv2.imshow('Marcador', marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
