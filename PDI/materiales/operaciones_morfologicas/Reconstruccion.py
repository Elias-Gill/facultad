# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:46:12 2024

@author: jvazquez
"""

import numpy as np
from scipy.ndimage import binary_dilation
import cv2

def morphological_reconstruction(marker, mask):
    """
    Realiza la reconstrucción morfológica binaria.
    
    :param marker: Imagen binaria que sirve como marcador.
    :param mask: Imagen binaria que sirve como máscara.
    :return: Imagen reconstruida.
    """
    # Verificar que las dimensiones de marker y mask sean las mismas
    if marker.shape != mask.shape:
        raise ValueError("Las dimensiones de marker y mask deben ser iguales.")
    
    # Inicializar la imagen reconstruida
    reconstructed = marker.copy()
    
    while True:
        previous = reconstructed.copy()
        # Aplicar la dilatación a la imagen reconstruida
        dilated = binary_dilation(reconstructed, structure=np.ones((3, 3)))
        # Realizar la intersección con la máscara usando AND lógico
        reconstructed = dilated & mask
        # Comprobar si hay cambios
        if np.array_equal(reconstructed, previous):
            break
    
    return reconstructed

# Leer las imágenes en escala de grises
marker = cv2.imread('Marker.png', 0)
mask = cv2.imread('Mask.png', 0)

# Convertir las imágenes a binarias
_, marker_bin = cv2.threshold(marker, 127, 1, cv2.THRESH_BINARY)
_, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

# Llamar a la función de reconstrucción morfológica
reconstructed = morphological_reconstruction(marker_bin, mask_bin)

# Convertir la imagen reconstruida a formato 8-bit para visualización
reconstructed = (reconstructed * 255).astype(np.uint8)

# Mostrar las imágenes
cv2.imshow('Marcador', marker_bin * 255)
cv2.imshow('Máscara', mask_bin * 255)
cv2.imshow('Reconstrucción', reconstructed)
cv2.waitKey(0)
cv2.destroyAllWindows()
