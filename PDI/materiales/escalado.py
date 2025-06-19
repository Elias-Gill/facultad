# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:52:51 2024

@author: jvazquez
"""

import cv2
import numpy as np

imagen = cv2.imread("1.jpg")  # Leemos la imagen a transformar y la guardamos en la variable imagen

if imagen is None:
    print("Error al cargar la imagen.")
else:
    h, w, c = imagen.shape  # obtenemos las dimensiones de la imagen

    escala = 2

    imagen2 = np.zeros((h * escala, w * escala, c), dtype=np.uint8)  # generamos la matriz que recibe la imagen escalada

    for i in range(h):
        for j in range(w):
            pixel = imagen[i, j]  # obtenemos el valor del pixel
            vector_pos = np.array([j, i, 1])  # generamos nuestro vector posición
            t_matrix = np.array([[escala, 0, 0], [0, escala, 0], [0, 0, 1]])  # generamos la matriz de transformación
            result_pp = np.dot(t_matrix, vector_pos)  # resolvemos el producto punto de la matriz de transformación con el vector posición

            # Asignamos el píxel a todas las nuevas posiciones que corresponde en la imagen escalada
            x_new, y_new = result_pp[0], result_pp[1]
            for dx in range(escala):
                for dy in range(escala):
                    imagen2[y_new + dy, x_new + dx] = pixel

    cv2.imshow("Original", imagen)
    cv2.imshow("Transformada", imagen2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



