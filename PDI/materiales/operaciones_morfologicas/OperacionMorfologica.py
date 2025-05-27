import cv2
import numpy as np

# # Cargar la imagen
# A = cv2.imread('broken_text.tif', 0)  # Asumiendo que es una imagen en escala de grises

# # Convertir la imagen a binario
# _, A_bin = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)

# # Crear el elemento estructurante
# B = np.array([[0, 1, 0],
#               [1, 1, 1],
#               [0, 1, 0]], dtype=np.uint8)

# # Aplicar la dilatación
# A2 = cv2.dilate(A_bin, B)

# # Mostrar la imagen resultante
# cv2.imshow('Original Image',A)
# cv2.imshow('Dilated Image', A2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# # Cargar la imagen
# A = cv2.imread('wirebond_mask.tif', 0)  # Cargar como imagen en escala de grises

# _, A = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
# # Crear el elemento estructurante para un disco de radio 10
# se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

# # Erosión con un disco de radio 10
# A2 = cv2.erode(A, se1)

# # Mostrar la imagen resultante
# cv2.imshow('Erosion with disk radius 10', A2)

# # Crear el elemento estructurante para un disco de radio 5
# se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# # Erosión con un disco de radio 5
# A3 = cv2.erode(A, se2)

# # Mostrar la imagen resultante
# cv2.imshow('Erosion with disk radius 5', A3)

# # Crear el elemento estructurante para un disco de radio 20
# se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

# # Erosión con un disco de radio 20
# A4 = cv2.erode(A, se3)

# # Mostrar la imagen resultante
# cv2.imshow('Erosion with disk radius 20', A4)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Rectangular Kernel
print(cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

# Elliptical Kernel
print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

# Cross-shaped Kernel
print(cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
