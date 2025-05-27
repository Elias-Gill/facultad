import cv2
import numpy as np
from scipy.ndimage import binary_dilation

def generate_marker(binary_image):
    
    # Invertir la imagen binaria
    inverted_image = cv2.bitwise_not(binary_image)

    # Crear la imagen marcador
    marker = np.zeros_like(binary_image)
    marker[0, :] = inverted_image[0, :]
    marker[-1, :] = inverted_image[-1, :]
    marker[:, 0] = inverted_image[:, 0]
    marker[:, -1] = inverted_image[:, -1]

    return marker

def morphological_reconstruction(marker, mask):

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



# Leer la imagen binaria
image_path = 'Fig0916(a)(region-filling-reflections).tif'
binary_image = cv2.imread(image_path, 0)

# Convertir la imagen a binaria
_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

# Rellenar los agujeros en la imagen binaria
marker = generate_marker(binary_image)

# Llamar a la función de reconstrucción morfológica
reconstructed = morphological_reconstruction(marker,cv2.bitwise_not(binary_image))


# Calcular el complemento y Convertir la imagen reconstruida a formato 8-bit para visualización
rellenada = (255- reconstructed * 255).astype(np.uint8)


# Guardar y mostrar la imagen resultado

cv2.imshow('Original Image', binary_image)
cv2.imshow('Filled Image', rellenada )
cv2.waitKey(0)
cv2.destroyAllWindows()
