import matplotlib.pyplot as plt

# Datos
data = [ 11,7,5,14,13,
        12,9,12,5,7,
        1,2,5,7,8,
        11,2,3,3,1,
        7,10,10,9,14]

# Rango completo de 0 a 14
bins = range(0, 16)  # De 0 a 14, el 15 es el límite superior

# Crear el histograma
plt.figure(figsize=(8, 6))
plt.hist(data, bins=bins, color='skyblue', edgecolor='black', align='left')

# Ajustar ejes para ver todos los valores
plt.xticks(range(0, 15))  # Marcas en el eje X de 0 a 14

# Etiquetas
plt.title('Histograma de valores')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')

# Guardar en PNG
plt.savefig('histograma_equalizado.eps')

# Mostrar (opcional)
plt.show()
