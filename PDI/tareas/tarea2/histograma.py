import matplotlib.pyplot as plt

# Datos
data = [12, 8, 5, 14, 14,
        13, 10, 13, 5, 8,
        1, 2, 5, 8, 8,
        12, 2, 4, 4, 1,
        8, 11, 11, 10, 14]

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
