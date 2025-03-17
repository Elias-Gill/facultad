lista = ['hola', 'abih', 'elias']

# Usando compresión de conjuntos
def mayusculas_compresion(palabras):
    return {palabra.upper() for palabra in palabras}

# Usando programación funcional
def mayusculas_funcional(palabras):
    return set(map(str.upper, palabras))

# Usando bucle for
def mayusculas_for(palabras):
    result = set()
    for palabra in palabras:
        result.add(palabra.upper())
    return result

# Pruebas
print(mayusculas_compresion(lista))
print(mayusculas_funcional(lista))
print(mayusculas_for(lista))
