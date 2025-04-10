def comunes(a, b):
    result = []
    for element in a:
        if element in b:
            result.append(element)
    return list(set(result))

def comunes_funcional(a, b):
    return list(filter(lambda e: e in b, a))

def comunes_conjuntos(a, b):
    return list(set(a) & set(b))

def comunes_compresion(a, b):
    return [e for e in a if e in b]

# Leer listas de números desde la entrada
a = list(map(int, input("Ingrese la primera lista de números separados por comas: ").split(",")))
b = list(map(int, input("Ingrese la segunda lista de números separados por comas: ").split(",")))

print(comunes(a, b))
print(comunes_funcional(a, b))
print(comunes_compresion(a, b))
print(comunes_conjuntos(a, b))
