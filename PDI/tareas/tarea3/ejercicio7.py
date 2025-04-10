def eliminar_duplicados_compresion(lista):
    return list(dict.fromkeys(lista))

def eliminar_duplicados_funcional(lista):
    return reduce(lambda acc, x: acc + [x] if x not in acc else acc, lista, [])

def eliminar_duplicados_for(lista):
    resultado = []
    for elemento in lista:
        if elemento not in resultado:
            resultado.append(elemento)
    return resultado


from functools import reduce

lista = [1, 2, 2, 3, 4, 4, 5]

print(eliminar_duplicados_compresion(lista))
print(eliminar_duplicados_funcional(lista))
print(eliminar_duplicados_for(lista))
