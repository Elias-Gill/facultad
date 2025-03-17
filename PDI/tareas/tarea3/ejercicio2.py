def largas(lista):
    result = []
    for i in lista:
        if len(i) > 5:
            result.append(i)

    return result

def largas_funcional(lista):
    return list(filter(lambda value: len(value) > 5, lista))

def largas_coleccion(lista):
    return [i for i in lista if len(i) > 5]

prueba = ['manzana', 'pera', 'banana']
print(largas(prueba))

print(largas_funcional(prueba))
print(largas_coleccion(prueba))
