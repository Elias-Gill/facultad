from functools import reduce

def contar_ocurrencias_compresion(lista):
    return {palabra: lista.count(palabra) for palabra in set(lista)}

def contar_ocurrencias_funcional(lista):
    return reduce(lambda acc, palabra: acc.update({palabra: acc.get(palabra, 0) + 1}) or acc, lista, {})

def contar_ocurrencias_for(lista):
    ocurrencias = {}
    for palabra in lista:
        if palabra in ocurrencias:
            ocurrencias[palabra] += 1
        else:
            ocurrencias[palabra] = 1
    return ocurrencias

lista_palabras_usuario = input("Ingrese las palabras separadas por comas: ").split(",")

print(contar_ocurrencias_compresion(lista_palabras_usuario))
print(contar_ocurrencias_funcional(lista_palabras_usuario))
print(contar_ocurrencias_for(lista_palabras_usuario))
