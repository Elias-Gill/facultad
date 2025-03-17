from functools import reduce

def combinar_diccionarios_compresion(d1, d2):
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}


def combinar_diccionarios_funcional(d1, d2):
    return reduce(
        lambda acc, k: acc.update({k: d1.get(k, 0) + d2.get(k, 0)}) or acc,
        set(d1) | set(d2),
        {},
    )


def combinar_diccionarios_for(d1, d2):
    resultado = d1.copy()
    for k, v in d2.items():
        resultado[k] = resultado.get(k, 0) + v
    return resultado


d1 = dict(
    item.split(":")
    for item in input("Primer diccionario: ").replace(" ", "").split(",")
)
d2 = dict(
    item.split(":")
    for item in input("Segundo diccionario: ").replace(" ", "").split(",")
)
d1 = {k: float(v) for k, v in d1.items()}
d2 = {k: float(v) for k, v in d2.items()}

print(combinar_diccionarios_compresion(d1, d2))
print(combinar_diccionarios_funcional(d1, d2))
print(combinar_diccionarios_for(d1, d2))
