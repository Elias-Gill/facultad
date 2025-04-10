def promedio(lista):
    suma = 0
    for i in lista:
        suma += i
    return suma / 10, suma

def moda(lista):
    repeticiones = {}
    moda = []
    veces_moda = 0

    for value in lista:
        # buscar si el valor ya existe en la tabla de repeticiones
        aux = 0
        if value in repeticiones:
            aux = repeticiones[value] + 1
        repeticiones[value] = aux

        # si el valor se repite mas que la moda, reemplaza el array
        if veces_moda < aux:
            moda = []
            moda.append(value)
            veces_moda = aux

        # si tiene igual repeticiones se agrega el valor al array de moda
        elif veces_moda == aux:
            moda.append(value)

    return moda
