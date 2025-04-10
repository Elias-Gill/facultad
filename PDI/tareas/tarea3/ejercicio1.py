def porcentaje(dic, porcentaje):
    result = {}
    for key in dic:
        value = dic[key]
        result[key] = value + value * (porcentaje / 100)

    return result

def porcentaje_funcional(dic, porcentaje):
    return dict(zip(
        prueba.keys(), 
        list(map(
            lambda key: dic[key] + dic[key] * (porcentaje / 100), dic))))


def porcentaje_compresion(dic, porcentaje):
    result = {}
    result = dict([(key, dic[key] + dic[key] * (porcentaje / 100)) for key in dic])
    return result


prueba = { "entrada": 100, "key2": 200, "key1": 3, "key3": 4 }

result = porcentaje(prueba, 10)
print(result)

result = porcentaje_funcional(prueba, 10)
print(result)

result = porcentaje_compresion(prueba, 10)
print(result)
