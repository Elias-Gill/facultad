def calsificar():
    i = input("ingrese: ").split(",")
    result = {}

    for x in i:
        value = x.split(":")
        nombre = value[0]
        edad = int(value[1])
        if edad < 13:
            result[nombre] = "criatura"
        elif edad < 20:
            result[nombre] = "joven"
        elif edad < 60:
            result[nombre] = "adulto"
        else:
            result[nombre] = "jubilado"

    return result


def calsificar_funcional():
    def aux(x):
        value = x.split(":")
        nombre = value[0]
        edad = int(value[1])
        if edad < 13:
            return nombre,"criatura"
        elif edad < 20:
            return nombre,"joven"
        elif edad < 60:
            return nombre,"adulto"
        else:
            return nombre,"jubilado"

    i = input("ingrese: ").split(",")
    return dict(map(aux, i))

def calsificar_compresion():
    i = input("ingrese: ").split(",")
    return {
            nombre: (
                "criatura" if (edad := int(edad_str)) < 13 else
                "joven" if edad < 20 else
                "adulto" if edad < 60 else
                "jubilado"
                )
            for nombre, edad_str in (x.split(":") for x in i)
            }

print(calsificar())
print(calsificar_funcional())
print(calsificar_compresion())
