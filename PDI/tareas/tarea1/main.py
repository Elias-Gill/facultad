# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:28:37 2021

@author: Jesus-Mtz
"""

import operaciones

lista = []
cuenta = 0
while cuenta < 10:
    cuenta = cuenta + 1
    numero = int(input("ingresa un numero: "))
    lista.append(numero)
    
promedio, suma = operaciones.promedio(lista)

moda = operaciones.moda(lista)

print(promedio, suma, moda)
