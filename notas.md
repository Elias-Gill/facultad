UNa imagenn se representa como una matriz que contiene valores discretos finitos escalares los cuales representan valores
como el color o la intensidad del pixel.

Tipos de operaciones
- operaciones sobre pixel
- operaciones de vecindad
- Transformadas o transformaciones (trasnforma el dominio de la matriz, por ejemplo de intencidad de color a frecuencias)

# Operaciones de piexeles

Solo es necesario saber la informacion de ese mismo piexl en la imagen de entrada para procesar y dar un resultado en la
imgen de salida.

# Operaciones de vecindad

Para generar el valor de un pixel debemos conocer los valores de los pixeles vecinos en la imagen de entrada.

# Transformadas

Transforma la imagen a otro dominio la cual retorna una matriz que cambia el dominio de la matriz original, con la cual se
pueden aplicar operaciones como filtros y luego aplicar la anti-transformada para generar la imagen de salida.

Una transformada es una funcion que es biyectiva e invertible la cual transforma un dominio en valores de otro dominio.

Se pueden realizar operaciones sobre la matriz pero se debe mantener el dominio de la funcion, pero si es que los valores se
pasan se deben **truncar**.

### operadores lineales

Es una funcion sobre matrices que cumple con:
- F(a) + F(B) = F(A + B)
- a.F(B) = F(a.B)

[operaciones sobre el espacio](./operaciones_sobre_el_espacio.md)
