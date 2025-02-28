# Operaciones sobre el dominio del espacio (sin transformaciones)

Las operaciones sobre el dominio espacial se pueden dividir en:
- Operaciones sobre un pixel
- Operaciones sobre una vecindad de pixeles

## Operaciones aritmeticas

Es aplicar directamente funciones sobre el pixel en cuestion. Se pueden hacer operaciones como
sumar, multiplicar por constantes, dividir por valores, aplicar el complemento, etc.

La transformacion se manteiene en el domoinio espacial y hay que definir las funciones a modo
de que no se salga del dominio del los valores de la imagen (obvio).

## Histogramas

El histograma es un grafico que indica la frecuencia de los niveles de intensidad de la imagen.
Es decir, es un conteo de las apariciones de los niveles de intensidad. Con este histograma se
puede por ejemplo diferenciar si una imagen es mas oscura o brillante viendo la grafica.

### Estiramiento de contraste

Una funcion lineal a tramos es una funcion que tiene cambios en forma de recta, pero a tramos,
es decir, que las ecuaciones de las rectas varian dependiendo de que rango de valores se toman.

Esta tecnica tiene problemas ya que requiere que se calcule una funcion para cada nuevo
histograma.

### Ecualizacino del histograma

Los nuevos valores de intensidad son iguales a la sumatoria de los valores anteriores de
intensidad / n (cantidad de valores de intensidad). Luego esto lo multiplico por (n / cantidad
de pixeles).

Entonces aplicamos esta formula por todos los valores del histograma y redondeamos los valores
al entero mas cercano. Esto nos los nuevos valores de intensidad donde directamente
reemplazamos dicho nivel de intensidad con el nuevo en la imagen.

Esto es util para realizar una imagen con un histograma mas distribuido, mejorando posiblemente
los detalles de la imagen.

## Tablas de busquedas

Es una tecnica para realizar transformaciones sobre imagenes de manera mas eficiente. POr
ejemplo una imagen con 256 valores de intensidad de gris diferentes. Podemos contar con una
tabla con 256 valores la cual podemos consultar la tabla en vez de realizar las operaciones por
cada pixel. 

El acceso a memoria es mas rapido que realizar operaciones.

## Filtrado espacial lineal y NO lineal

