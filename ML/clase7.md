# Introduccion a algoritmos NO supervisados

NOTA:
inducir es buscar es buscar una funcion que se parezca a la realidad.

En aprendizado, se tiene la x e y y se para a un modelo para que trate de inducir a la funcion
que describa mejor la realidad.

El NO supervisado trata de buscar un patron oculto o relacion entre los datos

- Supervisado:
  Enseñar con ejemplos
- Reforzado:
  Premiar o castigar segun rendimiento

# Clusterizacion

Es parecido al problema de categorizacion, pero en vez de vos darle los ejemplos con sus
resultado, lo que se hace es darle los resultados y el algoritmo va a tratar de de categorizar
el solo los datos.

El problema mas complejo del problema de Clusterizacion es el de calcular la distancia de
similaridad entre elementos.

## Definicion de Medida de proximidad

Es una generalizacion de dis-similaridad y similaridad.

Esta medida es:
- Simetrica
- Siempre positiva
- Desigualdad triangular (Suma de dos lados es siempre mayor o igual al lado restante)
- Reflexibidad ` D[x,y]=0 , si x=y `
- Hay una condicion mas que hace que sea ULTRA metrica.

Una `Funcion de similaridad` solo se da cuando no se cumple la desigualdad triangular, pero si
el resto.

Hay otra formula loca que si cumple la funcion de similaridad, entonecs es una `Metrica de
similaridad`
