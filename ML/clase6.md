# Continuacion. Arboles de decision

## Variables continuas en arboles de decision

Cuando tenemos variables numericas, se encuentra un problema sobre como calcular el arbol.

Primero ordenamos los valores de "entrenamiento" en la recta ordinal.
Luego se eligen los "puntos de corte" sobre las partes donde hay un cambio de clase en la
recta.

Luego se aplica el mismo algoritmo que vimos recien.
Pero en vez de decirno una decision, ahora nos da un `intervalo` por los cuales un valor toma
un valor de cierta clase.

## Impacto de la profundidad maxima

Un arbol mas grande lo que tiene de problema es de que esta cada vez mas especializado, es
decir, mas sesgado a solo los ejemplos de entrenamiento.
Esto es justamente el "sobreentrenamiento". 

Entonces lo mejor es tener un arbol lo suficientemente grande para ser efectivo, pero no lo
suficientemente grande como para sobreentrenar.

Para eso es que se realiza la poda del arbol.
Esto tambien existe para redes neuronales apagando neuronas.

NOTA:
la `CLASE` siempre es binaria.
La clase se refiere a la decision si es true o false, en cambio los atributos si pueden ser
continuos, enteros o discretos.

## Poda

El algoritmo de poda es el siguiente:

- Estimar la tasa de error del arbol original.
- Estimar el error de los arboles obtenidos despues de la poda.
- Elegir la poda que minimize la tasa de error.

## Transformacion del arbol de decision

El resultado de un arbol de decision es al final una mezcla de If's simplemente, que se puede
transformar en codigo fuente.

# Arboles de regresion

Los arboles de regresion a diferencia de los de decision, nos retornan una probabilidad de que
se de un caso, sabiendo datos del pasado (prediccion).

## Tecnica del random forest

Lo que se hace es dividir el dataset, tomando ciertas clases, y solo ciertos atributos,
entonces se arman arboles diferentes.
Luego la salida de los arboles pequenos individuales se "combina" y se da una salida final.

Sistemas ensamblados:
se combinan muchos sistemas pequenos, rapidos pero poco precisos, pero que trabajando en
conjunto son eficientes y robustos.
