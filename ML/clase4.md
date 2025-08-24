---
Titulo: Redes neuronales
---

NOTA:
esto es IA predictiva, no IA generativa.

# Redes neuronales (introduccion)

Se inspiran biologicamente (creada por un neurologo).
Las neuronas se conectan entre si a traves de la sinapsis(emisores) y dentritas(receptores)
mediantes cargas electricas.

Las descargas electricas se conectan con el axion (el cable).
Cuando la neurona se carga a un cierto nivel, entnoces se genera una descarga por el axion
hacia neuronas vecinas (la carga electrica varia entre neuronas, eso es lo que se llama como
"pesos" dentro de las redes neuronales digitales).

## Estructura
- Entradas (xn) y un umbral de activacion (F) (solo si se necesita una salida binaria).
- Pesos sinapticos (w).
- Regla de propagacion (s) cuenta con un campo "b" que es el punto de "energia" minimo.
  Esta funcion es una recta, es decir, una funcion lineal que es la suma de `xn * wn`.
- Una funcion de activacion (_f_):
  hay varias pero una de ellas es la funcion sigmoide `(1/(1+e^-S))`.
  Lo que hace es "normalizar" la funcion entre 0 y 1.
  Esto da una funcion no lineal.
- Salida (y) (0 o 1 si se cuenta con el umbral de activacion).

```txt
+----+     +----+     +---------------------+  f (cursiva)   +-------------------------+     +------------+
| xn | --> | wn | --> | S = x1 * w1 ... + b | -------------> | F (solo si se necesita) | --> | salida (y) |
+----+     +----+     +---------------------+                +-------------------------+     +------------+
```

Otros tipos de funicones de activacion:

- Sigmoide
- Identidad
- Escalon
- Lineal a tramos
- Tangencial


La parte mas complicada es determinar los pesos de cada W, la otra tambien es elegir la funcion
de activacion y el umbral.

Existen redes neuronales no recurrentes, recurrentes, tambien redes parcialmente conectadas o
completamente conectadas.

Clasificacion de RN de primera generacion:
- Por numero de capas
- Por tipo de conexion
- Por tipo de 

La propagacion de izquierda a derecha es forward propagration.

Lo dificil del "Perceptron" es calcular la cantidad neuronas de la "capa oculta" asi como sus
pesos, tambien la cantidad de "capas" de neuronas de la capa oculta.

## Errores de las redes

`Tasa de error = errores / total de muestras`.

Un valor de tasa de aciertos aceptable es de mas de 90%.
En el caso de aplicaciones medicas debe de ser mayor al 95%.

Pero lo mejor es usar la tecnica de diferencia de cuadrados para evitar errores de casos
extremos.

Pero hay mas tecnicas y ecuaciones para calcular la tasa de errores.

# Retro propagacion (Backpropagation of error)

Es un paso del entrenamiento que consiste en pasar la tasa de error medio hacia atras como
parametro de la red neuronal.
Entonces mediante eso se trata de calcular los nuevos pesos de las neuronas.

Lo que hace el proceso de entrenamiento es ajustar una formula que separa (clasifica) el
espacio de muestras posibles.

TODO:
buscar `Formula de Herbb` para aprendizaje y calculo de pesos.

Para construir una RN los "pasos" son:
- Disenar la arquitectura (capas y numero de neuronas)
- Hiperparametrizacion (cuales parametros usar)
- Entrenar (calcular los pesos)

Cada "etapa finalizada" en el proceso de entrenamiento (cuando ya veo todos mis entradas del
dataset) se llama `epoca`.

El problema de la propagacion hacia atras son los valles y maximos locales de las soluciones
encontradas (ver grafico de "landscape" y acordarse de primeagen).
Por el tema de las combinatorias, entonces el problema del entrenamiento es NP-completo.
