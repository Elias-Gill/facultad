# Medicion y evaluacion de rendimiento de modelos de ML

La tasa de error rara vez nos da una perspectiva completa del desempeno de un modelo.
Para ello debemos conocer otras metricas de medicion y evaluacion de clasificadores.

Para la clasificacion cuando tenemos el resultado esperado podemos tener 4 tipos de resultados

por parte del modelo:
- Real Positivo Tp
- Real Negativo Tn
- Falso Positivo Fp
- Falso Negativo Fn

Con esto podemos armar una matriz 2x2, llamada tabla de confusion.
Para tener un clasificador perfecto, los false positive y false negative deben de ser 0.
Es decir, la matriz debe de ser diagonal.

La metrica mas basica que podemos hacer es el Error Rating:
- `E = (Nfp + Nfn) / (Nfp + Nfn + Ntp + Ntn)`
- `Accuracy = 1 - E`
 
En Algunos casos nos interesa de que existan mas falsos positivos o mas falsos negativos, esto
por ejemplo en medicina, donde se quiere que los falsos positivos sean mayores a los falsos
negativos.

Por esto es que no es suficiente solo creerle al ratio de error.
Ademas, este ratio no nos dice que metricas son mas altas, cuales bajas y como afectan al
modelo.

## Otras metricas

Otras metricas son:
- `Precision = Ntp / (Ntp + Nfp)` con que probabilidad el clasificador "ignora" casos positivos
  reales.
- `Exhaustividad = Ntp / (Ntp + Nfn)` con que probabilidad el algoritmo clasifica correctamente
  un positivo.
  A veces llamada _Sensibilidad_.
- `Especificidad = Ntp / (Ntn + Nfp)` no anote esta parte.

Un desbalance en las metricas puede darse porque el dataset elegido esta mal, por lo que hay un
sesgo en los datos de entrenamiento, por tanto el Accuracy es alto, pero la precision y
Exhaustividad son pesimos.

```kt
PRECISION: De todos los correos que el modelo dijo que eran spam, ¿cuántos realmente eran spam?
   Foco: Confianza en las predicciones positivas.
   Fórmula: Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
   "¿Qué tan preciso soy cuando digo que algo es positivo?"

RECALL: De todos los correos que realmente eran spam, ¿cuántos logré detectar?
   Foco: Capacidad para encontrar todos los casos positivos.
   Fórmula: Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
   "¿Qué tan bueno soy para no dejarme escapar ningún positivo?"
```

En el caso de diagnosticos medicos por ejemplo, quiero que el recall mejore, porque no quiero
dejar escapar NINGUN caso positivo a ser posible, por tanto maximizamos el `Recall`.

En casos de sistemas de recomendacion de tiendas, no importa que dejemos pasar ciertos
positivos, pero nosotros queremos solo los positivos puros, entonces es mas confiable cada
prediccion, por eso importa las la `Precision`.

De esto salen las `Curvas ROC` una gráfica que visualiza el compromiso (trade-off) entre dos
métricas importantes (en este context, precision vs recall) a través de diferentes umbrales de
decisión:
Esto es util para comparar modelos.

Luego hay otras metricas que permiten dar pesos diferentes tanto al recall como a la precision,
entonce asi podemos definir cual metrica nos importa mas.

Un ejemplo mas conocido es el de `Fb = [(B^2 + 1) x Pr x Re]/(B^2 x Pr + Re)` Con B > 1
entonces damos importancia al recall, con valores entre 0 y 1 damos relevancia a la precision.
1 es la mitad, donde da mismo peso a ambas metricas.

Otra media es el promedio Geometrico `gmean = raiz(Accuracy pos X Accuracy neg) = raiz(Recall X
Especificidad)`

## Generalizacion para N etiquetas

Simplemente se hace la misma formula, pero el contador de falsos es el resto de etiquetas
basicamente, todo lo que no sea un True Positive es una suma directa de las demas etiquetas, y
luego se hace la media, dividiendo entre la cantidad de etiquetas que existen.

# Curvas de aprendizaje y costos computacionales

Algoritmos que pueden ser mejores, dependiendo de su curva de aprendizaje, no por tener peor
rendimiento en un momento especifico significa que son peores, puede ser que el tiempo de
entrenamiento no es suficiente.

La mayoria de veces, un conjunto de datos mas grande significa una mejor clasificacion.
Hay dos aspectos de costos para analizar un algoritmo.
- Costo de computacion
- Tiempo de entrenamiento

Cuando el conjunto de datos es pequeno, se pueden hacer particiones de los datos, de modo que
seleccionamos entradas aleatorios, por tanto el entrenamiento va a dar algoritmos diferentes.
