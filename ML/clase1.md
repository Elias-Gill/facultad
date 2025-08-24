---
Titulo: Introduccion al ML
---

# ML supervisado

Los grupos de entrenamiento tienen "atributos" (sinonimos:
predictores, variables).

El ML supervisado implica de que "alguien" te diga "que esta bien" y "que esta mal".
Por eso se dice que es supervisado, porque alguien ya te da los ejemplos y alguien ya
"resolvio" ejemplos del problema por mi.

Inducido significa que mediante el ML se quiere llegar a una funcion matematica (clasificador),
utilizando los datos de entrenamiento, que nos resuelva el problema que queremos encontrar.

El problema de conseguir un clasificador es NP completo, por tanto se utilizan heuristicas, la
fuerza bruta no nos sirve.

Los datos disponibles se divien en dos conjuntos, en uno de aprendizaje y otro de testeo, que
sirve para medir el rendimiento del clasificador.

Se debe tener cuidado con el conjunto de test, porque no debe ser demasiado parecido al dataset
de trainning, ya que los problemas que sean diferentes no se pueden, porque las respuestas se
"memorizaron".
Esto se llama sobre-entrenamiento.

Lo contrario es el mal entrenamiento, ya que puede ser que se construya un conjunto de
entrenamiento sesgado, por tanto el modelo no sabra resolver problemas diferentes.

Por ello la seleccion de los datasets de pruebas y de testing deben ser seleccionados
cuidadosamente y no es un problema trivial.

NO es suficiente que los algoritmos den resultados, tambien tienen que explicar el "por que" de
sus resultados.
Esto entra dentro de la "XAI (Explanable Artificial Intelligence)".

Tipos de ruido:
- Estocastico (aleatorio)
- Sistematico (todas las muestas tienen el mismo desface)
- Artefactos arbitrarios (el valor obtenido no tiene relacion alguna con la realidad)
