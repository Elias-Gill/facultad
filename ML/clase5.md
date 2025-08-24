NOTE:
el resto esta en mi cuaderno.

# Para resumir la clase 5

Para armar el arbol de decision, lo que se hace es utilizar un algoritmo recursivo que realiza
los siguientes pasos:

- Calcular el indice de informacion sobre cada atributo
- Tomar esa columa y armar las hojas en base a los posibles valores del atributo
- Pasar a las hojas y realizar el algoritmo de forma recursiva si es que su subgrupo de
  ejemplos aun no parte al espacio.
  Si es que todos los ejemplos forman parte de la misma clase, entonces creamos una hoja final
  y la etiquetamos con la clase.

El nivel de informacion se calcula como:
```txt 
I = D(p) - sumatoria{ D(px) * Nx/N }

Donde:
D = la funcion que calcula la impureza, pueden ser distintas funciones, pero una de ellas es la
entropia.

en cristiano:
I = Impureza global - sumatoria(impureza ponderada de cada clase)
```

Este arbol nos permite explicar facilmente preguntas sobre el problema.
