# Teoria de decision

## Decisiones bajo certidumbre

Programacion lineal: 

```
 Maxz - Minz
   s.a
 Restriccinoes funcionales y de no negatividad
```

Conocemos las funciones objetivo, hacia donde se apunta y las restricciones.

### Proceso de jerarquia analitica

- _Alternativas_:
  son las opciones para la toma de decision 
- _Criterios_:
  son los atributos que rodean a las alternativas 
- _Pesos_:
  son los niveles de importancia asignados a los criterios y alternativas

*Determinacion de los pesos:* El PJA se centra en la determinacion de los pesos relativos para
calificas las alternativas.
Para ello se reaiza una matriz de comparacion de criterios donde se encuentran los pesos.

|            | criterio1  | Ccriterio2 |
|----------- | ---------- | ---------- |
| criterio1  | a.12       | a.12       |
| criterio2  | a.21       | a.22       |

Se utiliza una escala numerica para marcar los pesos de los criterios de la siguietne forma:

- Si `a.ij = 1` entonces "i" y "j" son de **igual** importancia
- Si `a.ij = 3` entonces "i" es **debilmente** mas importante que "j"
- Si `a.ij = 5` entonces "i" es **fuertemente** mas importante que "j"
- Si `a.ij = 7` entonces "i" es **muy fuertemente** mas importante que "j"
- Si `a.ij = 9` entonces "i" es **extremadamente** mas importante que "j"

Los valores 2,4,6,8 se interpretan segun corresponde.

*La consistencia de un juicio*:
implica que si `a.ij = K`, entonces `a.ji = 1/k`.
(en cristiano, los elementos opuestos se parecen)

Ademas siempre se tiene que `a.ii = 1`, porque se califica el elemento consigo mismo.

#### Normalizacion de matrices de comparacion

La normalizacion sirve para estandarizar los pesos en valores que varian entre 0 y 1, de esa
menera se tendra unan idea cuantitativa mas clara del nivel de importancia asignado.

El proceso requiere **dividir los elementos de cada columna con la sumatoria de la misma**

#### Calculo de pesos relativos

Para poder calcular los pesos relativos de una matriz de comparacion, debemos **promediar los
pesos de cada fila de la matriz NORMALIZADA**.

#### Analisis de consistencia

Una matriz es consistente si las columnas de su matriz normalizada son iguales.
En caso contrario la matriz es NO consistente.

NOTA:
las matrices 2x2 siempre son consistentes

NOTA:
decimos que _la decision es consistente_ cuando todas sus matrices de comparacion son
consistentes o inconsistentes tolerables.

_Cuando la matriz es NO consistente_ se debe realizar lo siguiente y calcular la razon de
consistencia (RC).

Formula:
$$ A*W = n_{\text{max}}, \ n_{\text{max}} \geq n $$

Donde:
- _A_:
  es la matriz de comparacion.
- _W_:
  es el vector de pesos.
- _n_:
  dimension del vector 
- _nmax_:
  la suma de los elementos del vector resultante `A.W`

##### Calculo de la razon de consistencia

_CI_:
Indice de consistencia

Donde `CI = (nmax - n) / (n-1)`

_RI_:
Razon de _consistencia aleatoria_

Donde `RI = (1.98 * (n - 2)) / (n)`

_RC_:
Razon de consistencia

Donde `RC = CI / RI`

> Se dice que el nivel de inconsistencia es tolerable si es que `_RC_ < 0,1`

--- 
## Ejemplo

Un estudiane debe decidir entre tres universidades "A, B, C o D".
Los criterios son:
Ubicacion y Prestigio

Para el el prestigio es fuertemente mas importante que la ubicacion.

Sus preferencias en cuanto a la ubicacion son:

- La universidad B tiene relacion 2 con la ubicacion de A.
- La universidad C esta fuertemente mejor ubicada que A y su relacino con la B es de 2.

Sus preferencias en cuanto al prestigio son:

- La universidad A tiene relacion 2 con el prestigio de B.
- La universidad A tiene prestigio debilmente superior a C.
- La universidad B tiene prestigio con relacion 3/2 con C.

--- 
Primero escribimos la matriz de comparacion de criterios.
Esta matriz sera de tamanho 2x2

|            | prestigio  | ubicacion  |
|----------- | ---------- | ---------- |
| prestigio  | 1          | 5          |
| ubicacion  | 1/5        | 1          |

--- 
Ahora generamos la matriz de comparacion de alternativas segun **ubicacion**:

|     | A    | B    | C    |
|---- | ---- | ---- | ---- |
| A   | 1    | 1/2  | 1/5  |
| B   | 2    | 1    | 1/2  |
| C   | 5    | 2    | 1    |

POdemos notar como los opuestos de la diagonal son las inversas, sigue la regla que vimos mas
atras.
Asi tambien la diagonal es siempre 1, porque son los criterios relacionados consigo mismo.

--- 
Ahora generamos la matriz de comparacion de alternativas segun **prestigio**:

|     | A    | B    | C    |
|---- | ---- | ---- | ---- |
| A   | 1    | 2    | 3    |
| B   | 1/2  | 1    | 3/2  |
| C   | 1/3  | 2/3  | 1    |

---
Ahora pasamos a realizar la normalizacion de las matrices, lo cual consiste en simplemente
dividir los elementos de cada **Columna** y dividirlos entre la sumatoria de la columna.

Matriz de *criterios* normalizada: 
|            | prestigio  | ubicacion  |
|----------- | ---------- | ---------- |
| prestigio  | 1/6        | 1/6        |
| ubicacion  | 5/6        | 5/6        |

Matriz de comparacino de _ubicacion_ normalizada: 
|     | A    | B    | C    |
|---- | ---- | ---- | ---- |
| A   | 1/8  | 1/7  | 2/17 |
| B   | 1/4  | 2/7  | 5/17 |
| C   | 5/8  | 4/7  | 10/17|

Matriz de comparacino de _prestigio_ normalizada: 
|     | A    | B    | C    |
|---- | ---- | ---- | ---- |
| A   | 6/11 | 6/11 | 6/11 |
| B   | 3/11 | 3/11 | 3/11 |
| C   | 2/11 | 2/11 | 2/11 |

---
Ahora calculamos los pesos relativos de la matriz de _criterios_:
- Peso relativo de ubicacion:
  `1/6 ~= 0,17 => 17%`
- Peso relativo de prestigio:
  `5/6 ~= 0,83 => 83%`

Ahora calculamos los pesos relativos de la matriz de _ubicacion_:
- Peso relativo de A:
  `12,8%`
- Peso relativo B:
  `27,7%`
- Peso relativo C:
  `59,4%`

Esto nos dice que la universidad C es la mejor ubicada entre las 3, y la peor es la A.
Ahora calculamos los pesos relativos de la matriz de _prestigio_:
- Peso relativo de A:
  `54%`
- Peso relativo B:
  `27%`
- Peso relativo C:
  `18%`

---
Para saber cual alternativa elegir, finalmente realizamos el calculo de la sumas de sus pesos
de cada opcion.
En este caso `(peso ubicacion * peso criterio de ubicacion) + (peso prestigio * peso criterio
de prestigio)`

- Universidad A:
  `(0,17 * 0,128) + (0,83 * 0,54) = 0,499`
- Universidad B:
  `(0,17 * 0,277) + (0,83 * 0,27) = 0,271`
- Universidad C:
  `(0,17 * 0,594) + (0,83 * 0,18) = 0,25`

Por tanto la eleccion sera la universidad A por tener la mejor suma de pesos.

--- 
Ahora revisamos con el analisis de consisencia de las matrices.
Podemos deducir que:

- La matriz de comparacion de criterios es consistente
- La matriz de comparacion de criterios segun ubicacion es NO consistente, dado que las
  columnas de su matriz normalizada no son iguales.
- La matriz de comparacion de criterios segun prestigio es consistente

Como la matriz de _ubicacion_ es no consistente entonces tenemos que hacer el analisis de
inconsistencia:

```
|------|------|------|      -------
| 1/8  | 1/7  | 2/17 |      |0,129|
| 1/4  | 2/7  | 5/17 |  x   |0.277| 
| 5/8  | 4/7  | 10/17|      |0.594|
|------|------|------|      -------
```

Lo que es igual a 
```
-------
|0.386|
|0.832|
|1.793|
-------
```

Por tanto:

```
nmax = 3,011 (la sumatoria de las filas del resultado. Como es mayor que n seguimos)
CI = 5,5 x 10^-3
RI = 0,66
RC = 0,0856
```

Como `RC < 0,1` entonces se dice que la matriz es inconsistente tolerable.

Como todas las matrices de comparacion son consistentes o tolerables, entonces decimos que _la
decision es consistente_.
