---
Titulo: K-NN (Kesimo nearest neightbor)
---

# La regla de k-vecino mas cercano (k-NN)

Comparar con el ejemplo de entrenamiento mas parecido.

Similitud por vecetores.

Problema, si es que uso un solo "vecino" para calcular la distancia, entonces puede darse de
que se esta sesgando.
Nentonces se eligen k vecinos para trabajar sobre ellos.
Elegir el valor de K no es trivial.

Se necesita la tabla de distancias para los ejemplos comparados con el caso que se quiere
estudiar.
Igual que Bayesiano.

Se toma la clase (es decir, la decision) la cual aparezca mas veces entre los K vecinos.

El problema son los puntos limites, es decir, donde la "franja" que divide geometricamente (el
umbral) entre las clases, es donde no se puede decidir de manera clara a cual clase pertenece
la clasificacion.

Cuando no se puede partir geometricamente el espacio (por ejemplo, hay negativos dentro del
campo positivo), entonces significa que se necesitan mas caracteristicas dentro de mi estudio.

## TIpos de distancia

Distancia geometrica entre dos puntos (raiz de (delta x)^2 + (delta y)^2 de toda la vida).
Esto se puede usar siempre que las caracteristicas se representen en numeros ordinales.
La misma ecuacion se generaliza por n componentes (facil de geometria).

Para atributos continuos es la misma formula, para atributos discretos, la distancia entre 2
atributos es `0 si x = y` y `1 si x != y` (logica booleana boludo).

NO se pueden simplemente mezclar atributos de distintos dominios si es que hay demasiada
diferencia entre atributos del dominio.
Es decir, por ejemplo si la diferencia de un atributo es de millones, entonces si es que
tenemos otro atributo que es discreto, entonces esta columna no impacta en la formula.

Para poder mezclar entonces se normalizan los dominios (poner todo entre 0 y 1).

Se deben elegir correctamente las distancias entre las distancias del dominio entre cada
elemento, porque de esto depenede la presicion del clasificador.

Reglas:
- La distancia no puede ser negativa
- La distancia entre vectores identicos es 0
- La distancia de `x a y` debe de ser igual a la de `y a x`
- La metrica debe satisfacer la desigualdad triangular = `d(x,y) + d(y,z) >= d(x,z)`

Siguiendo, la eleccion de atributos relevantes tambien debe de ser cuidadosa, porque atributos
irrelevantes modifcan los valores del estudio.
Por ejemplo, queremos definir si un paciente esta enfermo o no, pero meto entre atributos el
talle de zapatos.

Para resolver todos los problemas se normaliza con la siguiente formula:
```tex
x = \div{x - MIN}{MAX-MIN}
```

La desventaja es que se pierde la interpretacion natural de los resultados (porque se "deforma"
la informacion).

Hay otro metodo que se llama estandarizacion, que no es lo mismo que normalizacion.

**Fórmula**:
```tex
\section*{Normalización (Min-Max)} 

La normalización escala los datos a un rango fijo, típicamente [0, 1]:
\[ x' = \frac{x - \min(x)}{\max(x) - \min(x)}\] 

donde:
\begin{itemize} 
    \item\(x\): valor original.
    \item\(\min(x)\): valor mínimo de la característica.
    \item\(\max(x)\): valor máximo de la característica.
\end{itemize}
```
Los datos resultantes tienen una distribución similar a una normal estándar, pero no están
limitados a un rango específico.

**Cuándo usarla**:

Ideal cuando las características tienen diferentes unidades o distribuciones (ej.
edad en años, ingreso en dólares).
k-NN se beneficia porque las distancias no se ven dominadas por características con rangos más
amplios.

### 2. **Normalización**

- **Fórmula** (normalización min-max):
```tex
% Defining the standardization formula
\section*{Estandarización}
La estandarización transforma los datos para que tengan media 0 y desviación estándar 1:
\[
x' = \frac{x - \mu}{\sigma}
\]
donde:
\begin{itemize}
\item \(x\): valor original.
\item \(\mu\): media de la característica.
\item \(\sigma\): desviación estándar de la característica.
\end{itemize}
```

- **Efecto**:
  Todos los valores se reescalan al rango [0, 1], preservando las proporciones relativas.
- **Cuándo usarla**:
  Útil cuando los datos no tienen una distribución normal o cuando se desea un rango
  específico.
  En k-NN, asegura que todas las características contribuyan equitativamente a la distancia.

### Diferencias clave

| **Aspecto**           | **Estandarización*                          | **Normalización**                            |
|-----------------------|---------------------------------------------|----------------------------------------------|
| **Base**              | Media y desviación estándar                 | Mínimo y máximo                              |
| **Rango de salida**   | No limitado (centrado en 0, desv. 1)        | Fijo, generalmente [0, 1]                    |
| **Distribución**      | Conserva la forma de la distribución        | No necesariamente conserva la distribución   |
| **Uso en k-NN**       | Si los datos tienen distribuciones gaussianas o escalas muy diferentes | Mejor si los datos requieren un rango uniforme o no son gaussianos |
| **Sensibilidad**      | Menos sensible a valores extremos           | Sensible a valores extremos (afecta min/max) |

## K-NN con Pesos

Lo que pasa es que pueden darse de que simplemente existan mas vecinos de una clase, pero que
realmente los mas cercanos son de otra clase.
Entonces damos mas pesos a los puntos mas cercanso al caso de estudio, para calcular esto se
hace con:

```
peso = (dmax - punto) / (dmax - dmin)
peso = 1 si es que dk = punto

punto = punto que se esta analizando
```

# Algoritmo para eliminar ruido y casos falsos

Tomek links se dan cuando:
- x es el mas cercano de y
- y es el mas cercano de x
- X e y son de clases distintas

Eliminar estos links nos ayuda a limpiar los casos anomalos y el ruido de nuestros datos de
entrenamiento.

1. Se tiene i = 1 y T un conjunto vacio
2. x es el i-esimo ejemplo de entrenamiento e Y es el vecino mas cercano de x.
3. Si x e Y son de la misma clase, saltar al paso 5.
4. Si x es vecino mas cercano de y, agregar a {x, y} a T
5. Si i <= N, volver al punto 2 (iterar por el resto de NN)
6. Eliminar los links tomek
