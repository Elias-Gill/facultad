Respuesta:
> Una expresión regular no se puede analizar directamente como una gramática LL(1), ya que las
> expresiones regulares describen lenguajes, no gramáticas.
> 
> Sin embargo, como todo lenguaje regular puede representarse mediante un AFD, y de todo AFD se
> puede construir una gramática lineal por la derecha, entonces:
> 
> Toda expresión regular tiene una gramática LL(1) equivalente.
> Por tanto, el lenguaje descrito por (a | b)*(ab | ba)? sí admite una gramática LL(1).

## Lenguaje vs Gramática

Un **lenguaje** es simplemente **un conjunto de cadenas** (strings) formadas por símbolos de un
alfabeto.

Por ejemplo:

* El lenguaje `{ "ab", "aabb", "abab", "ba" }` es un conjunto finito de cadenas.
* El lenguaje `L = { aⁿbⁿ | n ≥ 1 }` es un conjunto infinito de cadenas como `"ab", "aabb",
  "aaabbb", ..."`.

Una **gramática** es una forma de **describir cómo se generan las cadenas de un lenguaje**.
Es como una "receta" con reglas que permiten construir todas las cadenas válidas del lenguaje.

Una gramática se compone de:

* **Símbolos no terminales** (variables, como `S`, `A`, etc.)
* **Símbolos terminales** (los que aparecen en las cadenas finales, como `a`, `b`, etc.)
* **Reglas de producción** (como `S → aSb | ε`)
* **Símbolo inicial** (usualmente `S`)

| Concepto         | Lenguaje                    | Gramática                         |    |     |
| ---------------- | --------------------------- | --------------------------------- | -- | --- |
| ¿Qué es?         | Un conjunto de cadenas      | Un conjunto de reglas             |    |     |
| ¿Para qué sirve? | Representa lo que es válido | Describe cómo construir lo válido |    |     |
| Ejemplo          | `{ a, ab, abb }`            | \`S → aS                          | ab | ε\` |
| Tipo             | Concepto abstracto          | Herramienta formal                |    |     |

> El lenguaje es **el conjunto de frases** válidas de un idioma.
> La gramática es **el conjunto de reglas gramaticales** que indican cómo construir esas frases.
> **Una gramática es una herramienta para generar un lenguaje.**


Notas:
- "LL(1)" es una propiedad de gramáticas, no de lenguajes.
- Un lenguaje `puede tener o no una gramática LL(1) equivalente`.
- En el caso de los `lenguajes regulares`, siempre tienen al menos una gramática LL(1), porque
  las right-linear grammars (que derivan directo del AFD) son LL(1).
