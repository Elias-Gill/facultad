# Generadores del lenguaje

Los generadores son utiles para describir como se describe la sintaxis de un lenguaje

Los "reconocedores" sirven para tomar cadenas de lenguaje y nos dice si esa cadena pertenece a un lenguaje.
Los que veremos son las expresiones regulares y los automatas deterministas finitos.

Ambos reconocen y describen lenguajes formales REGULARES.

Los lenguajes regulares son lenguajes que pueden ser definidos por una expresion regular o reconocidos por 
un automata finito.

> El lenguaje {0n1n∣n≥0}{0n1n∣n≥0} representa el conjunto de cadenas que consisten en una cantidad 
> igual de ceros seguida por una cantidad igual de unos, como por ejemplo: "", "01", "0011", "000111", etc. 
> Este lenguaje es un ejemplo clásico de un lenguaje no regular según el lema del bombeo para lenguajes no regulares.

> Un ejemplo de lenguaje regular: {0n1m∣n,m≥0}, el cual puede ser representado por la expresion regular `0* 1*`

# Automatas finitos

En resumen, un DFA sigue un camino claro y predecible, como seguir un conjunto de reglas estrictas en un juego. 
Por otro lado, un NFA tiene opciones múltiples en cada paso, como explorar diferentes caminos sin estar seguro 
de cuál es el mejor. ¡Así que, dependiendo del juego (o problema), a veces uno es más útil que el otro!

Cada NFA se puede transformar en un DFA y todo DFA tiene representacion en un NFA.

## Deterministas (DFA)

- Es como seguir un camino fijo en el juego.
- Cada vez que estás en un estado (una situación), sabes exactamente qué hacer a continuación. 
No hay opciones, solo una ruta clara.
- Es como un juego con reglas estrictas donde cada decisión lleva a un resultado específico. Sin sorpresas.

## Deterministas (NFA)

Son simples para expresar lenguajes mas complejos, los cuales serian muy tediosos con los DFA.

- Es como tener varias opciones en un juego y no estar seguro de cuál es la mejor.
- En un estado, puedes tener múltiples opciones de movimiento. Puedes probar diferentes caminos al mismo tiempo.
- Es como explorar diferentes rutas sin saber cuál te llevará al final. Puedes probar suerte y ver qué pasa.

El simbolo `λ` representa un nuevo estado sin que la maquina lea un nuevo caracter.

# Pumping Lemma
Imagina que tienes un lenguaje de palabras en las que las letras son solo 0 y 1. El Pumping Lemma dice que si 
ese lenguaje es regular, entonces para cualquier palabra lo suficientemente larga, puedes dividirla en tres partes, 
digamos "antes", "medio" y "después". Así que, para una palabra en el lenguaje, XYZ, aquí está el trato:

"Antes" y "después" pueden ser cualquier combinación de 0s y 1s, pero la parte "medio" (la Y) no puede ser vacía.
La "medio" (Y) es la parte que podrías repetir (o "bombardear") tantas veces como quieras y aún así obtener 
una palabra en el lenguaje.

La palabra XYZ también debe estar en el lenguaje.

Entonces, si tomas una palabra larga en el lenguaje, como "0011001", según el Pumping Lemma, deberías poder 
dividirla en tres partes de una manera especial, y al repetir la parte del medio, aún obtendrías palabras 
en el lenguaje.

`No entendi nadaite lgmt`.

EJERCICIOS paginas 95 a 98 del libro de "principios de compiladores".
