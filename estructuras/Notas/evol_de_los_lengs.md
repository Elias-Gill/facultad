# Caracteristicas de los lenguajes

Un lenguaje tiene dos componentes:

- La sintaxis
- La semantica

Y tiene dos destinos de comunicacion:

- Computadoras
- Otros programadores

## Desempeno vs facilidad de escritura

El desempenho de un lenguaje es mas predecible si se puede conocer como sera compilado el codigo
en instrucciones maquinas. Los lenguajes faciles de escribir, generalmente tienen la contra de ser menos
predecible en cuanto a su comportamiento a la hora de la compilacion.

Al tener instrucciones mas "simples", los lenguajes de mas bajo nivel son mas "predecibles"
que los lenguajes de alto nivel.

## Lenguajes de mas bajo nivel. Assembler

El lenguaje de ensamblador es el nievl mas bajo antes del codigo maquina. Pero depende de la maquina
y es casi tan dificil como escribir codigo maquina. Se pueden generar macros para abreviar
la escritura de codigo comun.

Los _ensambladores_ transforman codigo assembler a instrucciones maquinas directas.

## Lenguajes de alto nivel

Incluyen construcciones de nivel superior como abstracciones de tipo, subrutinas y funciones,
estructuras de datos mas complejas, etc.

Tienen la ventaja de que son mas faciles de escribir, leer, depurar y detectar errores.

### Compiladores vs interpretes

Los compiladores, a diferencia de los ensambladores, transforman el codigo fuente a codigo ensamblador,
pero ademas realizan procesos de deteccion de errores sintacticos y SEMANTICOS, lo que permite
detectar errores en tiempo de compilacion, los cuales de otra manera serian validos en
codigo assembler.

Ademas, compiladores ofrecen rendimiento superior sobre los interpretes, ya que puede realizar
optimizaciones y no se requiere "traducir" cada vez el programa.

Los interpretes en cambio "traducen" el codigo cada vez que se ejecuta, por tanto
los tiempos de ejecucion son mas lentos (de 10 a 100 veces mas lento). En contra parte son mas "faciles"
de desarrollar, ademas de ser mas "amigables" con el usuario final.

# Historia de los lenguajes

## Fortran (1957)

[Leer](./historia_fortran.md)

## Algol

Un punto de inflexion en el paradigma de la programacion se da con algol. Durante este tiempo se demostro
la correlacion entre las sintaxis libres de contexto y sintaxis regulares. ASi como el desarrollo de la
teoria de los compiladores. Con esto se genero una explosion de lenguajes de programacion.

#### Lenguajes Algol-like

- deben ser algoritmicos para describir procesos computacioneles
- imperativo
- compilado
- los bloques funcdamentales son los bloques y procedimientos
