# Fortran

## Historia y origen

Las computadoras en la epoca de 1940 y 1950 no contaban con procesadores con operaciones de punto flotante, 
por lo que la "lentitud" de los lenguajes interpretados era bastante aceptable. 
Aun asi, los programadores preferian trabajar a bajo nivel con instrucciones maquina

En 1954 IBM anuncia su maquina 704 y a su vez el nuevo lenguaje FORTRAN, el cual hace referencia a "FORmula TRANslator". 

A pesar de estos trabajos anteriores, el primer lenguaje de alto nivel compilado y ampliamente
aceptado fue Fortran.

El entorno en el que se desarrolló Fortran fue el siguiente: (1) Las computadoras tenían pequeñas
memorias y eran lentas y relativamente poco confiables; (2) el uso principal de las computadoras
era para cálculos científicos; (3) no existían formas eficientes y efectivas de programar
computadoras; y (4) debido al alto costo de las computadoras en comparación con el costo de los
programadores, la velocidad del código objeto generado fue el objetivo principal de los primeros
compiladores Fortran.

## Versiones principales
### FORTRAN 1

Fortran I se lanzo en 1956 e incluyó:

- El formato de entrada / salida,
- Nombres de variables de hasta seis caracteres (solo había dos en Fortran 0), 
- Subrutinas definidas por el usuario, aunque no pudieron compilarse por separado, 
- La declaración de selección If
- La instrucción de bucle Do.

Todas las variables con nombre: I, J, K, L, M, N; tenian tipo entero.
Todas las demas variables toman el valor de punto con coma flotante.

La primera version de FORTRAN podia casi igualar el 50% de la eficiencia del codigo escrito a mano.

### FORTRAN 2

Fue lanzado en 1958 y corrigio algunos errores de la primera version. 

Tambien añadio la compilacion independiente de subrutinas, lo que permitia compilar programas mas largos e incluir versiones 
de subrutinas precompiladas al programa principal.

### FORTRAN 4
La version 3 fue bastante intrascendente, pero la 4 ser convirtio en uno de los lenguejes mas usados de su tiempo.

Incluia: 

- Tipado de datos explicito
- Constructor logico para las sentencias if.
- Capacidad de pasar subprogramas como parametro a otros subprogramas.

### FORTRAN 77 y FORTRAN 90

FORTRAN 77 añadio el manejo de cadenas de caracteres, sentencias de control de bucle logico (while) y una sentencia else opcional para el if.

FORTRAN 90 agrego matrices dinamicas, registros (structs), punteros, switch y modulos, ademas de soporte para recursividad.

Ademas, se elimino el formato rigido que venia de los tiempos donde se trabajaba con cartas perforadas 
(limite de caracteres, posicion especifica de escritura de los programas, etc).

### FORTRAN 95, 2003 y 2008

Se agrego una nueva construccion de iteracion llamada "ForAll" que facilitaba la paralelizacion de programas. Se agrego tambien soporte para programacion
orientada a objetos, tipos derivados parametrizados e interoperabilidad con el lenguaje C.

FORTRAN 2008 agrego soporte de ambitos locales, co-matrices que proporcionan un modulo de ejecucion paralelar

# Preguntas del trabajo:

2. Caracteristicas clave y paradigma:

Paradigma imperativo y procedural. 

- Tipado estatico fuerte
- Interoperabilidad con C
- Sistema de control de concurrencia y paralelizacion
- Versiones anteriores FORTRAN 90 su sintaxis estaba diseñadas para trabajar con "punch cards"

3. Impacto

Considerado como el primer lenguaje compilado de alto nivel y de uso masivo. Impulso el crecimiento de las ciencias informaticas.

4. Usos modernos

Software de prediccion meteorologica, aplicaciones matematicas y fisicas.

