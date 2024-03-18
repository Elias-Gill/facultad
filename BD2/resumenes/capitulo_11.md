# Almacenamiento

Los tipos de almacenamiento desde el mas rapido y costoso al mas lento y economico:

- Cache
- Memoria principal (RAM)
- Memoria flash (usb, ssd)
- Disco magnetico
- Disco optico (cd's)
- Almacenamiento en cinta 

## Medidas de rendimiento de los disps. de almacenamiento

- _Tiempo de acceso_: tiempo entre la solicitud de lectura o escritura hasta que se comienza a
transferir los datos.
- _Velocidad de trasnferencia de datos_
- _Tiempo medio de fallas_: tiempo medio que le toma a una unidad de memoria fallar.

## Bloques

Para optimizar el acceso al disco, las tablas en las bases de datos se guardan y leen
en "bloques". Los bloques son la minima cantidad de informacion transferida entre disco y
cpu.

Un bloque mas pequenho causa muchas mas operaciones I/O, un bloque mas grande
desperdicia memoria (puede ser que muchos de esos bloques quedan con "espacios
vacios"). El tamanho tipico de bloque suele ser de 4 a 16 KB.

## Algoritmo del ascensor

El algoritmo del ascensor es un algoritmo de planificacion de lecturas que permite que se
realicen la mayor cantidad de lecturas y escrituras en una sola "pasada" del disco.

Se intenta que se realicen la menor cantidad de cambios de direccion de lectura del
disco.

Este algoritmo se implementa en el controlador de la propia unidad de almacenamiento.

# Redundancia de discos y control de fallas

Si el _TMDF_ de un disco singular es de 100.000hs, entonces el de 100 discos trabajando a la
vez es de 1.000hs. Esto es una medida de probabilidad.

Cuantos mas discos individuales conformen una UNIDAD de almacenamiento, mas posibilidades
tiene la UNIDAD de fallar. Esto ya que hay muchas posibilidades de que uno de los discos
falle y corrompa la unidad entera.

Para solucionar esto se pueden utilizar discos espejados. Estos tienen un _TMPD_ de 57.000
anhos.

## Tecnicas RAID

"Redundant arrays of independent disks", es una tecnica de organizacion de discos que
provee la vision de que varios discos independientes trabajan como una sola unidad de almancenamiento.

Existen 6 niveles de raid, pero entre ellos tenemos:

- *RAID 0*: que no presenta ninguna medida de mitigacion de fallos.
- *RAID 1*: que utiliza discos trabajando de manera espejada. Su capacidad de memoria es de `memtotal/2`.
- *RAID 5*: guarda bits de paridad de los un disco en los demas discos. Para ello se reserva
un cierto espacio en los demas discos para guardar estos bits de paridad. Esto permite tan
solo gastar el espacio equivalente a solo un disco a cambio de contar con el mecanismo de
control de fallas. El espacio reservado en cada disco es de: `memDisco * (1/cant_discos)`

# Buffers de memoria

El _SGBD_ trata de optimizar el acceso al disco, por ello guarda en memoria principal
buffers de datos leidos del disco. Para ello cuenta con un sistema administrador de buffers.

El administrador se encarga de leer del disco los bloques requeridos, pero si esos bloques
ya se encuentran en la memoria principal, entonces los retorna sin la necesidad de
realizar nuevas lecturas del disco. De ser necesario este puede reemplazar bloques ya
cargados en memoria con los nuevos bloques que se necesitan.

## Tecnica de reemplazo

El administrador de buffers debe optar por una estrategia de reemplazo de bloques. Para
ello utiliza la tecnica de _MRU_. Esto se debe a que proporciana la mayor cantidad de
"aciertos" de cache al trabajar con operaciones en las bases de datos, minimizando el
uso del disco.

MRU es solo bueno para este caso, en lugares como diccionarios o indices, es menester
utilizar _LRU_. 

(elemental mi querido Gill, pensa y acordate de la explicacion del profesor).

# Organizacion de archivos

Para mejorar los tiempos de disco se requiere que los archivos contengan su memoria en
bloques contiguos del disco, asi son leidos con una sola pasada.

Esto no es posible por lo cambiante de los archivos, por tanto se produce la fragmentacion de
los datos por el disco, provocando el incremento del movimiento del peine del disco.

La solucion a ello es la utilizacion de herramientas de desfragmentacion.

`continuar`...

# Glosario

_RAID_: redundant arrays of independent disks
_TMDF_: tiempo medio de fallas.
_TMDR_: tiempo medio de reparacion.
_TMPD_: tiempo medio de perdida de datos. `formula:` (TMDF^2)/(cant_discos * TMDR)
_SGBD_: sistema gerenciador de bases de datos.
_MRU_: most recent used
_LRU_: last recent used
