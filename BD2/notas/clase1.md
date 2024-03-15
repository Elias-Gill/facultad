*Lectura*: capitulos 11 a 19.

# NQV
Dos discos espejados tienen un TMDF de 57 mil anhos. Para calcular el TMDF de un grupo de discos "independientes", 
se hace `TMDF del disco / cant. discos`

## Medios de almacenamiento
Mas abajo +capacidad -velocidad. Mas arribal +velocidad -capacidad.

- Cache (volatil)
- RAM (volatil)
- Memoria flash (ssd y usb)
- Disco magnetico
- Disco optico (cd)
- Cintas magneticas

## Las medidas de rendimiento de un medio de almacenamiento
- Tiempo de acceso: tiempo de lectura, busqueda y escritura.
- Taza de transferencia: volumen de datos por unidad de tiempo que se puede transferir al medio.
- Tiempo medio de falla (TMDF): tiempo medio que tarda un medio de almacenamiento en fallar (morir).

## RAID's
Redundant Arrays of Independent Disks. Tecnica para organizar varios discos para hacer que se comporten como 
"uno solo". 

Esto provee mayor confiabilidad, velocidad y capacidad. Funciona combinando grupos de discos "independientes" 
con grupos de discos "espejados".

_RAID 0_: No existe redundancia.

_RAID 1_: discos de pareado espejados. Discos que trabajan duplicando sus datos en otros discos "espejos".

_RAID 5_: se utiliza con paridad. Se sacrifica parte del almacenamiento de los discos para almacenar "bits de paridad" para los demas discos.
Esto permite recuperar los datos en casos de fallas. 
