# Proceso de cirrer una peticion SQL

Bases de datos son internamente archivos normales, organizados de una manera formal,
predecible

Cada tabla de la base de datos tiene asignado un archivo. Cada archivo se compone 
por un conjuntos de "bloques". Por tanto, la lectura de archivos se hace en bloques. 
Estos bloques contiene una cantidad x de filas de datos de la tabla

EL motor de la bd es un sistema experto, que trata de optimizar las consultas 
en tiempo de disco, que interpreta un lenguaje de 4ta generacion de algo nivel (sql).
