# zk_circuit_project

Este proyecto implementa un circuito zk-SNARK en Circom para verificar la operación:

```
c = (a² + b²) % p
```

donde:
- `a` y `b` son valores secretos.
- `p` es un número primo público.
- `c` es la salida pública.

## Estructura

- `circuits/` → Circuito en Circom.
- `artifacts/` → Archivos generados (WASM, claves, pruebas, etc.).
- `test/` → Pruebas e inputs de ejemplo.
- `public/` → Archivos para pruebas en el navegador.

## Requisitos

- Linux
- Node.js
- GNU Make

## Uso

### 1. Instalar dependencias y generar pruebas
```bash
make setup
```

### 2. Verificar la prueba generada
```bash
make verify
```

### 3. Probar en el navegador
```bash
make browser
```

### 4. Limpiar archivos generados
```bash
make clean
```
