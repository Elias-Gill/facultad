# zk_circuit_project

Este proyecto implementa un circuito zk-SNARK utilizando Circom y snarkjs para verificar la operación:

```
c = (a² + b²) % p
```

donde:
- `a` y `b` son números secretos.
- `p` es un número primo público.
- `c` es la salida pública.

## Estructura del Proyecto

- `circuits/`: Contiene el archivo del circuito en Circom.
- `scripts/`: Scripts para compilar y probar el circuito.
- `artifacts/`: Archivos generados (WASM, claves, pruebas, etc.).
- `test/`: Pruebas y ejemplos de uso.

## Requisitos

- Node.js
- Circom
- snarkjs

## Instrucciones

1. Instalar dependencias:

```bash
npm install -g circom snarkjs
```

2. Compilar el circuito:

```bash
make compile
```

3. Generar las claves y la prueba:

```bash
make setup
```

4. Verificar la prueba:

```bash
make verify
```
