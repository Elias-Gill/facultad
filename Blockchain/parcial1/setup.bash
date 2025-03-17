#!/bin/bash

# Nombre del proyecto
PROJECT_NAME="zk_circuit_project"

# Crear la estructura de directorios
mkdir -p ${PROJECT_NAME}/{circuits,scripts,artifacts,test}
cd ${PROJECT_NAME}

# Crear el archivo README.md
cat <<EOL > README.md
# ${PROJECT_NAME}

Este proyecto implementa un circuito zk-SNARK utilizando Circom y snarkjs para verificar la operación:

\`\`\`
c = (a² + b²) % p
\`\`\`

donde:
- \`a\` y \`b\` son números secretos.
- \`p\` es un número primo público.
- \`c\` es la salida pública.

## Estructura del Proyecto

- \`circuits/\`: Contiene el archivo del circuito en Circom.
- \`scripts/\`: Scripts para compilar y probar el circuito.
- \`artifacts/\`: Archivos generados (WASM, claves, pruebas, etc.).
- \`test/\`: Pruebas y ejemplos de uso.

## Requisitos

- Node.js
- Circom
- snarkjs

## Instrucciones

1. Instalar dependencias:

\`\`\`bash
npm install -g circom snarkjs
\`\`\`

2. Compilar el circuito:

\`\`\`bash
make compile
\`\`\`

3. Generar las claves y la prueba:

\`\`\`bash
make setup
\`\`\`

4. Verificar la prueba:

\`\`\`bash
make verify
\`\`\`
EOL

# Crear el archivo del circuito en Circom
cat <<EOL > circuits/circuit.circom
pragma circom 2.0.0;

template Main() {
// Entradas privadas
signal private input a;
signal private input b;

// Entrada pública (número primo)
signal input p;

// Salida pública
signal output c;

// Operaciones aritméticas
signal a_squared;
signal b_squared;
signal sum_of_squares;

a_squared <== a * a;
b_squared <== b * b;
sum_of_squares <== a_squared + b_squared;

// Reducción modular
c <== sum_of_squares % p;
}

component main = Main();
EOL

# Crear el archivo Makefile
cat <<EOL > Makefile
# Variables
CIRCUIT_NAME=circuit
CIRCUIT_DIR=circuits
ARTIFACTS_DIR=artifacts
INPUT_FILE=test/input.json

# Compilar el circuito
compile:
@echo "Compilando el circuito..."
circom \$(CIRCUIT_DIR)/\$(CIRCUIT_NAME).circom --r1cs --wasm --sym -o \$(ARTIFACTS_DIR)

# Generar las claves y la prueba
setup:
@echo "Generando claves y prueba..."
cd \$(ARTIFACTS_DIR)/\$(CIRCUIT_NAME)_js && \\
node generate_witness.js \$(CIRCUIT_NAME).wasm ../../\$(INPUT_FILE) witness.wtns && \\
snarkjs groth16 setup \$(CIRCUIT_NAME).r1cs ../../../powersOfTau28_hez_final_10.ptau \$(CIRCUIT_NAME)_0000.zkey && \\
snarkjs zkey contribute \$(CIRCUIT_NAME)_0000.zkey \$(CIRCUIT_NAME)_0001.zkey --name="Contribución 1" -v && \\
snarkjs zkey export verificationkey \$(CIRCUIT_NAME)_0001.zkey verification_key.json && \\
snarkjs groth16 prove \$(CIRCUIT_NAME)_0001.zkey witness.wtns proof.json public.json

# Verificar la prueba
verify:
@echo "Verificando la prueba..."
cd \$(ARTIFACTS_DIR)/\$(CIRCUIT_NAME)_js && \\
snarkjs groth16 verify verification_key.json public.json proof.json

# Limpiar archivos generados
clean:
@echo "Limpiando archivos..."
rm -rf \$(ARTIFACTS_DIR)/*
EOL

# Crear un archivo de entrada de prueba
cat <<EOL > test/input.json
{
    "a": 3,
    "b": 4,
    "p": 17
}
EOL

# Crear un script para instalar dependencias
cat <<EOL > scripts/setup.sh
#!/bin/bash

# Instalar dependencias
npm install -g circom snarkjs

# Descargar el archivo de potencias de Tau
wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau -O powersOfTau28_hez_final_10.ptau
EOL

# Dar permisos de ejecución al script de setup
chmod +x scripts/setup.sh

# Mensaje final
echo "Proyecto creado en la carpeta '${PROJECT_NAME}'."
echo "Instala las dependencias ejecutando:"
echo "  cd ${PROJECT_NAME} && ./scripts/setup.sh"
echo "Luego, compila y prueba el circuito con:"
echo "  make compile"
echo "  make setup"
echo "  make verify"
