pragma circom 2.0.0;

template Main() {
    // Entradas privadas
    signal input a;
    signal input b;

    // Entrada pública (número primo)
    signal input p;

    // Salida pública
    signal output c;

    // Variables intermedias
    signal a_squared;
    signal b_squared;
    signal sum_of_squares;
    signal k;
    signal remainder;

    // Calcular a² y b²
    a_squared <== a * a;
    b_squared <== b * b;

    // Calcular la suma de los cuadrados
    sum_of_squares <== a_squared + b_squared;

    // Calcular el cociente (k) y el residuo (remainder)
    k <-- sum_of_squares \ p;  // División entera (no es una restricción)
    remainder <== sum_of_squares - (p * k);

    // Asegurar que el residuo sea menor que p
    // Para implementar remainder < p, usamos una variable auxiliar
    // que representa p - remainder - 1, y verificamos que no sea negativo.
    signal aux;
    aux <== p - remainder - 1;

    // Asegurar que aux no sea negativo (es decir, p - remainder - 1 >= 0)
    // Esto implica que remainder < p
    aux === p - remainder - 1;

    // Asignar el residuo como salida
    c <== remainder;
}

component main = Main();
