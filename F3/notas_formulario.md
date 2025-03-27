# Clasificación de Fórmulas de Física por Tipo de Ejercicio

## 1. Movimiento Armónico Simple (MAS) - Oscilaciones Libres
**Fórmulas:** (1)-(10).
**Temas clave:**
- `x = A sin(ωt + α)` - Ecuación de posición
- `ω = √(k/m)` - Frecuencia angular natural
- `a = -ω²x` - Relación aceleración-posición
- `E = ½kA²` - Energía total del sistema

## 2. Oscilaciones Amortiguadas
**Fórmulas:** (11)-(19).

**Ecuaciones principales:**
- `x(t) = Ae^(-γt)sin(ω_d t + φ)` - Solución subamortiguada
- `τ = 1/(2γ)` - Constante de tiempo
- `Q = ω_d/(2γ)` - Factor de calidad

## 3. Oscilaciones Forzadas y Resonancia
**Fórmulas:** (20)-(22).

**Destacado:**
- `A = (F₀/m)/√[(ω₀²-ω²)² + (2γω)²]` - Amplitud en régimen estacionario

## 4. Ondas Mecánicas
**Fórmulas:** (23)-(36).

**Conceptos clave:**
- `v = √(T/μ)` - Velocidad en cuerdas
- `I = ½ρv(ωs_max)²` - Intensidad sonora
- `β = 10 log(I/I₀)` - Nivel de intensidad

## 5. Interferencia y Ondas Estacionarias
**Fórmulas:** (37)-(42).

**Condiciones importantes:**
- `Δr = nλ` - Máximos de interferencia
- `y = 2A sin(kx)cos(ωt)` - Onda estacionaria

## 6. Electromagnetismo (Ondas EM)
**Fórmulas:** (43)-(54).

**Ecuaciones fundamentales:**
- `v = 1/√(εμ)` - Velocidad ondas EM
- `S = (1/μ₀)E×B` - Vector de Poynting
- `p_rad = I/c` - Presión de radiación

## 7. Misceláneos
**Fórmulas:** (55)-(56).

**Incluye:** 
- Expansión binomial
- Momento de inercia de esfera (`I = (2/5)MR²`)

**Leyenda:**  
- `ω`:
  Frecuencia angular  
- `γ`:
  Coeficiente de amortiguamiento  
- `Q`:
  Factor de calidad  
- `λ`:
  Longitud de onda  
- `μ`:
  Densidad lineal (cuerdas)

---

# Tabla de Símbolos Físicos (Ondas y Oscilaciones)

## Símbolos Fundamentales

| Símbolo | Nombre                     | Unidades (SI)   | Definición/Relación                |
|---------|----------------------------|-----------------|------------------------------------|
| **x**   | Elongación                 | m               | Posición instantánea               |
| **A**   | Amplitud                   | m               | Máximo desplazamiento             |
| **T**   | Periodo                    | s               | Tiempo por ciclo                  |
| **f**   | Frecuencia                 | Hz (s⁻¹)        | `f = 1/T`                         |
| **ω**   | Frecuencia angular         | rad/s           | `ω = 2πf`                         |
| **φ**   | Fase inicial               | rad             | Ángulo en t=0                     |
| **k**   | Constante elástica         | N/m             | Ley de Hooke (`F = -kx`)           |

## Oscilaciones Amortiguadas/Forzadas

| Símbolo | Nombre                     | Unidades (SI)   | Definición/Relación                |
|---------|----------------------------|-----------------|------------------------------------|
| **γ**   | Coefic. amortiguamiento    | s⁻¹             | `γ = b/(2m)`                      |
| **τ**   | Constante de tiempo        | s               | `τ = 1/(2γ)`                      |
| **Q**   | Factor de calidad          | Adimensional    | `Q = ω₀/(2γ)`                     |
| **ω_d** | Frec. amortiguada          | rad/s           | `ω_d = √(ω₀² - γ²)`               |

## Ondas Mecánicas

| Símbolo | Nombre                     | Unidades (SI)   | Definición/Relación                |
|---------|----------------------------|-----------------|------------------------------------|
| **v**   | Velocidad de onda          | m/s             | `v = λf`                          |
| **λ**   | Longitud de onda           | m               | Distancia entre crestas           |
| **μ**   | Densidad lineal            | kg/m            | Masa por unidad de longitud       |
| **I**   | Intensidad                 | W/m²            | Potencia por área                 |
| **β**   | Nivel sonoro               | dB              | `β = 10 log(I/I₀)`                |

## Electromagnetismo

| Símbolo | Nombre                     | Unidades (SI)   | Definición/Relación                |
|---------|----------------------------|-----------------|------------------------------------|
| **E**   | Campo eléctrico            | V/m             | `E = E₀ cos(kx - ωt)`             |
| **B**   | Campo magnético            | T               | `B = B₀ cos(kx - ωt)`             |
| **S**   | Vector de Poynting         | W/m²            | `S = (1/μ₀)E×B`                   |
| **c**   | Velocidad de la luz        | m/s             | `c = 1/√(ε₀μ₀)`                   |

## Constantes Universales

| Símbolo | Valor                     | Unidades (SI)   |
|---------|---------------------------|-----------------|
| **ε₀**  | 8.85×10⁻¹²                | C²/N·m²         |
| **μ₀**  | 4π×10⁻⁷                   | N/A²            |
| **I₀**  | 1×10⁻¹²                   | W/m²            |

**Notas:**
1. Para ondas en cuerdas:
   `v = √(T/μ)`
2. Para sonido en aire:
   `v ≈ 331 + (0.6T) m/s` (T en °C)
3. En medios materiales:
   `c → v = c/n` (n = índice refracción)
