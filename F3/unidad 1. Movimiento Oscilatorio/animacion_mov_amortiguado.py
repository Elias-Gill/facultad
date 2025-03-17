import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Parámetros del sistema
m = 1  # masa
k = 1  # constante del resorte
omega_0 = np.sqrt(k / m)

# Tiempos
t = np.linspace(0, 20, 1000)

# Amortiguamientos
b_sub = 0.2  # Subamortiguado (b^2 < 4mk)
b_crit = 2 * np.sqrt(m * k)  # Críticamente amortiguado (b^2 = 4mk)
b_sobre = 3.0  # Sobreamortiguado (b^2 > 4mk)

# Soluciones
A = 1  # Amplitud inicial

# Subamortiguado
omega_d = np.sqrt(omega_0**2 - (b_sub / (2 * m)) ** 2)
x_sub = A * np.exp(-b_sub * t / (2 * m)) * np.cos(omega_d * t)

# Críticamente amortiguado
x_crit = A * np.exp(-b_crit * t / (2 * m)) * (1 + (b_crit / (2 * m)) * t)

# Sobreamortiguado
r1 = -b_sobre / (2 * m) + np.sqrt((b_sobre / (2 * m)) ** 2 - omega_0**2)
r2 = -b_sobre / (2 * m) - np.sqrt((b_sobre / (2 * m)) ** 2 - omega_0**2)
C1, C2 = A / 2, A / 2
x_sobre = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

# Configuración de la animación
fig, ax = plt.subplots()
ax.set_xlim(0, 20)
ax.set_ylim(-1.1, 1.1)
(line_sub,) = ax.plot([], [], label="Subamortiguado")
(line_crit,) = ax.plot([], [], label="Críticamente amortiguado")
(line_sobre,) = ax.plot([], [], label="Sobreamortiguado")
ax.legend()


def update(frame):
    line_sub.set_data(t[:frame], x_sub[:frame])
    line_crit.set_data(t[:frame], x_crit[:frame])
    line_sobre.set_data(t[:frame], x_sobre[:frame])
    return line_sub, line_crit, line_sobre


ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=20)
plt.xlabel("Tiempo")
plt.ylabel("Desplazamiento")
plt.grid(True)
plt.show()
