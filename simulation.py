import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameter dari soal
V1max = 5.0
Km1 = 2.0
Ki = 3.0
X = 10.0

k2 = 1.0
k3 = 0.8
k4 = 0.3

# Model ODE
def metabolic_model(t, y):
    A, B, P = y

    # Reaction rates
    v1 = (V1max * X) / ((Km1 + X) * (1 + P / Ki))
    v2 = k2 * A
    v3 = k3 * B
    v4 = k4 * A

    # Differential equations
    dA_dt = v1 - v2 - v4
    dB_dt = v2 - v3
    dP_dt = v3

    return [dA_dt, dB_dt, dP_dt]

# Initial conditions
A0 = 0
B0 = 0
P0 = 0

y0 = [A0, B0, P0]

# Simulasi 48 jam
t_span = (0, 48)
t_eval = np.linspace(0, 48, 500)

solution = solve_ivp(
    metabolic_model,
    t_span,
    y0,
    t_eval=t_eval
)

# Hasil
t = solution.t
A = solution.y[0]
B = solution.y[1]
P = solution.y[2]

# Visualisasi
plt.figure(figsize=(8,5))
plt.plot(t, A, label='Metabolite A')
plt.plot(t, B, label='Metabolite B')
plt.plot(t, P, label='Product P')

plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Metabolic Pathway Simulation')
plt.legend()
plt.grid(True)

# Simpan gambar
plt.savefig("simulation_result.png", dpi=300)

plt.show()