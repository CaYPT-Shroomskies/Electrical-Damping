import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from scipy.integrate import solve_ivp  # noqa

# Completed to first order approximation
# Dipole model, assume small angle across area of coil
# Given the motion is axial, the dipole equation simplifies to:
# B (tesla) = (u_0/4pi) * (2u/r^3)
pi: float = np.pi
u_0: float = (10**-7) * (4 * pi)
mass: float = 0.05  # kg
m: float = 0.6  # dipole moment
k: float = 10
radius: float = 0.02  # Assume same radius for magnet coil
r: float = 50  # ohms
N = 1  # number of coils
L: float = u_0 * pi * radius**2 * 10**2 / 0.02  # Inductance of the coil V = L (dI/dt)

initial: np.ndarray = np.array([0.1, 0, 0])  # position, velocity, I
runtime = 2


def step(t, state):
    x, x_dot, I = state
    V: float = (pi * radius**2 * N) * (1e-7) * 2 * m * x_dot * 3 / (x**4)
    # voltage is dPhi
    m_coil = I * N * pi * radius**2
    F_b = (10**-7) * 6 * m_coil * m / x**4

    return np.array([x_dot, -(k * x + F_b) / m, (V - I * r) / L])


time_array = np.linspace(0, runtime, int(runtime * 60))
returned = solve_ivp(
    step,
    [0, runtime],
    initial,
    t_eval=time_array,
    method="LSODA",
)

solve = returned.y.T

fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(time_array, solve[:, 0], label="Position")
ax1.plot(time_array, solve[:, 2], label="Current")
ax1.set_ylabel("Current (A)")
ax0.set_ylabel("Position (m)")
ax1.set_xlabel("Time (s)")
plt.show()
