import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from scipy.integrate import solve_ivp  # noqa

# Completed to first order approximation
# Dipole model, assume small angle across area of coil
# Given the motion is axial, the dipole equation simplifies to:
# B (tesla) = (u_0/4pi) * (2u/r^3)
#
# NOTE: The dipole approximation asymptotes as the magnet passes through the coil.
# This is not the case, so we enforce a minimum "distance"
# This causes a discontinuity in the current graph

pi: float = np.pi
u_0: float = (10**-7) * (4 * pi)
mass: float = 0.05  # kg
m: float = 0.6  # dipole moment
k: float = 10
radius: float = 0.02  # Assume same radius for magnet coil
r: float = 20  # ohms
N = 40  # number of coils
L: float = 100 * u_0 * pi * radius**2 * N**2 / 0.01  # Inductance of the coil
initial: np.ndarray = np.array([0.01, 0, 0])  # position, velocity, I
runtime = 40
dipole_threshold = 0.01  # where to stop the dipole asymptoting


def step(t, state):
    x, x_dot, I = state
    V: float = (
        (pi * radius**2 * N)
        * (1e-7)
        * 2
        * m
        * x_dot
        * 3
        * np.sign(x)
        / np.sqrt(x**2 + dipole_threshold**2) ** 4
    )

    # emf is dPhi
    m_coil = I * N * pi * radius**2
    F_b = (
        (10**-7)
        * np.sign(x)
        * 6
        * m_coil
        * m
        / np.sqrt(x**2 + dipole_threshold**2) ** 4
    )
    return np.array([x_dot, -(k * x + F_b) / m, (V - I * r) / L])


time_array = np.linspace(0, runtime, int(runtime * 1000))
returned = solve_ivp(
    step,
    [0, runtime],
    initial,
    t_eval=time_array,
    method="LSODA",
)

solve = returned.y.T

fig, (ax0, ax1, ax2) = plt.subplots(3)
ax0.plot(time_array, solve[:, 0], label="Position")
ax1.plot(time_array, solve[:, 1], label="Velocity")
ax2.plot(time_array, solve[:, 2], label="Current")
ax2.set_ylabel("Current (A)")
ax1.set_ylabel("Velocity (m/s)")
ax0.set_ylabel("Position (m)")
ax2.set_xlabel("Time (s)")
ax0.grid()
ax1.grid()
ax2.grid()
plt.tight_layout()
plt.show()
