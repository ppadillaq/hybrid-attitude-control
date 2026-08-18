import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from attitude_visualizer import AttitudeVisualizer
from scipy.integrate import cumulative_trapezoid


def skew(v):
    """Skew-symmetric matrix S(v)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quaternion_product(q, p):
    """
    Quaternion product q ⊗ p

    q = [eta, eps1, eps2, eps3]
    p = [zeta, rho1, rho2, rho3]
    """
    eta = q[0]
    eps = q[1:]

    zeta = p[0]
    rho = p[1:]

    scalar = eta * zeta - np.dot(eps, rho)
    vector = eta * rho + zeta * eps + np.cross(eps, rho)

    return np.concatenate(([scalar], vector))


def dynamics(t, x, h, J, c, K_omega):
    """
    Rigid-body attitude dynamics with energy-based control:

    q_dot = 1/2 * q ⊗ nu(omega)

    omega_dot = J^-1 * (S(J*omega)*omega + tau)

    tau = -c*h*eps - K_omega*omega

    State:
        x = [q, omega]
        q     = [eta, eps1, eps2, eps3]
        omega = [wx, wy, wz]
    """

    # States
    q = x[:4]
    omega = x[4:]

    #eta = q[0]
    eps = q[1:]

    # Energy-based control torque
    tau = -c * h * eps - K_omega @ omega

    # Quaternion kinematics
    nu_omega = np.concatenate(([0.0], omega))
    q_dot = 0.5 * quaternion_product(q, nu_omega)

    # Rigid-body dynamics
    J_omega = J @ omega

    omega_dot = np.linalg.solve(
        J,
        skew(J_omega) @ omega + tau
    )

    # Complete state derivative
    return np.concatenate((q_dot, omega_dot))


# -----------------------------------
# Parameters
# -----------------------------------

# Inertia matrix
J = np.diag([4.35, 4.33, 3.664])

# Energy-based controller parameters
c = 0.5
K_omega = 0.5 * np.eye(3)

# Hysteresis parameter
delta = 0.45

# Initial quaternion (must be unit norm)
q0 = np.array([
    0.1,
    0.8,
    0.4,
    0.43
])

q0 = q0 / np.linalg.norm(q0)

omega0 = np.array([0.0, 0.0, 0.0])
x0 = np.concatenate((q0, omega0))

# Logic variable
h = 1

# Simulation settings
t0 = 0.0
tf = 40.0
#dt = 1 / 1000  # 0.001 s
dt = 1 / 100  # 0.01 s

times = []
etas = []
h_values = []
omega_squared = []
quaternions = []
tau_squared = []

t_current = t0
x_current = x0.copy()

# -----------------------------------
# Hybrid simulation loop
# -----------------------------------

while t_current < tf:

    # Integrate small interval
    sol = solve_ivp(
        lambda t, x: dynamics(t, x, h, J, c, K_omega),
        [t_current, min(t_current + dt, tf)],
        x_current,
        method="RK45"
    )

    x_current = sol.y[:, -1]

    q_current = x_current[:4]
    omega_current = x_current[4:]

    # Normalize quaternion
    q_current = q_current / np.linalg.norm(q_current)
    x_current[:4] = q_current

    eta = q_current[0]

    eps = q_current[1:]
    #omega = -h * K_eps @ eps

    tau = -c * h * eps - K_omega @ omega_current
    tau_squared.append(tau.T @ tau)

    # Store results
    omega_squared.append(omega_current.T @ omega_current)
    quaternions.append(q_current.copy())

    # Hybrid jump condition
    if h * eta <= -delta:
        h = -h
        print(f"Jump at t = {sol.t[-1]:.2f}, new h = {h}")

    times.append(sol.t[-1])
    etas.append(eta)
    h_values.append(h)

    t_current = sol.t[-1]

actuator_effort = cumulative_trapezoid(
    tau_squared,
    times,
    initial=0.0
)

# -----------------------------------
# Plot
# -----------------------------------

fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

# Logic variable h
ax[0].plot(times, h_values)
ax[0].set_ylabel(r"$h$")
ax[0].grid(True)

# Quaternion scalar part eta
ax[1].plot(times, etas)
ax[1].set_ylabel(r"$\eta$")
ax[1].grid(True)

# Angular velocity squared
ax[2].plot(times, omega_squared)
ax[2].set_ylabel(r"$\omega^T\omega$")
ax[2].grid(True)

# Cumulative control effort
ax[3].plot(times, actuator_effort)
ax[3].set_ylabel(r"$\int_0^t \tau^T\tau\,dt$")
ax[3].set_xlabel("Time [s]")
ax[3].grid(True)

fig.suptitle("Energy-Based Hybrid Attitude Control")
plt.tight_layout()
plt.show()

visualizer = AttitudeVisualizer(
    times=times,
    quaternions=quaternions,
    h_values=h_values,
    etas=etas,
    omega_squared=omega_squared,
    actuator_effort=actuator_effort,
    frame_step=20
)

visualizer.animate()