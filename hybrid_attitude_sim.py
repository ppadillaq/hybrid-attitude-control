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

def noisy_quaternion_measurement(q, noise_bound=0.4):
    """
    Quaternion measurement with bounded random noise:

    q_tilde = (q + e) / ||q + e||

    with ||e||^2 <= noise_bound.
    """

    # Random direction in R^4
    e = np.random.uniform(-1.0, 1.0, size=4)

    # Normalize direction
    e = e / np.linalg.norm(e)

    # Random squared norm uniformly distributed in [0, noise_bound]
    e_norm_squared = np.random.uniform(0.0, noise_bound)

    # Set ||e||^2 = e_norm_squared
    e *= np.sqrt(e_norm_squared)

    # Noisy quaternion measurement
    q_tilde = q + e
    q_tilde /= np.linalg.norm(q_tilde)

    return q_tilde


def dynamics(t, x, J, tau):
    """
    Rigid-body attitude plant dynamics:

    q_dot = 1/2 * q ⊗ nu(omega)

    omega_dot = J^-1 * (S(J*omega)*omega + tau)

    State:
        x = [q, omega]
        q     = [eta, eps1, eps2, eps3]
        omega = [wx, wy, wz]

    Input:
        tau = control torque
    """

    # States
    q = x[:4]
    omega = x[4:]

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


def plot_results(results):
    """
    Plot and compare attitude-control simulation results.

    Parameters
    ----------
    results : list of dict
        Each dictionary must contain:
        'times', 'h_values', 'etas', 'omega_squared',
        'actuator_effort', and 'label'.
    """

    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    for result in results:

        times = result["times"]
        label = result["label"]

        # Logic variable h
        ax[0].plot(
            times,
            result["h_filtered"],
            label=label
        )

        # Quaternion scalar part eta
        ax[1].plot(
            times,
            result["etas"],
            label=label
        )

        # Angular velocity squared
        ax[2].plot(
            times,
            result["omega_squared"],
            label=label
        )

        # Cumulative control effort
        ax[3].plot(
            times,
            result["actuator_effort"],
            label=label
        )

    ax[0].set_ylabel(r"$\mathcal{F}h$")
    ax[1].set_ylabel(r"$\eta$")
    ax[2].set_ylabel(r"$\omega^T\omega$")
    ax[3].set_ylabel(r"$\int_0^t \tau^T\tau\,dt$")
    ax[3].set_xlabel("Time [s]")

    for axis in ax:
        axis.grid(True)

    ax[0].legend()

    fig.suptitle("Hybrid Attitude Control")
    plt.tight_layout()
    plt.show()


def simulate_energy(
    J, # inertia matrix
    c,
    K_omega,
    delta, # hysteresis parameter
    q0,
    omega0,
    h0=1, # Logic variable
    t0=0.0,
    tf=40.0,
    dt=0.01
):
    """
    Simulate the hybrid rigid-body attitude dynamics using
    the energy-based controller with noisy quaternion measurements.
    """

    q0 = q0 / np.linalg.norm(q0)
    x_current = np.concatenate((q0, omega0))

    h = h0
    t_current = t0

    times = []
    etas = []
    h_values = []
    omega_squared = []
    quaternions = []
    tau_squared = []

    while t_current < tf:

        # Current state
        q_current = x_current[:4]
        omega_current = x_current[4:]

        # One noisy measurement per sampling period
        q_tilde = noisy_quaternion_measurement(q_current)

        eta_tilde = q_tilde[0]
        eps_tilde = q_tilde[1:]

        # Hybrid jump logic
        if h * eta_tilde <= -delta:
            h = -h

        # Energy-based controller
        tau = -c * h * eps_tilde - K_omega @ omega_current

        # Integrate plant dynamics over one sampling period
        sol = solve_ivp(
            lambda t, x: dynamics(t, x, J, tau),
            [t_current, min(t_current + dt, tf)],
            x_current,
            method="RK45"
        )

        # Updated state
        x_current = sol.y[:, -1]

        q_current = x_current[:4]
        omega_current = x_current[4:]

        # Normalize quaternion
        q_current = q_current / np.linalg.norm(q_current)
        x_current[:4] = q_current

        eta = q_current[0]
        #eps = q_current[1:]

        # Store results
        times.append(sol.t[-1])
        etas.append(eta)
        h_values.append(h)
        omega_squared.append(omega_current.T @ omega_current)
        quaternions.append(q_current.copy())
        tau_squared.append(tau.T @ tau)

        t_current = sol.t[-1]

    actuator_effort = cumulative_trapezoid(
        tau_squared,
        times,
        initial=0.0
    )

    # -----------------------------------
    # Filter logic variable for plotting
    # -----------------------------------

    # First-order low-pass filter used in the paper:
    # F(s) = beta / (s + beta)
    beta = 10.0

    # Exact discrete-time decay factor for the sampling period dt
    alpha = np.exp(-beta * dt)

    # Initialize filtered signal with the same initial condition as h
    h_filtered = np.zeros(len(h_values), dtype=float)
    h_filtered[0] = h0

    # Apply first-order low-pass filter to h
    for k in range(1, len(h_values)):
        h_filtered[k] = (
            alpha * h_filtered[k - 1]
            + (1.0 - alpha) * h_values[k]
        )

    return {
        "times": np.asarray(times),
        "h_values": np.asarray(h_values),
        "h_filtered": np.asarray(h_filtered),
        "etas": np.asarray(etas),
        "omega_squared": np.asarray(omega_squared),
        "actuator_effort": np.asarray(actuator_effort),
        "quaternions": np.asarray(quaternions)
    }


def simulate_backstepping(
    J,
    c,
    K_eps,
    K_z,
    delta,
    q0,
    omega0,
    h0=1,
    t0=0.0,
    tf=40.0,
    dt=0.01
):
    """
    Simulate the hybrid rigid-body attitude dynamics using
    the backstepping controller.

    Backstepping variable:
        z = omega + h*K_eps*eps

    Control law:
        tau = -S(J*omega)*omega
              - (h/2)*J*K_eps*(eta*I + S(eps))*omega
              - K_z*z
              - c*h*eps

    Hybrid switching function:
        Phi(q, omega) = eta
                        - (1/(2*c))*omega^T*J*K_eps*eps

    Jump condition:
        h*Phi(q, omega) <= -delta
    """

    # Normalize initial quaternion
    q0 = q0 / np.linalg.norm(q0)
    x_current = np.concatenate((q0, omega0))

    h = h0
    t_current = t0

    times = []
    etas = []
    h_values = []
    omega_squared = []
    quaternions = []
    tau_squared = []

    while t_current < tf:

        # Current state
        q_current = x_current[:4]
        omega_current = x_current[4:]

        # One noisy quaternion measurement per sampling period
        q_tilde = noisy_quaternion_measurement(q_current)

        eta_tilde = q_tilde[0]
        eps_tilde = q_tilde[1:]

        # Backstepping switching function
        Phi = (
            eta_tilde
            - (1.0 / (2.0 * c))
            * omega_current.T @ J @ K_eps @ eps_tilde
        )

        # Hybrid jump logic
        if h * Phi <= -delta:
            h = -h

        # Backstepping variable
        z = omega_current + h * K_eps @ eps_tilde

        # Backstepping control torque
        tau = (
            -skew(J @ omega_current) @ omega_current
            - 0.5 * h * J @ K_eps
            @ (eta_tilde * np.eye(3) + skew(eps_tilde))
            @ omega_current
            - K_z @ z
            - c * h * eps_tilde
        )

        # Integrate plant dynamics over one sampling period
        sol = solve_ivp(
            lambda t, x: dynamics(t, x, J, tau),
            [t_current, min(t_current + dt, tf)],
            x_current,
            method="RK45"
        )

        # Updated state
        x_current = sol.y[:, -1]

        q_current = x_current[:4]
        omega_current = x_current[4:]

        # Normalize quaternion
        q_current = q_current / np.linalg.norm(q_current)
        x_current[:4] = q_current

        # Store results
        times.append(sol.t[-1])
        etas.append(q_current[0])
        h_values.append(h)
        omega_squared.append(omega_current.T @ omega_current)
        quaternions.append(q_current.copy())
        tau_squared.append(tau.T @ tau)

        t_current = sol.t[-1]

    # Cumulative control effort
    actuator_effort = cumulative_trapezoid(
        tau_squared,
        times,
        initial=0.0
    )

    # Filter h for visualization:
    # F(s) = beta / (s + beta)
    beta = 10.0
    alpha = np.exp(-beta * dt)

    h_filtered = np.zeros(len(h_values), dtype=float)
    h_filtered[0] = h0

    for k in range(1, len(h_values)):
        h_filtered[k] = (
            alpha * h_filtered[k - 1]
            + (1.0 - alpha) * h_values[k]
        )

    return {
        "times": np.asarray(times),
        "h_values": np.asarray(h_values),
        "h_filtered": np.asarray(h_filtered),
        "etas": np.asarray(etas),
        "omega_squared": np.asarray(omega_squared),
        "actuator_effort": np.asarray(actuator_effort),
        "quaternions": np.asarray(quaternions)
    }


# -----------------------------------
# Parameters
# -----------------------------------

# Inertia matrix
J = np.diag([4.35, 4.33, 3.664])

# Energy-based controller parameters
c = 0.5
K_omega = 0.5 * np.eye(3)

# -----------------------------------
# Initial conditions
# -----------------------------------

# Rotation axis used in the paper
v_hat = np.array([3.0, -4.0, 5.0])

# Unit rotation axis
v = v_hat / np.linalg.norm(v_hat)

# Initial quaternion:
# eta = 0 corresponds to a 180-degree rotation
# from the desired attitude
q0 = np.concatenate(([0.0], v))

# Initial angular velocity
omega0 = np.zeros(3)

# Initial logic variable
h0 = 1

# Simulations

results_discontinuous = simulate_energy(
    J, c, K_omega,
    delta=0.0,
    q0=q0,
    omega0=omega0
)

results_hysteresis = simulate_energy(
    J, c, K_omega,
    delta=0.45,
    q0=q0,
    omega0=omega0
)

results_discontinuous["label"] = r"Discontinuous ($\delta=0$)"
results_hysteresis["label"] = r"Hysteresis ($\delta=0.45$)"

plot_results([
    results_discontinuous,
    results_hysteresis
])


# visualizer = AttitudeVisualizer(
#     times=times,
#     quaternions=quaternions,
#     h_values=h_values,
#     etas=etas,
#     omega_squared=omega_squared,
#     actuator_effort=actuator_effort,
#     frame_step=20
# )

# visualizer.animate()



# -----------------------------------
# Simulation parameters - Fig. 2
# -----------------------------------

# Inertia matrix
J = np.diag([4.35, 4.33, 3.664])

# Common controller parameters
c = 1.0
delta = 0.45

# Energy-based controller gains
K_omega_energy = np.eye(3)

# Backstepping controller gains
K_eps = 0.5 * np.eye(3)
K_z = 0.25 * np.eye(3)

# Rotation axis
v_hat = np.array([3.0, -4.0, 5.0])
v = v_hat / np.linalg.norm(v_hat)

# Initial conditions for Fig. 2
q0 = np.array([1.0, 0.0, 0.0, 0.0])
omega0 = 2.0 * v
h0 = 1

# Simulation settings
tf = 20.0
dt = 1 / 1000


# -----------------------------------
# Energy-based controller
# -----------------------------------

results_energy = simulate_energy(
    J=J,
    c=c,
    K_omega=K_omega_energy,
    delta=delta,
    q0=q0,
    omega0=omega0,
    h0=h0,
    tf=tf,
    dt=dt
)

results_energy["label"] = "Energy-based"


# -----------------------------------
# Backstepping controller
# -----------------------------------

results_backstepping = simulate_backstepping(
    J=J,
    c=c,
    K_eps=K_eps,
    K_z=K_z,
    delta=delta,
    q0=q0,
    omega0=omega0,
    h0=h0,
    tf=tf,
    dt=dt
)

results_backstepping["label"] = "Backstepping"


# -----------------------------------
# Compare controllers
# -----------------------------------

plot_results([
    results_energy,
    results_backstepping
])