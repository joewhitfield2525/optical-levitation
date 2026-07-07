import time

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Physical parameters for a 1D Brownian harmonic oscillator
# ---------------------------------------------------------------------
kB = 1.380649e-23
T = 293.0

m = 1.0e-12          # kg
k = 1.0e-7           # N/m
damping_ratio = 0.2
b = damping_ratio * 2 * np.sqrt(m * k)

omega0 = np.sqrt(k / m)
frequency = omega0 / (2 * np.pi)

x_rms_exact = np.sqrt(kB * T / k)
v_rms_exact = np.sqrt(kB * T / m)

print("Brownian harmonic oscillator benchmark")
print("m =", m, "kg")
print("k =", k, "N/m")
print("b =", b, "kg/s")
print("damping ratio =", damping_ratio)
print("natural frequency =", frequency, "Hz")
print("exact x_rms =", x_rms_exact, "m")
print("exact x_rms =", x_rms_exact * 1e9, "nm")
print("exact v_rms =", v_rms_exact, "m/s")
print()


# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------
t_end = 2.0
burn_in_fraction = 0.5
seed = 4

dt_values = [2e-5, 1e-5, 5e-6, 2e-6, 1e-6]


# ---------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------
def simulate_euler_maruyama(dt, seed):
    """
    Euler-Maruyama update for the underdamped Langevin equation:

        dx = v dt
        dv = [-(k/m)x - (b/m)v] dt
             + sqrt(2 b kB T / m^2) dW
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + dt, dt)

    x = np.zeros_like(t)
    v = np.zeros_like(t)

    sigma_v = np.sqrt(2 * b * kB * T / m**2)

    for i in range(len(t) - 1):
        a_det = -(k / m) * x[i] - (b / m) * v[i]

        v[i + 1] = (
            v[i]
            + a_det * dt
            + sigma_v * np.sqrt(dt) * rng.normal()
        )
        x[i + 1] = x[i] + v[i + 1] * dt

    return t, x, v


def simulate_baoab(dt, seed):
    """
    BAOAB Langevin integrator.

    Splitting:
        B: half force kick
        A: half position drift
        O: exact Ornstein-Uhlenbeck thermostat
        A: half position drift
        B: half force kick

    This usually samples the configurational distribution very well for
    Langevin harmonic systems.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + dt, dt)

    x = np.zeros_like(t)
    v = np.zeros_like(t)

    gamma = b / m
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((kB * T / m) * (1 - c1**2))

    for i in range(len(t) - 1):
        xi = x[i]
        vi = v[i]

        # B
        vi = vi + 0.5 * dt * (-(k / m) * xi)

        # A
        xi = xi + 0.5 * dt * vi

        # O
        vi = c1 * vi + c2 * rng.normal()

        # A
        xi = xi + 0.5 * dt * vi

        # B
        vi = vi + 0.5 * dt * (-(k / m) * xi)

        x[i + 1] = xi
        v[i + 1] = vi

    return t, x, v


def simulate_stochastic_velocity_verlet(dt, seed):
    """
    Simple stochastic velocity-Verlet-style Langevin update.

    This is included as a practical comparison method. BAOAB is generally
    preferred for equilibrium sampling, but this method is useful as a
    familiar baseline.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + dt, dt)

    x = np.zeros_like(t)
    v = np.zeros_like(t)

    sigma_v = np.sqrt(2 * b * kB * T / m**2)

    def acceleration(x_val, v_val):
        return -(k / m) * x_val - (b / m) * v_val

    for i in range(len(t) - 1):
        a_i = acceleration(x[i], v[i])

        v_half = v[i] + 0.5 * a_i * dt
        v_half = v_half + sigma_v * np.sqrt(dt) * rng.normal()

        x[i + 1] = x[i] + v_half * dt

        a_next = acceleration(x[i + 1], v_half)
        v[i + 1] = v_half + 0.5 * a_next * dt

    return t, x, v


integrators = {
    "Euler-Maruyama": simulate_euler_maruyama,
    "BAOAB": simulate_baoab,
    "Stoch-Verlet": simulate_stochastic_velocity_verlet,
}


# ---------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------
results = []
trajectories_for_plot = {}

for method_name, simulator in integrators.items():
    for dt in dt_values:
        start = time.perf_counter()
        t, x, v = simulator(dt, seed)
        runtime = time.perf_counter() - start

        burn_mask = t > burn_in_fraction * t_end

        x_sample = x[burn_mask]
        v_sample = v[burn_mask]

        x_rms = np.std(x_sample)
        v_rms = np.std(v_sample)

        x_rms_error = x_rms - x_rms_exact
        v_rms_error = v_rms - v_rms_exact

        x_rms_rel_error = x_rms_error / x_rms_exact
        v_rms_rel_error = v_rms_error / v_rms_exact

        results.append({
            "method": method_name,
            "dt": dt,
            "runtime": runtime,
            "n_steps": len(t) - 1,
            "x_rms": x_rms,
            "v_rms": v_rms,
            "x_rms_rel_error": x_rms_rel_error,
            "v_rms_rel_error": v_rms_rel_error,
        })

        if dt == dt_values[-1]:
            trajectories_for_plot[method_name] = (t, x, v, burn_mask)


# ---------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------
print(
    f"{'method':<16} "
    f"{'dt/s':>10} "
    f"{'steps':>8} "
    f"{'runtime/s':>12} "
    f"{'x_rms/nm':>12} "
    f"{'x err/%':>10} "
    f"{'v err/%':>10}"
)

for row in results:
    print(
        f"{row['method']:<16} "
        f"{row['dt']:>10.1e} "
        f"{row['n_steps']:>8} "
        f"{row['runtime']:>12.5g} "
        f"{row['x_rms'] * 1e9:>12.5g} "
        f"{100 * row['x_rms_rel_error']:>10.4g} "
        f"{100 * row['v_rms_rel_error']:>10.4g}"
    )


# ---------------------------------------------------------------------
# Plot RMS position error vs timestep
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

for method_name in integrators:
    rows = [row for row in results if row["method"] == method_name]
    dts = np.array([row["dt"] for row in rows])
    errors = np.abs([row["x_rms_rel_error"] for row in rows])

    plt.loglog(dts, errors, "o-", label=method_name)

plt.gca().invert_xaxis()
plt.xlabel("Timestep dt / s")
plt.ylabel("Absolute relative error in x_rms")
plt.title("Brownian solver accuracy: position RMS")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Plot RMS velocity error vs timestep
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

for method_name in integrators:
    rows = [row for row in results if row["method"] == method_name]
    dts = np.array([row["dt"] for row in rows])
    errors = np.abs([row["v_rms_rel_error"] for row in rows])

    plt.loglog(dts, errors, "o-", label=method_name)

plt.gca().invert_xaxis()
plt.xlabel("Timestep dt / s")
plt.ylabel("Absolute relative error in v_rms")
plt.title("Brownian solver accuracy: velocity RMS")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Plot stationary position distributions at smallest dt
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

x_plot = np.linspace(-5 * x_rms_exact, 5 * x_rms_exact, 500)
pdf_exact = (
    1 / (np.sqrt(2 * np.pi) * x_rms_exact)
    * np.exp(-0.5 * (x_plot / x_rms_exact)**2)
)

plt.plot(x_plot * 1e9, pdf_exact / np.max(pdf_exact), "k", linewidth=2, label="exact Gaussian")

for method_name, (t, x, v, burn_mask) in trajectories_for_plot.items():
    plt.hist(
        x[burn_mask] * 1e9,
        bins=80,
        density=True,
        histtype="step",
        linewidth=1.4,
        label=method_name,
    )

plt.xlabel("x / nm")
plt.ylabel("Normalized density")
plt.title("Stationary position distribution")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------------------
# Example trajectories at smallest dt
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

for method_name, (t, x, v, burn_mask) in trajectories_for_plot.items():
    plt.plot(t, x * 1e9, linewidth=0.8, label=method_name)

plt.xlabel("Time / s")
plt.ylabel("x / nm")
plt.title("Example Brownian trajectories")
plt.legend()
plt.grid()
plt.show()
