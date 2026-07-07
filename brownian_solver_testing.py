import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import perf_counter


# =============================================================================
# Brownian damped harmonic oscillator solver benchmark
# =============================================================================
# Model:
#
#     m x'' + b x' + k x = F_th(t)
#
# with Brownian thermal forcing:
#
#     <F_th(t) F_th(t')> = 2 b k_B T delta(t - t')
#
# This script compares:
#
#     Euler-Maruyama
#     BAOAB Langevin integrator
#     stochastic Verlet-style update
#     RK45 with piecewise-constant Brownian force
#     DOP853 with piecewise-constant Brownian force
#     Radau with piecewise-constant Brownian force
#
# Important:
# RK45, DOP853, and Radau are deterministic ODE solvers. They are included here
# as a comparison only. They are not true SDE solvers, and they become slow when
# forced to resolve rapidly changing white-noise samples.


# =============================================================================
# Physical parameters
# =============================================================================
kB = 1.380649e-23
T = 293.0

radius = 5e-6
density = 2200.0
volume = (4.0 / 3.0) * np.pi * radius**3
m = density * volume

f0 = 50.0
omega0 = 2.0 * np.pi * f0
k = m * omega0**2

damping_ratio = 0.05
b = 2.0 * damping_ratio * m * omega0
gamma = b / m

sigma_v = np.sqrt(2.0 * b * kB * T) / m

x_rms_theory = np.sqrt(kB * T / k)
v_rms_theory = np.sqrt(kB * T / m)

print("Mass m =", m, "kg")
print("Spring constant k =", k, "N/m")
print("Damping coefficient b =", b, "kg/s")
print("Natural frequency f0 =", f0, "Hz")
print("Damping ratio =", damping_ratio)
print("Theoretical x_rms =", x_rms_theory, "m")
print("Theoretical v_rms =", v_rms_theory, "m/s")


# =============================================================================
# Benchmark controls
# =============================================================================
t_end = 20.0
dt_ref = 1e-6

# These must be integer multiples of dt_ref for clean noise aggregation.
dt_values = np.array([1e-3, 1e-4, 1e-5, 1e-6])

run_adaptive_ode_solvers = True
adaptive_methods = ["RK45", "DOP853", "Radau"]

# Keep the adaptive comparison short. Increase carefully.
adaptive_t_end = 0.05
adaptive_dt_values = np.array([1e-3, 5e-4])

seed = 7
x0 = 0.0
v0 = 0.0


# =============================================================================
# Noise construction
# =============================================================================
def make_reference_normals(total_time, dt, random_seed):
    n_steps = int(round(total_time / dt))
    rng = np.random.default_rng(random_seed)
    return rng.normal(size=n_steps)


def aggregate_normals(reference_normals, ratio):
    """
    Aggregate fine Brownian increments into coarser increments.

    If dW_i = sqrt(dt_ref) N_i, then for a coarse step:

        dW_coarse = sum_i dW_i

    and:

        N_coarse = dW_coarse / sqrt(dt_coarse)
    """
    usable = (len(reference_normals) // ratio) * ratio
    trimmed = reference_normals[:usable]
    grouped = trimmed.reshape(-1, ratio)
    return np.sum(grouped, axis=1) / np.sqrt(ratio)


reference_normals = make_reference_normals(t_end, dt_ref, seed)


# =============================================================================
# Integrators
# =============================================================================
def force_spring(x):
    return -k * x


def integrate_euler_maruyama(dt, normal_values):
    n_steps = len(normal_values)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0] = x0
    v[0] = v0

    noise_scale = sigma_v * np.sqrt(dt)

    start = perf_counter()

    for i in range(n_steps):
        deterministic_acceleration = (force_spring(x[i]) - b * v[i]) / m
        v[i + 1] = v[i] + deterministic_acceleration * dt + noise_scale * normal_values[i]
        x[i + 1] = x[i] + v[i + 1] * dt

    runtime = perf_counter() - start
    return t, x, v, runtime


def integrate_baoab(dt, normal_values):
    """
    BAOAB Langevin integrator.

    This is usually a better choice than Euler-Maruyama for Langevin dynamics
    because it treats the velocity damping/noise step as an exact
    Ornstein-Uhlenbeck update.
    """
    n_steps = len(normal_values)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0] = x0
    v[0] = v0

    exp_factor = np.exp(-gamma * dt)
    thermal_velocity_scale = np.sqrt((kB * T / m) * (1.0 - exp_factor**2))

    start = perf_counter()

    for i in range(n_steps):
        xi = x[i]
        vi = v[i]

        vi += 0.5 * dt * force_spring(xi) / m
        xi += 0.5 * dt * vi
        vi = exp_factor * vi + thermal_velocity_scale * normal_values[i]
        xi += 0.5 * dt * vi
        vi += 0.5 * dt * force_spring(xi) / m

        x[i + 1] = xi
        v[i + 1] = vi

    runtime = perf_counter() - start
    return t, x, v, runtime


def integrate_stochastic_verlet(dt, normal_values):
    """
    A simple stochastic velocity-Verlet-style update.

    This is a useful comparison method, but BAOAB is generally preferred for
    Langevin thermal equilibrium.
    """
    n_steps = len(normal_values)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0] = x0
    v[0] = v0

    noise_scale = sigma_v * np.sqrt(dt)

    start = perf_counter()

    for i in range(n_steps):
        acceleration = (force_spring(x[i]) - b * v[i]) / m
        random_kick = noise_scale * normal_values[i]

        v_half = v[i] + 0.5 * acceleration * dt + 0.5 * random_kick
        x[i + 1] = x[i] + v_half * dt

        acceleration_new = (force_spring(x[i + 1]) - b * v_half) / m
        v[i + 1] = v_half + 0.5 * acceleration_new * dt + 0.5 * random_kick

    runtime = perf_counter() - start
    return t, x, v, runtime


def integrate_adaptive_ode(method, dt_noise, normal_values):
    """
    Deterministic ODE solver driven by piecewise-constant Brownian force.

    The Brownian force for each interval is:

        F_th = sqrt(2 b kBT / dt_noise) N(0, 1)

    The solver is forced to use max_step <= dt_noise so it cannot skip over the
    noise intervals.
    """
    n_steps = len(normal_values)
    t_eval = np.linspace(0.0, n_steps * dt_noise, n_steps + 1)
    force_noise = np.sqrt(2.0 * b * kB * T / dt_noise) * normal_values

    def force_noise_at_time(time):
        index = int(time / dt_noise)
        index = np.clip(index, 0, n_steps - 1)
        return force_noise[index]

    def rhs(time, y):
        x = y[0]
        v = y[1]

        dxdt = v
        dvdt = (force_spring(x) - b * v + force_noise_at_time(time)) / m

        return [dxdt, dvdt]

    start = perf_counter()

    sol = solve_ivp(
        rhs,
        [0.0, n_steps * dt_noise],
        [x0, v0],
        method=method,
        t_eval=t_eval,
        rtol=1e-6,
        atol=[1e-12, 1e-9],
        max_step=dt_noise,
    )

    runtime = perf_counter() - start
    return sol.t, sol.y[0], sol.y[1], runtime, sol.nfev, sol.success


# =============================================================================
# Analysis helpers
# =============================================================================
def one_sided_psd(signal, dt):
    signal = signal - np.mean(signal)
    n = len(signal)

    window = np.hanning(n)
    window_power = np.mean(window**2)

    fft_values = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, dt)

    psd = (2.0 * dt / (n * window_power)) * np.abs(fft_values) ** 2
    psd[0] *= 0.5

    if n % 2 == 0:
        psd[-1] *= 0.5

    return freqs, psd


def theoretical_displacement_psd(freqs):
    omega = 2.0 * np.pi * freqs
    numerator = 4.0 * kB * T * b
    denominator = (k - m * omega**2) ** 2 + (b * omega) ** 2
    return numerator / denominator


def calculate_metrics(t, x, v, runtime, x_reference, v_reference, nfev):
    path_error = x - x_reference
    velocity_error = v - v_reference

    settled = t > 0.25 * t[-1]
    if np.count_nonzero(settled) < 10:
        settled = np.ones_like(t, dtype=bool)

    x_rms = np.std(x[settled])
    v_rms = np.std(v[settled])

    freqs, x_psd = one_sided_psd(x[settled], t[1] - t[0])
    x_psd_theory = theoretical_displacement_psd(freqs)

    psd_mask = (
        (freqs > max(1.0, 1.0 / t[-1]))
        & (freqs < min(1000.0, 0.45 / (t[1] - t[0])))
        & np.isfinite(x_psd)
        & np.isfinite(x_psd_theory)
        & (x_psd > 0)
        & (x_psd_theory > 0)
    )

    if np.count_nonzero(psd_mask) > 5:
        psd_log_rms_error = np.sqrt(
            np.mean((np.log10(x_psd[psd_mask]) - np.log10(x_psd_theory[psd_mask])) ** 2)
        )
    else:
        psd_log_rms_error = np.nan

    return {
        "runtime": runtime,
        "nfev": nfev,
        "rms_x_path_error": np.sqrt(np.mean(path_error**2)),
        "max_x_path_error": np.max(np.abs(path_error)),
        "rms_v_path_error": np.sqrt(np.mean(velocity_error**2)),
        "x_rms": x_rms,
        "v_rms": v_rms,
        "relative_x_rms_error": abs(x_rms - x_rms_theory) / x_rms_theory,
        "relative_v_rms_error": abs(v_rms - v_rms_theory) / v_rms_theory,
        "psd_log_rms_error": psd_log_rms_error,
    }


def max_finite_metric(rows, metric_name):
    values = np.array(
        [
            row["metrics"][metric_name]
            for row in rows
            if np.isfinite(row["metrics"][metric_name])
        ]
    )

    if len(values) == 0:
        return np.nan

    return np.max(values)


def normalised_value(value, maximum):
    if not np.isfinite(value) or not np.isfinite(maximum) or maximum == 0:
        return np.nan

    return value / maximum


def print_results_table(title, rows, interval_label):
    max_runtime = max_finite_metric(rows, "runtime")
    max_rms_path_error = max_finite_metric(rows, "rms_x_path_error")

    print("\n" + title)
    print("Normalised columns are scaled so the largest value in this table is 1.")
    print(
        f"{'method':<18}"
        f"{interval_label:>12}"
        f"{'runtime/s':>12}"
        f"{'rt norm':>10}"
        f"{'nfev':>10}"
        f"{'RMS x err/nm':>16}"
        f"{'err norm':>10}"
        f"{'max x err/nm':>16}"
        f"{'x_rms err':>12}"
        f"{'v_rms err':>12}"
        f"{'PSD log err':>13}"
    )

    for row in rows:
        metrics = row["metrics"]
        runtime_norm = normalised_value(metrics["runtime"], max_runtime)
        rms_path_error_norm = normalised_value(
            metrics["rms_x_path_error"],
            max_rms_path_error,
        )

        print(
            f"{row['method']:<18}"
            f"{row['dt']:>12.2e}"
            f"{metrics['runtime']:>12.5g}"
            f"{runtime_norm:>10.4g}"
            f"{str(metrics['nfev']):>10}"
            f"{metrics['rms_x_path_error'] * 1e9:>16.5g}"
            f"{rms_path_error_norm:>10.4g}"
            f"{metrics['max_x_path_error'] * 1e9:>16.5g}"
            f"{metrics['relative_x_rms_error']:>12.5g}"
            f"{metrics['relative_v_rms_error']:>12.5g}"
            f"{metrics['psd_log_rms_error']:>13.5g}"
        )


# =============================================================================
# Reference solution
# =============================================================================
print("\nBuilding fine reference solution using BAOAB at dt_ref =", dt_ref, "s")
t_ref, x_ref, v_ref, reference_runtime = integrate_baoab(dt_ref, reference_normals)
print("Reference runtime =", reference_runtime, "s")


# =============================================================================
# Fixed-step stochastic method benchmark
# =============================================================================
stochastic_integrators = {
    "Euler-Maruyama": integrate_euler_maruyama,
    "BAOAB": integrate_baoab,
    "Stoch-Verlet": integrate_stochastic_verlet,
}

stochastic_rows = []
stochastic_solutions = {}

for dt in dt_values:
    ratio = int(round(dt / dt_ref))

    if not np.isclose(ratio * dt_ref, dt):
        raise ValueError("Each dt in dt_values must be an integer multiple of dt_ref.")

    normal_values = aggregate_normals(reference_normals, ratio)
    x_reference = x_ref[::ratio][: len(normal_values) + 1]
    v_reference = v_ref[::ratio][: len(normal_values) + 1]

    for method_name, integrator in stochastic_integrators.items():
        t, x, v, runtime = integrator(dt, normal_values)

        metrics = calculate_metrics(
            t=t,
            x=x,
            v=v,
            runtime=runtime,
            x_reference=x_reference,
            v_reference=v_reference,
            nfev="fixed",
        )

        stochastic_rows.append(
            {
                "method": method_name,
                "dt": dt,
                "metrics": metrics,
            }
        )

        stochastic_solutions[(method_name, dt)] = {
            "t": t,
            "x": x,
            "v": v,
        }


print_results_table(
    "Fixed-step stochastic solvers",
    stochastic_rows,
    interval_label="step dt/s",
)


# =============================================================================
# Adaptive deterministic solver benchmark with piecewise-constant noise
# =============================================================================
adaptive_rows = []
adaptive_solutions = {}

if run_adaptive_ode_solvers:
    adaptive_reference_steps = int(round(adaptive_t_end / dt_ref))
    adaptive_reference_normals = reference_normals[:adaptive_reference_steps]
    t_ref_adaptive = t_ref[: adaptive_reference_steps + 1]
    x_ref_adaptive = x_ref[: adaptive_reference_steps + 1]
    v_ref_adaptive = v_ref[: adaptive_reference_steps + 1]

    for dt in adaptive_dt_values:
        ratio = int(round(dt / dt_ref))

        if not np.isclose(ratio * dt_ref, dt):
            raise ValueError("Each dt in adaptive_dt_values must be an integer multiple of dt_ref.")

        normal_values = aggregate_normals(adaptive_reference_normals, ratio)
        x_reference = x_ref_adaptive[::ratio][: len(normal_values) + 1]
        v_reference = v_ref_adaptive[::ratio][: len(normal_values) + 1]

        for method in adaptive_methods:
            print("Running adaptive method", method, "with dt_noise =", dt, "s")

            t, x, v, runtime, nfev, success = integrate_adaptive_ode(
                method=method,
                dt_noise=dt,
                normal_values=normal_values,
            )

            metrics = calculate_metrics(
                t=t,
                x=x,
                v=v,
                runtime=runtime,
                x_reference=x_reference,
                v_reference=v_reference,
                nfev=nfev,
            )

            adaptive_rows.append(
                {
                    "method": method + (" OK" if success else " FAIL"),
                    "dt": dt,
                    "metrics": metrics,
                }
            )

            adaptive_solutions[(method, dt)] = {
                "t": t,
                "x": x,
                "v": v,
            }

    print_results_table(
        "Adaptive deterministic ODE solvers with sampled noise",
        adaptive_rows,
        interval_label="noise dt/s",
    )


# =============================================================================
# Plots
# =============================================================================
plt.figure(figsize=(9, 5))
max_stochastic_path_error = max_finite_metric(stochastic_rows, "rms_x_path_error")

for method_name in stochastic_integrators:
    method_rows = [row for row in stochastic_rows if row["method"] == method_name]
    dts = np.array([row["dt"] for row in method_rows])
    errors = np.array([row["metrics"]["rms_x_path_error"] for row in method_rows])
    normalised_errors = errors / max_stochastic_path_error
    plt.loglog(dts, normalised_errors, marker="o", label=method_name)

plt.gca().invert_xaxis()
plt.xlabel("Timestep / s")
plt.ylabel("Normalised RMS path error")
plt.title("Pathwise displacement error vs timestep, worst = 1")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()


plt.figure(figsize=(9, 5))
runtime_rows_for_plot = stochastic_rows + adaptive_rows
max_runtime_for_plot = max_finite_metric(runtime_rows_for_plot, "runtime")

for method_name in stochastic_integrators:
    method_rows = [row for row in stochastic_rows if row["method"] == method_name]
    dts = np.array([row["dt"] for row in method_rows])
    runtimes = np.array([row["metrics"]["runtime"] for row in method_rows])
    normalised_runtimes = runtimes / max_runtime_for_plot
    plt.loglog(dts, normalised_runtimes, marker="o", label=method_name)

if adaptive_rows:
    for method in adaptive_methods:
        method_rows = [row for row in adaptive_rows if row["method"].startswith(method)]
        dts = np.array([row["dt"] for row in method_rows])
        runtimes = np.array([row["metrics"]["runtime"] for row in method_rows])
        normalised_runtimes = runtimes / max_runtime_for_plot
        plt.loglog(dts, normalised_runtimes, marker="x", linestyle="--", label=method + " sampled-noise")

plt.gca().invert_xaxis()
plt.xlabel("Timestep or noise interval / s")
plt.ylabel("Normalised runtime")
plt.title("Runtime comparison, slowest = 1")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()


example_dt = dt_values[-1]
plt.figure(figsize=(9, 5))
for method_name in stochastic_integrators:
    sol = stochastic_solutions[(method_name, example_dt)]
    plt.plot(sol["t"], sol["x"] * 1e9, linewidth=0.8, label=method_name)

plt.xlabel("Time / s")
plt.ylabel("Position / nm")
plt.title("Example Brownian trajectories, dt = " + str(example_dt) + " s")
plt.legend()
plt.grid()
plt.tight_layout()


plt.figure(figsize=(9, 5))
freqs_ref, psd_ref = one_sided_psd(x_ref[int(0.25 * len(x_ref)) :], dt_ref)
psd_theory_ref = theoretical_displacement_psd(freqs_ref)

plt.semilogy(freqs_ref[1:], psd_ref[1:] * 1e18, linewidth=0.8, label="reference BAOAB")
plt.semilogy(freqs_ref[1:], psd_theory_ref[1:] * 1e18, "k--", linewidth=1.4, label="theory")
plt.axvline(f0, color="black", linestyle=":", label="natural frequency")
plt.xscale("log")
plt.xlim(1.0, 2000.0)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Reference displacement PSD vs theory")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()


plt.show()
