import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# =============================================================================
# Brownian/Langevin stochastic solver comparison
# =============================================================================
# Test problem:
#
#     m x'' = -k x - b x' + F_B(t)
#
# with
#
#     <F_B(t) F_B(t')> = 2 b k_B T delta(t - t')
#
# The exact equilibrium statistics for the harmonic trap are:
#
#     var(x) = k_B T / k
#     var(v) = k_B T / m
#
# These statistical quantities, not point-by-point trajectories, are the right
# way to compare stochastic integrators.


# =============================================================================
# Output
# =============================================================================
output_dir = Path(
    "/Users/josephwhitfield/Documents/optical levitation/"
    "brownian_solver_validation_plots"
)
output_dir.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print("Saved plot:", output_path)


# =============================================================================
# Physical parameters
# =============================================================================
kB = 1.380649e-23
T = 300.0

radius = 6.3e-6
density = 1100.0
volume = (4.0 / 3.0) * np.pi * radius**3
m = density * volume

f0 = 50.0
omega0 = 2.0 * np.pi * f0
k = m * omega0**2

damping_ratio = 0.05
b = 2.0 * damping_ratio * m * omega0
gamma = b / m

x_rms_theory = np.sqrt(kB * T / k)
v_rms_theory = np.sqrt(kB * T / m)
x_variance_theory = x_rms_theory**2
v_variance_theory = v_rms_theory**2

print("Mass m =", m, "kg")
print("Spring constant k =", k, "N/m")
print("Damping coefficient b =", b, "kg/s")
print("Damping rate gamma =", gamma, "s^-1")
print("Natural frequency f0 =", f0, "Hz")
print("Damping ratio =", damping_ratio)
print("Theoretical x RMS =", x_rms_theory, "m")
print("Theoretical v RMS =", v_rms_theory, "m/s")


# =============================================================================
# Benchmark controls
# =============================================================================
default_t_end = 50.0
burn_in_fraction = 0.10

dt_values = np.array([5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 1.0e-5, 1.0e-6])
seeds = np.array([
    101, 203, 307, 409, 503, 607, 709, 811,
    907, 1009, 1103, 1201, 1301, 1409, 1511, 1601,
])
duration_by_dt = {
    1.0e-5: 10.0,
    1.0e-6: 0.5,
}
# The finest timestep remains short to keep the benchmark runtime practical.
# The printed table reports the simulated duration used for each timestep.
strong_error_t_end = 0.2
strong_error_dt_reference = 2.0e-7
strong_error_seeds = seeds[:4]

x0 = 0.0
v0 = 0.0

example_dt = 1.0e-4
example_seed = int(seeds[0])


# =============================================================================
# Integrators
# =============================================================================
def spring_force(x):
    return -k * x


def integrate_euler_maruyama(dt, normals):
    """
    Explicit Euler-Maruyama baseline.

    It is simple, but usually gives biased equilibrium statistics unless dt is
    very small.
    """
    n_steps = len(normals)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    noise_scale = np.sqrt(2.0 * b * kB * T) / m * np.sqrt(dt)

    start = time.perf_counter()
    for i in range(n_steps):
        x[i + 1] = x[i] + v[i] * dt
        acceleration = (spring_force(x[i]) - b * v[i]) / m
        v[i + 1] = v[i] + acceleration * dt + noise_scale * normals[i]

    runtime = time.perf_counter() - start
    return x, v, runtime


def integrate_euler_cromer(dt, normals):
    """
    Semi-implicit Euler variant.

    Velocity is updated before position. This is often more stable than fully
    explicit Euler for oscillators, but it is still a low-order stochastic
    method.
    """
    n_steps = len(normals)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    noise_scale = np.sqrt(2.0 * b * kB * T) / m * np.sqrt(dt)

    start = time.perf_counter()
    for i in range(n_steps):
        acceleration = (spring_force(x[i]) - b * v[i]) / m
        v[i + 1] = v[i] + acceleration * dt + noise_scale * normals[i]
        x[i + 1] = x[i] + v[i + 1] * dt

    runtime = time.perf_counter() - start
    return x, v, runtime


def integrate_bbk(dt, normals):
    """
    Brünger-Brooks-Karplus/Langevin-Verlet style update.

    This is a common Langevin velocity-Verlet comparison method.
    """
    n_steps = len(normals)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    c0 = 1.0 / (1.0 + 0.5 * gamma * dt)
    c1 = 1.0 - 0.5 * gamma * dt
    noise_scale = np.sqrt(2.0 * gamma * kB * T / m * dt)

    start = time.perf_counter()
    for i in range(n_steps):
        acceleration = spring_force(x[i]) / m
        v_half = v[i] + 0.5 * dt * acceleration
        x[i + 1] = x[i] + dt * v_half
        acceleration_new = spring_force(x[i + 1]) / m
        v[i + 1] = c0 * (
            c1 * v[i]
            + 0.5 * dt * (acceleration + acceleration_new)
            + noise_scale * normals[i]
        )

    runtime = time.perf_counter() - start
    return x, v, runtime


def integrate_baoab(dt, normals):
    """
    BAOAB Langevin splitting.

    The damping/noise step is an exact Ornstein-Uhlenbeck update, which is why
    BAOAB is expected to sample equilibrium configurational statistics well.
    """
    n_steps = len(normals)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    exp_factor = np.exp(-gamma * dt)
    thermal_velocity_scale = np.sqrt(
        (kB * T / m) * (1.0 - exp_factor**2)
    )

    start = time.perf_counter()
    for i in range(n_steps):
        xi = x[i]
        vi = v[i]

        vi += 0.5 * dt * spring_force(xi) / m
        xi += 0.5 * dt * vi
        vi = exp_factor * vi + thermal_velocity_scale * normals[i]
        xi += 0.5 * dt * vi
        vi += 0.5 * dt * spring_force(xi) / m

        x[i + 1] = xi
        v[i + 1] = vi

    runtime = time.perf_counter() - start
    return x, v, runtime


integrators = {
    "Euler-Maruyama": integrate_euler_maruyama,
    "Euler-Cromer": integrate_euler_cromer,
    "BBK / Langevin Verlet": integrate_bbk,
    "BAOAB": integrate_baoab,
}


# Small horizontal offsets make overlapping timestep curves visible. The
# statistics are still evaluated at the unshifted dt values.
method_styles = {
    "Euler-Maruyama": {
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "x_offset": 0.93,
    },
    "Euler-Cromer": {
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
        "x_offset": 0.98,
    },
    "BBK / Langevin Verlet": {
        "color": "tab:green",
        "marker": "^",
        "linestyle": "-.",
        "x_offset": 1.03,
    },
    "BAOAB": {
        "color": "tab:red",
        "marker": "D",
        "linestyle": ":",
        "x_offset": 1.08,
    },
}


# =============================================================================
# Analysis helpers
# =============================================================================
def analytic_displacement_psd(freqs_hz):
    omega = 2.0 * np.pi * freqs_hz
    denominator = (omega0**2 - omega**2) ** 2 + gamma**2 * omega**2
    return 4.0 * kB * T * gamma / (m * denominator)


def welch_psd(signal, dt):
    signal = signal - np.mean(signal)
    sampling_frequency = 1.0 / dt
    nperseg = min(len(signal), max(256, int(round(2.0 * sampling_frequency))))
    noverlap = min(nperseg // 2, nperseg - 1)

    freqs, psd = welch(
        signal,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    positive = (freqs > 0) & (psd > 0)
    return freqs[positive], psd[positive]


def analyse_solution(x, v, dt, runtime):
    burn_in_index = int(round(burn_in_fraction * len(x)))
    burn_in_index = min(burn_in_index, len(x) - 2)
    x_analysis = x[burn_in_index:] - np.mean(x[burn_in_index:])
    v_analysis = v[burn_in_index:] - np.mean(v[burn_in_index:])

    x_variance = np.var(x_analysis)
    v_variance = np.var(v_analysis)
    x_rms_ratio = np.sqrt(x_variance / x_variance_theory)
    v_rms_ratio = np.sqrt(v_variance / v_variance_theory)
    unstable = (
        not np.isfinite(x_rms_ratio)
        or not np.isfinite(v_rms_ratio)
        or x_rms_ratio > 10.0
        or v_rms_ratio > 10.0
    )

    return {
        "runtime": runtime,
        "unstable": unstable,
        "x_variance": x_variance,
        "v_variance": v_variance,
        "x_sample_count": len(x_analysis),
        "v_sample_count": len(v_analysis),
        "x_rms_ratio": x_rms_ratio,
        "v_rms_ratio": v_rms_ratio,
        "x_rms_percent_error": abs(x_rms_ratio - 1.0) * 100.0,
        "v_rms_percent_error": abs(v_rms_ratio - 1.0) * 100.0,
        "x_variance_error": abs(x_variance / x_variance_theory - 1.0),
        "v_variance_error": abs(v_variance / v_variance_theory - 1.0),
    }


def grouped_metric(rows, method_name, dt, metric):
    values = np.array(
        [
            row[metric]
            for row in rows
            if row["method"] == method_name and np.isclose(row["dt"], dt)
            and not row["unstable"]
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    return np.mean(values), np.std(values, ddof=1)


def grouped_stable_fraction(rows, method_name, dt):
    selected = [
        row
        for row in rows
        if row["method"] == method_name and np.isclose(row["dt"], dt)
    ]
    if len(selected) == 0:
        return np.nan
    return np.mean([not row["unstable"] for row in selected])


def grouped_runtime_per_simulated_second(rows, method_name, dt):
    values = np.array(
        [
            row["runtime"] / row["simulated_duration"]
            for row in rows
            if row["method"] == method_name and np.isclose(row["dt"], dt)
            and row["simulated_duration"] > 0
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    return np.mean(values), np.std(values, ddof=1)


def grouped_pooled_rms_percent_error(
    rows,
    method_name,
    dt,
    variance_metric,
    count_metric,
    variance_theory,
    *,
    n_bootstrap=500,
):
    selected = [
        row
        for row in rows
        if row["method"] == method_name and np.isclose(row["dt"], dt)
        and not row["unstable"]
        and np.isfinite(row[variance_metric])
    ]
    if len(selected) == 0:
        return np.nan, np.nan

    variances = np.array([row[variance_metric] for row in selected], dtype=float)
    counts = np.array([row[count_metric] for row in selected], dtype=float)
    pooled_variance = np.average(variances, weights=counts)
    pooled_error = abs(np.sqrt(pooled_variance / variance_theory) - 1.0) * 100.0

    if len(selected) == 1:
        return pooled_error, 0.0

    bootstrap_errors = []
    rng = np.random.default_rng(12345)
    indices = np.arange(len(selected))
    for _ in range(n_bootstrap):
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        sample_variance = np.average(
            variances[sample_indices],
            weights=counts[sample_indices],
        )
        bootstrap_errors.append(
            abs(np.sqrt(sample_variance / variance_theory) - 1.0) * 100.0
        )

    return pooled_error, np.std(bootstrap_errors, ddof=1)


def offset_timestep_values(dt_array, method_name):
    style = method_styles.get(method_name, {})
    return np.asarray(dt_array, dtype=float) * style.get("x_offset", 1.0)


def plot_log_errorbar(
    x_values,
    y_values,
    y_std_values,
    label,
    *,
    offset_timestep=False,
):
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    y_std_values = np.asarray(y_std_values)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0)
    if np.count_nonzero(finite) == 0:
        return

    x_values = x_values[finite]
    y_values = y_values[finite]
    y_std_values = np.where(np.isfinite(y_std_values[finite]), y_std_values[finite], 0.0)
    if offset_timestep:
        x_values = offset_timestep_values(x_values, label)
    lower = np.minimum(y_std_values, 0.8 * y_values)
    upper = y_std_values
    style = method_styles.get(label, {})

    plt.errorbar(
        x_values,
        y_values,
        yerr=np.vstack([lower, upper]),
        color=style.get("color"),
        marker=style.get("marker", "o"),
        linestyle=style.get("linestyle", "-"),
        linewidth=1.4,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        capsize=4,
        label=label,
    )


def set_log_x_decimal_ticks(ax):
    tick_values = np.array([1.0e-6, 1.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4])
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{value:g}" for value in tick_values])


def aggregate_brownian_normals(reference_normals, step_ratio):
    n_coarse_steps = len(reference_normals) // step_ratio
    trimmed = reference_normals[:n_coarse_steps * step_ratio]
    return trimmed.reshape(n_coarse_steps, step_ratio).sum(axis=1) / np.sqrt(step_ratio)


def grouped_strong_error(rows, method_name, dt, metric):
    values = np.array(
        [
            row[metric]
            for row in rows
            if row["method"] == method_name and np.isclose(row["dt"], dt)
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    return np.mean(values), np.std(values, ddof=1)


# =============================================================================
# Run benchmark
# =============================================================================
rows = []
example_solutions = {}

print("\nRunning stochastic solver benchmark")
print(
    f"{'method':<24} "
    f"{'dt/s':>10} "
    f"{'duration/s':>12} "
    f"{'seed':>8} "
    f"{'runtime/s':>12} "
    f"{'status':>10} "
    f"{'x RMS/theory':>14} "
    f"{'v RMS/theory':>14} "
    f"{'x var err':>12}"
)

for dt in dt_values:
    simulated_duration_target = duration_by_dt.get(float(dt), default_t_end)
    n_steps = int(round(simulated_duration_target / dt))
    simulated_duration = n_steps * dt

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        normal_values = rng.normal(size=n_steps)

        for method_name, integrator in integrators.items():
            x, v, runtime = integrator(dt, normal_values)
            metrics = analyse_solution(x, v, dt, runtime)

            row = {
                "method": method_name,
                "dt": dt,
                "simulated_duration": simulated_duration,
                "seed": int(seed),
                **metrics,
            }
            rows.append(row)

            if np.isclose(dt, example_dt) and int(seed) == example_seed:
                example_solutions[method_name] = {
                    "x": x,
                    "v": v,
                    "dt": dt,
                }

            status = "unstable" if metrics["unstable"] else "ok"
            if metrics["unstable"]:
                print(
                    f"{method_name:<24} "
                    f"{dt:>10.1e} "
                    f"{simulated_duration:>12.3g} "
                    f"{int(seed):>8} "
                    f"{runtime:>12.5g} "
                    f"{status:>10} "
                    f"{'--':>14} "
                    f"{'--':>14} "
                    f"{'--':>12}"
                )
            else:
                print(
                    f"{method_name:<24} "
                    f"{dt:>10.1e} "
                    f"{simulated_duration:>12.3g} "
                    f"{int(seed):>8} "
                    f"{runtime:>12.5g} "
                    f"{status:>10} "
                    f"{metrics['x_rms_ratio']:>14.5f} "
                    f"{metrics['v_rms_ratio']:>14.5f} "
                    f"{metrics['x_variance_error']:>12.5f}"
                )


print("\nSummary: mean +/- std across random seeds")
print(
    f"{'method':<24} "
    f"{'dt/s':>10} "
    f"{'stable':>10} "
    f"{'x var err':>22} "
    f"{'runtime/s':>22}"
)
for method_name in integrators:
    for dt in dt_values:
        mean_error, std_error = grouped_metric(rows, method_name, dt, "x_variance_error")
        mean_runtime, std_runtime = grouped_metric(rows, method_name, dt, "runtime")
        stable_fraction = grouped_stable_fraction(rows, method_name, dt)
        error_text = (
            f"{mean_error:>10.4g} +/- {std_error:<8.3g}"
            if np.isfinite(mean_error)
            else f"{'unstable':>22}"
        )
        runtime_text = (
            f"{mean_runtime:>10.4g} +/- {std_runtime:<8.3g}"
            if np.isfinite(mean_runtime)
            else f"{'--':>22}"
        )
        print(
            f"{method_name:<24} "
            f"{dt:>10.1e} "
            f"{stable_fraction:>10.2f} "
            f"{error_text} "
            f"{runtime_text}"
        )


# =============================================================================
# Strong trajectory error against a fine BAOAB reference
# =============================================================================
strong_error_rows = []
reference_steps = int(round(strong_error_t_end / strong_error_dt_reference))

print("\nStrong trajectory error against fine-BAOAB reference")
print(
    f"{'method':<24} "
    f"{'dt/s':>10} "
    f"{'seed':>8} "
    f"{'max |dx|/nm':>14} "
    f"{'RMS |dx|/nm':>14}"
)

for seed in strong_error_seeds:
    rng = np.random.default_rng(int(seed))
    reference_normals = rng.normal(size=reference_steps)
    x_reference, _, _ = integrate_baoab(
        strong_error_dt_reference,
        reference_normals,
    )

    for dt in dt_values:
        step_ratio = int(round(dt / strong_error_dt_reference))

        if step_ratio < 1 or not np.isclose(
            step_ratio * strong_error_dt_reference,
            dt,
            rtol=0.0,
            atol=1.0e-15
        ):
            continue

        coarse_normals = aggregate_brownian_normals(reference_normals, step_ratio)
        x_reference_at_dt = x_reference[::step_ratio][:len(coarse_normals) + 1]

        for method_name, integrator in integrators.items():
            x_test, _, _ = integrator(dt, coarse_normals)
            displacement_error = x_test[:len(x_reference_at_dt)] - x_reference_at_dt
            max_error_nm = np.max(np.abs(displacement_error)) * 1.0e9
            rms_error_nm = np.sqrt(np.mean(displacement_error**2)) * 1.0e9

            strong_error_rows.append(
                {
                    "method": method_name,
                    "dt": dt,
                    "seed": int(seed),
                    "max_position_error_nm": max_error_nm,
                    "rms_position_error_nm": rms_error_nm,
                }
            )
            print(
                f"{method_name:<24} "
                f"{dt:>10.1e} "
                f"{int(seed):>8} "
                f"{max_error_nm:>14.5g} "
                f"{rms_error_nm:>14.5g}"
            )


# =============================================================================
# Plot 1: variance error vs timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_errors = []
    std_errors = []

    for dt in dt_values:
        mean_error, std_error = grouped_metric(
            rows,
            method_name,
            dt,
            "x_variance_error",
        )
        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plot_log_errorbar(
        dt_values,
        mean_errors,
        std_errors,
        method_name,
        offset_timestep=True,
    )

ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
set_log_x_decimal_ticks(ax)
plt.xlabel("Timestep / s")
plt.ylabel(r"mean $|\mathrm{var}(x)_\mathrm{sim}/\mathrm{var}(x)_\mathrm{theory} - 1|$")
plt.title("Stochastic solver variance error vs timestep")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_variance_error_vs_timestep.png")


# =============================================================================
# Plot 2: velocity variance error vs timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_errors = []
    std_errors = []

    for dt in dt_values:
        mean_error, std_error = grouped_metric(
            rows,
            method_name,
            dt,
            "v_variance_error",
        )
        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plot_log_errorbar(
        dt_values,
        mean_errors,
        std_errors,
        method_name,
        offset_timestep=True,
    )

ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
set_log_x_decimal_ticks(ax)
plt.xlabel("Timestep / s")
plt.ylabel(r"mean $|\mathrm{var}(v)_\mathrm{sim}/\mathrm{var}(v)_\mathrm{theory} - 1|$")
plt.title("Stochastic solver velocity variance error vs timestep")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_velocity_variance_error_vs_timestep.png")


# =============================================================================
# Plot 3: strong trajectory error vs timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_errors = []
    std_errors = []

    for dt in dt_values:
        mean_error, std_error = grouped_strong_error(
            strong_error_rows,
            method_name,
            dt,
            "max_position_error_nm",
        )
        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plot_log_errorbar(
        dt_values,
        mean_errors,
        std_errors,
        method_name,
        offset_timestep=True,
    )

ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
set_log_x_decimal_ticks(ax)
plt.xlabel("Timestep / s")
plt.ylabel("Mean maximum trajectory error / nm")
plt.title("Strong trajectory error vs fine BAOAB reference")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_strong_position_error_vs_timestep.png")


# =============================================================================
# Plot 4: accuracy vs runtime
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_errors = []
    std_errors = []
    mean_runtimes = []

    for dt in dt_values:
        mean_error, std_error = grouped_metric(
            rows,
            method_name,
            dt,
            "x_variance_error",
        )
        mean_runtime, _ = grouped_metric(rows, method_name, dt, "runtime")
        mean_errors.append(mean_error)
        std_errors.append(std_error)
        mean_runtimes.append(mean_runtime)

    plot_log_errorbar(mean_runtimes, mean_errors, std_errors, method_name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Runtime per trajectory / s")
plt.ylabel("mean x variance error")
plt.title("Stochastic solver accuracy vs runtime")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_accuracy_vs_runtime.png")


# =============================================================================
# Plot 5: runtime per simulated second vs timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_runtime_rates = []
    std_runtime_rates = []

    for dt in dt_values:
        mean_runtime_rate, std_runtime_rate = grouped_runtime_per_simulated_second(
            rows,
            method_name,
            dt,
        )
        mean_runtime_rates.append(mean_runtime_rate)
        std_runtime_rates.append(std_runtime_rate)

    style = method_styles.get(method_name, {})
    finite = np.isfinite(mean_runtime_rates) & (np.asarray(mean_runtime_rates) > 0)
    plt.errorbar(
        offset_timestep_values(dt_values[finite], method_name),
        np.asarray(mean_runtime_rates)[finite],
        yerr=np.asarray(std_runtime_rates)[finite],
        color=style.get("color"),
        marker=style.get("marker", "o"),
        linestyle=style.get("linestyle", "-"),
        linewidth=1.4,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        capsize=4,
        label=method_name,
    )

ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
set_log_x_decimal_ticks(ax)
plt.xlabel("Timestep / s")
plt.ylabel("Runtime per simulated second")
plt.title("Stochastic solver computational cost")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_runtime_per_simulated_second.png")


# =============================================================================
# Plot 6: RMS percentage error vs timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    mean_errors = []
    std_errors = []

    for dt in dt_values:
        mean_error, std_error = grouped_pooled_rms_percent_error(
            rows,
            method_name,
            dt,
            "x_variance",
            "x_sample_count",
            x_variance_theory,
        )
        mean_errors.append(mean_error)
        std_errors.append(std_error)

    plot_log_errorbar(
        dt_values,
        mean_errors,
        std_errors,
        method_name,
        offset_timestep=True,
    )

ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
set_log_x_decimal_ticks(ax)
plt.xlabel("Timestep / s")
plt.ylabel("RMS error / %")
plt.title("Pooled thermal RMS error vs timestep")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_pooled_rms_percentage_error_vs_timestep.png")


# =============================================================================
# Plot 7: stability fraction
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name in integrators:
    stable_fractions = [
        grouped_stable_fraction(rows, method_name, dt)
        for dt in dt_values
    ]
    style = method_styles.get(method_name, {})
    plt.plot(
        offset_timestep_values(dt_values, method_name),
        stable_fractions,
        color=style.get("color"),
        marker=style.get("marker", "o"),
        linestyle=style.get("linestyle", "-"),
        linewidth=1.4,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label=method_name,
    )

ax = plt.gca()
ax.set_xscale("log")
set_log_x_decimal_ticks(ax)
plt.ylim(-0.05, 1.05)
plt.xlabel("Timestep / s")
plt.ylabel("fraction of seeds stable")
plt.title("Stochastic solver stability across random seeds")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_stability_fraction.png")


# =============================================================================
# Plot 7: PSD comparison for representative timestep
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name, solution in example_solutions.items():
    dt = solution["dt"]
    burn_in_index = int(round(burn_in_fraction * len(solution["x"])))
    freqs, psd = welch_psd(solution["x"][burn_in_index:], dt)
    style = method_styles.get(method_name, {})
    plt.loglog(
        freqs,
        psd,
        color=style.get("color"),
        linestyle=style.get("linestyle", "-"),
        linewidth=1.2,
        label=method_name,
    )

theory_freqs = np.logspace(0, np.log10(0.45 / example_dt), 700)
plt.loglog(
    theory_freqs,
    analytic_displacement_psd(theory_freqs),
    "k--",
    linewidth=1.6,
    label="analytic Langevin PSD",
)
plt.axvline(f0, color="black", linestyle=":", linewidth=1.1, label="f0")
plt.xlim(1.0, min(5000.0, 0.45 / example_dt))
plt.ylim(bottom=1.0e-28)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / m^2 Hz^-1")
plt.title(f"PSD comparison at dt = {example_dt:g} s")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_psd_comparison.png")


# =============================================================================
# Plot 8: example trajectories
# =============================================================================
plt.figure(figsize=(9, 5))

for method_name, solution in example_solutions.items():
    dt = solution["dt"]
    t = np.arange(len(solution["x"])) * dt
    plot_mask = t <= 2.0
    style = method_styles.get(method_name, {})
    plt.plot(
        t[plot_mask],
        solution["x"][plot_mask] * 1e9,
        color=style.get("color"),
        linestyle=style.get("linestyle", "-"),
        linewidth=1.0,
        label=method_name,
    )

plt.xlabel("Time / s")
plt.ylabel("Displacement / nm")
plt.title(f"Example Brownian trajectories, dt = {example_dt:g} s")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.35)
plt.tight_layout()
save_plot("brownian_solver_example_trajectories.png")


print("\nFigures written to:", output_dir)
if "agg" not in plt.get_backend().lower():
    plt.show()
