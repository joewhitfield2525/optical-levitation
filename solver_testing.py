import csv
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
from scipy.integrate import solve_ivp

# --- Publication style ---
fontsize = 7
mpl.rcParams.update({

    # Figure
    "figure.figsize": (3.4, 2.6),  # single column
    "figure.dpi": 300,

    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": fontsize,
    "axes.labelsize": fontsize + 2,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize - 1,
    "ytick.labelsize": fontsize - 1,
    "legend.fontsize": fontsize - 1,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # Axes
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    # Grid
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,

    # Remove top/right spine? (optional)
    # "axes.spines.top": False,
    # "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
m = 1.0
k = 1000.0
damping_ratio = 0.1

b = damping_ratio * 2 * np.sqrt(m * k)

x0 = 1.0
v0 = 0.0

t_start = 0.0
t_end = 5.0
t_eval = np.linspace(t_start, t_end, 5000)

output_dir = Path("/Users/josephwhitfield/Documents/optical levitation/solver_validation_plots")
output_dir.mkdir(parents=True, exist_ok=True)
accuracy_runtime_csv = output_dir / "deterministic_solver_accuracy_vs_runtime.csv"


def format_decimal_tick(value, _position):
    if value <= 0:
        return ""
    return f"{value:.3g}"


def use_more_numeric_log_x_ticks(ax, fixed_ticks=None):
    if fixed_ticks is None:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1, 2, 3, 5, 7), numticks=20))
    else:
        ax.xaxis.set_major_locator(FixedLocator(np.sort(fixed_ticks)))

    ax.xaxis.set_major_formatter(FuncFormatter(format_decimal_tick))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10), numticks=50))
    ax.tick_params(axis="x", which="major", labelrotation=0)


def read_accuracy_runtime_rows(path):
    rows = []

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "solver": row["solver"],
                    "rtol": float(row["rtol"]),
                    "runtime_s": float(row["runtime_s"]),
                    "max_error_nm": float(row["max_error_nm"]),
                }
            )

    return rows


def plot_accuracy_runtime_from_csv(path, rows, accuracy_target_nm=10.0):
    fig, ax = plt.subplots()
    markers = {
        "RK45": "o",
        "DOP853": "s",
        "Radau": "^",
        "BDF": "D",
        "LSODA": "P",
    }

    for solver, marker in markers.items():
        solver_rows = sorted(
            [row for row in rows if row["solver"] == solver],
            key=lambda row: row["rtol"],
            reverse=True,
        )

        if not solver_rows:
            continue

        ax.plot(
            [row["runtime_s"] for row in solver_rows],
            [row["max_error_nm"] for row in solver_rows],
            marker=marker,
            label=solver,
        )

    ax.axhline(
        accuracy_target_nm,
        color="0.35",
        linestyle="--",
        label=f"{accuracy_target_nm:g} nm requirement",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime / s")
    ax.set_ylabel("Maximum position error / nm")
    ax.set_title("Deterministic solver accuracy-runtime trade-off")
    ax.grid(True, which="both")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_solver_validation(
    path,
    energy_results,
    energy_time_s,
    accuracy_runtime_rows,
    accuracy_target_nm=10.0
):
    fig, (ax_energy, ax_accuracy) = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.6),
        gridspec_kw={"wspace": 0.42}
    )
    markers = {
        "RK45": "o",
        "DOP853": "s",
        "Radau": "^",
        "BDF": "D",
        "LSODA": "P",
    }
    colours = {
        method: f"C{index}"
        for index, method in enumerate(markers)
    }
    legend_handles = []
    legend_labels = []

    for method in markers:
        if method not in energy_results:
            continue

        line, = ax_energy.plot(
            energy_time_s,
            energy_results[method]["relative_energy_error"],
            color=colours[method],
            label=method,
        )
        legend_handles.append(line)
        legend_labels.append(method)

    ax_energy.set_xlabel("Time / s")
    ax_energy.set_ylabel("Relative energy change")
    ax_energy.set_ylim(-5e-8, 3.4e-8)
    ax_energy.grid(True)

    for method, marker in markers.items():
        solver_rows = sorted(
            [row for row in accuracy_runtime_rows if row["solver"] == method],
            key=lambda row: row["rtol"],
            reverse=True,
        )

        if not solver_rows:
            continue

        ax_accuracy.plot(
            [row["runtime_s"] for row in solver_rows],
            [row["max_error_nm"] for row in solver_rows],
            marker=marker,
            color=colours[method],
            label=method,
        )

    requirement_line = ax_accuracy.axhline(
        accuracy_target_nm,
        color="0.35",
        linestyle="--",
        label=f"{accuracy_target_nm:g} nm requirement",
    )
    legend_handles.append(requirement_line)
    legend_labels.append(f"{accuracy_target_nm:g} nm requirement")
    ax_accuracy.set_xscale("log")
    ax_accuracy.set_yscale("log")
    ax_accuracy.set_xlabel("Runtime / s")
    ax_accuracy.set_ylabel("Maximum position error / nm")
    ax_accuracy.grid(True, which="both")
    use_more_numeric_log_x_ticks(ax_accuracy)

    ax_energy.text(
        -0.16,
        1.05,
        "(a)",
        transform=ax_energy.transAxes,
        fontsize=fontsize + 3,
        ha="left",
        va="bottom",
    )
    ax_accuracy.text(
        -0.16,
        1.05,
        "(b)",
        transform=ax_accuracy.transAxes,
        fontsize=fontsize + 3,
        ha="left",
        va="bottom",
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(legend_labels),
        columnspacing=1.2,
        handlelength=2.2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.84)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------
omega0 = np.sqrt(k / m)
gamma = b / (2 * m)

if damping_ratio >= 1:
    raise ValueError("This analytic example is currently written for the underdamped case.")

omega_d = np.sqrt(omega0**2 - gamma**2)


def analytic_solution(t):
    """
    Analytic solution of:

        m x'' + b x' + k x = 0

    for the underdamped case.
    """
    A = x0
    B = (v0 + gamma * x0) / omega_d

    exp_part = np.exp(-gamma * t)
    x = exp_part * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    v = exp_part * (
        -gamma * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        + (-A * omega_d * np.sin(omega_d * t) + B * omega_d * np.cos(omega_d * t))
    )

    return x, v


x_exact, v_exact = analytic_solution(t_eval)


# ---------------------------------------------------------------------
# Numerical model
# ---------------------------------------------------------------------
def damped_oscillator(t, Y):
    x, v = Y
    dxdt = v
    dvdt = -(b / m) * v - (k / m) * x
    return [dxdt, dvdt]


methods = ["RK45", "DOP853", "Radau", "BDF", "LSODA"]

results = {}
step_results = {}

for method in methods:
    start = time.perf_counter()

    sol = solve_ivp(
        damped_oscillator,
        [t_start, t_end],
        [x0, v0],
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=[1e-12, 1e-12],
    )

    runtime = time.perf_counter() - start

    if not sol.success:
        print(method, "failed:", sol.message)
        continue

    x_num = sol.y[0]
    v_num = sol.y[1]

    x_error = np.abs(x_num - x_exact)
    v_error = np.abs(v_num - v_exact)
    x_cummax_error = np.maximum.accumulate(x_error)
    v_cummax_error = np.maximum.accumulate(v_error)

    results[method] = {
        "x": x_num,
        "v": v_num,
        "x_error": x_error,
        "v_error": v_error,
        "x_cummax_error": x_cummax_error,
        "v_cummax_error": v_cummax_error,
        "max_x_error": np.max(x_error),
        "max_v_error": np.max(v_error),
        "rms_x_error": np.sqrt(np.mean(x_error**2)),
        "runtime": runtime,
        "nfev": sol.nfev,
    }

    # Run again without t_eval so sol.t contains the solver's accepted
    # internal timesteps rather than the user-requested output times.
    step_sol = solve_ivp(
        damped_oscillator,
        [t_start, t_end],
        [x0, v0],
        method=method,
        rtol=1e-9,
        atol=[1e-12, 1e-12],
    )

    if step_sol.success:
        internal_steps = np.diff(step_sol.t)
        step_results[method] = {
            "t": step_sol.t,
            "steps": internal_steps,
            "n_steps": len(internal_steps),
            "min_step": np.min(internal_steps),
            "max_step": np.max(internal_steps),
            "mean_step": np.mean(internal_steps),
            "median_step": np.median(internal_steps),
            "nfev": step_sol.nfev,
        }


# ---------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------
print("Damped oscillator solver benchmark")
print("m =", m)
print("k =", k)
print("b =", b)
print("damping ratio =", damping_ratio)
print("omega0 =", omega0)
print("omega_d =", omega_d)
print()
print(f"{'method':<8} {'runtime/s':>12} {'nfev':>8} {'max |x err|':>14} {'RMS |x err|':>14} {'max |v err|':>14}")

for method, data in results.items():
    print(
        f"{method:<8} "
        f"{data['runtime']:>12.5g} "
        f"{data['nfev']:>8} "
        f"{data['max_x_error']:>14.5e} "
        f"{data['rms_x_error']:>14.5e} "
        f"{data['max_v_error']:>14.5e}"
    )

print()
print("Final cumulative max errors")
print(f"{'method':<8} {'cummax |x err|':>18} {'cummax |v err|':>18}")

for method, data in results.items():
    print(
        f"{method:<8} "
        f"{data['x_cummax_error'][-1]:>18.5e} "
        f"{data['v_cummax_error'][-1]:>18.5e}"
    )

print()
print("Accepted internal timestep summary")
print(
    f"{'method':<8} "
    f"{'steps':>8} "
    f"{'nfev':>8} "
    f"{'min dt':>12} "
    f"{'median dt':>12} "
    f"{'mean dt':>12} "
    f"{'max dt':>12}"
)

for method, data in step_results.items():
    print(
        f"{method:<8} "
        f"{data['n_steps']:>8} "
        f"{data['nfev']:>8} "
        f"{data['min_step']:>12.5e} "
        f"{data['median_step']:>12.5e} "
        f"{data['mean_step']:>12.5e} "
        f"{data['max_step']:>12.5e}"
    )


# ---------------------------------------------------------------------
# Plot displacement comparison
# ---------------------------------------------------------------------
plt.figure()
plt.plot(t_eval, x_exact, color="black", linewidth=2, label="analytic")

for method, data in results.items():
    plt.plot(t_eval, data["x"], "--", linewidth=1, label=method)

plt.xlabel("Time / s")
plt.ylabel("Displacement x")
plt.title("Damped harmonic oscillator: analytic vs numerical")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------------------
# Plot absolute error as a function of time
# ---------------------------------------------------------------------
plt.figure()

for method, data in results.items():
    plt.semilogy(t_eval, data["x_error"], label=method)

plt.xlabel("Time / s")
plt.ylabel("Absolute displacement error")
plt.title("Error vs time")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Energy conservation test for the undamped oscillator
# ---------------------------------------------------------------------
# This is a separate test for numerical heating. With b = 0, the exact
# mechanical energy should be constant:
#
#     E = 0.5*m*v^2 + 0.5*k*x^2
#
# Upward drift in E indicates numerical heating.


def undamped_oscillator(t, Y):
    x, v = Y
    dxdt = v
    dvdt = -(k / m) * x
    return [dxdt, dvdt]


energy_results = {}

for method in methods:
    sol = solve_ivp(
        undamped_oscillator,
        [t_start, t_end],
        [x0, v0],
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=[1e-12, 1e-12],
    )

    if not sol.success:
        print(method, "failed in energy test:", sol.message)
        continue

    x_num = sol.y[0]
    v_num = sol.y[1]

    E_num = 0.5 * m * v_num**2 + 0.5 * k * x_num**2
    relative_energy_error = (E_num - E_num[0]) / E_num[0]

    energy_results[method] = {
        "E": E_num,
        "relative_energy_error": relative_energy_error,
        "max_abs_relative_energy_error": np.max(np.abs(relative_energy_error)),
        "final_relative_energy_error": relative_energy_error[-1],
    }


print()
print("Undamped energy conservation test")
print(
    f"{'method':<8} "
    f"{'max |dE/E0|':>16} "
    f"{'final dE/E0':>16}"
)

for method, data in energy_results.items():
    print(
        f"{method:<8} "
        f"{data['max_abs_relative_energy_error']:>16.5e} "
        f"{data['final_relative_energy_error']:>16.5e}"
    )


plt.figure()

for method, data in energy_results.items():
    plt.plot(t_eval, data["relative_energy_error"], label=method)

plt.xlabel("Time / s")
plt.ylabel("(E(t) - E(0)) / E(0)")
plt.title("Relative energy drift, undamped oscillator")
plt.ylim(-5e-8, 3.4e-8)
plt.legend(loc="upper left")
plt.grid()
plt.tight_layout()
plt.savefig(output_dir / "relative_energy_drift_publication.png")
plt.savefig(output_dir / "relative_energy_drift_publication.pdf")
plt.show()


plt.figure()

for method, data in energy_results.items():
    plt.semilogy(
        t_eval,
        np.abs(data["relative_energy_error"]),
        label=method,
    )

plt.xlabel("Time / s")
plt.ylabel("|(E(t) - E(0)) / E(0)|")
plt.title("Absolute relative energy error, undamped oscillator")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Deterministic timestep-convergence test
# ---------------------------------------------------------------------
# This test removes stochastic terms and compares the damped harmonic
# oscillator against its analytic solution while restricting the maximum
# timestep used by each deterministic solver.
#
# The plotted error is normalised by the initial displacement amplitude:
#
#     RMS trajectory error = sqrt(mean((x_num - x_exact)^2)) / |x0|
#
# A straight line on the log-log plot indicates power-law convergence.

validation_period = 2.0 * np.pi / omega0
convergence_dt_values = validation_period / np.array(
    [4, 6, 8, 12, 16, 24, 32, 48, 64],
    dtype=float,
)
convergence_results = {}

for method in methods:
    method_dt_values = []
    method_rms_errors = []
    method_max_errors = []
    method_runtimes = []

    for dt_value in convergence_dt_values:
        convergence_t_eval = np.arange(
            t_start,
            t_end + 0.5 * dt_value,
            dt_value,
        )
        convergence_t_eval = convergence_t_eval[convergence_t_eval <= t_end]
        if convergence_t_eval[-1] < t_end:
            convergence_t_eval = np.append(convergence_t_eval, t_end)

        x_exact_dt, _ = analytic_solution(convergence_t_eval)

        start = time.perf_counter()
        sol = solve_ivp(
            damped_oscillator,
            [t_start, t_end],
            [x0, v0],
            t_eval=convergence_t_eval,
            method=method,
            max_step=dt_value,
            rtol=1.0e-3,
            atol=[1.0e-9, 1.0e-9],
        )
        runtime = time.perf_counter() - start

        if not sol.success:
            print(method, "failed in convergence test:", sol.message)
            continue

        displacement_error = sol.y[0] - x_exact_dt
        normalisation = max(abs(x0), 1.0e-30)
        rms_error = np.sqrt(np.mean(displacement_error**2)) / normalisation
        max_error = np.max(np.abs(displacement_error)) / normalisation

        method_dt_values.append(dt_value)
        method_rms_errors.append(rms_error)
        method_max_errors.append(max_error)
        method_runtimes.append(runtime)

    method_dt_values = np.asarray(method_dt_values)
    method_rms_errors = np.asarray(method_rms_errors)
    method_max_errors = np.asarray(method_max_errors)
    method_runtimes = np.asarray(method_runtimes)

    finite_mask = (
        np.isfinite(method_dt_values)
        & np.isfinite(method_rms_errors)
        & (method_dt_values > 0)
        & (method_rms_errors > 0)
    )

    if np.count_nonzero(finite_mask) >= 3:
        # Fit the smallest timesteps, where the asymptotic convergence trend is
        # most likely to dominate.
        fit_indices = np.where(finite_mask)[0][-4:]
        observed_order, log_prefactor = np.polyfit(
            np.log(method_dt_values[fit_indices]),
            np.log(method_rms_errors[fit_indices]),
            1,
        )
    else:
        observed_order = np.nan
        log_prefactor = np.nan

    convergence_results[method] = {
        "dt": method_dt_values,
        "rms_error": method_rms_errors,
        "max_error": method_max_errors,
        "runtime": method_runtimes,
        "observed_order": observed_order,
        "log_prefactor": log_prefactor,
    }

print()
print("Deterministic solver convergence test")
print(
    f"{'method':<8} "
    f"{'observed order':>16} "
    f"{'RMS err @ min dt':>18} "
    f"{'max err @ min dt':>18}"
)

for method, data in convergence_results.items():
    if len(data["dt"]) == 0:
        continue
    min_dt_index = int(np.argmin(data["dt"]))
    print(
        f"{method:<8} "
        f"{data['observed_order']:>16.5g} "
        f"{data['rms_error'][min_dt_index]:>18.5e} "
        f"{data['max_error'][min_dt_index]:>18.5e}"
    )

plt.figure()

for method, data in convergence_results.items():
    if len(data["dt"]) == 0:
        continue
    label = method
    if np.isfinite(data["observed_order"]):
        label += f" (order {data['observed_order']:.2g})"
    plt.loglog(
        data["dt"],
        data["rms_error"],
        "o-",
        label=label,
    )

plt.xlabel("Maximum solver timestep / s")
plt.ylabel("Normalised RMS displacement error")
plt.title("Deterministic solver convergence")
plt.legend()
plt.grid(True, which="both")
use_more_numeric_log_x_ticks(
    plt.gca(),
    fixed_ticks=np.array([0.003, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.05]),
)
plt.tight_layout()
plt.savefig(output_dir / "deterministic_solver_convergence_error_vs_timestep.png")
plt.show()

if accuracy_runtime_csv.exists():
    accuracy_runtime_rows = read_accuracy_runtime_rows(accuracy_runtime_csv)
    plot_accuracy_runtime_from_csv(
        output_dir / "deterministic_solver_accuracy_vs_runtime_publication.png",
        accuracy_runtime_rows,
        accuracy_target_nm=10.0
    )
    plot_combined_solver_validation(
        output_dir / "combined_solver_validation_publication.png",
        energy_results,
        t_eval,
        accuracy_runtime_rows,
        accuracy_target_nm=10.0
    )
else:
    print("Accuracy-runtime CSV not found:", accuracy_runtime_csv)


# ---------------------------------------------------------------------
# Plot cumulative maximum displacement error
# ---------------------------------------------------------------------
plt.figure()

for method, data in results.items():
    plt.semilogy(t_eval, data["x_cummax_error"], label=method)

plt.xlabel("Time / s")
plt.ylabel("Cumulative max displacement error")
plt.title("Worst displacement error reached up to time t")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Plot adaptive internal timestep for each solver
# ---------------------------------------------------------------------
plt.figure()

for method, data in step_results.items():
    plt.semilogy(data["t"][:-1], data["steps"], label=method)

plt.xlabel("Time / s")
plt.ylabel("Accepted internal timestep / s")
plt.title("Adaptive solver timestep vs time")
plt.legend()
plt.grid(True, which="both")
plt.show()


# ---------------------------------------------------------------------
# Plot histogram of accepted internal timesteps
# ---------------------------------------------------------------------
plt.figure()

for method, data in step_results.items():
    plt.hist(
        data["steps"],
        bins=40,
        histtype="step",
        linewidth=1.4,
        label=method,
    )

plt.xlabel("Accepted internal timestep / s")
plt.ylabel("Count")
plt.title("Distribution of accepted solver timesteps")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------------------
# Plot velocity error as a function of time
# ---------------------------------------------------------------------
plt.figure()

for method, data in results.items():
    plt.semilogy(t_eval, data["v_error"], label=method)

plt.xlabel("Time / s")
plt.ylabel("Absolute velocity error")
plt.title("Velocity error vs time")
plt.legend()
plt.grid(True, which="both")
plt.show()
