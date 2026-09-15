import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, least_squares
from scipy.signal import welch
from time import perf_counter
from pathlib import Path
import os
import re
import subprocess
import sys

try:
    import baoab_3d_cython as _baoab_3d_cython
    solve_baoab_3d_lookup_cython = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_lookup_cython",
        None
    )
    solve_baoab_3d_lookup_cython_positions_into = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_lookup_cython_positions_into",
        None
    )
    solve_baoab_3d_feedback_lookup_cython = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_feedback_lookup_cython",
        None
    )
    cython_baoab_available = solve_baoab_3d_lookup_cython is not None
    cython_baoab_positions_into_available = (
        solve_baoab_3d_lookup_cython_positions_into is not None
    )
    cython_feedback_baoab_available = solve_baoab_3d_feedback_lookup_cython is not None
except ImportError:
    solve_baoab_3d_lookup_cython = None
    solve_baoab_3d_lookup_cython_positions_into = None
    solve_baoab_3d_feedback_lookup_cython = None
    cython_baoab_available = False
    cython_baoab_positions_into_available = False
    cython_feedback_baoab_available = False

script_start_time = perf_counter()

# ****************************************************************************************************************************************************
# Constants
# ****************************************************************************************************************************************************
g = 9.81
c_light = 299792458

# ****************************************************************************************************************************************************
# User-adjustable parameters
# ****************************************************************************************************************************************************
# Particle properties
radius = 5e-6          # m
density = 1100         # kg/m^3
n_particle = 1.55      # particle refractive index
n_medium = 1.00027     # surrounding medium refractive index, air

# Gas properties
p = 100         # pressure, Pa
pressure_override_pa = os.environ.get("PROJECT_3D_LINEWIDTH_PRESSURE_PA")
if pressure_override_pa is not None:
    p = float(pressure_override_pa)
T = 300           # impinging/ambient gas temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
kB = 1.38e-23
d_air = 3.7e-10

# Heated Brownian bath from a chosen particle surface temperature
use_input_surface_temperature_for_brownian = True
input_surface_temperature = 500.0   # K; set equal to T for the cold-particle case
surface_temperature_override_k = os.environ.get(
    "PROJECT_3D_LINEWIDTH_SURFACE_TEMPERATURE_K"
)
if surface_temperature_override_k is not None:
    input_surface_temperature = float(surface_temperature_override_k)
brownian_surface_temperature_accommodation_alpha = 0.777

# Laser / force parameters
w0 = 3e-6        # beam waist, m
wavelength = 532e-9
M2 = 1.2
use_m2_rayleigh_range = False
zR_manual = 100e-6
P_laser = 0.1             # W, example laser power
use_laser_power_noise = False
laser_noise_fraction = 0.01
laser_noise_frequency=300
laser_noise_step_duration = 1/laser_noise_frequency     # s

# Ashkin ray-optics sampling
ray_grid_points = 100

# Photophoretic force parameters
k_particle = 1.4          # W/(m K), approximate silica
alpha_acc = 1.0           # thermal accommodation coefficient
kappa_t = 1.14            # thermal creep coefficient
absorption_fraction = 1e-3

# 3D force lookup table
use_force_lookup_table = True
force_lookup_grid_points_r = 101
force_lookup_grid_points_z = 201
force_lookup_r_min = 0.0
force_lookup_r_base_max = 80e-6
force_lookup_r_displacement_factor = 4
force_lookup_r_thermal_factor = 8
force_lookup_z_base_half_width = 150e-6
force_lookup_z_displacement_factor = 4
force_lookup_z_thermal_factor = 8

# Equilibrium and derivative settings
x_equilibrium = 0.0
y_equilibrium = 0.0
equilibrium_z_min = -10e-3
equilibrium_z_max = 10e-3
equilibrium_scan_points = 3000
equilibrium_root_index = 0
root_finding_xtol = 1e-15
root_finding_rtol = 1e-15
numerical_derivative_step = 1e-9

# Damping model
drag_model = "auto"
# Choose one of:
#   "stokes"       continuum Stokes drag, best for Kn << 1
#   "cunningham"  Stokes drag with slip correction, useful in transition regime
#   "epstein"     free-molecular Epstein drag, best for Kn >> 1
#   "auto"        pick a model from Knudsen number
cunningham_A = 1.257
cunningham_B = 0.4
cunningham_C = 1.1
epstein_accommodation_alpha = 1.0

# Initial conditions relative to the chosen equilibrium
x_displacement = 0e-6
y_displacement = 0e-6
z_displacement = 0e-6
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0

# Time integration
t_start = 0
t_end = 100.0
dt_baoab = 1 / 20000
brownian_seed = 9
laser_noise_seed = 12

# Optional laser feedback loop
# Set this to True to add a delayed PD loop that reads z and varies laser power.
use_pd_feedback = False
feedback_update_frequency = 20000              # Hz, 50 microsecond updates
feedback_noise_seed = 17
use_feedback_position_noise = True
feedback_position_noise_rms = 0e-9           # m RMS
feedback_total_loop_frequency=20000
feedback_total_loop_delay = 1/feedback_total_loop_frequency            # s
feedback_kp_multiplier = 0.2                  # Kp = this number * axial spring constant kz
feedback_kd_multiplier = 8.0                  # Kd = this number * gas damping coefficient b
feedback_power_min_factor = 0.80              # minimum command = this * nominal power
feedback_power_max_factor = 1.20              # maximum command = this * nominal power
feedback_velocity_filter_alpha = 0.25         # lower values smooth velocity more

# Numerical tolerances
focus_zero_tolerance = 1e-30
radial_zero_tolerance = 1e-30
near_axis_tolerance = 1e-12

# Plotting and diagnostic ranges
display_plots = False
max_plot_points = 200000
simple_atmospheric_trajectory_psd_plot = False
simple_atmospheric_trajectory_psd_plot_filename = (
    os.environ.get(
        "PROJECT_3D_SIMPLE_ATMOSPHERIC_PLOT",
        "atmospheric_trajectory_and_psd.png",
    )
)
psd_plot_max_frequency_override = None
psd_min_frequency_factor = 0.95
psd_min_value = 1e-30
psd_max_value = None
psd_segment_duration_seconds = None
psd_segment_samples = None
psd_overlap_fraction = 0.0
psd_overlap_samples = None
use_disk_backed_large_arrays = False
disk_array_directory = "baoab_disk_arrays"
disk_array_chunk_size = 1_000_000
use_adaptive_psd_timing = True
target_bins_across_linewidth = 10.0
minimum_independent_welch_segments = 20.0
minimum_psd_segment_duration_seconds = 20.0
maximum_psd_segment_duration_seconds = None
maximum_t_end_for_adaptive_psd = 600.0
include_z_detector_resolution_psd = True
detector_position_resolution_nm = 100.0
fit_z_detector_resolution_psd = True
surface_temperature_impinging_gas_temperature = None  # None uses gas temperature T
surface_temperature_accommodation_alpha = brownian_surface_temperature_accommodation_alpha
use_physical_psd_c_initial_guess = True
fit_psd_over_whole_frequency_range = False
psd_fit_frequency_min_hz = None
psd_fit_frequency_max_hz = None
psd_fit_peak_band_half_width_hz = 1.0
psd_fit_peak_weight = 0.0
psd_fit_max_points = None

# Linewidth-multiplier sweep. The fit window is chosen from a rough PSD fit,
# not from the known input surface temperature. The known input is used only
# afterwards to score this synthetic recovery test.
run_linewidth_multiplier_sweep = True
linewidth_fit_multiplier = 1
linewidth_fit_multipliers = [linewidth_fit_multiplier]
linewidth_sweep_rough_fit_half_width_hz = 1.0
linewidth_sweep_sources = ("true", "measured")
minimum_fit_bins_each_side = 8
use_pressure_adaptive_psd_fit = True
use_constrained_psd_fit = False
pressure_adaptive_linewidth_multiplier = 2.0
pressure_adaptive_retry_width_scales = (1.0, 2.0, 4.0)
pressure_adaptive_min_half_width_hz = None
pressure_adaptive_max_half_width_hz = None
pressure_adaptive_gamma_ratio_bounds = (0.1, 10.0)
use_psd_area_surface_temperature_estimator = True
use_variance_surface_temperature_estimator = True
subtract_detector_quantisation_variance = True
run_temperature_pressure_error_heatmap = False
temperature_pressure_heatmap_surface_temperatures = np.linspace(300.0, 1200.0, 50)
temperature_pressure_heatmap_pressures = np.logspace(0, 5, 50)
temperature_pressure_heatmap_estimator = "psd_measured"
temperature_pressure_heatmap_plot_filename = (
    os.environ.get(
        "PROJECT_3D_TEMPERATURE_PRESSURE_HEATMAP_PLOT",
        "temperature_pressure_recovered_error_heatmap.png",
    )
)
run_pressure_temperature_error_vs_pressure_sweep = True
pressure_temperature_error_sweep_values = pressure_values = np.logspace(0, 5,20)
pressure_temperature_error_sweep_surface_temperatures = np.array(
    [500.0, 700.0, 900.0],
    dtype=float,
)
pressure_temperature_sweep_surface_temperatures_override = os.environ.get(
    "PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_SURFACE_TEMPERATURES"
)
if pressure_temperature_sweep_surface_temperatures_override:
    pressure_temperature_error_sweep_surface_temperatures = np.array(
        [
            float(value.strip())
            for value in pressure_temperature_sweep_surface_temperatures_override.split(",")
            if value.strip()
        ],
        dtype=float,
    )
pressure_sweep_values_override = os.environ.get(
    "PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_VALUES"
)
if pressure_sweep_values_override:
    pressure_temperature_error_sweep_values = np.array(
        [
            float(value.strip())
            for value in pressure_sweep_values_override.split(",")
            if value.strip()
        ],
        dtype=float,
    )
pressure_temperature_error_sweep_estimators = (
    "psd_measured",
)
pressure_temperature_error_sweep_plot_filename = (
    os.environ.get(
        "PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_PLOT",
        "pressure_recovered_temperature_error.png",
    )
)
transverse_force_x_min = -40e-6
transverse_force_x_max = 40e-6
transverse_force_points = 401
vertical_force_half_width = 500e-6
vertical_force_points = 1000
force_field_x_min = -60e-6
force_field_x_max = 60e-6
force_field_x_points = 31
force_field_z_half_width = 100e-6
force_field_z_points = 31
force_field_contour_levels = 30
force_field_quiver_scale = 55
on_axis_check_z_min = -200e-6
on_axis_check_z_max = 300e-6
on_axis_check_points = 1000
time_trace_figsize = (8, 8)
single_diagnostic_figsize = (8, 4)
spectrum_figsize = (8, 5)
trajectory_3d_figsize = (8, 6)
trajectory_projection_figsize = (13, 4)
force_check_figsize = (8, 5)
force_field_figsize = (8, 6)


t_end = 100.0
psd_segment_duration_seconds = 20.0
linewidth_sweep_rough_fit_half_width_hz = 10.0

linewidth_fit_multipliers = [0.1,1, 2, 3, 5, 7, 10, 12, 15, 20,100,500,1000]

pressure_temperature_error_sweep_child = (
    os.environ.get("PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_CHILD") == "1"
)
pressure_temperature_error_variance_only_child = (
    pressure_temperature_error_sweep_child
    and all(
        estimator in ("variance_true", "variance_measured")
        for estimator in pressure_temperature_error_sweep_estimators
    )
)
heatmap_surface_temperatures_override = os.environ.get(
    "PROJECT_3D_HEATMAP_SURFACE_TEMPERATURES"
)
if heatmap_surface_temperatures_override:
    temperature_pressure_heatmap_surface_temperatures = np.array(
        [
            float(value.strip())
            for value in heatmap_surface_temperatures_override.split(",")
            if value.strip()
        ],
        dtype=float,
    )
heatmap_pressures_override = os.environ.get("PROJECT_3D_HEATMAP_PRESSURES")
if heatmap_pressures_override:
    temperature_pressure_heatmap_pressures = np.array(
        [
            float(value.strip())
            for value in heatmap_pressures_override.split(",")
            if value.strip()
        ],
        dtype=float,
    )
if simple_atmospheric_trajectory_psd_plot:
    use_adaptive_psd_timing = False
if pressure_temperature_error_sweep_child:
    run_temperature_pressure_error_heatmap = False
    run_pressure_temperature_error_vs_pressure_sweep = False
    run_linewidth_multiplier_sweep = False

def finish_plot():
    if display_plots:
        plt.show()
    else:
        plt.close()


script_directory = Path(__file__).resolve().parent
disk_array_run_directory = None


def pressure_temperature_error_sweep_plot_path():
    path = Path(pressure_temperature_error_sweep_plot_filename)
    if not path.is_absolute():
        path = script_directory / path
    return path


def temperature_pressure_heatmap_plot_path():
    path = Path(temperature_pressure_heatmap_plot_filename)
    if not path.is_absolute():
        path = script_directory / path
    return path


def parse_pressure_temperature_error_results(output_text):
    result_pattern = re.compile(
        r"PRESSURE_TEMPERATURE_ERROR_RESULT "
        r"pressure_pa=(?P<pressure>\S+) "
        r"estimator=(?P<estimator>\S+) "
        r"recovered_T_surface_K=(?P<recovered>\S+) "
        r"error_percent=(?P<error>\S+)"
    )

    rows = []
    for match in result_pattern.finditer(output_text):
        rows.append(
            {
                "pressure_pa": float(match.group("pressure")),
                "estimator": match.group("estimator"),
                "recovered_surface_temperature": float(match.group("recovered")),
                "error_percent": float(match.group("error")),
            }
        )
    return rows


def heatmap_axis_edges(values, log_scale=False):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        width = 0.5 * values[0] if values[0] != 0.0 else 0.5
        return np.array([values[0] - width, values[0] + width], dtype=float)

    if log_scale:
        log_values = np.log10(values)
        log_edges = np.empty(len(values) + 1, dtype=float)
        log_edges[1:-1] = 0.5 * (log_values[:-1] + log_values[1:])
        log_edges[0] = log_values[0] - 0.5 * (log_values[1] - log_values[0])
        log_edges[-1] = log_values[-1] + 0.5 * (log_values[-1] - log_values[-2])
        return 10.0**log_edges

    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def run_temperature_pressure_error_heatmap_parent_sweep():
    child_env_name = "PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_CHILD"
    pressure_env_name = "PROJECT_3D_LINEWIDTH_PRESSURE_PA"
    surface_temperature_env_name = "PROJECT_3D_LINEWIDTH_SURFACE_TEMPERATURE_K"
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / (
        "project_3d_linewidth_matplotlib"
    )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    surface_temperatures = np.asarray(
        temperature_pressure_heatmap_surface_temperatures,
        dtype=float,
    )
    pressures = np.asarray(temperature_pressure_heatmap_pressures, dtype=float)
    error_grid = np.full((len(pressures), len(surface_temperatures)), np.nan)

    print("\nPressure-temperature recovered surface-temperature error heat map:")
    print("pressure_Pa, input_T_surface_K, estimator, signed_error_percent")

    for pressure_index, pressure_value in enumerate(pressures):
        for temperature_index, surface_temperature in enumerate(surface_temperatures):
            print(
                f"Running p = {pressure_value:g} Pa, "
                f"T_surface = {surface_temperature:g} K"
            )
            child_environment = os.environ.copy()
            child_environment[child_env_name] = "1"
            child_environment[pressure_env_name] = repr(float(pressure_value))
            child_environment[surface_temperature_env_name] = repr(
                float(surface_temperature)
            )
            child_environment.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(script_directory),
                env=child_environment,
                text=True,
                capture_output=True,
            )
            child_rows = parse_pressure_temperature_error_results(completed.stdout)
            matching_rows = [
                row
                for row in child_rows
                if row["estimator"] == temperature_pressure_heatmap_estimator
            ]

            if matching_rows:
                error_grid[pressure_index, temperature_index] = matching_rows[-1][
                    "error_percent"
                ]
                print(
                    pressure_value,
                    surface_temperature,
                    temperature_pressure_heatmap_estimator,
                    error_grid[pressure_index, temperature_index],
                    sep=", ",
                )
            else:
                print(
                    pressure_value,
                    surface_temperature,
                    temperature_pressure_heatmap_estimator,
                    np.nan,
                    sep=", ",
                )
                if completed.returncode != 0:
                    tail_lines = (completed.stdout + completed.stderr).splitlines()[-20:]
                    for line in tail_lines:
                        print("  " + line)

    temperature_edges = heatmap_axis_edges(surface_temperatures)
    pressure_edges = heatmap_axis_edges(pressures, log_scale=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(
        temperature_edges,
        pressure_edges,
        error_grid,
        shading="auto",
        cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Particle surface temperature / K")
    ax.set_ylabel("Pressure / Pa")
    ax.set_title(
        "Recovered surface-temperature percentage-error heat map "
        f"({temperature_pressure_heatmap_estimator})"
    )
    ax.set_xticks(surface_temperatures)
    ax.set_xticklabels([f"{value:g}" for value in surface_temperatures], rotation=45)
    ax.set_yticks(pressures)
    ax.set_yticklabels([f"{value:g}" for value in pressures])
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Signed error in recovered surface temperature / %")
    fig.tight_layout()

    output_path = temperature_pressure_heatmap_plot_path()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finish_plot()
    print("\nSaved pressure-temperature error heat map to", output_path)
    return error_grid


def run_pressure_temperature_error_vs_pressure_parent_sweep():
    rows = []
    child_env_name = "PROJECT_3D_LINEWIDTH_PRESSURE_SWEEP_CHILD"
    pressure_env_name = "PROJECT_3D_LINEWIDTH_PRESSURE_PA"
    surface_temperature_env_name = "PROJECT_3D_LINEWIDTH_SURFACE_TEMPERATURE_K"
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / (
        "project_3d_linewidth_matplotlib"
    )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    print("\nPressure sweep for recovered surface-temperature error:")
    print(
        "pressure_Pa, input_T_surface_K, estimator, "
        "recovered_T_surface_K, signed_error_percent"
    )

    for surface_temperature in pressure_temperature_error_sweep_surface_temperatures:
        for pressure_value in pressure_temperature_error_sweep_values:
            print(
                f"\nRunning pressure case p = {pressure_value:g} Pa, "
                f"T_surface = {surface_temperature:g} K"
            )
            child_environment = os.environ.copy()
            child_environment[child_env_name] = "1"
            child_environment[pressure_env_name] = repr(float(pressure_value))
            child_environment[surface_temperature_env_name] = repr(
                float(surface_temperature)
            )
            child_environment.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(script_directory),
                env=child_environment,
                text=True,
                capture_output=True,
            )

            child_rows = parse_pressure_temperature_error_results(completed.stdout)
            wanted_child_rows = [
                row
                for row in child_rows
                if row["estimator"] in pressure_temperature_error_sweep_estimators
            ]

            if completed.returncode != 0 and not wanted_child_rows:
                print(
                    f"  pressure case failed with return code {completed.returncode}"
                )
                tail_lines = (completed.stdout + completed.stderr).splitlines()[-40:]
                for line in tail_lines:
                    print("  " + line)
                for estimator in pressure_temperature_error_sweep_estimators:
                    rows.append(
                        {
                            "pressure_pa": float(pressure_value),
                            "surface_temperature": float(surface_temperature),
                            "estimator": estimator,
                            "recovered_surface_temperature": np.nan,
                            "error_percent": np.nan,
                        }
                    )
                continue

            if not wanted_child_rows:
                print("  no recovered-temperature marker was found in child output")
                for estimator in pressure_temperature_error_sweep_estimators:
                    rows.append(
                        {
                            "pressure_pa": float(pressure_value),
                            "surface_temperature": float(surface_temperature),
                            "estimator": estimator,
                            "recovered_surface_temperature": np.nan,
                            "error_percent": np.nan,
                        }
                    )
                continue

            for row in wanted_child_rows:
                row["surface_temperature"] = float(surface_temperature)
                rows.append(row)
                print(
                    row["pressure_pa"],
                    row["surface_temperature"],
                    row["estimator"],
                    row["recovered_surface_temperature"],
                    row["error_percent"],
                    sep=", ",
                )

    fig, ax_error = plt.subplots(figsize=spectrum_figsize)
    for surface_temperature in pressure_temperature_error_sweep_surface_temperatures:
        estimator_rows = [
            row
            for row in rows
            if row["estimator"] == "psd_measured"
            and np.isclose(row["surface_temperature"], surface_temperature)
        ]
        pressures = np.array([row["pressure_pa"] for row in estimator_rows], dtype=float)
        errors = np.array([row["error_percent"] for row in estimator_rows], dtype=float)
        finite_mask = np.isfinite(pressures) & np.isfinite(errors)

        if np.any(finite_mask):
            ax_error.semilogx(
                pressures[finite_mask],
                errors[finite_mask],
                marker="o",
                linewidth=1.5,
                label=f"{surface_temperature:g} K measured PSD",
            )

    ax_error.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_error.set_xlabel("Pressure / Pa")
    ax_error.set_ylabel("Signed error in recovered surface temperature / %")
    ax_error.set_title("Recovered surface-temperature error vs pressure")
    ax_error.grid(True, which="both")

    handles_error, labels_error = ax_error.get_legend_handles_labels()
    ax_error.legend(
        handles_error,
        labels_error,
        loc="best",
    )
    fig.tight_layout()
    output_path = pressure_temperature_error_sweep_plot_path()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finish_plot()
    print("\nSaved pressure-temperature recovery error plot to", output_path)
    return rows


if run_temperature_pressure_error_heatmap:
    run_temperature_pressure_error_heatmap_parent_sweep()
    raise SystemExit(0)


if run_pressure_temperature_error_vs_pressure_sweep:
    run_pressure_temperature_error_vs_pressure_parent_sweep()
    raise SystemExit(0)


def get_disk_array_run_directory():
    global disk_array_run_directory

    if disk_array_run_directory is None:
        base_directory = Path(disk_array_directory)
        if not base_directory.is_absolute():
            base_directory = script_directory / base_directory
        base_directory.mkdir(parents=True, exist_ok=True)

        run_directory = base_directory / f"run_pid_{os.getpid()}"
        suffix = 1
        while run_directory.exists():
            run_directory = base_directory / f"run_pid_{os.getpid()}_{suffix}"
            suffix += 1
        run_directory.mkdir(parents=True)
        disk_array_run_directory = run_directory
        print("Disk-backed arrays written to =", disk_array_run_directory)

    return disk_array_run_directory


def flush_large_array(array):
    if isinstance(array, np.memmap):
        array.flush()


def fill_large_array(array, value):
    for start in range(0, len(array), disk_array_chunk_size):
        end = min(start + disk_array_chunk_size, len(array))
        array[start:end] = value
    flush_large_array(array)


def make_large_array(name, length, fill_value=None):
    length = int(length)

    if use_disk_backed_large_arrays:
        path = get_disk_array_run_directory() / f"{name}.dat"
        array = np.memmap(path, dtype=np.float64, mode="w+", shape=(length,))
    else:
        array = np.empty(length, dtype=np.float64)

    if fill_value is not None:
        fill_large_array(array, fill_value)

    return array


def make_time_array(name):
    sample_count = int(np.floor((t_end - t_start) / dt_baoab + 0.5)) + 1
    array = make_large_array(name, sample_count)

    for start in range(0, sample_count, disk_array_chunk_size):
        end = min(start + disk_array_chunk_size, sample_count)
        array[start:end] = t_start + dt_baoab * np.arange(start, end)

    flush_large_array(array)
    return array


def make_random_normal_array(name, random_generator, length):
    array = make_large_array(name, length)

    for start in range(0, length, disk_array_chunk_size):
        end = min(start + disk_array_chunk_size, length)
        array[start:end] = random_generator.normal(size=end - start)

    flush_large_array(array)
    return array


def make_difference_array(name, left, right):
    array = make_large_array(name, len(left))

    for start in range(0, len(left), disk_array_chunk_size):
        end = min(start + disk_array_chunk_size, len(left))
        array[start:end] = left[start:end] - right[start:end]

    flush_large_array(array)
    return array


def make_quantised_displacement_array(name, position, origin, resolution):
    array = make_large_array(name, len(position))

    if resolution <= 0.0:
        for start in range(0, len(position), disk_array_chunk_size):
            end = min(start + disk_array_chunk_size, len(position))
            array[start:end] = position[start:end] - origin
    else:
        for start in range(0, len(position), disk_array_chunk_size):
            end = min(start + disk_array_chunk_size, len(position))
            displacement = position[start:end] - origin
            array[start:end] = np.round(displacement / resolution) * resolution

    flush_large_array(array)
    return array


# ****************************************************************************************************************************************************
# Derived quantities
# ****************************************************************************************************************************************************
volume = (4/3) * np.pi * radius**3
m = density * volume

rho_g = p * M_air / (R * T)
lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

zR_m2 = np.pi * w0**2 / (M2 * wavelength)
zR = zR_m2 if use_m2_rayleigh_range else zR_manual

I0 = 2 * P_laser / (np.pi * w0**2)   # Gaussian peak intensity, W/m^2

# print("Particle mass =", m, "kg")
# print("Weight mg =", m*g, "N")
# print("Gas density =", rho_g, "kg/m^3")
# print("Mean free path =", lambda_mfp, "m")
# print("Knudsen number =", Kn)
# print("Laser power =", P_laser, "W")
# print("Wavelength =", wavelength, "m")
# print("Beam quality M2 =", M2)
# print("Using M2-derived Rayleigh range =", use_m2_rayleigh_range)
# print("Manual Rayleigh range zR_manual =", zR_manual, "m")
# print("M2-derived Rayleigh range zR_m2 =", zR_m2, "m")
# print("Rayleigh range used zR =", zR, "m")
# print("Peak intensity I0 =", I0, "W/m^2")
# print("Paraxial divergence half-angle w0/zR =", w0 / zR, "rad")


def beam_width(z):
   
    return w0 * np.sqrt(1 + (z / zR)**2)


def intensity(x, z):
   
    s = 1 + (z / zR)**2
    return (1 / s) * np.exp(-2 * x**2 / (w0**2 * s))


def physical_intensity(x, z, power_factor=1.0):
    return power_factor * I0 * intensity(x, z)


def wavefront_radius(z):
    """
    Gaussian beam wavefront radius of curvature.

    R(z) is negative before the focus, positive after the focus, and infinite
    at the focus. The local paraxial ray direction is approximately:

        s_hat proportional to (x / R(z), 1)
    """
    if abs(z) < focus_zero_tolerance:
        return np.inf

    return z * (1 + (zR / z)**2)


# ****************************************************************************************************************************************************
# Ashkin ray-optics force model
# ****************************************************************************************************************************************************
# This replaces the old phenomenological gradient and scattering forces.
# The sphere's projected disk is sampled by many ray bundles. Each ray
# contributes:
#
#     dF = (n_medium / c) dP (-Q_g e_perp + Q_s s_hat)
#
# where Q_s and Q_g are Ashkin ray-optics force efficiencies, s_hat is the
# local focused-ray direction, and e_perp is perpendicular to s_hat in the
# x-z plane. This gives the gradient-like part an axial component when the
# beam rays are converging or diverging.

u_values = np.linspace(-radius, radius, ray_grid_points)
v_values = np.linspace(-radius, radius, ray_grid_points)
du = u_values[1] - u_values[0]
dv = v_values[1] - v_values[0]
dA = du * dv

U, V = np.meshgrid(u_values, v_values, indexing="ij")
rho = np.sqrt(U**2 + V**2)
ray_mask = rho < radius

U_hit = U[ray_mask]
V_hit = V[ray_mask]
rho_hit = rho[ray_mask]

sin_theta_i = rho_hit / radius
theta_i = np.arcsin(np.clip(sin_theta_i, 0.0, 1.0))
sin_theta_r = (n_medium / n_particle) * sin_theta_i
theta_r = np.arcsin(np.clip(sin_theta_r, 0.0, 1.0))

cos_i = np.cos(theta_i)
cos_r = np.cos(theta_r)

Rs = ((n_medium * cos_i - n_particle * cos_r) / (n_medium * cos_i + n_particle * cos_r))**2
Rp = ((n_medium * cos_r - n_particle * cos_i) / (n_medium * cos_r + n_particle * cos_i))**2


def ashkin_efficiencies_from_reflectance(fresnel_R):
    """Ashkin scattering-like and gradient-like efficiencies for one polarization."""
    fresnel_T = 1 - fresnel_R
    denom = 1 + fresnel_R**2 + 2 * fresnel_R * np.cos(2 * theta_r)

    Q_scat = (
        1
        + fresnel_R * np.cos(2 * theta_i)
        - (
            fresnel_T**2
            * (np.cos(2 * theta_i - 2 * theta_r) + fresnel_R * np.cos(2 * theta_i))
            / denom
        )
    )

    Q_grad = (
        fresnel_R * np.sin(2 * theta_i)
        - (
            fresnel_T**2
            * (np.sin(2 * theta_i - 2 * theta_r) + fresnel_R * np.sin(2 * theta_i))
            / denom
        )
    )
    return Q_scat, Q_grad


Q_s_s_pol, Q_g_s_pol = ashkin_efficiencies_from_reflectance(Rs)
Q_s_p_pol, Q_g_p_pol = ashkin_efficiencies_from_reflectance(Rp)

# Unpolarised light is a 50/50 mixture of s and p polarisations.  Average the
# final force efficiencies, rather than averaging Fresnel R before applying the
# nonlinear multiple-reflection expression.
Q_s = 0.5 * (Q_s_s_pol + Q_s_p_pol)
Q_g = 0.5 * (Q_g_s_pol + Q_g_p_pol)

rho_safe = np.where(rho_hit > 0, rho_hit, 1.0)
u_hat = U_hit / rho_safe


def F_ray_optics_2d_scalar(x, z, power_factor=1.0):
    """
    Ashkin-style ray-optics force for the x-z model.

    The full projected disk is integrated, but only Fx and Fz are returned.
    """
    Fx_scat, Fz_scat, Fx_grad, Fz_grad = F_ray_optics_2d_components_scalar(
        x,
        z,
        power_factor
    )
    return Fx_scat + Fx_grad, Fz_scat + Fz_grad


def F_ray_optics_2d_components_scalar(x, z, power_factor=1.0):
    """
    Ray-optics force split into scattering-like and gradient-like parts.

    Returns:
        Fx_scat, Fz_scat, Fx_grad, Fz_grad

    In this Ashkin ray model, Q_s is treated as the scattering-like component
    along the local ray direction, while Q_g is treated as the gradient-like
    component perpendicular to the local ray direction.
    """
    ray_x = x + U_hit
    dP = physical_intensity(ray_x, z, power_factor) * dA
    prefactor = n_medium / c_light

    R_wavefront = wavefront_radius(z)
    if np.isinf(R_wavefront):
        s_x = np.zeros_like(ray_x)
    else:
        s_x = ray_x / R_wavefront

    s_z = np.ones_like(ray_x)
    s_norm = np.sqrt(s_x**2 + s_z**2)
    s_x = s_x / s_norm
    s_z = s_z / s_norm

    # Unit vector perpendicular to the local ray direction in the x-z plane.
    # For a parallel ray, this becomes e_perp = +x_hat.
    e_perp_x = s_z
    e_perp_z = -s_x

    Fx_scat = prefactor * np.sum(dP * Q_s * s_x)
    Fz_scat = prefactor * np.sum(dP * Q_s * s_z)

    # The transverse gradient-like force points toward higher intensity.
    # With u_hat defined radially outward from the particle centre, the
    # restoring direction is -u_hat for a particle displaced to +x.
    Fx_grad = prefactor * np.sum(dP * (-Q_g * u_hat * e_perp_x))
    Fz_grad = prefactor * np.sum(dP * (-Q_g * u_hat * e_perp_z))

    return Fx_scat, Fz_scat, Fx_grad, Fz_grad


def F_ray_optics_2d(x, z, power_factor=1.0):
    """
    Vectorized wrapper around the scalar ray-optics force.
    """
    x_arr, z_arr, power_factor_arr = np.broadcast_arrays(x, z, power_factor)

    if x_arr.shape == ():
        return F_ray_optics_2d_scalar(
            float(x_arr),
            float(z_arr),
            float(power_factor_arr)
        )

    Fx = np.zeros_like(x_arr, dtype=float)
    Fz = np.zeros_like(z_arr, dtype=float)

    for index in np.ndindex(x_arr.shape):
        Fx[index], Fz[index] = F_ray_optics_2d_scalar(
            x_arr[index],
            z_arr[index],
            power_factor_arr[index]
        )

    return Fx, Fz


Fx_focus, Fz_focus = F_ray_optics_2d(0.0, 0.0)
# print("Ray-optics force at focus Fx =", Fx_focus, "N")
# print("Ray-optics force at focus Fz =", Fz_focus, "N")
# print("Ray-optics Fz at focus / mg =", Fz_focus / (m*g))


# ****************************************************************************************************************************************************
# Photophoretic force model
# ****************************************************************************************************************************************************

def mean_thermal_speed():
    return np.sqrt(8 * R * T / (np.pi * M_air))


def photophoretic_D():
    c_bar = mean_thermal_speed()
    return (np.pi * c_bar * eta / (2 * T)) * np.sqrt(np.pi * kappa_t / 3)


def photophoretic_p_max():
    D = photophoretic_D()
    return (3 * T / (np.pi * radius)) * D * np.sqrt(2 / alpha_acc)


def absorbed_intensity(x, z, power_factor=1.0):
    return absorption_fraction * physical_intensity(x, z, power_factor)


def photophoretic_force_magnitude(x, z, power_factor=1.0):
    D = photophoretic_D()
    p_max_ph = photophoretic_p_max()

    I_abs = absorbed_intensity(x, z, power_factor)

    F_max = (
        0.5
        * radius**2
        * D
        * np.sqrt(alpha_acc / 2)
        * I_abs
        / k_particle
    )

    return 2 * F_max / ((p / p_max_ph) + (p_max_ph / p))


def F_photo_2d(x, z, power_factor=1.0):
    """
    Improved photophoretic force.
    Positive sign means force points upward, in +z.
    """
    return 0.0, photophoretic_force_magnitude(x, z, power_factor)


def F_optical_2d_direct(x, z, power_factor=1.0):
    
    Fx_ray, Fz_ray = F_ray_optics_2d(x, z, power_factor)
    Fx_photo, Fz_photo = F_photo_2d(x, z, power_factor)

    Fx = Fx_ray + Fx_photo
    Fz = Fz_ray + Fz_photo

    return Fx, Fz


force_lookup_ready = False
force_lookup_r_values = None
force_lookup_z_values = None
force_lookup_Fr_interpolator = None
force_lookup_Fz_interpolator = None
force_lookup_Fr_table = None
force_lookup_Fz_table = None


def build_force_lookup_table(r_min, r_max, z_min, z_max):
    """
    Precompute the optical + photophoretic force on a cylindrical r-z grid.

    The expensive ray-optics calculation is only performed for y = 0 and
    x = r. Cylindrical symmetry is then used during the 3D simulation:

        Fx = Fr x/r
        Fy = Fr y/r
        Fz = Fz(r,z)

    The table is built at nominal laser power. This is valid because both the
    ray-optics force and the current photophoretic force scale linearly with
    laser power in this model, so laser noise is applied by multiplying the
    interpolated nominal force by power_factor.
    """
    global force_lookup_ready
    global force_lookup_r_values
    global force_lookup_z_values
    global force_lookup_Fr_interpolator
    global force_lookup_Fz_interpolator
    global force_lookup_Fr_table
    global force_lookup_Fz_table

    force_lookup_start_time = perf_counter()

    force_lookup_r_values = np.linspace(
        r_min,
        r_max,
        force_lookup_grid_points_r
    )
    force_lookup_z_values = np.linspace(
        z_min,
        z_max,
        force_lookup_grid_points_z
    )

    Fr_table = np.zeros(
        (force_lookup_grid_points_r, force_lookup_grid_points_z)
    )
    Fz_table = np.zeros_like(Fr_table)

    for ir, r_value in enumerate(force_lookup_r_values):
        for iz, z_value in enumerate(force_lookup_z_values):
            Fr_table[ir, iz], Fz_table[ir, iz] = F_optical_2d_direct(
                r_value,
                z_value,
                power_factor=1.0
            )

    force_lookup_Fr_interpolator = RegularGridInterpolator(
        (force_lookup_r_values, force_lookup_z_values),
        Fr_table,
        bounds_error=True
    )
    force_lookup_Fz_interpolator = RegularGridInterpolator(
        (force_lookup_r_values, force_lookup_z_values),
        Fz_table,
        bounds_error=True
    )
    force_lookup_Fr_table = np.ascontiguousarray(Fr_table, dtype=np.float64)
    force_lookup_Fz_table = np.ascontiguousarray(Fz_table, dtype=np.float64)

    force_lookup_ready = True

    # print("Cylindrical force lookup table enabled =", use_force_lookup_table)
    # print("Force lookup grid r points =", force_lookup_grid_points_r)
    # print("Force lookup grid z points =", force_lookup_grid_points_z)
    # print(
        # "Force lookup r range =",
        # force_lookup_r_values[0] * 1e6,
        # "to",
        # force_lookup_r_values[-1] * 1e6,
        # "micrometres"
    # )
    # print(
        # "Force lookup z range =",
        # force_lookup_z_values[0] * 1e6,
        # "to",
        # force_lookup_z_values[-1] * 1e6,
        # "micrometres"
    # )
    # print(
        # "Force lookup table build runtime =",
        # perf_counter() - force_lookup_start_time,
        # "s"
    # )


def F_optical_3d_direct(x, y, z, power_factor=1.0):
    x_arr, y_arr, z_arr, power_factor_arr = np.broadcast_arrays(
        x,
        y,
        z,
        power_factor
    )

    r_arr = np.sqrt(x_arr**2 + y_arr**2)

    if x_arr.shape == ():
        Fr, Fz = F_optical_2d_direct(
            float(r_arr),
            float(z_arr),
            float(power_factor_arr)
        )

        if r_arr < radial_zero_tolerance:
            return 0.0, 0.0, Fz

        Fx = Fr * float(x_arr) / float(r_arr)
        Fy = Fr * float(y_arr) / float(r_arr)
        return Fx, Fy, Fz

    Fx = np.zeros_like(x_arr, dtype=float)
    Fy = np.zeros_like(y_arr, dtype=float)
    Fz = np.zeros_like(z_arr, dtype=float)

    for index in np.ndindex(x_arr.shape):
        r_value = r_arr[index]
        Fr_value, Fz_value = F_optical_2d_direct(
            r_value,
            z_arr[index],
            power_factor_arr[index]
        )

        if r_value > radial_zero_tolerance:
            Fx[index] = Fr_value * x_arr[index] / r_value
            Fy[index] = Fr_value * y_arr[index] / r_value

        Fz[index] = Fz_value

    return Fx, Fy, Fz


def F_optical_3d_lookup(x, y, z, power_factor=1.0):
    x_arr, y_arr, z_arr, power_factor_arr = np.broadcast_arrays(
        x,
        y,
        z,
        power_factor
    )

    r_arr = np.sqrt(x_arr**2 + y_arr**2)

    r_inside = (
        np.min(r_arr) >= force_lookup_r_values[0]
        and np.max(r_arr) <= force_lookup_r_values[-1]
    )
    z_inside = (
        np.min(z_arr) >= force_lookup_z_values[0]
        and np.max(z_arr) <= force_lookup_z_values[-1]
    )

    if not (r_inside and z_inside):
        return F_optical_3d_direct(x, y, z, power_factor)

    points = np.column_stack((r_arr.ravel(), z_arr.ravel()))

    Fr_nominal = force_lookup_Fr_interpolator(points).reshape(r_arr.shape)
    Fz_nominal = force_lookup_Fz_interpolator(points).reshape(z_arr.shape)

    Fr = power_factor_arr * Fr_nominal
    Fz = power_factor_arr * Fz_nominal

    Fx = np.zeros_like(r_arr, dtype=float)
    Fy = np.zeros_like(r_arr, dtype=float)
    nonzero_r = r_arr > radial_zero_tolerance

    Fx[nonzero_r] = Fr[nonzero_r] * x_arr[nonzero_r] / r_arr[nonzero_r]
    Fy[nonzero_r] = Fr[nonzero_r] * y_arr[nonzero_r] / r_arr[nonzero_r]

    if Fx.shape == ():
        return float(Fx), float(Fy), float(Fz)

    return Fx, Fy, Fz


def F_optical_3d_lookup_scalar_clamped(x, y, z, power_factor=1.0):
    """
    Fast scalar force lookup for time-stepping loops.

    This avoids the per-step overhead of RegularGridInterpolator for closed-loop
    feedback. Positions outside the lookup table are clamped to the table edge,
    matching the fast Cython loop's behaviour.
    """
    r = np.sqrt(x**2 + y**2)

    r_min = force_lookup_r_values[0]
    r_max = force_lookup_r_values[-1]
    z_min = force_lookup_z_values[0]
    z_max = force_lookup_z_values[-1]

    r_lookup = min(max(r, r_min), r_max)
    z_lookup = min(max(z, z_min), z_max)
    out_of_bounds = r_lookup != r or z_lookup != z

    n_r = len(force_lookup_r_values)
    n_z = len(force_lookup_z_values)

    if n_r < 2 or n_z < 2:
        raise ValueError("Force lookup table needs at least two grid points per axis.")

    dr = (r_max - r_min) / (n_r - 1)
    dz = (z_max - z_min) / (n_z - 1)

    r_grid_position = (r_lookup - r_min) / dr
    z_grid_position = (z_lookup - z_min) / dz

    ir = int(r_grid_position)
    iz = int(z_grid_position)

    if ir >= n_r - 1:
        ir = n_r - 2
        r_weight = 1.0
    else:
        r_weight = r_grid_position - ir

    if iz >= n_z - 1:
        iz = n_z - 2
        z_weight = 1.0
    else:
        z_weight = z_grid_position - iz

    Fr00 = force_lookup_Fr_table[ir, iz]
    Fr10 = force_lookup_Fr_table[ir + 1, iz]
    Fr01 = force_lookup_Fr_table[ir, iz + 1]
    Fr11 = force_lookup_Fr_table[ir + 1, iz + 1]

    Fz00 = force_lookup_Fz_table[ir, iz]
    Fz10 = force_lookup_Fz_table[ir + 1, iz]
    Fz01 = force_lookup_Fz_table[ir, iz + 1]
    Fz11 = force_lookup_Fz_table[ir + 1, iz + 1]

    Fr_nominal = (
        (1 - r_weight) * (1 - z_weight) * Fr00
        + r_weight * (1 - z_weight) * Fr10
        + (1 - r_weight) * z_weight * Fr01
        + r_weight * z_weight * Fr11
    )
    Fz_nominal = (
        (1 - r_weight) * (1 - z_weight) * Fz00
        + r_weight * (1 - z_weight) * Fz10
        + (1 - r_weight) * z_weight * Fz01
        + r_weight * z_weight * Fz11
    )

    Fr = power_factor * Fr_nominal
    Fz = power_factor * Fz_nominal

    if r > radial_zero_tolerance:
        Fx = Fr * x / r
        Fy = Fr * y / r
    else:
        Fx = 0.0
        Fy = 0.0

    return Fx, Fy, Fz, out_of_bounds


def F_optical_3d(x, y, z, power_factor=1.0):
    if use_force_lookup_table and force_lookup_ready:
        return F_optical_3d_lookup(x, y, z, power_factor)

    return F_optical_3d_direct(x, y, z, power_factor)


def F_optical_2d(x, z, power_factor=1.0):
    Fx, _, Fz = F_optical_3d(x, 0.0, z, power_factor)
    return Fx, Fz


def Fz_net_on_axis(z):
    
    _, Fz = F_optical_2d(0.0, z)
    return Fz - m*g


# ****************************************************************************************************************************************************
# Find on-axis equilibrium
# ****************************************************************************************************************************************************
z_scan = np.linspace(equilibrium_z_min, equilibrium_z_max, equilibrium_scan_points)
F_scan = Fz_net_on_axis(z_scan)

roots = []

for i in range(len(z_scan) - 1):
    if F_scan[i] * F_scan[i + 1] < 0:
        root = brentq(
            Fz_net_on_axis,
            z_scan[i],
            z_scan[i + 1],
            xtol=root_finding_xtol,
            rtol=root_finding_rtol
        )
        roots.append(root)

# print("On-axis equilibrium positions / micrometres:")
for root in roots:
    pass
    # print(root * 1e6)


def numerical_derivative_1d(func, z, h=numerical_derivative_step):
    return (func(z + h) - func(z - h)) / (2*h)


stable_roots = []

for root in roots:
    slope = numerical_derivative_1d(Fz_net_on_axis, root)
    if slope < 0:
        stable_roots.append(root)

if len(stable_roots) == 0:
    raise ValueError("No stable on-axis equilibrium found.")

if not -len(stable_roots) <= equilibrium_root_index < len(stable_roots):
    raise IndexError(
        "equilibrium_root_index is outside the stable root list. "
        f"Found {len(stable_roots)} stable root(s)."
    )

x_eq = x_equilibrium
y_eq = y_equilibrium
z_eq = stable_roots[equilibrium_root_index]

# print("Chosen equilibrium x =", x_eq * 1e6, "micrometres")
# print("Chosen equilibrium y =", y_eq * 1e6, "micrometres")
# print("Chosen equilibrium z =", z_eq * 1e6, "micrometres")
# print("Fz_net_on_axis(z_eq) =", Fz_net_on_axis(z_eq), "N")

# ****************************************************************************************************************************************************
# Local spring constants
# ****************************************************************************************************************************************************


def Fx_at_x(x):
    Fx, _ = F_optical_2d(x, z_eq)
    return Fx


def Fz_net_at_z(z):
    _, Fz = F_optical_2d(x_eq, z)
    return Fz - m*g


kx = -numerical_derivative_1d(Fx_at_x, x_eq)
ky = kx
kz = -numerical_derivative_1d(Fz_net_at_z, z_eq)

if kx <= 0 or kz <= 0:
    raise ValueError(
        "The selected equilibrium is not stable. "
        f"kx={kx:.3e} N/m, kz={kz:.3e} N/m. "
        "Check the ray-optics force signs, laser power, search range, or chosen root."
    )

omega_x = np.sqrt(kx / m)
omega_y = np.sqrt(ky / m)
omega_z = np.sqrt(kz / m)

# print("kx =", kx, "N/m")
# print("ky =", ky, "N/m")
# print("kz =", kz, "N/m")
# print("fx =", omega_x / (2*np.pi), "Hz")
# print("fy =", omega_y / (2*np.pi), "Hz")
# print("fz =", omega_z / (2*np.pi), "Hz")

x_rms_thermal = np.sqrt(kB * T / kx)
y_rms_thermal = np.sqrt(kB * T / ky)
z_rms_thermal = np.sqrt(kB * T / kz)

# print("Expected x thermal RMS =", x_rms_thermal * 1e6, "micrometres")
# print("Expected y thermal RMS =", y_rms_thermal * 1e6, "micrometres")
# print("Expected z thermal RMS =", z_rms_thermal * 1e6, "micrometres")

# ****************************************************************************************************************************************************
# Damping
# ****************************************************************************************************************************************************
def gas_density(pressure):
    return pressure * M_air / (R * T)


def mean_free_path(pressure):
    return kB * T / (np.sqrt(2) * np.pi * d_air**2 * pressure)


def knudsen_number(pressure):
    return mean_free_path(pressure) / radius


def cunningham_correction(Kn):
    if Kn <= 0:
        return 1.0

    return 1 + Kn * (
        cunningham_A
        + cunningham_B * np.exp(-cunningham_C / Kn)
    )


def damping_coefficient_stokes():
    """
    Continuum Stokes drag:

        F_drag = -b v
        b = 6 pi eta a

    This is appropriate when Kn << 1.
    """
    return 6 * np.pi * eta * radius


def damping_coefficient_stokes_cunningham(pressure):
    """
    Stokes drag with Cunningham slip correction:

        b = 6 pi eta a / Cc

    This is useful in the slip/transition regime.
    """
    Kn = knudsen_number(pressure)
    Cc = cunningham_correction(Kn)
    return damping_coefficient_stokes() / Cc


def damping_coefficient_epstein(pressure):
    """
    Epstein drag in the free-molecular regime:

        b = (4/3) pi a^2 rho_g c_bar (1 + pi alpha_E / 8)

    alpha_E is an accommodation factor. alpha_E = 1 is a common simple
    diffuse-reflection estimate.
    """
    rho = gas_density(pressure)
    c_bar = mean_thermal_speed()
    accommodation_factor = 1 + np.pi * epstein_accommodation_alpha / 8

    return (4 / 3) * np.pi * radius**2 * rho * c_bar * accommodation_factor


def choose_drag_model(pressure):
    """
    Pick a drag model from Knudsen number.
    """
    Kn = knudsen_number(pressure)

    if Kn < 0.1:
        return "stokes"
    if Kn < 10:
        return "cunningham"
    return "epstein"


def damping_coefficient(pressure, model="auto"):
    """
    Return damping coefficient b for the chosen drag model.
    """
    if model == "auto":
        model = choose_drag_model(pressure)

    if model == "stokes":
        return damping_coefficient_stokes(), model
    if model == "cunningham":
        return damping_coefficient_stokes_cunningham(pressure), model
    if model == "epstein":
        return damping_coefficient_epstein(pressure), model

    raise ValueError(
        "Unknown drag_model. Use 'stokes', 'cunningham', 'epstein', or 'auto'."
    )


b_stokes = damping_coefficient_stokes()
b_cunningham = damping_coefficient_stokes_cunningham(p)
b_epstein = damping_coefficient_epstein(p)

b, drag_model_used = damping_coefficient(p, drag_model)

# print("Drag model requested =", drag_model)
# print("Drag model used =", drag_model_used)
# print("Stokes damping b =", b_stokes, "kg/s")
# print("Cunningham-corrected Stokes damping b =", b_cunningham, "kg/s")
# print("Epstein damping b =", b_epstein, "kg/s")
# print("Selected pressure-dependent damping b =", b, "kg/s")
# print("Damping ratio x =", b / (2 * np.sqrt(m * kx)))
# print("Damping ratio y =", b / (2 * np.sqrt(m * ky)))
# print("Damping ratio z =", b / (2 * np.sqrt(m * kz)))

# ****************************************************************************************************************************************************
# Initial conditions
# ****************************************************************************************************************************************************
x0 = x_eq + x_displacement
y0 = y_eq + y_displacement
z0 = z_eq + z_displacement

Fx_initial, Fy_initial, Fz_initial = F_optical_3d(x0, y0, z0)
ax_initial = Fx_initial / m
ay_initial = Fy_initial / m
az_initial = (Fz_initial - m*g) / m

# print("Initial x displacement =", x_displacement * 1e6, "micrometres")
# print("Initial y displacement =", y_displacement * 1e6, "micrometres")
# print("Initial z displacement =", z_displacement * 1e6, "micrometres")
# print("Initial Fx / mg =", Fx_initial / (m*g))
# print("Initial Fy / mg =", Fy_initial / (m*g))
# print("Initial Fz_net / mg =", (Fz_initial - m*g) / (m*g))
# print("Initial ax =", ax_initial, "m/s^2")
# print("Initial ay =", ay_initial, "m/s^2")
# print("Initial az =", az_initial, "m/s^2")

if use_force_lookup_table:
    # Keep the lookup region local to the trap. If the particle leaves this
    # region, F_optical_3d automatically falls back to the direct ray sum.
    force_lookup_r_max = max(
        force_lookup_r_base_max,
        force_lookup_r_displacement_factor
        * np.sqrt(x_displacement**2 + y_displacement**2),
        force_lookup_r_thermal_factor * x_rms_thermal
    )
    force_lookup_z_half_width = max(
        force_lookup_z_base_half_width,
        force_lookup_z_displacement_factor * abs(z_displacement),
        force_lookup_z_thermal_factor * z_rms_thermal
    )

    build_force_lookup_table(
        force_lookup_r_min,
        force_lookup_r_max,
        z_eq - force_lookup_z_half_width,
        z_eq + force_lookup_z_half_width
    )

    Fx_initial_direct, Fy_initial_direct, Fz_initial_direct = F_optical_3d_direct(x0, y0, z0)
    Fx_initial_lookup, Fy_initial_lookup, Fz_initial_lookup = F_optical_3d_lookup(x0, y0, z0)

    # print(
        # "Lookup check at initial position: |Fx error| / mg =",
        # abs(Fx_initial_lookup - Fx_initial_direct) / (m*g)
    # )
    # print(
        # "Lookup check at initial position: |Fy error| / mg =",
        # abs(Fy_initial_lookup - Fy_initial_direct) / (m*g)
    # )
    # print(
        # "Lookup check at initial position: |Fz error| / mg =",
        # abs(Fz_initial_lookup - Fz_initial_direct) / (m*g)
    # )
else:
    pass
    # print("Force lookup table enabled = False")


def expected_gamma_cm_for_current_brownian_inputs():
    gamma0 = b / m

    if not use_input_surface_temperature_for_brownian:
        return gamma0

    if input_surface_temperature <= 0.0:
        raise ValueError("input_surface_temperature must be positive.")
    if brownian_surface_temperature_accommodation_alpha <= 0.0:
        raise ValueError(
            "brownian_surface_temperature_accommodation_alpha must be positive."
        )

    temperature_emerging = (
        T
        + brownian_surface_temperature_accommodation_alpha
        * (input_surface_temperature - T)
    )
    if temperature_emerging <= 0.0:
        raise ValueError(
            "The chosen surface temperature/accommodation gives a non-positive "
            "emerging gas temperature."
        )

    emerging_temperature_ratio = temperature_emerging / T
    paper_temperature_factor = np.pi / 8.0
    return gamma0 * (
        1.0 + paper_temperature_factor * np.sqrt(emerging_temperature_ratio)
    ) / (1.0 + paper_temperature_factor)


def configure_psd_timing_from_linewidth():
    global t_end, psd_segment_duration_seconds, psd_segment_samples

    if not use_adaptive_psd_timing:
        return

    gamma_for_planning = expected_gamma_cm_for_current_brownian_inputs()
    linewidth_hz = gamma_for_planning / (2.0 * np.pi)
    if not np.isfinite(linewidth_hz) or linewidth_hz <= 0.0:
        print("Adaptive PSD timing skipped: non-positive linewidth.")
        return

    if psd_segment_samples is not None:
        current_segment_duration = psd_segment_samples * dt_baoab
    elif psd_segment_duration_seconds is not None:
        current_segment_duration = psd_segment_duration_seconds
    else:
        current_segment_duration = t_end - t_start

    required_segment_duration = target_bins_across_linewidth / linewidth_hz
    chosen_segment_duration = max(
        current_segment_duration,
        minimum_psd_segment_duration_seconds,
        required_segment_duration,
    )

    if maximum_psd_segment_duration_seconds is not None:
        chosen_segment_duration = min(
            chosen_segment_duration,
            maximum_psd_segment_duration_seconds,
        )

    required_t_end = t_start + minimum_independent_welch_segments * chosen_segment_duration
    chosen_t_end = max(t_end, required_t_end)
    if maximum_t_end_for_adaptive_psd is not None:
        chosen_t_end = max(t_end, min(chosen_t_end, maximum_t_end_for_adaptive_psd))

    available_duration = max(chosen_t_end - t_start, dt_baoab)
    if chosen_segment_duration > available_duration:
        chosen_segment_duration = available_duration

    psd_segment_samples = None
    psd_segment_duration_seconds = chosen_segment_duration
    t_end = chosen_t_end

    achieved_bins_across_linewidth = chosen_segment_duration * linewidth_hz
    achieved_independent_segments = (t_end - t_start) / chosen_segment_duration

    print("\nAdaptive PSD timing:")
    print("  expected linewidth =", linewidth_hz, "Hz")
    print("  PSD segment duration =", psd_segment_duration_seconds, "s")
    print("  t_end =", t_end, "s")
    print("  bins across linewidth =", achieved_bins_across_linewidth)
    print("  approximate independent Welch segments =", achieved_independent_segments)
    if achieved_bins_across_linewidth < target_bins_across_linewidth:
        print(
            "  Warning: target bins across linewidth was not reached. "
            "Increase psd_segment_duration_seconds or maximum_t_end_for_adaptive_psd."
        )
    if achieved_independent_segments < minimum_independent_welch_segments:
        print(
            "  Warning: target Welch averaging was not reached. "
            "Increase t_end or maximum_t_end_for_adaptive_psd."
        )


# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
configure_psd_timing_from_linewidth()
t_baoab = make_time_array("t_baoab")

# print("BAOAB requested timestep =", dt_baoab, "s")
# print("BAOAB sampling frequency =", 1 / dt_baoab, "Hz")
# print("BAOAB number of samples =", len(t_baoab))

# ****************************************************************************************************************************************************
# BAOAB solution with Brownian motion and optional stepwise laser-power noise
# ****************************************************************************************************************************************************
rng = np.random.default_rng(seed=brownian_seed)
laser_rng = np.random.default_rng(seed=laser_noise_seed)

if use_laser_power_noise:
    laser_step_samples = max(1, int(round(laser_noise_step_duration / dt_baoab)))
    laser_step_duration_actual = laser_step_samples * dt_baoab
    laser_allowed_power_factors = np.array(
        [1 - laser_noise_fraction, 1.0, 1 + laser_noise_fraction]
    )
    n_laser_steps = int(np.ceil(len(t_baoab) / laser_step_samples))

    laser_step_power_factors = np.zeros(n_laser_steps)
    laser_step_power_factors[0] = laser_rng.choice(laser_allowed_power_factors)

    for j in range(1, n_laser_steps):
        previous_factor = laser_step_power_factors[j - 1]

        if np.isclose(previous_factor, 1 + laser_noise_fraction):
            possible_factors = np.array([1.0, 1 + laser_noise_fraction])
        elif np.isclose(previous_factor, 1 - laser_noise_fraction):
            possible_factors = np.array([1 - laser_noise_fraction, 1.0])
        else:
            possible_factors = laser_allowed_power_factors

        laser_step_power_factors[j] = laser_rng.choice(possible_factors)

    laser_power_factor = make_large_array("laser_power_factor", len(t_baoab))
    for j, step_power_factor in enumerate(laser_step_power_factors):
        start = j * laser_step_samples
        end = min(start + laser_step_samples, len(t_baoab))
        if start >= len(t_baoab):
            break
        laser_power_factor[start:end] = step_power_factor
    flush_large_array(laser_power_factor)
else:
    laser_step_samples = len(t_baoab)
    laser_step_duration_actual = t_baoab[-1] - t_baoab[0]
    laser_allowed_power_factors = np.array([1.0])
    laser_step_power_factors = np.array([1.0])
    laser_power_factor = make_large_array("laser_power_factor", len(t_baoab), 1.0)

laser_power_time = None

instantaneous_z_equilibrium_by_power_factor = {}

for power_factor_value in laser_allowed_power_factors:
    def Fz_net_on_axis_at_power(z):
        _, Fz = F_optical_2d_direct(0.0, z, power_factor_value)
        return Fz - m*g

    F_scan_power = Fz_net_on_axis_at_power(z_scan)
    roots_power = []

    for i in range(len(z_scan) - 1):
        if F_scan_power[i] * F_scan_power[i + 1] < 0:
            root = brentq(
                Fz_net_on_axis_at_power,
                z_scan[i],
                z_scan[i + 1],
                xtol=root_finding_xtol,
                rtol=root_finding_rtol
            )
            roots_power.append(root)

    stable_roots_power = []

    for root in roots_power:
        slope = numerical_derivative_1d(Fz_net_on_axis_at_power, root)
        if slope < 0:
            stable_roots_power.append(root)

    if len(stable_roots_power) == 0:
        raise ValueError(
            "No stable on-axis equilibrium found for "
            f"laser power factor {power_factor_value:.5f}."
        )

    if not -len(stable_roots_power) <= equilibrium_root_index < len(stable_roots_power):
        raise IndexError(
            "equilibrium_root_index is outside the stable root list for "
            f"laser power factor {power_factor_value:.5f}. "
            f"Found {len(stable_roots_power)} stable root(s)."
        )

    instantaneous_z_equilibrium_by_power_factor[power_factor_value] = (
        stable_roots_power[equilibrium_root_index]
    )

z_eq_laser_noise_time = None

gamma_0_baoab = b / m
brownian_impinging_temperature = T
brownian_surface_temperature = T
brownian_emerging_temperature = T
brownian_centre_of_mass_temperature = T
gamma_baoab = gamma_0_baoab

if use_input_surface_temperature_for_brownian:
    if input_surface_temperature <= 0.0:
        raise ValueError("input_surface_temperature must be positive.")
    if brownian_surface_temperature_accommodation_alpha <= 0.0:
        raise ValueError(
            "brownian_surface_temperature_accommodation_alpha must be positive."
        )

    brownian_surface_temperature = input_surface_temperature
    brownian_emerging_temperature = (
        brownian_impinging_temperature
        + brownian_surface_temperature_accommodation_alpha
        * (brownian_surface_temperature - brownian_impinging_temperature)
    )

    if brownian_emerging_temperature <= 0.0:
        raise ValueError(
            "The chosen surface temperature/accommodation gives a non-positive "
            "emerging gas temperature."
        )

    emerging_temperature_ratio = (
        brownian_emerging_temperature / brownian_impinging_temperature
    )
    paper_temperature_factor = np.pi / 8.0
    gamma_baoab = gamma_0_baoab * (
        1.0 + paper_temperature_factor * np.sqrt(emerging_temperature_ratio)
    ) / (1.0 + paper_temperature_factor)
    brownian_centre_of_mass_temperature = (
        brownian_impinging_temperature**1.5
        + paper_temperature_factor * brownian_emerging_temperature**1.5
    ) / (
        np.sqrt(brownian_impinging_temperature)
        + paper_temperature_factor * np.sqrt(brownian_emerging_temperature)
    )

baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
baoab_thermal_velocity_scale = np.sqrt(
    (kB * brownian_centre_of_mass_temperature / m)
    * (1 - baoab_damping_factor**2)
)

print("Brownian bath uses input surface temperature =", use_input_surface_temperature_for_brownian)
print("Brownian T_imp =", brownian_impinging_temperature, "K")
print("Brownian T_surface input =", brownian_surface_temperature, "K")
print("Brownian T_em =", brownian_emerging_temperature, "K")
print("Brownian T_CM =", brownian_centre_of_mass_temperature, "K")
print("Cold Gamma_0 =", gamma_0_baoab, "s^-1")
print("Trajectory Gamma_CM =", gamma_baoab, "s^-1")

# print("BAOAB timestep =", dt_baoab, "s")
# print("BAOAB damping factor =", baoab_damping_factor)
# print("BAOAB thermal velocity kick scale =", baoab_thermal_velocity_scale, "m/s")
# print("Laser noise fraction =", laser_noise_fraction)
# print("Requested laser noise step duration =", laser_noise_step_duration, "s")
# print("Actual laser noise step duration =", laser_step_duration_actual, "s")

# print("Instantaneous stable z equilibria from laser noise:")
for power_factor_value, z_eq_power in instantaneous_z_equilibrium_by_power_factor.items():
    pass
    # print(
        # "  power factor =",
        # power_factor_value,
        # ", z_eq =",
        # z_eq_power * 1e6,
        # "micrometres"
    # )


def deterministic_force_no_drag_3d(x, y, z, power_factor=1.0):
    Fx, Fy, Fz = F_optical_3d(x, y, z, power_factor)
    return Fx, Fy, Fz - m*g


def solve_baoab_3d_with_power(
    power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z
):
    x_out = np.zeros_like(t_baoab)
    y_out = np.zeros_like(t_baoab)
    z_out = np.zeros_like(t_baoab)
    vx_out = np.zeros_like(t_baoab)
    vy_out = np.zeros_like(t_baoab)
    vz_out = np.zeros_like(t_baoab)

    x_out[0] = x0
    y_out[0] = y0
    z_out[0] = z0
    vx_out[0] = vx0
    vy_out[0] = vy0
    vz_out[0] = vz0

    for i in range(len(t_baoab) - 1):
        x_i = x_out[i]
        y_i = y_out[i]
        z_i = z_out[i]
        vx_i = vx_out[i]
        vy_i = vy_out[i]
        vz_i = vz_out[i]
        power_factor_i = power_factor_time[i]

        Fx_i, Fy_i, Fz_i = deterministic_force_no_drag_3d(
            x_i,
            y_i,
            z_i,
            power_factor_i
        )
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        vx_i = (
            baoab_damping_factor * vx_i
            + baoab_thermal_velocity_scale * brownian_normals_x[i]
        )
        vy_i = (
            baoab_damping_factor * vy_i
            + baoab_thermal_velocity_scale * brownian_normals_y[i]
        )
        vz_i = (
            baoab_damping_factor * vz_i
            + baoab_thermal_velocity_scale * brownian_normals_z[i]
        )

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        Fx_i, Fy_i, Fz_i = deterministic_force_no_drag_3d(
            x_i,
            y_i,
            z_i,
            power_factor_i
        )
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_out[i + 1] = x_i
        y_out[i + 1] = y_i
        z_out[i + 1] = z_i
        vx_out[i + 1] = vx_i
        vy_out[i + 1] = vy_i
        vz_out[i + 1] = vz_i

    return x_out, y_out, z_out, vx_out, vy_out, vz_out


def solve_baoab_3d_fast_with_power(
    power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z,
    output_label="baoab",
):
    """
    Use the compiled Cython BAOAB loop when available.

    If the Cython extension has not been built yet, this falls back to the
    original Python implementation so the script remains runnable.
    """
    if use_disk_backed_large_arrays:
        if (
            cython_baoab_positions_into_available
            and use_force_lookup_table
            and force_lookup_ready
        ):
            x_out = make_large_array(f"{output_label}_x", len(t_baoab))
            y_out = make_large_array(f"{output_label}_y", len(t_baoab))
            z_out = make_large_array(f"{output_label}_z", len(t_baoab))

            out_of_bounds_count = solve_baoab_3d_lookup_cython_positions_into(
                np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
                np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
                np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
                np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
                power_factor_time,
                brownian_normals_x,
                brownian_normals_y,
                brownian_normals_z,
                dt_baoab,
                m,
                m * g,
                baoab_damping_factor,
                baoab_thermal_velocity_scale,
                x0,
                y0,
                z0,
                vx0,
                vy0,
                vz0,
                x_out,
                y_out,
                z_out,
                radial_zero_tolerance,
            )

            flush_large_array(x_out)
            flush_large_array(y_out)
            flush_large_array(z_out)

            if out_of_bounds_count > 0:
                pass

            return x_out, y_out, z_out, None, None, None

        raise RuntimeError(
            "use_disk_backed_large_arrays=True needs the rebuilt Cython "
            "positions-into solver. Run: python3 setup_baoab_3d.py build_ext --inplace"
        )

    if cython_baoab_available and use_force_lookup_table and force_lookup_ready:
        result = solve_baoab_3d_lookup_cython(
            np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
            np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
            np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
            np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
            np.ascontiguousarray(power_factor_time, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_x, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_y, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_z, dtype=np.float64),
            dt_baoab,
            m,
            m * g,
            baoab_damping_factor,
            baoab_thermal_velocity_scale,
            x0,
            y0,
            z0,
            vx0,
            vy0,
            vz0,
            radial_zero_tolerance,
            False,
            np.ascontiguousarray(force_lookup_z_values[[0, -1]], dtype=np.float64),
            np.ascontiguousarray(np.zeros(2), dtype=np.float64),
            0.0,
            0.0,
        )

        (
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out,
            out_of_bounds_count,
        ) = result[:7]

        if out_of_bounds_count > 0:
            pass
            # print(
                # "Warning: Cython BAOAB clamped",
                # out_of_bounds_count,
                # "force lookups to the edge of the lookup table.",
            # )
            # print(
                # "Consider increasing force_lookup_r_max or "
                # "force_lookup_z_half_width."
            # )

        return x_out, y_out, z_out, vx_out, vy_out, vz_out

    # print("Cython BAOAB unavailable; using slower Python BAOAB loop.")
    return solve_baoab_3d_with_power(
        power_factor_time,
        brownian_normals_x,
        brownian_normals_y,
        brownian_normals_z
    )


def measure_z_position_for_feedback(z_actual, feedback_rng):
    """
    Synthetic z-position measurement used by the feedback loop.

    The controller sees the delayed true z position plus optional calibrated
    readout noise. The delay is applied in the controller loop.
    """
    z_measured = z_actual

    if use_feedback_position_noise and feedback_position_noise_rms > 0:
        z_measured += feedback_rng.normal(0.0, feedback_position_noise_rms)

    return z_measured


def solve_baoab_3d_with_pd_feedback(
    disturbance_power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z
):
    """
    Run BAOAB with a delayed PD feedback loop controlling laser power.

    The feedback loop reads z, estimates z velocity, asks for a force
    correction, converts that correction to a laser-power change, and then
    applies the requested power after clipping to the allowed power range.
    """
    if not (use_force_lookup_table and force_lookup_ready):
        raise ValueError(
            "PD feedback requires use_force_lookup_table=True so the closed-loop "
            "solver can use the fast scalar force lookup."
        )

    feedback_rng = np.random.default_rng(seed=feedback_noise_seed)
    baoab_sampling_frequency = 1 / dt_baoab
    control_decimation = int(round(baoab_sampling_frequency / feedback_update_frequency))

    if control_decimation < 1:
        raise ValueError("feedback_update_frequency cannot exceed the BAOAB sampling frequency.")

    if not np.isclose(baoab_sampling_frequency / feedback_update_frequency, control_decimation):
        raise ValueError("Choose feedback_update_frequency so it divides the BAOAB sampling frequency.")

    if not 0 <= feedback_velocity_filter_alpha <= 1:
        raise ValueError("feedback_velocity_filter_alpha must be between 0 and 1.")

    dt_control = control_decimation * dt_baoab

    if feedback_total_loop_delay <= 0:
        loop_delay_control_steps = 0
    else:
        loop_delay_control_steps = int(np.ceil(feedback_total_loop_delay / dt_control))

    loop_delay_baoab_steps = loop_delay_control_steps * control_decimation
    actual_loop_delay = loop_delay_control_steps * dt_control

    if P_laser <= 0:
        raise ValueError("P_laser must be positive for laser-power feedback.")

    _, _, Fz_per_nominal_power, _ = F_optical_3d_lookup_scalar_clamped(
        x_eq,
        y_eq,
        z_eq,
        power_factor=1.0
    )
    force_per_watt = Fz_per_nominal_power / P_laser

    if abs(force_per_watt) < radial_zero_tolerance:
        raise ValueError("Force per watt is too close to zero for laser-power feedback.")

    feedback_kp = feedback_kp_multiplier * kz
    feedback_kd = feedback_kd_multiplier * b
    feedback_power_min = feedback_power_min_factor * P_laser
    feedback_power_max = feedback_power_max_factor * P_laser

    x_out = np.zeros_like(t_baoab)
    y_out = np.zeros_like(t_baoab)
    z_out = np.zeros_like(t_baoab)
    vx_out = np.zeros_like(t_baoab)
    vy_out = np.zeros_like(t_baoab)
    vz_out = np.zeros_like(t_baoab)
    command_power_time = np.zeros_like(t_baoab)
    actual_power_time = np.zeros_like(t_baoab)

    control_times = []
    measurement_times = []
    measured_positions = []
    filtered_positions = []
    requested_powers = []

    x_out[0] = x0
    y_out[0] = y0
    z_out[0] = z0
    vx_out[0] = vx0
    vy_out[0] = vy0
    vz_out[0] = vz0

    command_power = P_laser
    filtered_z_previous = None
    out_of_bounds_count = 0

    for i in range(len(t_baoab) - 1):
        if i % control_decimation == 0:
            delayed_i = max(0, i - loop_delay_baoab_steps)
            z_measured = measure_z_position_for_feedback(z_out[delayed_i], feedback_rng)

            if filtered_z_previous is None:
                z_filtered = z_measured
                measured_vz = 0.0
            else:
                z_filtered = (
                    feedback_velocity_filter_alpha * z_measured
                    + (1 - feedback_velocity_filter_alpha) * filtered_z_previous
                )
                measured_vz = (z_filtered - filtered_z_previous) / dt_control

            z_error = z_filtered - z_eq
            feedback_force = -feedback_kp * z_error - feedback_kd * measured_vz
            power_change = feedback_force / force_per_watt
            command_power = np.clip(
                P_laser + power_change,
                feedback_power_min,
                feedback_power_max
            )

            control_times.append(t_baoab[i])
            measurement_times.append(t_baoab[delayed_i])
            measured_positions.append(z_measured)
            filtered_positions.append(z_filtered)
            requested_powers.append(command_power)

            filtered_z_previous = z_filtered

        command_factor = command_power / P_laser
        actual_power_factor = command_factor * disturbance_power_factor_time[i]
        actual_power_factor = np.clip(
            actual_power_factor,
            feedback_power_min_factor,
            feedback_power_max_factor
        )

        x_i = x_out[i]
        y_i = y_out[i]
        z_i = z_out[i]
        vx_i = vx_out[i]
        vy_i = vy_out[i]
        vz_i = vz_out[i]

        Fx_i, Fy_i, Fz_i, force_was_clamped = F_optical_3d_lookup_scalar_clamped(
            x_i,
            y_i,
            z_i,
            actual_power_factor
        )
        out_of_bounds_count += int(force_was_clamped)
        Fz_i -= m * g
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        vx_i = (
            baoab_damping_factor * vx_i
            + baoab_thermal_velocity_scale * brownian_normals_x[i]
        )
        vy_i = (
            baoab_damping_factor * vy_i
            + baoab_thermal_velocity_scale * brownian_normals_y[i]
        )
        vz_i = (
            baoab_damping_factor * vz_i
            + baoab_thermal_velocity_scale * brownian_normals_z[i]
        )

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        Fx_i, Fy_i, Fz_i, force_was_clamped = F_optical_3d_lookup_scalar_clamped(
            x_i,
            y_i,
            z_i,
            actual_power_factor
        )
        out_of_bounds_count += int(force_was_clamped)
        Fz_i -= m * g
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_out[i + 1] = x_i
        y_out[i + 1] = y_i
        z_out[i + 1] = z_i
        vx_out[i + 1] = vx_i
        vy_out[i + 1] = vy_i
        vz_out[i + 1] = vz_i
        command_power_time[i] = command_power
        actual_power_time[i] = P_laser * actual_power_factor

    command_power_time[-1] = command_power
    actual_power_time[-1] = actual_power_time[-2]

    return {
        "t": t_baoab,
        "x": x_out,
        "y": y_out,
        "z": z_out,
        "vx": vx_out,
        "vy": vy_out,
        "vz": vz_out,
        "command_power": command_power_time,
        "actual_power": actual_power_time,
        "control_t": np.array(control_times),
        "measurement_t": np.array(measurement_times),
        "z_measured": np.array(measured_positions),
        "z_filtered": np.array(filtered_positions),
        "requested_power": np.array(requested_powers),
        "dt_control": dt_control,
        "loop_delay_control_steps": loop_delay_control_steps,
        "actual_loop_delay": actual_loop_delay,
        "force_per_watt": force_per_watt,
        "out_of_bounds_count": out_of_bounds_count,
        "feedback_kp": feedback_kp,
        "feedback_kd": feedback_kd,
        "feedback_power_min": feedback_power_min,
        "feedback_power_max": feedback_power_max,
    }

def solve_baoab_3d_fast_with_pd_feedback(
    disturbance_power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z
):
    """
    Use the compiled Cython feedback loop when available.

    The pure-Python implementation remains as a fallback so the script still
    runs before the extension has been built.
    """
    if cython_feedback_baoab_available and use_force_lookup_table and force_lookup_ready:
        feedback_rng = np.random.default_rng(seed=feedback_noise_seed)
        baoab_sampling_frequency = 1 / dt_baoab
        control_decimation = int(round(baoab_sampling_frequency / feedback_update_frequency))

        if control_decimation < 1:
            raise ValueError("feedback_update_frequency cannot exceed the BAOAB sampling frequency.")

        if not np.isclose(baoab_sampling_frequency / feedback_update_frequency, control_decimation):
            raise ValueError("Choose feedback_update_frequency so it divides the BAOAB sampling frequency.")

        if not 0 <= feedback_velocity_filter_alpha <= 1:
            raise ValueError("feedback_velocity_filter_alpha must be between 0 and 1.")

        dt_control = control_decimation * dt_baoab

        if feedback_total_loop_delay <= 0:
            loop_delay_control_steps = 0
        else:
            loop_delay_control_steps = int(np.ceil(feedback_total_loop_delay / dt_control))

        loop_delay_baoab_steps = loop_delay_control_steps * control_decimation
        actual_loop_delay = loop_delay_control_steps * dt_control

        if P_laser <= 0:
            raise ValueError("P_laser must be positive for laser-power feedback.")

        _, _, Fz_per_nominal_power, _ = F_optical_3d_lookup_scalar_clamped(
            x_eq,
            y_eq,
            z_eq,
            power_factor=1.0
        )
        force_per_watt = Fz_per_nominal_power / P_laser

        if abs(force_per_watt) < radial_zero_tolerance:
            raise ValueError("Force per watt is too close to zero for laser-power feedback.")

        feedback_kp = feedback_kp_multiplier * kz
        feedback_kd = feedback_kd_multiplier * b
        feedback_power_min = feedback_power_min_factor * P_laser
        feedback_power_max = feedback_power_max_factor * P_laser

        if len(t_baoab) >= 2:
            n_control_updates = ((len(t_baoab) - 2) // control_decimation) + 1
        else:
            n_control_updates = 0

        if use_feedback_position_noise and feedback_position_noise_rms > 0:
            feedback_position_noise = feedback_rng.normal(
                0.0,
                feedback_position_noise_rms,
                size=n_control_updates
            )
        else:
            feedback_position_noise = np.zeros(n_control_updates)

        result = solve_baoab_3d_feedback_lookup_cython(
            np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
            np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
            np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
            np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
            np.ascontiguousarray(disturbance_power_factor_time, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_x, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_y, dtype=np.float64),
            np.ascontiguousarray(brownian_normals_z, dtype=np.float64),
            np.ascontiguousarray(feedback_position_noise, dtype=np.float64),
            dt_baoab,
            m,
            m * g,
            baoab_damping_factor,
            baoab_thermal_velocity_scale,
            x0,
            y0,
            z0,
            vx0,
            vy0,
            vz0,
            z_eq,
            P_laser,
            feedback_kp,
            feedback_kd,
            force_per_watt,
            feedback_power_min_factor,
            feedback_power_max_factor,
            feedback_velocity_filter_alpha,
            control_decimation,
            loop_delay_baoab_steps,
            dt_control,
            radial_zero_tolerance,
        )

        (
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out,
            command_power_time,
            actual_power_time,
            control_times,
            measurement_times,
            measured_positions,
            filtered_positions,
            requested_powers,
            out_of_bounds_count,
        ) = result

        return {
            "t": t_baoab,
            "x": x_out,
            "y": y_out,
            "z": z_out,
            "vx": vx_out,
            "vy": vy_out,
            "vz": vz_out,
            "command_power": command_power_time,
            "actual_power": actual_power_time,
            "control_t": control_times,
            "measurement_t": measurement_times,
            "z_measured": measured_positions,
            "z_filtered": filtered_positions,
            "requested_power": requested_powers,
            "dt_control": dt_control,
            "loop_delay_control_steps": loop_delay_control_steps,
            "actual_loop_delay": actual_loop_delay,
            "force_per_watt": force_per_watt,
            "out_of_bounds_count": out_of_bounds_count,
            "feedback_kp": feedback_kp,
            "feedback_kd": feedback_kd,
            "feedback_power_min": feedback_power_min,
            "feedback_power_max": feedback_power_max,
        }

    return solve_baoab_3d_with_pd_feedback(
        disturbance_power_factor_time,
        brownian_normals_x,
        brownian_normals_y,
        brownian_normals_z
    )


brownian_normals_x = make_random_normal_array(
    "brownian_normals_x",
    rng,
    len(t_baoab) - 1,
)
brownian_normals_y = make_random_normal_array(
    "brownian_normals_y",
    rng,
    len(t_baoab) - 1,
)
brownian_normals_z = make_random_normal_array(
    "brownian_normals_z",
    rng,
    len(t_baoab) - 1,
)
if pressure_temperature_error_variance_only_child:
    zero_brownian_normals = None
else:
    zero_brownian_normals = make_large_array(
        "zero_brownian_normals",
        len(t_baoab) - 1,
        0.0,
    )

if use_laser_power_noise:
    constant_power_factor = make_large_array(
        "constant_power_factor",
        len(t_baoab),
        1.0,
    )
else:
    constant_power_factor = laser_power_factor

baoab_constant_power_start_time = perf_counter()
(
    x_baoab_constant_power,
    y_baoab_constant_power,
    z_baoab_constant_power,
    vx_baoab_constant_power,
    vy_baoab_constant_power,
    vz_baoab_constant_power,
) = solve_baoab_3d_fast_with_power(
    constant_power_factor,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z,
    output_label="constant_power",
)
baoab_constant_power_runtime = perf_counter() - baoab_constant_power_start_time

if use_laser_power_noise:
    baoab_laser_noise_start_time = perf_counter()
    (
        x_baoab,
        y_baoab,
        z_baoab,
        vx_baoab,
        vy_baoab,
        vz_baoab,
    ) = solve_baoab_3d_fast_with_power(
        laser_power_factor,
        brownian_normals_x,
        brownian_normals_y,
        brownian_normals_z,
        output_label="laser_noise",
    )
    baoab_laser_noise_runtime = perf_counter() - baoab_laser_noise_start_time
else:
    x_baoab = x_baoab_constant_power
    y_baoab = y_baoab_constant_power
    z_baoab = z_baoab_constant_power
    vx_baoab = vx_baoab_constant_power
    vy_baoab = vy_baoab_constant_power
    vz_baoab = vz_baoab_constant_power
    baoab_laser_noise_runtime = 0.0

if pressure_temperature_error_variance_only_child:
    x_baoab_no_brownian = None
    y_baoab_no_brownian = None
    z_baoab_no_brownian = None
    vx_baoab_no_brownian = None
    vy_baoab_no_brownian = None
    vz_baoab_no_brownian = None
    baoab_laser_noise_without_brownian_runtime = 0.0
else:
    baoab_laser_noise_without_brownian_start_time = perf_counter()
    (
        x_baoab_no_brownian,
        y_baoab_no_brownian,
        z_baoab_no_brownian,
        vx_baoab_no_brownian,
        vy_baoab_no_brownian,
        vz_baoab_no_brownian,
    ) = solve_baoab_3d_fast_with_power(
        laser_power_factor,
        zero_brownian_normals,
        zero_brownian_normals,
        zero_brownian_normals,
        output_label="no_brownian",
    )
    baoab_laser_noise_without_brownian_runtime = (
        perf_counter() - baoab_laser_noise_without_brownian_start_time
    )

feedback_result = None
baoab_feedback_runtime = 0.0

if use_pd_feedback:
    if use_disk_backed_large_arrays:
        raise RuntimeError(
            "Disk-backed array mode does not support the PD-feedback solver yet. "
            "Set use_pd_feedback=False or use_disk_backed_large_arrays=False."
        )

    baoab_feedback_start_time = perf_counter()
    feedback_result = solve_baoab_3d_fast_with_pd_feedback(
        laser_power_factor,
        brownian_normals_x,
        brownian_normals_y,
        brownian_normals_z,
    )
    baoab_feedback_runtime = perf_counter() - baoab_feedback_start_time

    x_baoab_feedback = feedback_result["x"]
    y_baoab_feedback = feedback_result["y"]
    z_baoab_feedback = feedback_result["z"]
    vx_baoab_feedback = feedback_result["vx"]
    vy_baoab_feedback = feedback_result["vy"]
    vz_baoab_feedback = feedback_result["vz"]
else:
    x_baoab_feedback = None
    y_baoab_feedback = None
    z_baoab_feedback = None
    vx_baoab_feedback = None
    vy_baoab_feedback = None
    vz_baoab_feedback = None

baoab_total_runtime = (
    baoab_constant_power_runtime
    + baoab_laser_noise_runtime
    + baoab_laser_noise_without_brownian_runtime
    + baoab_feedback_runtime
)

if pressure_temperature_error_variance_only_child:
    x_laser_noise_difference = None
    y_laser_noise_difference = None
    z_laser_noise_difference = None
    x_brownian_difference = None
    y_brownian_difference = None
    z_brownian_difference = None
    x_baoab = None
    y_baoab = None
    vx_baoab = None
    vy_baoab = None
    vz_baoab = None
    x_baoab_constant_power = None
    y_baoab_constant_power = None
    vx_baoab_constant_power = None
    vy_baoab_constant_power = None
    vz_baoab_constant_power = None
    brownian_normals_x = None
    brownian_normals_y = None
    brownian_normals_z = None
    zero_brownian_normals = None
    laser_power_factor = None
    constant_power_factor = None
elif use_laser_power_noise:
    x_laser_noise_difference = make_difference_array(
        "x_laser_noise_difference",
        x_baoab,
        x_baoab_constant_power,
    )
    y_laser_noise_difference = make_difference_array(
        "y_laser_noise_difference",
        y_baoab,
        y_baoab_constant_power,
    )
    z_laser_noise_difference = make_difference_array(
        "z_laser_noise_difference",
        z_baoab,
        z_baoab_constant_power,
    )
else:
    x_laser_noise_difference = None
    y_laser_noise_difference = None
    z_laser_noise_difference = None

if not pressure_temperature_error_variance_only_child:
    x_brownian_difference = make_difference_array(
        "x_brownian_difference",
        x_baoab,
        x_baoab_no_brownian,
    )
    y_brownian_difference = make_difference_array(
        "y_brownian_difference",
        y_baoab,
        y_baoab_no_brownian,
    )
    z_brownian_difference = make_difference_array(
        "z_brownian_difference",
        z_baoab,
        z_baoab_no_brownian,
    )

# print("BAOAB constant-power runtime =", baoab_constant_power_runtime, "s")
# print("BAOAB laser-noise runtime =", baoab_laser_noise_runtime, "s")
# print("BAOAB laser-noise without-Brownian runtime =", baoab_laser_noise_without_brownian_runtime, "s")
# print("BAOAB feedback runtime =", baoab_feedback_runtime, "s")
# print("BAOAB total runtime =", baoab_total_runtime, "s")
print("Total calculation runtime, excluding graph-viewing time =", perf_counter() - script_start_time, "s")

if pressure_temperature_error_variance_only_child:
    def pressure_sweep_chunked_variance(position, origin, resolution_m=0.0):
        count = 0
        total = 0.0
        total_square = 0.0

        for start in range(0, len(position), disk_array_chunk_size):
            end = min(start + disk_array_chunk_size, len(position))
            displacement_chunk = position[start:end] - origin
            if resolution_m > 0.0:
                displacement_chunk = (
                    np.round(displacement_chunk / resolution_m) * resolution_m
                )
            total += np.sum(displacement_chunk)
            total_square += np.sum(displacement_chunk * displacement_chunk)
            count += end - start

        if count == 0:
            return np.nan

        mean = total / count
        return total_square / count - mean * mean

    def pressure_sweep_centre_of_mass_temperature_from_emerging(
        temperature_emerging,
        temperature_impinging,
    ):
        paper_temperature_factor = np.pi / 8.0
        return (
            temperature_impinging**1.5
            + paper_temperature_factor * temperature_emerging**1.5
        ) / (
            np.sqrt(temperature_impinging)
            + paper_temperature_factor * np.sqrt(temperature_emerging)
        )

    def pressure_sweep_emerging_temperature_from_centre_of_mass(
        temperature_cm,
        temperature_impinging,
    ):
        if temperature_cm <= 0.0 or temperature_impinging <= 0.0:
            return np.nan

        def residual(temperature_emerging):
            return (
                pressure_sweep_centre_of_mass_temperature_from_emerging(
                    temperature_emerging,
                    temperature_impinging,
                )
                - temperature_cm
            )

        lower = max(1e-12 * temperature_impinging, 1e-12)
        upper = max(2.0 * temperature_impinging, 2.0 * temperature_cm, lower * 10.0)

        if temperature_cm >= temperature_impinging:
            lower = temperature_impinging
            lower_residual = residual(lower)
            upper_residual = residual(upper)
            while lower_residual * upper_residual > 0.0 and upper < 1e9:
                upper *= 2.0
                upper_residual = residual(upper)

            if lower_residual * upper_residual <= 0.0:
                return brentq(residual, lower, upper)

        temperature_grid = np.geomspace(lower, upper, 500)
        residual_grid = np.array([residual(value) for value in temperature_grid])
        sign_change_indices = np.where(
            residual_grid[:-1] * residual_grid[1:] <= 0.0
        )[0]

        if len(sign_change_indices) == 0:
            return np.nan

        index = sign_change_indices[-1]
        return brentq(residual, temperature_grid[index], temperature_grid[index + 1])

    def pressure_sweep_print_variance_marker(estimator, resolution_m=0.0):
        variance_m2 = pressure_sweep_chunked_variance(
            z_baoab,
            z_eq,
            resolution_m,
        )
        if subtract_detector_quantisation_variance and resolution_m > 0.0:
            variance_m2 -= resolution_m**2 / 12.0
        variance_m2 = max(variance_m2, 0.0)

        temperature_cm = m * omega_z**2 * variance_m2 / kB
        temperature_impinging = (
            T
            if surface_temperature_impinging_gas_temperature is None
            else surface_temperature_impinging_gas_temperature
        )
        temperature_emerging = (
            pressure_sweep_emerging_temperature_from_centre_of_mass(
                temperature_cm,
                temperature_impinging,
            )
        )
        recovered_surface_temperature = temperature_impinging + (
            temperature_emerging - temperature_impinging
        ) / surface_temperature_accommodation_alpha

        if (
            np.isfinite(recovered_surface_temperature)
            and brownian_surface_temperature != 0.0
        ):
            error_percent = 100.0 * (
                recovered_surface_temperature - brownian_surface_temperature
            ) / brownian_surface_temperature
        else:
            error_percent = np.nan

        print(
            "PRESSURE_TEMPERATURE_ERROR_RESULT",
            f"pressure_pa={p}",
            f"estimator={estimator}",
            f"recovered_T_surface_K={recovered_surface_temperature}",
            f"error_percent={error_percent}",
        )

    requested_pressure_estimators = set(pressure_temperature_error_sweep_estimators)
    if "variance_true" in requested_pressure_estimators:
        pressure_sweep_print_variance_marker("variance_true", 0.0)
    if (
        "variance_measured" in requested_pressure_estimators
        and include_z_detector_resolution_psd
    ):
        pressure_sweep_print_variance_marker(
            "variance_measured",
            detector_position_resolution_nm * 1e-9,
        )
    raise SystemExit(0)

# print(
    # "BAOAB constant-power RMS x from equilibrium =",
    # np.std(x_baoab_constant_power - x_eq) * 1e6,
    # "micrometres"
# )
# print(
    # "BAOAB constant-power RMS y from equilibrium =",
    # np.std(y_baoab_constant_power - y_eq) * 1e6,
    # "micrometres"
# )
# print(
    # "BAOAB constant-power RMS z from equilibrium =",
    # np.std(z_baoab_constant_power - z_eq) * 1e6,
    # "micrometres"
# )
# print(
    # "RMS isolated laser-noise effect in x =",
    # np.std(x_laser_noise_difference) * 1e9,
    # "nm"
# )
# print(
    # "RMS isolated laser-noise effect in y =",
    # np.std(y_laser_noise_difference) * 1e9,
    # "nm"
# )
# print(
    # "RMS isolated laser-noise effect in z =",
    # np.std(z_laser_noise_difference) * 1e9,
    # "nm"
# )
# print(
    # "Max isolated laser-noise effect in x =",
    # np.max(np.abs(x_laser_noise_difference)) * 1e9,
    # "nm"
# )
# print(
    # "Max isolated laser-noise effect in y =",
    # np.max(np.abs(y_laser_noise_difference)) * 1e9,
    # "nm"
# )
# print(
    # "Max isolated laser-noise effect in z =",
    # np.max(np.abs(z_laser_noise_difference)) * 1e9,
    # "nm"
# )

plot_stride = max(1, len(t_baoab) // max_plot_points)
plot_slice = slice(None, None, plot_stride)
t_plot = t_baoab[plot_slice]

# print("Plot stride =", plot_stride)
# print("Number of plotted time samples =", len(t_plot))



















# ****************************************************************************************************************************************************
# Plot x(t), y(t), and z(t)
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

axes[0].plot(
    t_plot,
    x_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[0].set_ylabel("x / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    t_plot,
    y_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
axes[1].axhline(y_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[1].set_ylabel("y / micrometres")
axes[1].legend()
axes[1].grid()

axes[2].plot(
    t_plot,
    z_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
axes[2].axhline(z_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[2].set_xlabel("Time / s")
axes[2].set_ylabel("z / micrometres")
axes[2].legend()
axes[2].grid()

fig.suptitle("3D BAOAB ray-optics motion")
plt.tight_layout()
finish_plot()

# ****************************************************************************************************************************************************
# Laser-power noise diagnostic
# ****************************************************************************************************************************************************
plt.figure(figsize=single_diagnostic_figsize)
plt.step(t_plot, P_laser * laser_power_factor[plot_slice], where="post", label="laser power")
plt.axhline(P_laser, linestyle="--", color="black", label="nominal power")
plt.axhline(P_laser * (1 + laser_noise_fraction), linestyle=":", color="tab:red")
plt.axhline(P_laser * (1 - laser_noise_fraction), linestyle=":", color="tab:red")
plt.xlabel("Time / s")
plt.ylabel("Laser power / W")
plt.title("Stepwise laser-power noise")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot()


# ****************************************************************************************************************************************************
# Optional PD feedback laser-power diagnostic
# ****************************************************************************************************************************************************
if use_pd_feedback:
    plt.figure(figsize=single_diagnostic_figsize)
    plt.plot(
        t_plot,
        feedback_result["command_power"][plot_slice] * 1e3,
        label="PD command power"
    )
    plt.plot(
        t_plot,
        feedback_result["actual_power"][plot_slice] * 1e3,
        alpha=0.65,
        label="actual power after laser noise"
    )
    plt.axhline(P_laser * 1e3, linestyle="--", color="black", label="nominal power")
    plt.axhline(
        feedback_result["feedback_power_min"] * 1e3,
        linestyle=":",
        color="tab:red",
        label="feedback limits"
    )
    plt.axhline(feedback_result["feedback_power_max"] * 1e3, linestyle=":", color="tab:red")
    plt.xlabel("Time / s")
    plt.ylabel("Laser power / mW")
    plt.title("PD feedback laser power")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot()

    feedback_control_stride = max(
        1,
        len(feedback_result["control_t"]) // max_plot_points
    )
    feedback_control_slice = slice(None, None, feedback_control_stride)

    plt.figure(figsize=force_check_figsize)
    plt.plot(
        feedback_result["control_t"][feedback_control_slice],
        (feedback_result["z_measured"][feedback_control_slice] - z_eq) * 1e9,
        alpha=0.45,
        label="delayed measured z"
    )
    plt.plot(
        feedback_result["control_t"][feedback_control_slice],
        (feedback_result["z_filtered"][feedback_control_slice] - z_eq) * 1e9,
        label="filtered z used by controller"
    )
    plt.axhline(0, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("Time / s")
    plt.ylabel("Measured z displacement / nm")
    plt.title("PD feedback measurement signal")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot()

# ****************************************************************************************************************************************************
# Compare Brownian trajectories with and without laser-power noise
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

axes[0].plot(
    t_plot,
    x_baoab_constant_power[plot_slice] * 1e6,
    label="BAOAB Brownian, constant laser power"
)
axes[0].plot(
    t_plot,
    x_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    label="BAOAB Brownian + laser noise"
)
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="equilibrium")
axes[0].set_ylabel("x / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    t_plot,
    y_baoab_constant_power[plot_slice] * 1e6,
    label="BAOAB Brownian, constant laser power"
)
axes[1].plot(
    t_plot,
    y_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    label="BAOAB Brownian + laser noise"
)
axes[1].axhline(y_eq * 1e6, linestyle=":", color="black", label="equilibrium")
axes[1].set_ylabel("y / micrometres")
axes[1].legend()
axes[1].grid()

axes[2].plot(
    t_plot,
    z_baoab_constant_power[plot_slice] * 1e6,
    label="BAOAB Brownian, constant laser power"
)
axes[2].plot(
    t_plot,
    z_baoab[plot_slice] * 1e6,
    linewidth=0.9,
    label="BAOAB Brownian + laser noise"
)
axes[2].axhline(z_eq * 1e6, linestyle=":", color="black", label="equilibrium")
axes[2].set_xlabel("Time / s")
axes[2].set_ylabel("z / micrometres")
axes[2].legend()
axes[2].grid()

fig.suptitle("BAOAB motion with and without laser-power noise")
plt.tight_layout()
finish_plot()


# ****************************************************************************************************************************************************
# Optional PD feedback trajectory comparison
# ****************************************************************************************************************************************************
if use_pd_feedback:
    fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

    axes[0].plot(
        t_plot,
        x_baoab[plot_slice] * 1e6,
        label="without feedback"
    )
    axes[0].plot(
        t_plot,
        x_baoab_feedback[plot_slice] * 1e6,
        linewidth=0.9,
        label="with PD feedback"
    )
    axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="equilibrium")
    axes[0].set_ylabel("x / micrometres")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(
        t_plot,
        y_baoab[plot_slice] * 1e6,
        label="without feedback"
    )
    axes[1].plot(
        t_plot,
        y_baoab_feedback[plot_slice] * 1e6,
        linewidth=0.9,
        label="with PD feedback"
    )
    axes[1].axhline(y_eq * 1e6, linestyle=":", color="black", label="equilibrium")
    axes[1].set_ylabel("y / micrometres")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(
        t_plot,
        z_baoab[plot_slice] * 1e6,
        label="without feedback"
    )
    axes[2].plot(
        t_plot,
        z_baoab_feedback[plot_slice] * 1e6,
        linewidth=0.9,
        label="with PD feedback"
    )
    axes[2].axhline(z_eq * 1e6, linestyle=":", color="black", label="equilibrium")
    axes[2].set_xlabel("Time / s")
    axes[2].set_ylabel("z / micrometres")
    axes[2].legend()
    axes[2].grid()

    fig.suptitle("BAOAB motion with and without PD feedback")
    plt.tight_layout()
    finish_plot()

if use_laser_power_noise:
    # ****************************************************************************************************************************************************
    # Isolated laser-noise effect
    # ****************************************************************************************************************************************************
    fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

    axes[0].plot(t_plot, x_laser_noise_difference[plot_slice] * 1e9)
    axes[0].axhline(0, linestyle=":", color="black")
    axes[0].set_ylabel("x difference / nm")
    axes[0].grid()

    axes[1].plot(t_plot, y_laser_noise_difference[plot_slice] * 1e9)
    axes[1].axhline(0, linestyle=":", color="black")
    axes[1].set_ylabel("y difference / nm")
    axes[1].grid()

    axes[2].plot(t_plot, z_laser_noise_difference[plot_slice] * 1e9)
    axes[2].axhline(0, linestyle=":", color="black")
    axes[2].set_xlabel("Time / s")
    axes[2].set_ylabel("z difference / nm")
    axes[2].grid()

    fig.suptitle("Isolated displacement effect from laser-power noise")
    plt.tight_layout()
    finish_plot()

# ****************************************************************************************************************************************************
# Isolated Brownian-motion effect
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

axes[0].plot(t_plot, x_brownian_difference[plot_slice] * 1e9)
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x difference / nm")
axes[0].grid()

axes[1].plot(t_plot, y_brownian_difference[plot_slice] * 1e9)
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_ylabel("y difference / nm")
axes[1].grid()

axes[2].plot(t_plot, z_brownian_difference[plot_slice] * 1e9)
axes[2].axhline(0, linestyle=":", color="black")
axes[2].set_xlabel("Time / s")
axes[2].set_ylabel("z difference / nm")
axes[2].grid()

fig.suptitle("Isolated displacement effect from Brownian motion")
plt.tight_layout()
finish_plot()

#****************************************************************************************************************************************************
# Power spectral density
#****************************************************************************************************************************************************
sampling_frequency = 1 / dt_baoab
nyquist_frequency = sampling_frequency / 2
frequency_bin_size = 1 / (len(t_baoab) * dt_baoab)
psd_plot_max_frequency = (
    nyquist_frequency
    if psd_plot_max_frequency_override is None
    else psd_plot_max_frequency_override
)

# print("PSD sampling frequency =", sampling_frequency, "Hz")
# print("PSD Nyquist frequency =", nyquist_frequency, "Hz")
# print("PSD full-record frequency bin size =", frequency_bin_size, "Hz")


def positive_psd(signal, time_values):
    dt = time_values[1] - time_values[0]
    fs = 1 / dt
    if psd_segment_samples is not None:
        nperseg = min(psd_segment_samples, len(signal))
    elif psd_segment_duration_seconds is not None:
        nperseg = min(
            max(2, int(round(psd_segment_duration_seconds * fs))),
            len(signal)
        )
    else:
        nperseg = len(signal)

    if nperseg < 2:
        raise ValueError("PSD segment settings must leave at least two samples.")

    if psd_overlap_samples is not None:
        noverlap = min(psd_overlap_samples, nperseg - 1)
    elif nperseg == len(signal):
        noverlap = 0
    else:
        noverlap = min(int(round(psd_overlap_fraction * nperseg)), nperseg - 1)

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        return_onesided=True
    )

    positive_mask = (frequencies > 0) & (psd > 0)
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]

    return frequencies, psd


def apply_psd_axes():
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(minimum_plot_frequency, psd_plot_max_frequency)

    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(psd_min_value, psd_max_value)

    plt.xlabel("Frequency / Hz")
    plt.ylabel("PSD / m^2 Hz^-1")


def quantise_position_resolution(position_m, resolution_m):
    if resolution_m <= 0.0:
        return np.array(position_m, copy=True)
    return np.round(position_m / resolution_m) * resolution_m


def psd_fit_function(frequency_hz, a, b, c, d):
    return a / ((b - frequency_hz**2)**2 + c * frequency_hz**2) + d


def fit_psd_curve(
    frequency_hz,
    psd_m2_per_hz,
    label,
    fit_frequency_min_hz=None,
    fit_frequency_max_hz=None,
    resonance_frequency_hz=None,
):
    psd_nm2_per_hz = psd_m2_per_hz * 1e18
    active_fit_frequency_min_hz = (
        fit_frequency_min_hz
        if fit_frequency_min_hz is not None
        else frequency_hz[0]
    )
    active_fit_frequency_max_hz = (
        fit_frequency_max_hz
        if fit_frequency_max_hz is not None
        else psd_plot_max_frequency
    )

    fit_mask = (
        (frequency_hz > 0.0)
        & np.isfinite(psd_nm2_per_hz)
        & (psd_nm2_per_hz > 0.0)
        & (frequency_hz >= active_fit_frequency_min_hz)
        & (frequency_hz <= active_fit_frequency_max_hz)
    )

    fit_frequency = frequency_hz[fit_mask]
    fit_psd = psd_nm2_per_hz[fit_mask]

    if psd_fit_max_points is not None and len(fit_frequency) > psd_fit_max_points:
        peak_index_full = int(np.argmax(fit_psd))
        peak_window = np.arange(
            max(0, peak_index_full - 10),
            min(len(fit_frequency), peak_index_full + 11),
        )
        raw_indices = np.geomspace(
            1,
            len(fit_frequency),
            psd_fit_max_points,
        ).astype(int) - 1
        keep_indices = np.unique(
            np.concatenate(([0], raw_indices, peak_window, [len(fit_frequency) - 1]))
        )
        fit_frequency = fit_frequency[keep_indices]
        fit_psd = fit_psd[keep_indices]

    if len(fit_frequency) < 5:
        raise ValueError(f"Not enough PSD points to fit {label}.")

    peak_index = np.argmax(fit_psd)
    peak_frequency = fit_frequency[peak_index]
    frequency_spacing = np.median(np.diff(fit_frequency))

    d_guess = max(np.percentile(fit_psd, 10.0), 1e-300)
    peak_height_guess = max(fit_psd[peak_index] - d_guess, 0.1 * fit_psd[peak_index])
    b_guess = peak_frequency**2
    if use_physical_psd_c_initial_guess:
        c_initial_from_damping = (gamma_baoab / (2.0 * np.pi))**2
    else:
        c_initial_from_damping = (0.1 * peak_frequency)**2
    c_guess = max(c_initial_from_damping, frequency_spacing**2, 1e-300)
    a_guess = peak_height_guess * c_guess * max(peak_frequency**2, 1e-300)

    b_lower = max(active_fit_frequency_min_hz**2, 1e-300)
    b_upper = max(active_fit_frequency_max_hz**2, b_lower * (1.0 + 1e-9))
    lower = np.array([1e-300, b_lower, 1e-300, 1e-300])
    upper = np.array([np.inf, b_upper, np.inf, np.inf])
    initial = np.array([a_guess, b_guess, c_guess, d_guess])
    initial = np.maximum(initial, lower * (1.0 + 1e-9))
    initial[1] = min(initial[1], upper[1] * (1.0 - 1e-9))

    if resonance_frequency_hz is None:
        fit_weights = np.ones_like(fit_frequency)
    else:
        peak_sigma_hz = max(
            psd_fit_peak_band_half_width_hz / 3.0,
            frequency_spacing,
        )
        fit_weights = 1.0 + psd_fit_peak_weight * np.exp(
            -0.5 * ((fit_frequency - resonance_frequency_hz) / peak_sigma_hz)**2
        )

    def log_residual(log_parameters):
        parameters = np.exp(log_parameters)
        model = psd_fit_function(fit_frequency, *parameters)
        return fit_weights * (np.log10(model) - np.log10(fit_psd))

    result = least_squares(
        log_residual,
        np.log(initial),
        bounds=(np.log(lower), np.log(upper)),
        max_nfev=20000,
    )

    parameters = np.exp(result.x)
    fitted_curve_nm2 = psd_fit_function(frequency_hz, *parameters)
    return parameters, fitted_curve_nm2 / 1e18


def fit_psd_curve_fixed_frequency_and_damping(
    frequency_hz,
    psd_m2_per_hz,
    label,
    fit_frequency_min_hz,
    fit_frequency_max_hz,
    fixed_f0_hz,
    fixed_gamma_rad_per_s,
):
    psd_nm2_per_hz = psd_m2_per_hz * 1e18
    fit_mask = (
        (frequency_hz > 0.0)
        & np.isfinite(psd_nm2_per_hz)
        & (psd_nm2_per_hz > 0.0)
        & (frequency_hz >= fit_frequency_min_hz)
        & (frequency_hz <= fit_frequency_max_hz)
    )

    fit_frequency = frequency_hz[fit_mask]
    fit_psd = psd_nm2_per_hz[fit_mask]

    if psd_fit_max_points is not None and len(fit_frequency) > psd_fit_max_points:
        raw_indices = np.geomspace(
            1,
            len(fit_frequency),
            psd_fit_max_points,
        ).astype(int) - 1
        keep_indices = np.unique(
            np.concatenate(([0], raw_indices, [len(fit_frequency) - 1]))
        )
        fit_frequency = fit_frequency[keep_indices]
        fit_psd = fit_psd[keep_indices]

    if len(fit_frequency) < 5:
        raise ValueError(f"Not enough PSD points to fit {label}.")

    b_fixed = fixed_f0_hz**2
    c_fixed = (fixed_gamma_rad_per_s / (2.0 * np.pi))**2
    denominator = (b_fixed - fit_frequency**2)**2 + c_fixed * fit_frequency**2
    shape = 1.0 / denominator

    d_guess = max(np.percentile(fit_psd, 10.0), 1e-300)
    a_guess = max(np.max((fit_psd - d_guess) / shape), 1e-300)
    initial = np.array([a_guess, d_guess])
    lower = np.array([1e-300, 1e-300])
    upper = np.array([np.inf, np.inf])

    def log_residual(log_parameters):
        a_fit, d_fit = np.exp(log_parameters)
        model = a_fit * shape + d_fit
        return np.log10(model) - np.log10(fit_psd)

    result = least_squares(
        log_residual,
        np.log(initial),
        bounds=(np.log(lower), np.log(upper)),
        max_nfev=20000,
    )

    a_fit, d_fit = np.exp(result.x)
    parameters = np.array([a_fit, b_fixed, c_fixed, d_fit])
    fitted_curve_nm2 = psd_fit_function(frequency_hz, *parameters)
    return parameters, fitted_curve_nm2 / 1e18

def paper_centre_of_mass_temperature_from_emerging(
    temperature_emerging,
    temperature_impinging,
):
    paper_temperature_factor = np.pi / 8.0
    return (
        temperature_impinging**1.5
        + paper_temperature_factor * temperature_emerging**1.5
    ) / (
        np.sqrt(temperature_impinging)
        + paper_temperature_factor * np.sqrt(temperature_emerging)
    )


def paper_damping_from_emerging(
    temperature_emerging,
    temperature_impinging,
    cold_damping_rate,
):
    paper_temperature_factor = np.pi / 8.0
    return cold_damping_rate * (
        1.0
        + paper_temperature_factor
        * np.sqrt(temperature_emerging / temperature_impinging)
    ) / (1.0 + paper_temperature_factor)


def paper_emerging_temperature_from_centre_of_mass(
    temperature_cm,
    temperature_impinging,
):
    if temperature_cm <= 0.0 or temperature_impinging <= 0.0:
        return np.nan

    def residual(temperature_emerging):
        return (
            paper_centre_of_mass_temperature_from_emerging(
                temperature_emerging,
                temperature_impinging,
            )
            - temperature_cm
        )

    lower = max(1e-12 * temperature_impinging, 1e-12)
    upper = max(2.0 * temperature_impinging, 2.0 * temperature_cm, lower * 10.0)

    if temperature_cm >= temperature_impinging:
        lower = temperature_impinging
        lower_residual = residual(lower)
        upper_residual = residual(upper)
        while lower_residual * upper_residual > 0.0 and upper < 1e9:
            upper *= 2.0
            upper_residual = residual(upper)

        if lower_residual * upper_residual <= 0.0:
            return brentq(residual, lower, upper)

    temperature_grid = np.geomspace(lower, upper, 500)
    residual_grid = np.array([residual(value) for value in temperature_grid])
    sign_change_indices = np.where(
        residual_grid[:-1] * residual_grid[1:] <= 0.0
    )[0]

    if len(sign_change_indices) == 0:
        return np.nan

    index = sign_change_indices[-1]
    return brentq(residual, temperature_grid[index], temperature_grid[index + 1])


def psd_fit_temperature_quantities(fit_parameters):
    a_nm2, b_fit, c_fit, _ = fit_parameters

    if a_nm2 <= 0.0 or b_fit <= 0.0 or c_fit <= 0.0:
        return None

    a_m2 = a_nm2 * 1e-18
    f0_hz = np.sqrt(b_fit)
    omega0_rad_per_s = 2.0 * np.pi * f0_hz
    gamma_cm_rad_per_s = 2.0 * np.pi * np.sqrt(c_fit)
    variance_m2 = np.pi * a_m2 / (2.0 * b_fit * np.sqrt(c_fit))
    temperature_cm = m * omega0_rad_per_s**2 * variance_m2 / kB

    return {
        "f0_hz": f0_hz,
        "omega0_rad_per_s": omega0_rad_per_s,
        "gamma_cm_rad_per_s": gamma_cm_rad_per_s,
        "variance_m2": variance_m2,
        "rms_nm": np.sqrt(variance_m2) * 1e9,
        "temperature_cm": temperature_cm,
    }


def surface_temperature_from_psd_fit(fit_parameters):
    quantities = psd_fit_temperature_quantities(fit_parameters)
    if quantities is None:
        return None

    temperature_impinging = (
        T
        if surface_temperature_impinging_gas_temperature is None
        else surface_temperature_impinging_gas_temperature
    )
    gamma_0_rad_per_s = gamma_0_baoab
    alpha = surface_temperature_accommodation_alpha

    if temperature_impinging <= 0.0 or gamma_0_rad_per_s <= 0.0 or alpha <= 0.0:
        return None

    gamma_cm_rad_per_s = quantities["gamma_cm_rad_per_s"]
    temperature_cm = quantities["temperature_cm"]
    temperature_emerging = paper_emerging_temperature_from_centre_of_mass(
        temperature_cm,
        temperature_impinging,
    )
    temperature_surface = temperature_impinging + (
        temperature_emerging - temperature_impinging
    ) / alpha
    paper_predicted_gamma_cm = paper_damping_from_emerging(
        temperature_emerging,
        temperature_impinging,
        gamma_0_rad_per_s,
    )
    gamma_cm_percentage_difference = 100.0 * (
        gamma_cm_rad_per_s - paper_predicted_gamma_cm
    ) / paper_predicted_gamma_cm

    quantities.update(
        {
            "temperature_impinging": temperature_impinging,
            "gamma_0_rad_per_s": gamma_0_rad_per_s,
            "temperature_emerging": temperature_emerging,
            "temperature_surface": temperature_surface,
            "paper_predicted_gamma_cm_rad_per_s": paper_predicted_gamma_cm,
            "gamma_cm_percentage_difference": gamma_cm_percentage_difference,
            "thermal_accommodation_alpha": alpha,
        }
    )
    return quantities


def percentage_error(measured, reference):
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return 100.0 * (measured - reference) / reference


def signed_percent_error(value, reference):
    if reference == 0.0 or not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    return 100.0 * (value - reference) / reference


def score_linewidth_fit_result(result):
    if result is None:
        return np.inf, np.nan, np.nan, np.nan

    error_surface = signed_percent_error(
        result["temperature_surface"],
        brownian_surface_temperature,
    )
    error_cm = signed_percent_error(
        result["temperature_cm"],
        brownian_centre_of_mass_temperature,
    )
    error_gamma = signed_percent_error(
        result["gamma_cm_rad_per_s"],
        gamma_baoab,
    )

    errors = np.array([error_surface, error_cm, error_gamma], dtype=float)
    if not np.all(np.isfinite(errors)):
        return np.inf, error_surface, error_cm, error_gamma

    return (
        np.sum(np.abs(errors)),
        error_surface,
        error_cm,
        error_gamma,
    )


def fit_psd_with_data_driven_linewidth_window(
    frequency_hz,
    psd_m2_per_hz,
    label,
    linewidth_multiplier,
):
    observed_peak_frequency = frequency_hz[int(np.argmax(psd_m2_per_hz))]
    rough_min_hz = max(
        frequency_hz[0],
        observed_peak_frequency - linewidth_sweep_rough_fit_half_width_hz,
    )
    rough_max_hz = min(
        psd_plot_max_frequency,
        observed_peak_frequency + linewidth_sweep_rough_fit_half_width_hz,
    )

    rough_parameters, _ = fit_psd_curve(
        frequency_hz,
        psd_m2_per_hz,
        f"{label} rough linewidth estimate",
        rough_min_hz,
        rough_max_hz,
        observed_peak_frequency,
    )

    rough_f0_hz = np.sqrt(max(rough_parameters[1], 1e-300))
    rough_linewidth_hz = np.sqrt(max(rough_parameters[2], 1e-300))
    frequency_spacing = np.median(np.diff(frequency_hz))

    if not np.isfinite(rough_linewidth_hz) or rough_linewidth_hz <= 0.0:
        rough_linewidth_hz = frequency_spacing

    fit_half_width_hz = max(
        linewidth_multiplier * rough_linewidth_hz,
        minimum_fit_bins_each_side * frequency_spacing,
        2.0 * frequency_spacing,
    )
    fit_min_hz = max(frequency_hz[0], rough_f0_hz - fit_half_width_hz)
    fit_max_hz = min(psd_plot_max_frequency, rough_f0_hz + fit_half_width_hz)

    fit_parameters, fit_curve = fit_psd_curve(
        frequency_hz,
        psd_m2_per_hz,
        f"{label} linewidth multiplier {linewidth_multiplier:g}",
        fit_min_hz,
        fit_max_hz,
        rough_f0_hz,
    )
    temperature_result = surface_temperature_from_psd_fit(fit_parameters)
    score, error_surface, error_cm, error_gamma = score_linewidth_fit_result(
        temperature_result
    )

    return {
        "linewidth_multiplier": linewidth_multiplier,
        "rough_fit_min_hz": rough_min_hz,
        "rough_fit_max_hz": rough_max_hz,
        "rough_f0_hz": rough_f0_hz,
        "rough_linewidth_hz": rough_linewidth_hz,
        "fit_half_width_hz": fit_half_width_hz,
        "fit_min_hz": fit_min_hz,
        "fit_max_hz": fit_max_hz,
        "fit_parameters": fit_parameters,
        "fit_curve": fit_curve,
        "temperature_result": temperature_result,
        "score": score,
        "error_surface_percent": error_surface,
        "error_cm_percent": error_cm,
        "error_gamma_percent": error_gamma,
    }


def physically_usable_surface_temperature_result(result):
    if result is None:
        return False, "no temperature result"

    required_keys = (
        "f0_hz",
        "gamma_cm_rad_per_s",
        "temperature_cm",
        "temperature_emerging",
        "temperature_surface",
    )
    for key in required_keys:
        if key not in result or not np.isfinite(result[key]):
            return False, f"{key} is not finite"

    if result["temperature_cm"] <= 0.0:
        return False, "T_CM is non-positive"
    if result["temperature_emerging"] <= 0.0:
        return False, "T_em is non-positive"
    if result["temperature_surface"] <= 0.0:
        return False, "T_surface is non-positive"

    temperature_impinging = result["temperature_impinging"]
    if result["temperature_cm"] < temperature_impinging:
        return False, "T_CM is below the impinging gas temperature"

    lower_ratio, upper_ratio = pressure_adaptive_gamma_ratio_bounds
    gamma_ratio = result["gamma_cm_rad_per_s"] / gamma_baoab
    if (
        not np.isfinite(gamma_ratio)
        or gamma_ratio < lower_ratio
        or gamma_ratio > upper_ratio
    ):
        return False, "Gamma_CM is far from the pressure-model value"

    return True, "accepted"


def expected_psd_linewidth_hz():
    return gamma_baoab / (2.0 * np.pi)


def pressure_adaptive_base_half_width_hz(frequency_hz):
    frequency_spacing = np.median(np.diff(frequency_hz))
    minimum_width = (
        pressure_adaptive_min_half_width_hz
        if pressure_adaptive_min_half_width_hz is not None
        else minimum_fit_bins_each_side * frequency_spacing
    )
    half_width_hz = max(
        pressure_adaptive_linewidth_multiplier * expected_psd_linewidth_hz(),
        minimum_width,
        2.0 * frequency_spacing,
    )
    if pressure_adaptive_max_half_width_hz is not None:
        half_width_hz = min(half_width_hz, pressure_adaptive_max_half_width_hz)
    return half_width_hz


def fit_psd_with_pressure_adaptive_window(
    frequency_hz,
    psd_m2_per_hz,
    label,
):
    expected_f0_hz = omega_z / (2.0 * np.pi)
    base_half_width_hz = pressure_adaptive_base_half_width_hz(frequency_hz)
    attempted_rows = []

    for width_scale in pressure_adaptive_retry_width_scales:
        fit_half_width_hz = base_half_width_hz * width_scale
        if pressure_adaptive_max_half_width_hz is not None:
            fit_half_width_hz = min(
                fit_half_width_hz,
                pressure_adaptive_max_half_width_hz,
            )

        fit_min_hz = max(frequency_hz[0], expected_f0_hz - fit_half_width_hz)
        fit_max_hz = min(psd_plot_max_frequency, expected_f0_hz + fit_half_width_hz)

        try:
            if use_constrained_psd_fit:
                fit_parameters, fit_curve = fit_psd_curve_fixed_frequency_and_damping(
                    frequency_hz,
                    psd_m2_per_hz,
                    f"{label} constrained pressure-adaptive width scale {width_scale:g}",
                    fit_min_hz,
                    fit_max_hz,
                    expected_f0_hz,
                    gamma_baoab,
                )
            else:
                fit_parameters, fit_curve = fit_psd_curve(
                    frequency_hz,
                    psd_m2_per_hz,
                    f"{label} pressure-adaptive width scale {width_scale:g}",
                    fit_min_hz,
                    fit_max_hz,
                    expected_f0_hz,
                )
            temperature_result = surface_temperature_from_psd_fit(fit_parameters)
            score, error_surface, error_cm, error_gamma = score_linewidth_fit_result(
                temperature_result
            )
            validation_ok, validation_reason = (
                physically_usable_surface_temperature_result(temperature_result)
            )
        except Exception as exc:
            fit_parameters = None
            fit_curve = np.full_like(frequency_hz, np.nan, dtype=float)
            temperature_result = None
            score, error_surface, error_cm, error_gamma = (
                np.inf,
                np.nan,
                np.nan,
                np.nan,
            )
            validation_ok = False
            validation_reason = f"{type(exc).__name__}: {exc}"

        row = {
            "linewidth_multiplier": pressure_adaptive_linewidth_multiplier,
            "retry_width_scale": width_scale,
            "rough_fit_min_hz": np.nan,
            "rough_fit_max_hz": np.nan,
            "rough_f0_hz": expected_f0_hz,
            "rough_linewidth_hz": expected_psd_linewidth_hz(),
            "fit_half_width_hz": fit_half_width_hz,
            "fit_min_hz": fit_min_hz,
            "fit_max_hz": fit_max_hz,
            "fit_parameters": fit_parameters,
            "fit_curve": fit_curve,
            "temperature_result": temperature_result,
            "score": score,
            "error_surface_percent": error_surface,
            "error_cm_percent": error_cm,
            "error_gamma_percent": error_gamma,
            "validation_ok": validation_ok,
            "validation_reason": validation_reason,
            "expected_linewidth_hz": expected_psd_linewidth_hz(),
            "constrained_fit": use_constrained_psd_fit,
        }
        attempted_rows.append(row)

        if validation_ok:
            return row

    finite_rows = [row for row in attempted_rows if np.isfinite(row["score"])]
    if finite_rows:
        return min(finite_rows, key=lambda row: row["score"])
    return attempted_rows[-1]


def surface_temperature_result_from_variance_m2(
    variance_m2,
    label,
):
    variance_m2 = max(variance_m2, 0.0)
    temperature_cm = m * omega_z**2 * variance_m2 / kB
    temperature_impinging = (
        T
        if surface_temperature_impinging_gas_temperature is None
        else surface_temperature_impinging_gas_temperature
    )
    temperature_emerging = paper_emerging_temperature_from_centre_of_mass(
        temperature_cm,
        temperature_impinging,
    )
    temperature_surface = temperature_impinging + (
        temperature_emerging - temperature_impinging
    ) / surface_temperature_accommodation_alpha
    paper_predicted_gamma_cm = paper_damping_from_emerging(
        temperature_emerging,
        temperature_impinging,
        gamma_0_baoab,
    )

    if np.isfinite(paper_predicted_gamma_cm) and paper_predicted_gamma_cm != 0.0:
        gamma_cm_percentage_difference = 100.0 * (
            gamma_baoab - paper_predicted_gamma_cm
        ) / paper_predicted_gamma_cm
    else:
        gamma_cm_percentage_difference = np.nan

    result = {
        "f0_hz": omega_z / (2.0 * np.pi),
        "omega0_rad_per_s": omega_z,
        "gamma_cm_rad_per_s": gamma_baoab,
        "variance_m2": variance_m2,
        "rms_nm": np.sqrt(variance_m2) * 1e9,
        "temperature_cm": temperature_cm,
        "temperature_impinging": temperature_impinging,
        "gamma_0_rad_per_s": gamma_0_baoab,
        "temperature_emerging": temperature_emerging,
        "temperature_surface": temperature_surface,
        "paper_predicted_gamma_cm_rad_per_s": paper_predicted_gamma_cm,
        "gamma_cm_percentage_difference": gamma_cm_percentage_difference,
        "thermal_accommodation_alpha": surface_temperature_accommodation_alpha,
        "label": label,
    }

    return result


def package_surface_temperature_estimator_result(result):
    score, error_surface, error_cm, error_gamma = score_linewidth_fit_result(result)
    validation_ok, validation_reason = physically_usable_surface_temperature_result(
        result
    )

    return {
        "temperature_result": result,
        "score": score,
        "error_surface_percent": error_surface,
        "error_cm_percent": error_cm,
        "error_gamma_percent": error_gamma,
        "validation_ok": validation_ok,
        "validation_reason": validation_reason,
    }


def print_pressure_temperature_error_result_marker(estimator, estimator_result):
    if not pressure_temperature_error_sweep_child:
        return

    result = None
    error_percent = np.nan
    if estimator_result is not None:
        result = estimator_result.get("temperature_result")
        error_percent = estimator_result.get("error_surface_percent", np.nan)

    recovered_surface_temperature = (
        result["temperature_surface"]
        if result is not None and "temperature_surface" in result
        else np.nan
    )
    print(
        "PRESSURE_TEMPERATURE_ERROR_RESULT",
        f"pressure_pa={p}",
        f"estimator={estimator}",
        f"recovered_T_surface_K={recovered_surface_temperature}",
        f"error_percent={error_percent}",
    )


def surface_temperature_from_z_variance(
    z_values,
    label,
    detector_resolution_m=0.0,
):
    z_values = np.asarray(z_values, dtype=float)
    variance_m2 = np.var(z_values - np.mean(z_values))

    if subtract_detector_quantisation_variance and detector_resolution_m > 0.0:
        variance_m2 -= detector_resolution_m**2 / 12.0

    result = surface_temperature_result_from_variance_m2(variance_m2, label)
    return package_surface_temperature_estimator_result(result)


def surface_temperature_from_psd_area(
    frequency_hz,
    psd_m2_per_hz,
    label,
    detector_resolution_m=0.0,
):
    area_mask = (
        (frequency_hz > 0.0)
        & np.isfinite(frequency_hz)
        & np.isfinite(psd_m2_per_hz)
        & (psd_m2_per_hz > 0.0)
        & (frequency_hz <= psd_plot_max_frequency)
    )
    area_frequency = frequency_hz[area_mask]
    area_psd = psd_m2_per_hz[area_mask]

    if len(area_frequency) < 2:
        variance_m2 = np.nan
    else:
        integrate_trapezoid = getattr(np, "trapezoid", np.trapz)
        variance_m2 = integrate_trapezoid(area_psd, area_frequency)

    if subtract_detector_quantisation_variance and detector_resolution_m > 0.0:
        variance_m2 -= detector_resolution_m**2 / 12.0

    result = surface_temperature_result_from_variance_m2(variance_m2, label)
    result["integrated_psd_points"] = len(area_frequency)
    return package_surface_temperature_estimator_result(result)


def print_variance_surface_temperature_result(label, variance_result):
    result = variance_result["temperature_result"]
    print(f"\nVariance/equipartition surface-temperature estimate from {label}:")
    print("  trap f0 =", result["f0_hz"], "Hz")
    print("  model Gamma_CM =", result["gamma_cm_rad_per_s"], "s^-1")
    print("  RMS displacement =", result["rms_nm"], "nm")
    print("  T_CM =", result["temperature_cm"], "K")
    print("  T_surface =", result["temperature_surface"], "K")
    print(
        "  signed T_surface error =",
        variance_result["error_surface_percent"],
        "%",
    )
    print(
        "  validation =",
        variance_result["validation_reason"],
    )


def print_psd_area_surface_temperature_result(label, area_result):
    result = area_result["temperature_result"]
    print(f"\nPSD-area surface-temperature estimate from {label}:")
    print("  trap f0 =", result["f0_hz"], "Hz")
    print("  model Gamma_CM =", result["gamma_cm_rad_per_s"], "s^-1")
    print("  integrated PSD RMS displacement =", result["rms_nm"], "nm")
    print("  integrated PSD points =", result["integrated_psd_points"])
    print("  T_CM =", result["temperature_cm"], "K")
    print("  T_surface =", result["temperature_surface"], "K")
    print(
        "  signed T_surface error =",
        area_result["error_surface_percent"],
        "%",
    )
    print(
        "  validation =",
        area_result["validation_reason"],
    )


def print_linewidth_multiplier_sweep(source_label, frequency_hz, psd_m2_per_hz):
    print(f"\nLinewidth-multiplier sweep for {source_label}:")
    print(
        "N, rough_f0_Hz, rough_linewidth_Hz, fit_min_Hz, fit_max_Hz, "
        "recovered_T_surface_K, recovered_T_CM_K, recovered_Gamma_CM_s^-1, "
        "error_T_surface_percent, error_T_CM_percent, error_Gamma_CM_percent, score"
    )

    rows = []
    for linewidth_multiplier in linewidth_fit_multipliers:
        try:
            row = fit_psd_with_data_driven_linewidth_window(
                frequency_hz,
                psd_m2_per_hz,
                source_label,
                float(linewidth_multiplier),
            )
            result = row["temperature_result"]
            rows.append(row)
            print(
                row["linewidth_multiplier"],
                row["rough_f0_hz"],
                row["rough_linewidth_hz"],
                row["fit_min_hz"],
                row["fit_max_hz"],
                result["temperature_surface"] if result is not None else np.nan,
                result["temperature_cm"] if result is not None else np.nan,
                result["gamma_cm_rad_per_s"] if result is not None else np.nan,
                row["error_surface_percent"],
                row["error_cm_percent"],
                row["error_gamma_percent"],
                row["score"],
                sep=", "
            )
        except Exception as exc:
            print(
                linewidth_multiplier,
                f"error: {type(exc).__name__}: {exc}",
                sep=", "
            )

    finite_rows = [row for row in rows if np.isfinite(row["score"])]
    if finite_rows:
        best_row = min(finite_rows, key=lambda row: row["score"])
        print(
            f"Best {source_label} multiplier by synthetic score =",
            best_row["linewidth_multiplier"],
            "with score =",
            best_row["score"],
        )
        print(
            "Note: the score uses the known synthetic input only after fitting. "
            "The fit windows above were chosen from rough PSD fits to the data."
        )
    else:
        print(f"No finite linewidth-multiplier sweep fits for {source_label}.")

    return rows


def run_pressure_temperature_error_child_analysis():
    requested_estimators = set(pressure_temperature_error_sweep_estimators)
    z_displacement_child = make_quantised_displacement_array(
        "pressure_sweep_z_displacement",
        z_baoab,
        z_eq,
        0.0,
    )
    needs_measured_displacement = any(
        estimator.endswith("_measured") for estimator in requested_estimators
    )
    needs_true_psd = bool(
        {"psd_true", "psd_area_true"} & requested_estimators
    )
    needs_measured_psd = bool(
        {"psd_measured", "psd_area_measured"} & requested_estimators
    )
    needs_any_psd = needs_true_psd or needs_measured_psd

    if "variance_true" in requested_estimators:
        print_pressure_temperature_error_result_marker(
            "variance_true",
            surface_temperature_from_z_variance(
                z_displacement_child,
                "true z displacement",
            ),
        )

    z_displacement_measured_child = None
    detector_position_resolution_m = detector_position_resolution_nm * 1e-9
    if include_z_detector_resolution_psd and needs_measured_displacement:
        z_displacement_measured_child = make_quantised_displacement_array(
            "pressure_sweep_z_displacement_measured",
            z_baoab,
            z_eq,
            detector_position_resolution_m,
        )

    if (
        "variance_measured" in requested_estimators
        and z_displacement_measured_child is not None
    ):
        print_pressure_temperature_error_result_marker(
            "variance_measured",
            surface_temperature_from_z_variance(
                z_displacement_measured_child,
                f"{detector_position_resolution_nm:g} nm z displacement",
                detector_position_resolution_m,
            ),
        )

    if not needs_any_psd:
        return

    baoab_z_freqs_child, baoab_z_psd_child = positive_psd(
        z_displacement_child,
        t_baoab,
    )

    measured_z_freqs_child = None
    measured_z_psd_child = None
    if include_z_detector_resolution_psd and needs_measured_psd:
        if z_displacement_measured_child is None:
            z_displacement_measured_child = make_quantised_displacement_array(
                "pressure_sweep_z_displacement_measured",
                z_baoab,
                z_eq,
                detector_position_resolution_m,
            )
        measured_z_freqs_child, measured_z_psd_child = positive_psd(
            z_displacement_measured_child,
            t_baoab,
        )

    if "psd_true" in requested_estimators:
        if use_pressure_adaptive_psd_fit:
            fit_result = fit_psd_with_pressure_adaptive_window(
                baoab_z_freqs_child,
                baoab_z_psd_child,
                "true z PSD",
            )
        else:
            fit_result = fit_psd_with_data_driven_linewidth_window(
                baoab_z_freqs_child,
                baoab_z_psd_child,
                "true z PSD fixed linewidth multiplier",
                linewidth_fit_multiplier,
            )
        print_pressure_temperature_error_result_marker("psd_true", fit_result)

    if "psd_area_true" in requested_estimators:
        print_pressure_temperature_error_result_marker(
            "psd_area_true",
            surface_temperature_from_psd_area(
                baoab_z_freqs_child,
                baoab_z_psd_child,
                "true z PSD",
            ),
        )

    if not include_z_detector_resolution_psd:
        return

    if "psd_measured" in requested_estimators and measured_z_psd_child is not None:
        if use_pressure_adaptive_psd_fit:
            measured_fit_result = fit_psd_with_pressure_adaptive_window(
                measured_z_freqs_child,
                measured_z_psd_child,
                f"{detector_position_resolution_nm:g} nm z PSD",
            )
        else:
            measured_fit_result = fit_psd_with_data_driven_linewidth_window(
                measured_z_freqs_child,
                measured_z_psd_child,
                f"{detector_position_resolution_nm:g} nm z PSD fixed linewidth multiplier",
                linewidth_fit_multiplier,
            )
        print_pressure_temperature_error_result_marker(
            "psd_measured",
            measured_fit_result,
        )

    if "psd_area_measured" in requested_estimators and measured_z_psd_child is not None:
        print_pressure_temperature_error_result_marker(
            "psd_area_measured",
            surface_temperature_from_psd_area(
                measured_z_freqs_child,
                measured_z_psd_child,
                f"{detector_position_resolution_nm:g} nm z PSD",
                detector_position_resolution_m,
            ),
        )


if pressure_temperature_error_sweep_child:
    run_pressure_temperature_error_child_analysis()
    raise SystemExit(0)


z_displacement = make_quantised_displacement_array(
    "z_displacement",
    z_baoab,
    z_eq,
    0.0,
)

baoab_x_freqs, baoab_x_psd = positive_psd(x_baoab, t_baoab)
baoab_y_freqs, baoab_y_psd = positive_psd(y_baoab, t_baoab)
baoab_z_freqs, baoab_z_psd = positive_psd(z_displacement, t_baoab)

minimum_resolvable_frequency = baoab_x_freqs[0]
minimum_plot_frequency = psd_min_frequency_factor * minimum_resolvable_frequency

# print("Minimum resolvable non-zero frequency =", minimum_resolvable_frequency, "Hz")
# print("Lower frequency shown on plot =", minimum_plot_frequency, "Hz")

if simple_atmospheric_trajectory_psd_plot:
    output_path = Path(simple_atmospheric_trajectory_psd_plot_filename)
    if not output_path.is_absolute():
        output_path = script_directory / output_path

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(spectrum_figsize[0], 1.45 * spectrum_figsize[1]),
    )

    axes[0].plot(
        t_plot,
        (z_baoab[plot_slice] - z_eq) * 1e6,
        color="tab:orange",
        linewidth=0.9,
    )
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("z - z_eq / micrometres")
    axes[0].set_title("Atmospheric-pressure z trajectory")
    axes[0].grid(True)

    axes[1].loglog(
        baoab_z_freqs,
        baoab_z_psd,
        color="tab:blue",
        linewidth=1.0,
    )
    axes[1].axvline(
        omega_z / (2.0 * np.pi),
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="trap frequency",
    )
    axes[1].set_xlim(minimum_plot_frequency, psd_plot_max_frequency)
    if psd_max_value is None:
        axes[1].set_ylim(bottom=psd_min_value)
    else:
        axes[1].set_ylim(psd_min_value, psd_max_value)
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("PSD / m^2 Hz^-1")
    axes[1].set_title("Atmospheric-pressure z PSD")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.suptitle(f"Trajectory and PSD at atmospheric pressure ({p:g} Pa)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finish_plot()
    print("Saved atmospheric trajectory and PSD plot to", output_path)
    raise SystemExit(0)

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_freqs, baoab_x_psd, label="BAOAB x PSD")
apply_psd_axes()
plt.title("PSD of BAOAB x motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_y_freqs, baoab_y_psd, color="tab:green", label="BAOAB y PSD")
apply_psd_axes()
plt.title("PSD of BAOAB y motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="BAOAB z Welch-averaged PSD")
apply_psd_axes()
plt.title("PSD of BAOAB z motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

if include_z_detector_resolution_psd:
    detector_position_resolution_m = detector_position_resolution_nm * 1e-9
    z_displacement_measured = make_quantised_displacement_array(
        "z_displacement_measured",
        z_baoab,
        z_eq,
        detector_position_resolution_m,
    )
    measured_z_freqs, measured_z_psd = positive_psd(
        z_displacement_measured,
        t_baoab
    )

    if fit_z_detector_resolution_psd:
        true_z_peak_frequency = baoab_z_freqs[int(np.argmax(baoab_z_psd))]
        if fit_psd_over_whole_frequency_range:
            default_z_fit_frequency_min_hz = baoab_z_freqs[0]
            default_z_fit_frequency_max_hz = psd_plot_max_frequency
        else:
            default_z_fit_frequency_min_hz = max(
                baoab_z_freqs[0],
                true_z_peak_frequency - psd_fit_peak_band_half_width_hz,
            )
            default_z_fit_frequency_max_hz = min(
                psd_plot_max_frequency,
                true_z_peak_frequency + psd_fit_peak_band_half_width_hz,
            )

        active_z_fit_frequency_min_hz = (
            psd_fit_frequency_min_hz
            if psd_fit_frequency_min_hz is not None
            else default_z_fit_frequency_min_hz
        )
        active_z_fit_frequency_max_hz = (
            psd_fit_frequency_max_hz
            if psd_fit_frequency_max_hz is not None
            else default_z_fit_frequency_max_hz
        )
        true_z_fit_parameters, true_z_fit_curve = fit_psd_curve(
            baoab_z_freqs,
            baoab_z_psd,
            "true z PSD",
            active_z_fit_frequency_min_hz,
            active_z_fit_frequency_max_hz,
            true_z_peak_frequency,
        )
        measured_z_fit_parameters, measured_z_fit_curve = fit_psd_curve(
            measured_z_freqs,
            measured_z_psd,
            f"{detector_position_resolution_nm:g} nm z PSD",
            active_z_fit_frequency_min_hz,
            active_z_fit_frequency_max_hz,
            true_z_peak_frequency,
        )
        z_fit_percentage_error = percentage_error(
            measured_z_fit_parameters,
            true_z_fit_parameters
        )

        print(
            "z PSD fit frequency range =",
            active_z_fit_frequency_min_hz,
            "to",
            active_z_fit_frequency_max_hz,
            "Hz"
        )
        print("True z PSD fit a, b, c, d =", *true_z_fit_parameters)
        print(
            f"{detector_position_resolution_nm:g} nm z PSD fit a, b, c, d =",
            *measured_z_fit_parameters
        )
        print(
            f"{detector_position_resolution_nm:g} nm z PSD fit percentage errors a, b, c, d =",
            *z_fit_percentage_error
        )

        if use_pressure_adaptive_psd_fit:
            fixed_true_fit = fit_psd_with_pressure_adaptive_window(
                baoab_z_freqs,
                baoab_z_psd,
                "true z PSD",
            )
            fixed_measured_fit = fit_psd_with_pressure_adaptive_window(
                measured_z_freqs,
                measured_z_psd,
                f"{detector_position_resolution_nm:g} nm z PSD",
            )
        else:
            fixed_true_fit = fit_psd_with_data_driven_linewidth_window(
                baoab_z_freqs,
                baoab_z_psd,
                "true z PSD fixed linewidth multiplier",
                linewidth_fit_multiplier,
            )
            fixed_measured_fit = fit_psd_with_data_driven_linewidth_window(
                measured_z_freqs,
                measured_z_psd,
                f"{detector_position_resolution_nm:g} nm z PSD fixed linewidth multiplier",
                linewidth_fit_multiplier,
            )
        fixed_true_result = fixed_true_fit["temperature_result"]
        fixed_measured_result = fixed_measured_fit["temperature_result"]

        if use_pressure_adaptive_psd_fit:
            print(
                "Pressure-adaptive PSD fit result: linewidth multiplier =",
                pressure_adaptive_linewidth_multiplier,
                ", constrained fit =",
                use_constrained_psd_fit,
                ", retry width scale =",
                fixed_true_fit["retry_width_scale"],
                ", expected linewidth =",
                fixed_true_fit["expected_linewidth_hz"],
                "Hz, fit half-width =",
                fixed_true_fit["fit_half_width_hz"],
                "Hz, fit range =",
                fixed_true_fit["fit_min_hz"],
                "to",
                fixed_true_fit["fit_max_hz"],
                "Hz",
            )
            print(
                "Pressure-adaptive true PSD validation =",
                fixed_true_fit["validation_reason"],
            )
            print(
                f"Pressure-adaptive {detector_position_resolution_nm:g} nm PSD validation =",
                fixed_measured_fit["validation_reason"],
            )
        else:
            print(
                "Fixed linewidth multiplier result: N =",
                linewidth_fit_multiplier,
                ", fit range =",
                fixed_true_fit["fit_min_hz"],
                "to",
                fixed_true_fit["fit_max_hz"],
                "Hz",
            )
        if fixed_true_result is not None:
            print(
                "PSD true recovered T_surface =",
                fixed_true_result["temperature_surface"],
                "K, signed error =",
                fixed_true_fit["error_surface_percent"],
                "%",
            )
        print_pressure_temperature_error_result_marker("psd_true", fixed_true_fit)
        if fixed_measured_result is not None:
            print(
                f"PSD {detector_position_resolution_nm:g} nm recovered T_surface =",
                fixed_measured_result["temperature_surface"],
                "K, signed error =",
                fixed_measured_fit["error_surface_percent"],
                "%",
            )
        print_pressure_temperature_error_result_marker(
            "psd_measured",
            fixed_measured_fit,
        )

        if use_psd_area_surface_temperature_estimator:
            true_psd_area_result = surface_temperature_from_psd_area(
                baoab_z_freqs,
                baoab_z_psd,
                "true z PSD",
            )
            measured_psd_area_result = surface_temperature_from_psd_area(
                measured_z_freqs,
                measured_z_psd,
                f"{detector_position_resolution_nm:g} nm z PSD",
                detector_position_resolution_m,
            )
            print_psd_area_surface_temperature_result(
                "true z PSD",
                true_psd_area_result,
            )
            print_pressure_temperature_error_result_marker(
                "psd_area_true",
                true_psd_area_result,
            )
            print_psd_area_surface_temperature_result(
                f"{detector_position_resolution_nm:g} nm z PSD",
                measured_psd_area_result,
            )
            print_pressure_temperature_error_result_marker(
                "psd_area_measured",
                measured_psd_area_result,
            )

        if use_variance_surface_temperature_estimator:
            true_variance_result = surface_temperature_from_z_variance(
                z_displacement,
                "true z displacement",
            )
            measured_variance_result = surface_temperature_from_z_variance(
                z_displacement_measured,
                f"{detector_position_resolution_nm:g} nm z displacement",
                detector_position_resolution_m,
            )
            print_variance_surface_temperature_result(
                "true z displacement",
                true_variance_result,
            )
            print_pressure_temperature_error_result_marker(
                "variance_true",
                true_variance_result,
            )
            print_variance_surface_temperature_result(
                f"{detector_position_resolution_nm:g} nm z displacement",
                measured_variance_result,
            )
            print_pressure_temperature_error_result_marker(
                "variance_measured",
                measured_variance_result,
            )

        if run_linewidth_multiplier_sweep:
            print(
                "\nLinewidth sweep expected values from the input heated-Brownian bath:"
            )
            print("input_T_surface_K =", brownian_surface_temperature)
            print("expected_T_CM_K =", brownian_centre_of_mass_temperature)
            print("expected_Gamma_CM_s^-1 =", gamma_baoab)

            if "true" in linewidth_sweep_sources:
                true_linewidth_sweep_rows = print_linewidth_multiplier_sweep(
                    "true z PSD",
                    baoab_z_freqs,
                    baoab_z_psd,
                )
            if "measured" in linewidth_sweep_sources:
                measured_linewidth_sweep_rows = print_linewidth_multiplier_sweep(
                    f"{detector_position_resolution_nm:g} nm z PSD",
                    measured_z_freqs,
                    measured_z_psd,
                )

        if pressure_temperature_error_sweep_child:
            raise SystemExit(0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(spectrum_figsize[0], 1.6 * spectrum_figsize[1]),
        sharex=False
    )

    axes[0].plot(
        t_plot,
        z_displacement[plot_slice] * 1e9,
        color="tab:orange",
        alpha=0.45,
        linewidth=0.9,
        label="true z displacement"
    )
    axes[0].step(
        t_plot,
        z_displacement_measured[plot_slice] * 1e9,
        where="post",
        color="tab:green",
        linewidth=1.0,
        label=f"{detector_position_resolution_nm:g} nm resolution"
    )
    axes[0].axhline(0, linestyle=":", color="black")
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("z displacement / nm")
    axes[0].set_title("z trajectory with detector resolution")
    axes[0].legend()
    axes[0].grid(True)

    plt.sca(axes[1])
    axes[1].loglog(
        baoab_z_freqs,
        baoab_z_psd,
        color="tab:orange",
        alpha=0.45,
        label="true z Welch-averaged PSD"
    )
    axes[1].loglog(
        measured_z_freqs,
        measured_z_psd,
        color="tab:green",
        alpha=0.45,
        label=f"z Welch-averaged PSD with {detector_position_resolution_nm:g} nm resolution"
    )
    if fit_z_detector_resolution_psd:
        true_fit_plot_mask = (
            (baoab_z_freqs >= fixed_true_fit["fit_min_hz"])
            & (baoab_z_freqs <= fixed_true_fit["fit_max_hz"])
        )
        measured_fit_plot_mask = (
            (measured_z_freqs >= fixed_measured_fit["fit_min_hz"])
            & (measured_z_freqs <= fixed_measured_fit["fit_max_hz"])
        )
        axes[1].loglog(
            baoab_z_freqs[true_fit_plot_mask],
            fixed_true_fit["fit_curve"][true_fit_plot_mask],
            color="tab:orange",
            linewidth=2.0,
            label="true z fitted curve over fit range"
        )
        axes[1].loglog(
            measured_z_freqs[measured_fit_plot_mask],
            fixed_measured_fit["fit_curve"][measured_fit_plot_mask],
            color="tab:green",
            linewidth=2.0,
            label=f"{detector_position_resolution_nm:g} nm fitted curve over fit range"
        )
        axes[1].axvline(
            fixed_true_fit["fit_min_hz"],
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="fit range"
        )
        axes[1].axvline(
            fixed_true_fit["fit_max_hz"],
            color="black",
            linestyle="--",
            linewidth=0.9
        )
    apply_psd_axes()
    axes[1].set_title("Welch-averaged PSD of BAOAB z motion with detector resolution")
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.tight_layout()
    finish_plot()

if use_laser_power_noise:
    laser_effect_x_freqs, laser_effect_x_psd = positive_psd(
        x_laser_noise_difference,
        t_baoab
    )
    laser_effect_y_freqs, laser_effect_y_psd = positive_psd(
        y_laser_noise_difference,
        t_baoab
    )
    laser_effect_z_freqs, laser_effect_z_psd = positive_psd(
        z_laser_noise_difference,
        t_baoab
    )

    plt.figure(figsize=spectrum_figsize)
    plt.loglog(laser_effect_x_freqs, laser_effect_x_psd, label="x laser-noise effect")
    plt.loglog(
        laser_effect_y_freqs,
        laser_effect_y_psd,
        color="tab:green",
        label="y laser-noise effect"
    )
    plt.loglog(
        laser_effect_z_freqs,
        laser_effect_z_psd,
        color="tab:orange",
        label="z laser-noise effect"
    )
    apply_psd_axes()
    plt.title("PSD of isolated laser-noise displacement effect")
    plt.legend()
    plt.grid(True, which="both")
    finish_plot()


# ****************************************************************************************************************************************************
# Plot 3D trajectory and projections
# ****************************************************************************************************************************************************
fig = plt.figure(figsize=trajectory_3d_figsize)
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x_baoab[plot_slice] * 1e6,
    y_baoab[plot_slice] * 1e6,
    z_baoab[plot_slice] * 1e6,
    linewidth=0.8,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)

if use_pd_feedback:
    ax.plot(
        x_baoab_feedback[plot_slice] * 1e6,
        y_baoab_feedback[plot_slice] * 1e6,
        z_baoab_feedback[plot_slice] * 1e6,
        linewidth=0.9,
        alpha=0.85,
        label="BAOAB with PD feedback"
    )

ax.scatter(
    [x_eq * 1e6],
    [y_eq * 1e6],
    [z_eq * 1e6],
    color="black",
    s=30,
    label="equilibrium"
)

ax.set_xlabel("x / micrometres")
ax.set_ylabel("y / micrometres")
ax.set_zlabel("z / micrometres")
ax.set_title("3D trajectory")
ax.legend()
plt.tight_layout()
finish_plot()

fig, axes = plt.subplots(1, 3, figsize=trajectory_projection_figsize)

axes[0].plot(
    x_baoab[plot_slice] * 1e6,
    y_baoab[plot_slice] * 1e6,
    linewidth=0.8
)
axes[0].scatter([x_eq * 1e6], [y_eq * 1e6], color="black", s=20)
axes[0].set_xlabel("x / micrometres")
axes[0].set_ylabel("y / micrometres")
axes[0].set_title("x-y projection")
axes[0].axis("equal")
axes[0].grid()

axes[1].plot(
    x_baoab[plot_slice] * 1e6,
    z_baoab[plot_slice] * 1e6,
    linewidth=0.8
)
axes[1].scatter([x_eq * 1e6], [z_eq * 1e6], color="black", s=20)
axes[1].set_xlabel("x / micrometres")
axes[1].set_ylabel("z / micrometres")
axes[1].set_title("x-z projection")
axes[1].axis("equal")
axes[1].grid()

axes[2].plot(
    y_baoab[plot_slice] * 1e6,
    z_baoab[plot_slice] * 1e6,
    linewidth=0.8
)
axes[2].scatter([y_eq * 1e6], [z_eq * 1e6], color="black", s=20)
axes[2].set_xlabel("y / micrometres")
axes[2].set_ylabel("z / micrometres")
axes[2].set_title("y-z projection")
axes[2].axis("equal")
axes[2].grid()

fig.suptitle("3D trajectory projections")
plt.tight_layout()
finish_plot()

# ****************************************************************************************************************************************************
# Transverse restoring-force check
# ****************************************************************************************************************************************************
# x_scan = np.linspace(
#     transverse_force_x_min,
#     transverse_force_x_max,
#     transverse_force_points
# )
# Fx_scan = np.array([F_optical_2d(x_i, z_eq)[0] for x_i in x_scan])

# restoring = Fx_scan * x_scan < 0
# near_axis = np.abs(x_scan) < near_axis_tolerance
# restoring_or_axis = restoring | near_axis

# plt.figure(figsize=force_check_figsize)
# plt.plot(x_scan * 1e6, Fx_scan / (m*g), label="Fx at z_eq")
# plt.axhline(0, color="black", linewidth=0.8)
# plt.axvline(0, linestyle=":", color="black", label="beam axis")
# plt.axvline(x_displacement * 1e6, linestyle="--", label="initial x displacement")
# plt.axvline(-x_displacement * 1e6, linestyle="--")
# plt.xlabel("x / micrometres")
# plt.ylabel("Fx / mg")
# plt.title("Transverse force check at z = z_eq")
# plt.legend()
# plt.grid()
# plt.show()

# if not np.interp(abs(x_displacement), x_scan[x_scan >= 0], restoring_or_axis[x_scan >= 0].astype(float)) > 0.5:
#     pass
#     # print(
#         # "Warning: the chosen x displacement may be outside the local restoring region. "
#         # "Try reducing x_displacement."
#     # )

# # ****************************************************************************************************************************************************
# # Vertical force breakdown along the beam axis
# # ****************************************************************************************************************************************************
# z_force_values = np.linspace(
#     z_eq - vertical_force_half_width,
#     z_eq + vertical_force_half_width,
#     vertical_force_points
# )

# Fz_ray_scat_values = []
# Fz_ray_grad_values = []

# for z_i in z_force_values:
#     _, Fz_scat, _, Fz_grad = F_ray_optics_2d_components_scalar(0.0, z_i)
#     Fz_ray_scat_values.append(Fz_scat)
#     Fz_ray_grad_values.append(Fz_grad)

# Fz_ray_scat_values = np.array(Fz_ray_scat_values)
# Fz_ray_grad_values = np.array(Fz_ray_grad_values)
# Fz_ray_values = Fz_ray_scat_values + Fz_ray_grad_values
# Fz_photo_values = np.array([F_photo_2d(0.0, z_i)[1] for z_i in z_force_values])
# Fz_gravity_values = -m * g * np.ones_like(z_force_values)
# Fz_total_net_values = Fz_ray_values + Fz_photo_values + Fz_gravity_values

# plt.figure(figsize=force_check_figsize)
# plt.plot(
#     z_force_values * 1e6,
#     Fz_ray_scat_values / (m*g),
#     label="Ray scattering-like axial force"
# )
# plt.plot(
#     z_force_values * 1e6,
#     Fz_ray_grad_values / (m*g),
#     label="Ray gradient-like axial force"
# )
# plt.plot(
#     z_force_values * 1e6,
#     Fz_ray_values / (m*g),
#     linestyle="--",
#     label="Total ray-optics axial force"
# )
# plt.plot(z_force_values * 1e6, Fz_photo_values / (m*g), label="Photophoretic force")
# plt.plot(z_force_values * 1e6, Fz_gravity_values / (m*g), label="Gravity")
# plt.plot(
#     z_force_values * 1e6,
#     Fz_total_net_values / (m*g),
#     linewidth=2,
#     label="Total net vertical force"
# )
# plt.axhline(0, color="black", linewidth=0.8)
# plt.axvline(z_eq * 1e6, linestyle=":", color="black", label="equilibrium")
# plt.xlabel("z / micrometres")
# plt.ylabel("Force / mg")
# plt.title("Vertical force breakdown along beam axis")
# plt.legend()
# plt.grid()
# plt.show()

# # ****************************************************************************************************************************************************
# # Plot an x-z force-field slice through y = 0
# # ****************************************************************************************************************************************************
# x_values = np.linspace(force_field_x_min, force_field_x_max, force_field_x_points)
# z_values = np.linspace(
#     z_eq - force_field_z_half_width,
#     z_eq + force_field_z_half_width,
#     force_field_z_points
# )
# X, Z = np.meshgrid(x_values, z_values)

# Fx_field, Fz_field = F_optical_2d(X, Z)
# Fz_net_field = Fz_field - m*g

# force_scale = m*g

# plt.figure(figsize=force_field_figsize)
# plt.contourf(
#     X * 1e6,
#     Z * 1e6,
#     intensity(X, Z),
#     levels=force_field_contour_levels,
#     cmap="viridis",
#     alpha=0.75
# )
# plt.colorbar(label="Normalized intensity")
# plt.quiver(
#     X * 1e6,
#     Z * 1e6,
#     Fx_field / force_scale,
#     Fz_net_field / force_scale,
#     color="white",
#     pivot="mid",
#     scale=force_field_quiver_scale
# )
# plt.scatter([x_eq * 1e6], [z_eq * 1e6], color="red", s=35, label="equilibrium")

# plt.xlabel("x / micrometres")
# plt.ylabel("z / micrometres")
# plt.title("x-z force-field slice over intensity, y = 0")
# plt.legend()
# plt.grid()
# plt.show()

# # ****************************************************************************************************************************************************
# # On-axis force check
# # ****************************************************************************************************************************************************
# # z_axis = np.linspace(on_axis_check_z_min, on_axis_check_z_max, on_axis_check_points)
# # Fx_axis, Fz_axis = F_optical_2d(0.0, z_axis)

# # plt.figure(figsize=force_check_figsize)
# # plt.plot(z_axis * 1e6, Fz_axis, label="upward optical + photophoretic force")
# # plt.plot(z_axis * 1e6, Fz_axis - m*g, label="net vertical force")
# # plt.axhline(m*g, linestyle="--", label="gravity mg")
# # plt.axhline(0, linewidth=0.8, color="black")
# # plt.axvline(z_eq * 1e6, linestyle=":", label="equilibrium")

# # plt.xlabel("z / micrometres")
# # plt.ylabel("Force / N")
# # plt.title("On-axis vertical force check")
# # plt.legend()
# # plt.grid()
# # plt.show()
