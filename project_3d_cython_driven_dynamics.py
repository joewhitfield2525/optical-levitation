import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import welch
from time import perf_counter
import csv
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Publication style ---
fontsize = 13
axis_label_fontsize = 13
legend_fontsize = 11
mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": fontsize,
    "axes.labelsize": axis_label_fontsize,
    "axes.titlesize": axis_label_fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": legend_fontsize,
})

_unit_parentheses_pattern = re.compile(r"^(?P<label>.*?)\s+\((?P<unit>[^()]*)\)$")


def format_axis_label(label):
    if not isinstance(label, str):
        return label

    match = _unit_parentheses_pattern.match(label)
    if match is None:
        return label

    return f"{match.group('label')} / {match.group('unit')}"


_original_axes_set_xlabel = Axes.set_xlabel
_original_axes_set_ylabel = Axes.set_ylabel
_original_axes3d_set_zlabel = Axes3D.set_zlabel
_original_colorbar_set_label = Colorbar.set_label
_original_figure_savefig = Figure.savefig


def set_xlabel_without_unit_parentheses(self, xlabel, *args, **kwargs):
    return _original_axes_set_xlabel(self, format_axis_label(xlabel), *args, **kwargs)


def set_ylabel_without_unit_parentheses(self, ylabel, *args, **kwargs):
    return _original_axes_set_ylabel(self, format_axis_label(ylabel), *args, **kwargs)


def set_zlabel_without_unit_parentheses(self, zlabel, *args, **kwargs):
    return _original_axes3d_set_zlabel(self, format_axis_label(zlabel), *args, **kwargs)


def set_colorbar_label_without_unit_parentheses(self, label, *args, **kwargs):
    return _original_colorbar_set_label(self, format_axis_label(label), *args, **kwargs)


def remove_plot_title(*args, **kwargs):
    return None


def apply_plot_text_style(fig):
    for axis in fig.axes:
        axis.tick_params(axis="both", which="both", labelsize=axis_label_fontsize)
        if hasattr(axis, "zaxis"):
            axis.tick_params(axis="z", which="both", labelsize=axis_label_fontsize)


def savefig_with_plot_text_style(self, *args, **kwargs):
    apply_plot_text_style(self)
    return _original_figure_savefig(self, *args, **kwargs)


Axes.set_xlabel = set_xlabel_without_unit_parentheses
Axes.set_ylabel = set_ylabel_without_unit_parentheses
Axes.set_title = remove_plot_title
Axes3D.set_zlabel = set_zlabel_without_unit_parentheses
Colorbar.set_label = set_colorbar_label_without_unit_parentheses
Figure.suptitle = remove_plot_title
Figure.savefig = savefig_with_plot_text_style

try:
    import baoab_3d_cython as _baoab_3d_cython
    solve_baoab_3d_lookup_cython = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_lookup_cython",
        None
    )
    solve_baoab_3d_feedback_lookup_cython = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_feedback_lookup_cython",
        None
    )
    classify_baoab_3d_lookup_cython = getattr(
        _baoab_3d_cython,
        "classify_baoab_3d_lookup_cython",
        None
    )
    cython_baoab_available = solve_baoab_3d_lookup_cython is not None
    cython_feedback_baoab_available = solve_baoab_3d_feedback_lookup_cython is not None
    cython_capture_classifier_available = classify_baoab_3d_lookup_cython is not None
except ImportError:
    solve_baoab_3d_lookup_cython = None
    solve_baoab_3d_feedback_lookup_cython = None
    classify_baoab_3d_lookup_cython = None
    cython_baoab_available = False
    cython_feedback_baoab_available = False
    cython_capture_classifier_available = False

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
radius = 6.3e-6          # m
density = 1100         # kg/m^3
n_particle = 1.555      # particle refractive index
n_medium = 1.00027     # surrounding medium refractive index, air

# Gas properties
p = 100000        # pressure, Pa
T = 300           # impinging/ambient gas temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
kB = 1.38e-23
d_air = 3.7e-10

# Laser / force parameters
w0 = 2.97e-6        # beam waist, m
wavelength = 532e-9
M2 = 1.2
use_m2_rayleigh_range = False
zR_manual = 100e-6
P_laser = 0.075             # W, example laser power
use_laser_power_noise = True
laser_noise_fraction = 0.01
laser_noise_frequency=300
laser_noise_step_duration = 1/laser_noise_frequency     # s

# Driven-dynamics study
run_driven_dynamics_study = True
driven_modulation_fraction = 0.03
driven_single_cycle_start_time = 5.0
driven_short_cycle_duration = 0.10
driven_long_cycle_duration = 2.0
driven_continuous_frequencies_hz = (1.0, 100.0, 1000.0)
driven_use_brownian_motion = False
driven_include_laser_power_noise = False
driven_save_continuous_response_metrics = True
driven_square_asymmetry_max_frequency_hz = 10.0
driven_plot_bode = False
driven_bode_start_hz = 1.0
driven_bode_end_hz = 1000.0
driven_bode_points = 500
driven_bode_power_factor_derivative_step = 1e-4
driven_plot_amplitude_sweep = False
driven_amplitude_sweep_waveform = "sine"
driven_amplitude_sweep_min_factor = 0.5
driven_amplitude_sweep_max_factor = 3.0
driven_amplitude_sweep_points = 41
driven_amplitude_sweep_settle_fraction = 0.5
driven_amplitude_sweep_settle_cycles = 80.0
driven_amplitude_sweep_natural_band_fraction = 0.12
driven_amplitude_sweep_natural_band_min_hz = 2.0
driven_plot_parametric_comparison = False
driven_parametric_comparison_frequency_factor = 2.0
driven_plot_square_frequency_amplitude_sweep = False
driven_square_frequency_sweep_start_hz = 1.0
driven_square_frequency_sweep_end_hz = 100.0
driven_square_frequency_sweep_points = 100
driven_plot_resonant_peak_trajectories = False
driven_plot_modulation_amplitude_sweep = False
driven_modulation_amplitude_sweep_frequency_factor = 2.0
driven_modulation_amplitude_sweep_min_fraction = 0.005
driven_modulation_amplitude_sweep_max_fraction = 0.10
driven_modulation_amplitude_sweep_points = 40
driven_plot_power_step_response = False
driven_step_response_power_change_fraction = 0.20
driven_step_response_start_time = 5.0
driven_step_response_hold_time = 5.0
driven_step_response_settling_fraction = 0.02
driven_step_response_settling_abs_min_um = 0.05
driven_save_current_pressure_step_response_metrics = False
driven_response_time_step_fraction = driven_modulation_fraction
driven_plot_power_sine_response = False
driven_power_sine_start_time = 5.0
driven_power_sine_period = 10.0
driven_run_pressure_step_settling_table = False
driven_pressure_step_pressures_pa = (1, 10, 100, 1000, 10000, 100000)
driven_plot_standard_driven_cases = True
driven_standard_waveforms = ("square",)
driven_plot_single_cycle_cases = False
driven_plot_frequency_sweep = False
driven_frequency_sweep_start_hz = 1.0
driven_frequency_sweep_end_hz = 200.0
driven_frequency_sweep_duration = 1000.0
driven_run_frequency_sweep_trajectory = False
driven_frequency_sweep_sampling_frequency = 5000.0
driven_save_figures = True
driven_output_directory = os.path.join(SCRIPT_DIR, "driven dynamics")
driven_output_prefix = "project_3d_cython_driven_dynamics"

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
t_end =10.0
dt_baoab = 1 / 200000
brownian_seed = 9
laser_noise_seed = 12

# Trap-loss termination
# The automatic limits are intentionally wider than the thermal motion but
# local to the trapped region. In the Cython loop this check is cheap, so the
# default checks every sample and catches high-speed escapes promptly. Set either
# manual limit to a number in metres to override the corresponding automatic value.
# In energy_barrier mode, axial loss uses the lower unstable axial equilibrium
# and the on-axis potential barrier instead of a manually chosen z distance.
# The distance/acceleration limit remains available for comparison.
# Transverse loss is also one-sided in phase space: the particle must be outside
# the x limit and still moving farther away from x equilibrium.
terminate_on_trap_loss = True
trap_loss_check_interval_steps = 1
trap_loss_axial_mode = "energy_barrier"
# Choose one of:
#   "energy_barrier"          use the lower axial saddle and potential barrier
#   "distance_acceleration"   old z_eq-z distance limit plus az < 0
trap_loss_x_outward_limit_manual = None
trap_loss_below_equilibrium_limit_manual = None
trap_loss_x_outward_beam_waists = 20.0
trap_loss_below_equilibrium_rayleigh_ranges = 3.0

# Optional z-vz capture-basin scan
# This fixes x, y, vx, and vy, then scans initial z displacement and vz0.
# Set z_phase_space_scan_only=True to make the script stop after writing the
# scan plot/CSV instead of continuing into the usual trajectory and PSD plots.
use_z_phase_space_capture_scan = False
z_phase_space_scan_only = True
z_phase_space_x_displacement = 0.0  # metres, relative to equilibrium
z_phase_space_y_displacement = 0.0  # metres, relative to equilibrium
z_phase_space_vx0 = 0.0  # m/s
z_phase_space_vy0 = 0.0  # m/s
z_phase_space_z_displacement_values_um = np.linspace(-500.0, 500.0, 301)
z_phase_space_vz0_values = np.linspace(-0.25, 0.25, 201)  # m/s
z_phase_space_test_duration = 0.25  # seconds
z_phase_space_output_png = os.path.join(
    driven_output_directory,
    "z_phase_space_capture.png",
)
z_phase_space_output_csv = os.path.join(
    driven_output_directory,
    "z_phase_space_capture.csv",
)
z_phase_space_progress_interval = 1000

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
psd_plot_max_frequency_override = None
psd_min_frequency_factor = 0.95
psd_min_value = 1e-30
psd_max_value = None
psd_segment_duration_seconds = 20.0
welch_average_segment_duration_seconds = 2.0
psd_segment_samples = None
psd_overlap_fraction = 0.5
psd_overlap_samples = None
include_z_detector_resolution_psd = True
include_analytic_langevin_validation_psd = True
detector_position_resolution_nm = 100.0
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

# Experimental first graph
plot_experimental_first_graph = True
experimental_data_npy = "/Users/josephwhitfield/Masters/Summer Project/Particle_02_WP_00.npy"
experimental_use_recorded_time = True
experimental_dt_s = 1 / 20000
experimental_adc_min = 32500.0
experimental_adc_max = 31.0
experimental_position_scale_um = 1750.0 / 6.0
experimental_plot_seconds = 5.0


def finish_plot():
    apply_plot_text_style(plt.gcf())
    if display_plots:
        plt.show()
    else:
        plt.close()


def finite_experimental_trace(time_s, position_um):
    time_s = np.asarray(time_s, dtype=float)
    position_um = np.asarray(position_um, dtype=float)
    finite_mask = np.isfinite(time_s) & np.isfinite(position_um)
    time_s = time_s[finite_mask]
    position_um = position_um[finite_mask]

    if len(time_s) < 2:
        raise ValueError("Experimental trace has fewer than two finite samples.")

    return time_s - time_s[0], position_um


def downsample_for_plot(x_values, y_values, max_points):
    stride = max(1, len(x_values) // max_points)
    return x_values[::stride], y_values[::stride]


def load_experimental_photodiode_motion():
    data = np.load(experimental_data_npy)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D experimental NPY array, got {data.shape}.")

    if data.shape[0] >= 5:
        channels = data
    elif data.shape[1] >= 5:
        channels = data.T
    else:
        raise ValueError(
            "Experimental NPY must contain four photodiode channels plus time."
        )

    sample_count = channels.shape[1]
    adc_delta = experimental_adc_min - experimental_adc_max

    channel_1 = channels[0].astype(float) - adc_delta
    channel_2 = channels[1].astype(float) - adc_delta
    channel_3 = channels[2].astype(float) - adc_delta
    channel_4 = channels[3].astype(float) - adc_delta

    if experimental_use_recorded_time:
        time_s = channels[4].astype(float) / 1e9
        time_s = time_s - time_s[0]
    else:
        time_s = np.arange(sample_count, dtype=float) * experimental_dt_s

    with np.errstate(divide="ignore", invalid="ignore"):
        vertical_um = (
            (channel_2 - channel_1)
            / (channel_1 + channel_2)
            * experimental_position_scale_um
        )
        horizontal_um = (
            (channel_4 - channel_3)
            / (channel_3 + channel_4)
            * experimental_position_scale_um
        )

    horizontal_time_s, horizontal_um = finite_experimental_trace(
        time_s,
        horizontal_um
    )
    vertical_time_s, vertical_um = finite_experimental_trace(
        time_s,
        vertical_um
    )

    return {
        "horizontal_time_s": horizontal_time_s,
        "horizontal_um": horizontal_um,
        "vertical_time_s": vertical_time_s,
        "vertical_um": vertical_um,
    }


def plot_experimental_simulation_first_graph():
    if not plot_experimental_first_graph:
        return

    try:
        experiment = load_experimental_photodiode_motion()
    except (OSError, ValueError) as exc:
        print("Skipping experimental first graph:", exc)
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    comparisons = (
        (
            axes[0],
            "Horizontal",
            experiment["horizontal_time_s"],
            experiment["horizontal_um"],
            t_baoab,
            x_baoab * 1e6,
            "experiment horizontal",
            "simulation x",
        ),
        (
            axes[1],
            "Vertical",
            experiment["vertical_time_s"],
            experiment["vertical_um"],
            t_baoab,
            z_baoab * 1e6,
            "experiment vertical",
            "simulation z",
        ),
    )

    for (
        ax,
        axis_label,
        experiment_time_s,
        experiment_position_um,
        simulation_time_s,
        simulation_position_um,
        experiment_label,
        simulation_label,
    ) in comparisons:
        experiment_displacement_um = (
            experiment_position_um - np.mean(experiment_position_um)
        )
        simulation_displacement_um = (
            simulation_position_um - np.mean(simulation_position_um)
        )

        if experimental_plot_seconds is None:
            experiment_mask = np.ones_like(experiment_time_s, dtype=bool)
            simulation_mask = np.ones_like(simulation_time_s, dtype=bool)
        else:
            experiment_mask = experiment_time_s <= experimental_plot_seconds
            simulation_mask = simulation_time_s <= experimental_plot_seconds

        if (
            np.count_nonzero(experiment_mask) < 2
            or np.count_nonzero(simulation_mask) < 2
        ):
            ax.text(
                0.5,
                0.5,
                "Not enough samples in requested plot window",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        exp_time_plot, exp_disp_plot = downsample_for_plot(
            experiment_time_s[experiment_mask],
            experiment_displacement_um[experiment_mask],
            max_plot_points,
        )
        sim_time_plot, sim_disp_plot = downsample_for_plot(
            simulation_time_s[simulation_mask],
            simulation_displacement_um[simulation_mask],
            max_plot_points,
        )

        ax.plot(
            exp_time_plot,
            exp_disp_plot,
            linewidth=0.65,
            label=experiment_label,
        )
        ax.plot(
            sim_time_plot,
            sim_disp_plot,
            linewidth=0.9,
            label=simulation_label,
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Displacement / micrometres")
        ax.set_title(f"{axis_label} time-domain comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time / s")
    fig.suptitle("Experimental data compared with BAOAB simulation")
    plt.tight_layout()
    finish_plot()


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


root_slopes = []
stable_roots = []
unstable_roots = []

for root in roots:
    slope = numerical_derivative_1d(Fz_net_on_axis, root)
    root_slopes.append((root, slope))
    if slope < 0:
        stable_roots.append(root)
    elif slope > 0:
        unstable_roots.append(root)

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

lower_axial_escape_candidates = [
    root for root in unstable_roots
    if root < z_eq
]
upper_axial_escape_candidates = [
    root for root in unstable_roots
    if root > z_eq
]
axial_lower_escape_z = (
    max(lower_axial_escape_candidates)
    if lower_axial_escape_candidates
    else None
)
axial_upper_escape_z = (
    min(upper_axial_escape_candidates)
    if upper_axial_escape_candidates
    else None
)

if trap_loss_axial_mode not in ("energy_barrier", "distance_acceleration"):
    raise ValueError(
        "trap_loss_axial_mode must be 'energy_barrier' "
        "or 'distance_acceleration'."
    )

if trap_loss_axial_mode == "energy_barrier" and axial_lower_escape_z is None:
    raise ValueError(
        "Energy-barrier axial loss needs an unstable axial equilibrium below "
        "the chosen stable equilibrium. Try widening equilibrium_z_min/max or "
        "use trap_loss_axial_mode='distance_acceleration'."
    )

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
        force_lookup_z_thermal_factor * z_rms_thermal,
        (
            abs(z_eq - axial_lower_escape_z)
            if axial_lower_escape_z is not None
            else 0.0
        ),
        (
            np.max(np.abs(z_phase_space_z_displacement_values_um)) * 1e-6
            if use_z_phase_space_capture_scan
            else 0.0
        )
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

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)

trap_loss_check_interval_steps = max(1, int(trap_loss_check_interval_steps))

trap_loss_x_outward_limit_auto = trap_loss_x_outward_beam_waists * w0
trap_loss_below_equilibrium_limit_auto = (
    trap_loss_below_equilibrium_rayleigh_ranges * zR
)

trap_loss_x_outward_limit = (
    trap_loss_x_outward_limit_auto
    if trap_loss_x_outward_limit_manual is None
    else trap_loss_x_outward_limit_manual
)
trap_loss_below_equilibrium_limit = (
    trap_loss_below_equilibrium_limit_auto
    if trap_loss_below_equilibrium_limit_manual is None
    else trap_loss_below_equilibrium_limit_manual
)

if trap_loss_x_outward_limit <= 0:
    raise ValueError("trap_loss_x_outward_limit must be positive.")

if trap_loss_below_equilibrium_limit <= 0:
    raise ValueError("trap_loss_below_equilibrium_limit must be positive.")

use_axial_energy_loss = trap_loss_axial_mode == "energy_barrier"
axial_energy_z_values = np.array([z_eq - 1.0, z_eq + 1.0], dtype=np.float64)
axial_potential_values = np.zeros_like(axial_energy_z_values)
axial_lower_escape_energy = np.inf


def cumulative_trapezoid_from_left(y_values, x_values):
    y_values = np.asarray(y_values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    areas = 0.5 * (y_values[:-1] + y_values[1:]) * np.diff(x_values)
    return np.concatenate(([0.0], np.cumsum(areas)))


def build_axial_energy_barrier():
    if not use_axial_energy_loss:
        return

    global axial_energy_z_values
    global axial_potential_values
    global axial_lower_escape_energy

    lookup_covers_axial_barrier = (
        use_force_lookup_table
        and force_lookup_ready
        and force_lookup_z_values[0] <= axial_lower_escape_z <= force_lookup_z_values[-1]
        and force_lookup_z_values[0] <= z_eq <= force_lookup_z_values[-1]
    )

    if lookup_covers_axial_barrier:
        z_values = np.ascontiguousarray(force_lookup_z_values, dtype=np.float64)
        Fz_net_values = np.asarray(force_lookup_Fz_table[0, :] - m * g, dtype=float)
    else:
        z_min = min(equilibrium_z_min, axial_lower_escape_z, z_eq)
        z_max = max(equilibrium_z_max, axial_lower_escape_z, z_eq)
        z_values = np.linspace(z_min, z_max, equilibrium_scan_points)
        Fz_net_values = np.asarray(Fz_net_on_axis(z_values), dtype=float)

    if not (z_values[0] <= axial_lower_escape_z <= z_values[-1]):
        raise ValueError(
            "The axial potential table does not include the lower escape "
            "saddle. Increase the force lookup z range."
        )

    if not (z_values[0] <= z_eq <= z_values[-1]):
        raise ValueError(
            "The axial potential table does not include the chosen equilibrium."
        )

    potential_from_left = -cumulative_trapezoid_from_left(
        Fz_net_values,
        z_values
    )
    potential_at_equilibrium = np.interp(
        z_eq,
        z_values,
        potential_from_left
    )
    axial_energy_z_values = z_values
    axial_potential_values = np.ascontiguousarray(
        potential_from_left - potential_at_equilibrium,
        dtype=np.float64
    )
    axial_lower_escape_energy = float(
        np.interp(
            axial_lower_escape_z,
            axial_energy_z_values,
            axial_potential_values
        )
    )

    if not np.isfinite(axial_lower_escape_energy) or axial_lower_escape_energy <= 0:
        raise ValueError(
            "Computed axial lower escape barrier energy is not positive. "
            "Check the force curve and equilibrium root."
        )


def axial_potential_at_z(z):
    return float(np.interp(z, axial_energy_z_values, axial_potential_values))


def axial_total_energy(z, vz):
    return 0.5 * m * vz**2 + axial_potential_at_z(z)


build_axial_energy_barrier()
axial_barrier_speed_at_equilibrium = (
    np.sqrt(2.0 * axial_lower_escape_energy / m)
    if use_axial_energy_loss
    else np.nan
)

trap_loss_events = {}


def trap_loss_reason(
    x,
    y,
    z,
    vx,
    vz,
    axial_acceleration=None,
    power_factor=1.0
):
    if not terminate_on_trap_loss:
        return None

    if not np.all(np.isfinite([x, y, z])):
        return "position became non-finite"

    if not np.all(np.isfinite([vx, vz])):
        return "velocity became non-finite"

    x_offset = x - x_eq
    below_equilibrium_displacement = z_eq - z
    if axial_acceleration is None and not use_axial_energy_loss:
        axial_acceleration = deterministic_force_no_drag_3d(
            x,
            y,
            z,
            power_factor
        )[2] / m

    if (
        axial_acceleration is not None
        and not np.isfinite(axial_acceleration)
    ):
        return "axial acceleration became non-finite"

    if (
        abs(x_offset) > trap_loss_x_outward_limit
        and x_offset * vx > 0.0
    ):
        return (
            "x displacement "
            f"{x_offset * 1e6:.3g} micrometres exceeded "
            f"{trap_loss_x_outward_limit * 1e6:.3g} micrometres "
            "while moving farther from equilibrium"
        )

    if use_axial_energy_loss:
        axial_energy = axial_total_energy(z, vz)
        if (
            (
                axial_energy >= axial_lower_escape_energy
                and vz < 0.0
            )
            or (
                z <= axial_lower_escape_z
                and vz <= 0.0
            )
        ):
            return (
                "axial energy "
                f"{axial_energy:.3e} J reached lower escape barrier "
                f"{axial_lower_escape_energy:.3e} J "
                f"at z = {z * 1e6:.3g} micrometres"
            )

    elif (
        below_equilibrium_displacement > trap_loss_below_equilibrium_limit
        and axial_acceleration < 0.0
    ):
        return (
            "distance below equilibrium "
            f"{below_equilibrium_displacement * 1e6:.3g} micrometres exceeded "
            f"{trap_loss_below_equilibrium_limit * 1e6:.3g} micrometres "
            "while accelerating farther below equilibrium"
        )

    return None


def should_check_trap_loss(sample_index):
    return (
        terminate_on_trap_loss
        and (
            sample_index % trap_loss_check_interval_steps == 0
            or sample_index == len(t_baoab) - 1
        )
    )


def record_trap_loss(run_label, sample_index, x, y, z, reason):
    trap_loss_events[run_label] = {
        "sample_index": sample_index,
        "time": t_baoab[sample_index],
        "x": x,
        "y": y,
        "z": z,
        "reason": reason,
    }
    print(
        f"{run_label}: particle lost from trap at t = "
        f"{t_baoab[sample_index]:.6g} s; {reason}."
    )


def truncate_outputs(stop_index, *arrays):
    return tuple(array[:stop_index + 1] for array in arrays)


if terminate_on_trap_loss:
    print("Trap-loss termination enabled = True")
    print(
        "Trap-loss check interval =",
        trap_loss_check_interval_steps,
        "steps =",
        trap_loss_check_interval_steps * dt_baoab,
        "s"
    )
    print(
        "Trap-loss outward x displacement limit =",
        trap_loss_x_outward_limit * 1e6,
        "micrometres"
    )
    print("Trap-loss axial mode =", trap_loss_axial_mode)
    if use_axial_energy_loss:
        print(
            "Axial lower escape saddle =",
            axial_lower_escape_z * 1e6,
            "micrometres"
        )
        print(
            "Axial lower escape barrier energy =",
            axial_lower_escape_energy,
            "J"
        )
        print(
            "Equivalent lower-barrier speed at equilibrium =",
            axial_barrier_speed_at_equilibrium,
            "m/s"
        )
    else:
        print(
            "Trap-loss below-equilibrium displacement limit =",
            trap_loss_below_equilibrium_limit * 1e6,
            "micrometres"
        )
        print("Trap-loss below-equilibrium acceleration condition = az < 0")


def axis_edges_from_centres(values):
    values = np.asarray(values, dtype=float)

    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    middle = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (middle[0] - values[0])
    last = values[-1] + (values[-1] - middle[-1])
    return np.concatenate(([first], middle, [last]))


def classify_z_phase_space_capture():
    if not cython_capture_classifier_available:
        raise RuntimeError(
            "z-vz capture scan requires classify_baoab_3d_lookup_cython. "
            "Rebuild baoab_3d_cython.pyx with: "
            "python3 setup_baoab_3d.py build_ext --inplace"
        )

    if not (use_force_lookup_table and force_lookup_ready):
        raise RuntimeError("z-vz capture scan requires use_force_lookup_table=True.")

    scan_dt = dt_baoab
    scan_steps = max(1, int(round(z_phase_space_test_duration / scan_dt)))
    scan_damping_factor = np.exp(-(b / m) * scan_dt)
    scan_x0 = x_eq + z_phase_space_x_displacement
    scan_y0 = y_eq + z_phase_space_y_displacement
    force_lookup_r_values_c = np.ascontiguousarray(
        force_lookup_r_values,
        dtype=np.float64
    )
    force_lookup_z_values_c = np.ascontiguousarray(
        force_lookup_z_values,
        dtype=np.float64
    )
    force_lookup_Fr_table_c = np.ascontiguousarray(
        force_lookup_Fr_table,
        dtype=np.float64
    )
    force_lookup_Fz_table_c = np.ascontiguousarray(
        force_lookup_Fz_table,
        dtype=np.float64
    )
    axial_energy_z_values_c = np.ascontiguousarray(
        axial_energy_z_values,
        dtype=np.float64
    )
    axial_potential_values_c = np.ascontiguousarray(
        axial_potential_values,
        dtype=np.float64
    )
    axial_lower_escape_z_c = (
        z_eq
        if axial_lower_escape_z is None
        else axial_lower_escape_z
    )

    loaded = np.zeros(
        (
            len(z_phase_space_vz0_values),
            len(z_phase_space_z_displacement_values_um),
        ),
        dtype=bool
    )
    stop_time = np.full(loaded.shape, scan_steps * scan_dt, dtype=float)

    total_points = loaded.size
    completed_points = 0
    scan_start_time = perf_counter()

    for vz_index, vz0_scan in enumerate(z_phase_space_vz0_values):
        for z_index, z_displacement_um in enumerate(z_phase_space_z_displacement_values_um):
            scan_z0 = z_eq + z_displacement_um * 1e-6

            (
                lost_flag,
                stop_index,
                _x_final,
                _y_final,
                _z_final,
                _vx_final,
                _vy_final,
                _vz_final,
                _out_of_bounds_count,
            ) = classify_baoab_3d_lookup_cython(
                force_lookup_r_values_c,
                force_lookup_z_values_c,
                force_lookup_Fr_table_c,
                force_lookup_Fz_table_c,
                scan_steps,
                scan_dt,
                m,
                m * g,
                scan_damping_factor,
                scan_x0,
                scan_y0,
                scan_z0,
                z_phase_space_vx0,
                z_phase_space_vy0,
                float(vz0_scan),
                1.0,
                radial_zero_tolerance,
                use_axial_energy_loss,
                axial_energy_z_values_c,
                axial_potential_values_c,
                axial_lower_escape_z_c,
                axial_lower_escape_energy,
                trap_loss_check_interval_steps,
                x_eq,
                y_eq,
                z_eq,
                trap_loss_x_outward_limit,
                trap_loss_below_equilibrium_limit,
            )

            loaded[vz_index, z_index] = lost_flag == 0
            stop_time[vz_index, z_index] = stop_index * scan_dt

            completed_points += 1
            if (
                z_phase_space_progress_interval > 0
                and (
                    completed_points % z_phase_space_progress_interval == 0
                    or completed_points == total_points
                )
            ):
                elapsed = perf_counter() - scan_start_time
                print(
                    "Classified",
                    completed_points,
                    "/",
                    total_points,
                    "z-vz scan points in",
                    f"{elapsed:.1f}",
                    "s"
                )

    return loaded, stop_time, scan_dt, scan_steps


def write_z_phase_space_capture_csv(loaded, stop_time):
    os.makedirs(driven_output_directory, exist_ok=True)
    with open(z_phase_space_output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "z_displacement_um",
                "vz0_m_per_s",
                "loaded",
                "stop_time_s",
            ]
        )

        for vz_index, vz0_scan in enumerate(z_phase_space_vz0_values):
            for z_index, z_displacement_um in enumerate(z_phase_space_z_displacement_values_um):
                writer.writerow(
                    [
                        z_displacement_um,
                        vz0_scan,
                        int(loaded[vz_index, z_index]),
                        stop_time[vz_index, z_index],
                    ]
                )


def plot_z_phase_space_capture(loaded, scan_dt, scan_steps):
    os.makedirs(driven_output_directory, exist_ok=True)
    z_edges = axis_edges_from_centres(z_phase_space_z_displacement_values_um)
    vz_edges = axis_edges_from_centres(z_phase_space_vz0_values)
    if use_axial_energy_loss:
        loss_annotation = (
            f"z_eq = {z_eq * 1e6:.3g} um\n"
            f"lower saddle = {axial_lower_escape_z * 1e6:.3g} um\n"
            f"Ebar speed = {axial_barrier_speed_at_equilibrium:.3g} m/s"
        )
    else:
        loss_annotation = (
            f"z_eq = {z_eq * 1e6:.3g} um\n"
            f"loss: z_eq-z > {trap_loss_below_equilibrium_limit * 1e6:.3g} um\n"
            "and az < 0"
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cmap = ListedColormap(["#d9d9d9", "#1b9e77"])
    mesh = ax.pcolormesh(
        z_edges,
        vz_edges,
        loaded.astype(float),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat"
    )

    ax.contour(
        z_phase_space_z_displacement_values_um,
        z_phase_space_vz0_values,
        loaded.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.2
    )

    cbar = fig.colorbar(mesh, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["lost", "loaded"])

    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Initial z displacement from equilibrium / micrometres")
    ax.set_ylabel("Initial axial velocity vz0 / m s$^{-1}$")
    ax.set_title(
        "Axial capture basin "
        f"(x={z_phase_space_x_displacement * 1e6:g} um, "
        f"vx={z_phase_space_vx0:g} m/s)"
    )
    ax.text(
        0.02,
        0.02,
        (
            f"duration = {scan_steps * scan_dt:.3g} s\n"
            f"{loss_annotation}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )
    ax.grid(color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(z_phase_space_output_png, dpi=300)
    finish_plot()


if use_z_phase_space_capture_scan:
    loaded_z_phase_space, z_phase_space_stop_time, scan_dt, scan_steps = (
        classify_z_phase_space_capture()
    )
    write_z_phase_space_capture_csv(
        loaded_z_phase_space,
        z_phase_space_stop_time
    )
    plot_z_phase_space_capture(
        loaded_z_phase_space,
        scan_dt,
        scan_steps
    )

    print("Wrote z-vz capture plot to", z_phase_space_output_png)
    print("Wrote z-vz capture data to", z_phase_space_output_csv)
    print("Loaded fraction =", np.mean(loaded_z_phase_space))

    if z_phase_space_scan_only:
        print("z-vz capture scan complete.")
        raise SystemExit(0)

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

    laser_power_factor = np.repeat(laser_step_power_factors, laser_step_samples)[:len(t_baoab)]
else:
    laser_step_samples = len(t_baoab)
    laser_step_duration_actual = t_baoab[-1] - t_baoab[0]
    laser_allowed_power_factors = np.array([1.0])
    laser_step_power_factors = np.array([1.0])
    laser_power_factor = np.ones_like(t_baoab)

laser_power_time = P_laser * laser_power_factor

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

z_eq_laser_noise_time = np.array(
    [
        instantaneous_z_equilibrium_by_power_factor[power_factor_value]
        for power_factor_value in laser_power_factor
    ]
)

gamma_baoab = b / m
baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
baoab_thermal_velocity_scale = np.sqrt(
    (kB * T / m)
    * (1 - baoab_damping_factor**2)
)

print("Brownian bath temperature =", T, "K")
print("Trajectory damping rate gamma =", gamma_baoab, "s^-1")


def update_baoab_damping_for_pressure(pressure):
    global p
    global b_cunningham
    global b_epstein
    global b
    global drag_model_used
    global gamma_baoab
    global baoab_damping_factor
    global baoab_thermal_velocity_scale

    p = pressure
    b_cunningham = damping_coefficient_stokes_cunningham(pressure)
    b_epstein = damping_coefficient_epstein(pressure)
    b, drag_model_used = damping_coefficient(pressure, drag_model)
    gamma_baoab = b / m
    baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
    baoab_thermal_velocity_scale = np.sqrt(
        (kB * T / m)
        * (1 - baoab_damping_factor**2)
    )


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
    brownian_normals_z,
    run_label="BAOAB"
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

    initial_loss_reason = trap_loss_reason(
        x_out[0],
        y_out[0],
        z_out[0],
        vx_out[0],
        vz_out[0],
        power_factor=power_factor_time[0]
    )
    if initial_loss_reason is not None:
        record_trap_loss(
            run_label,
            0,
            x_out[0],
            y_out[0],
            z_out[0],
            initial_loss_reason
        )
        return truncate_outputs(
            0,
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out
        )

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

        sample_index = i + 1
        if should_check_trap_loss(sample_index):
            reason = trap_loss_reason(
                x_i,
                y_i,
                z_i,
                vx_i,
                vz_i,
                axial_acceleration=Fz_i / m,
                power_factor=power_factor_i
            )
            if reason is not None:
                record_trap_loss(run_label, sample_index, x_i, y_i, z_i, reason)
                return truncate_outputs(
                    sample_index,
                    x_out,
                    y_out,
                    z_out,
                    vx_out,
                    vy_out,
                    vz_out
                )

    return x_out, y_out, z_out, vx_out, vy_out, vz_out


def solve_baoab_3d_fast_with_power(
    power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z,
    run_label="BAOAB"
):
    """
    Use the compiled Cython BAOAB loop when available.

    If the Cython extension has not been built yet, this falls back to the
    original Python implementation so the script remains runnable.
    """
    global cython_baoab_available

    if cython_baoab_available and use_force_lookup_table and force_lookup_ready:
        try:
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
                use_axial_energy_loss,
                np.ascontiguousarray(axial_energy_z_values, dtype=np.float64),
                np.ascontiguousarray(axial_potential_values, dtype=np.float64),
                z_eq if axial_lower_escape_z is None else axial_lower_escape_z,
                axial_lower_escape_energy,
                terminate_on_trap_loss,
                trap_loss_check_interval_steps,
                x_eq,
                y_eq,
                z_eq,
                trap_loss_x_outward_limit,
                trap_loss_below_equilibrium_limit,
            )
        except TypeError as exc:
            print(
                "Compiled Cython BAOAB solver has an incompatible signature; "
                "using the Python solver instead.",
                exc,
            )
            cython_baoab_available = False
            return solve_baoab_3d_with_power(
                power_factor_time,
                brownian_normals_x,
                brownian_normals_y,
                brownian_normals_z,
                run_label
            )

        (
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out,
            out_of_bounds_count,
            stop_index,
            lost_flag,
        ) = result

        if lost_flag:
            reason = trap_loss_reason(
                x_out[stop_index],
                y_out[stop_index],
                z_out[stop_index],
                vx_out[stop_index],
                vz_out[stop_index],
                power_factor=power_factor_time[stop_index]
            )
            if reason is None:
                reason = "trap-loss boundary exceeded"
            record_trap_loss(
                run_label,
                stop_index,
                x_out[stop_index],
                y_out[stop_index],
                z_out[stop_index],
                reason
            )
            return truncate_outputs(
                stop_index,
                x_out,
                y_out,
                z_out,
                vx_out,
                vy_out,
                vz_out
            )

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
        brownian_normals_z,
        run_label
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
    brownian_normals_z,
    run_label="BAOAB with PD feedback"
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

    def build_feedback_result(stop_index):
        result_slice = slice(None, stop_index + 1)
        return {
            "t": t_baoab[result_slice],
            "x": x_out[result_slice],
            "y": y_out[result_slice],
            "z": z_out[result_slice],
            "vx": vx_out[result_slice],
            "vy": vy_out[result_slice],
            "vz": vz_out[result_slice],
            "command_power": command_power_time[result_slice],
            "actual_power": actual_power_time[result_slice],
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

    initial_loss_reason = trap_loss_reason(
        x_out[0],
        y_out[0],
        z_out[0],
        vx_out[0],
        vz_out[0],
        power_factor=actual_power_time[0] / P_laser
    )
    if initial_loss_reason is not None:
        record_trap_loss(
            run_label,
            0,
            x_out[0],
            y_out[0],
            z_out[0],
            initial_loss_reason
        )
        return build_feedback_result(0)

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
        command_power_time[i + 1] = command_power
        actual_power_time[i + 1] = P_laser * actual_power_factor

        sample_index = i + 1
        if should_check_trap_loss(sample_index):
            reason = trap_loss_reason(
                x_i,
                y_i,
                z_i,
                vx_i,
                vz_i,
                axial_acceleration=Fz_i / m,
                power_factor=actual_power_factor
            )
            if reason is not None:
                record_trap_loss(run_label, sample_index, x_i, y_i, z_i, reason)
                return build_feedback_result(sample_index)

    command_power_time[-1] = command_power
    actual_power_time[-1] = actual_power_time[-2]

    return build_feedback_result(len(t_baoab) - 1)

def solve_baoab_3d_fast_with_pd_feedback(
    disturbance_power_factor_time,
    brownian_normals_x,
    brownian_normals_y,
    brownian_normals_z,
    run_label="BAOAB with PD feedback"
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
            use_axial_energy_loss,
            np.ascontiguousarray(axial_energy_z_values, dtype=np.float64),
            np.ascontiguousarray(axial_potential_values, dtype=np.float64),
            z_eq if axial_lower_escape_z is None else axial_lower_escape_z,
            axial_lower_escape_energy,
            terminate_on_trap_loss,
            trap_loss_check_interval_steps,
            x_eq,
            y_eq,
            trap_loss_x_outward_limit,
            trap_loss_below_equilibrium_limit,
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
            stop_index,
            lost_flag,
        ) = result

        if lost_flag:
            reason = trap_loss_reason(
                x_out[stop_index],
                y_out[stop_index],
                z_out[stop_index],
                vx_out[stop_index],
                vz_out[stop_index],
                power_factor=actual_power_time[stop_index] / P_laser
            )
            if reason is None:
                reason = "trap-loss boundary exceeded"
            record_trap_loss(
                run_label,
                stop_index,
                x_out[stop_index],
                y_out[stop_index],
                z_out[stop_index],
                reason
            )
            result_slice = slice(None, stop_index + 1)
            x_out = x_out[result_slice]
            y_out = y_out[result_slice]
            z_out = z_out[result_slice]
            vx_out = vx_out[result_slice]
            vy_out = vy_out[result_slice]
            vz_out = vz_out[result_slice]
            command_power_time = command_power_time[result_slice]
            actual_power_time = actual_power_time[result_slice]

        return {
            "t": t_baoab[:len(x_out)],
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
        brownian_normals_z,
        run_label
    )


def driven_power_factor_time(
    waveform,
    frequency_hz,
    single_cycle=False,
    include_drive=True,
    include_laser_noise=None,
    modulation_fraction=None,
):
    if include_laser_noise is None:
        include_laser_noise = driven_include_laser_power_noise
    if modulation_fraction is None:
        modulation_fraction = driven_modulation_fraction

    phase_time = t_baoab.copy()

    if single_cycle:
        elapsed = t_baoab - driven_single_cycle_start_time
        active = (elapsed >= 0.0) & (elapsed < 1.0 / frequency_hz)
        phase_time = elapsed
    else:
        elapsed = t_baoab - driven_single_cycle_start_time
        active = elapsed >= 0.0
        phase_time = elapsed

    phase = 2.0 * np.pi * frequency_hz * phase_time
    power_factor = np.ones_like(t_baoab)

    if waveform == "sine":
        modulation = np.sin(phase)
    elif waveform == "square":
        cycle_fraction = (frequency_hz * phase_time) % 1.0
        modulation = np.where(cycle_fraction < 0.5, 1.0, -1.0)
    else:
        raise ValueError(f"Unknown driven waveform: {waveform}")

    if include_drive:
        power_factor[active] = 1.0 + modulation_fraction * modulation[active]

    if include_laser_noise:
        power_factor = power_factor * laser_power_factor

    return power_factor


def positive_driven_psd(signal, time_values):
    dt = time_values[1] - time_values[0]
    fs = 1.0 / dt
    signal = signal - np.mean(signal)

    nperseg = min(
        max(2, int(round(psd_segment_duration_seconds * fs))),
        len(signal)
    )
    noverlap = 0 if nperseg == len(signal) else min(
        int(round(psd_overlap_fraction * nperseg)),
        nperseg - 1
    )

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True
    )

    positive_mask = frequencies > 0.0
    return frequencies[positive_mask], np.maximum(psd[positive_mask], psd_min_value)


def driven_brownian_normals(use_brownian_motion=None):
    if use_brownian_motion is None:
        use_brownian_motion = driven_use_brownian_motion

    if use_brownian_motion:
        return brownian_normals_x, brownian_normals_y, brownian_normals_z

    return zero_brownian_normals, zero_brownian_normals, zero_brownian_normals


def run_driven_case(
    waveform,
    frequency_hz,
    single_cycle=False,
    label=None,
    include_drive=True,
    use_brownian_motion=None,
    include_laser_noise=None,
    modulation_fraction=None,
):
    power_factor = driven_power_factor_time(
        waveform,
        frequency_hz,
        single_cycle=single_cycle,
        include_drive=include_drive,
        include_laser_noise=include_laser_noise,
        modulation_fraction=modulation_fraction,
    )
    normals_x, normals_y, normals_z = driven_brownian_normals(use_brownian_motion)

    x_out, y_out, z_out, vx_out, vy_out, vz_out = solve_baoab_3d_fast_with_power(
        power_factor,
        normals_x,
        normals_y,
        normals_z,
        label or f"{waveform} {frequency_hz:g} Hz drive",
    )

    case_sample_count = min(len(t_baoab), len(z_out), len(power_factor))
    time_out = t_baoab[:case_sample_count]
    z_displacement = z_out[:case_sample_count] - z_eq
    frequencies, psd = positive_driven_psd(z_displacement, time_out)

    return {
        "time": time_out,
        "power_factor": power_factor[:case_sample_count],
        "z_displacement": z_displacement,
        "frequencies": frequencies,
        "psd": psd,
    }


def apply_driven_psd_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("z PSD / m^2 Hz^-1")
    ax.grid(True, which="both", alpha=0.35)


def save_and_finish_driven_figure(fig, filename, tight_layout_rect=None):
    if tight_layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_layout_rect)
    if driven_save_figures:
        os.makedirs(driven_output_directory, exist_ok=True)
        output_path = os.path.join(driven_output_directory, filename)
        fig.savefig(output_path, dpi=200)
        print("Wrote", output_path)
    finish_plot()


def plot_driven_trajectory(ax, time_values, z_displacement):
    plot_time, plot_z = downsample_for_plot(
        time_values,
        z_displacement * 1e6,
        max_plot_points,
    )
    ax.plot(plot_time, plot_z, linewidth=0.85, color="tab:blue")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("z displacement / micrometres")
    ax.grid(True, alpha=0.35)


def coherent_component(time_values, signal_values, frequency_hz, start_time):
    active = time_values >= start_time
    if np.count_nonzero(active) < 2:
        return np.nan + 1j * np.nan

    time_active = time_values[active] - start_time
    signal_active = signal_values[active] - np.mean(signal_values[active])
    reference = np.exp(-1j * 2.0 * np.pi * frequency_hz * time_active)
    return np.mean(signal_active * reference)


def phase_delay_from_complex_ratio(response_over_drive, frequency_hz):
    phase_lag_rad = np.angle(response_over_drive)
    phase_delay_s = -phase_lag_rad / (2.0 * np.pi * frequency_hz)

    period = 1.0 / frequency_hz
    if phase_delay_s < 0.0:
        phase_delay_s += period

    return phase_lag_rad * 180.0 / np.pi, phase_delay_s


def linearized_response_time_metrics(frequency_hz):
    response = axial_bode_response(np.array([frequency_hz], dtype=float))[0]
    phase_lag_deg = np.angle(response) * 180.0 / np.pi
    phase_delay_s = -np.angle(response) / (2.0 * np.pi * frequency_hz)
    if phase_delay_s < 0.0:
        phase_delay_s += 1.0 / frequency_hz

    return {
        "linearized_magnitude_nm_per_1_percent": abs(response) * 0.01 * 1e9,
        "linearized_phase_lag_deg": phase_lag_deg,
        "linearized_phase_delay_s": phase_delay_s,
    }


def median_signal_level(time_values, signal_values, start_time, end_time):
    mask = (time_values >= start_time) & (time_values <= end_time)
    if np.count_nonzero(mask) == 0:
        return np.nan
    return np.median(signal_values[mask])


def first_threshold_crossing_time(
    time_values,
    signal_values,
    start_time,
    end_time,
    threshold,
    direction,
):
    mask = (time_values >= start_time) & (time_values <= end_time)
    if np.count_nonzero(mask) < 2:
        return np.nan

    segment_time = time_values[mask]
    segment_signal = signal_values[mask]
    if direction == "up":
        crossed = segment_signal >= threshold
    elif direction == "down":
        crossed = segment_signal <= threshold
    else:
        raise ValueError("direction must be 'up' or 'down'.")

    crossing_indices = np.flatnonzero(crossed)
    if len(crossing_indices) == 0:
        return np.nan

    index = crossing_indices[0]
    if index == 0:
        return segment_time[index]

    t0 = segment_time[index - 1]
    t1 = segment_time[index]
    y0 = segment_signal[index - 1]
    y1 = segment_signal[index]

    if y1 == y0:
        return t1

    fraction = (threshold - y0) / (y1 - y0)
    return t0 + np.clip(fraction, 0.0, 1.0) * (t1 - t0)


def square_wave_cycle_asymmetry_rows(time_values, z_displacement, frequency_hz):
    time_values = np.asarray(time_values, dtype=float)
    z_displacement = np.asarray(z_displacement, dtype=float)
    finite_mask = np.isfinite(time_values) & np.isfinite(z_displacement)
    time_values = time_values[finite_mask]
    z_displacement = z_displacement[finite_mask]

    if (
        frequency_hz <= 0
        or frequency_hz > driven_square_asymmetry_max_frequency_hz
        or len(time_values) < 4
    ):
        return []

    period = 1.0 / frequency_hz
    half_period = 0.5 * period
    level_window = 0.25 * half_period
    first_repeated_cycle = 1
    final_time = time_values[-1]
    rows = []

    cycle_index = first_repeated_cycle
    while True:
        rise_start = driven_single_cycle_start_time + cycle_index * period
        fall_start = rise_start + half_period
        next_rise_start = rise_start + period

        if next_rise_start > final_time:
            break

        low_before = median_signal_level(
            time_values,
            z_displacement,
            rise_start - level_window,
            rise_start,
        )
        high_level = median_signal_level(
            time_values,
            z_displacement,
            fall_start - level_window,
            fall_start,
        )
        low_after = median_signal_level(
            time_values,
            z_displacement,
            next_rise_start - level_window,
            next_rise_start,
        )

        if not (
            np.isfinite(low_before)
            and np.isfinite(high_level)
            and np.isfinite(low_after)
        ):
            cycle_index += 1
            continue

        rise_direction = "up" if high_level >= low_before else "down"
        fall_direction = "down" if low_after <= high_level else "up"

        rise_10 = low_before + 0.10 * (high_level - low_before)
        rise_90 = low_before + 0.90 * (high_level - low_before)
        fall_90 = low_after + 0.90 * (high_level - low_after)
        fall_10 = low_after + 0.10 * (high_level - low_after)

        rise_10_time = first_threshold_crossing_time(
            time_values,
            z_displacement,
            rise_start,
            fall_start,
            rise_10,
            rise_direction,
        )
        rise_90_time = first_threshold_crossing_time(
            time_values,
            z_displacement,
            rise_start,
            fall_start,
            rise_90,
            rise_direction,
        )
        fall_90_time = first_threshold_crossing_time(
            time_values,
            z_displacement,
            fall_start,
            next_rise_start,
            fall_90,
            fall_direction,
        )
        fall_10_time = first_threshold_crossing_time(
            time_values,
            z_displacement,
            fall_start,
            next_rise_start,
            fall_10,
            fall_direction,
        )

        rise_time = rise_90_time - rise_10_time
        fall_time = fall_10_time - fall_90_time
        if not np.isfinite(rise_time) or rise_time < 0:
            rise_time = np.nan
        if not np.isfinite(fall_time) or fall_time < 0:
            fall_time = np.nan

        reference_midpoint = 0.5 * (high_level + low_after)
        high_mask = (time_values >= rise_start) & (time_values < fall_start)
        low_mask = (time_values >= fall_start) & (time_values < next_rise_start)
        if np.count_nonzero(high_mask) >= 2 and np.count_nonzero(low_mask) >= 2:
            upper_area = np.trapz(
                np.maximum(z_displacement[high_mask] - reference_midpoint, 0.0),
                time_values[high_mask],
            )
            lower_area = np.trapz(
                np.maximum(reference_midpoint - z_displacement[low_mask], 0.0),
                time_values[low_mask],
            )
        else:
            upper_area = np.nan
            lower_area = np.nan

        area_sum = upper_area + lower_area
        if np.isfinite(area_sum) and area_sum > 0:
            area_asymmetry = (upper_area - lower_area) / area_sum
        else:
            area_asymmetry = np.nan

        if np.isfinite(rise_time) and np.isfinite(fall_time) and (rise_time + fall_time) > 0:
            time_asymmetry = (rise_time - fall_time) / (rise_time + fall_time)
            rise_fall_ratio = rise_time / fall_time if fall_time > 0 else np.nan
        else:
            time_asymmetry = np.nan
            rise_fall_ratio = np.nan

        rows.append(
            {
                "cycle_index": cycle_index,
                "cycle_start_s": rise_start,
                "rise_time_10_90_s": rise_time,
                "fall_time_90_10_s": fall_time,
                "rise_fall_time_ratio": rise_fall_ratio,
                "time_asymmetry": time_asymmetry,
                "upper_half_area_um_s": upper_area * 1e6,
                "lower_half_area_um_s": lower_area * 1e6,
                "area_asymmetry": area_asymmetry,
                "low_before_um": low_before * 1e6,
                "high_level_um": high_level * 1e6,
                "low_after_um": low_after * 1e6,
            }
        )
        cycle_index += 1

    return rows


def summarize_square_wave_cycle_asymmetry(time_values, z_displacement, frequency_hz):
    rows = square_wave_cycle_asymmetry_rows(
        time_values,
        z_displacement,
        frequency_hz,
    )

    summary = {
        "cycle_count_for_asymmetry": len(rows),
        "mean_rise_time_10_90_s": np.nan,
        "mean_fall_time_90_10_s": np.nan,
        "mean_rise_fall_time_ratio": np.nan,
        "mean_time_asymmetry": np.nan,
        "mean_area_asymmetry": np.nan,
    }

    if not rows:
        return summary, rows

    for output_key, row_key in (
        ("mean_rise_time_10_90_s", "rise_time_10_90_s"),
        ("mean_fall_time_90_10_s", "fall_time_90_10_s"),
        ("mean_rise_fall_time_ratio", "rise_fall_time_ratio"),
        ("mean_time_asymmetry", "time_asymmetry"),
        ("mean_area_asymmetry", "area_asymmetry"),
    ):
        values = np.array([row[row_key] for row in rows], dtype=float)
        if np.any(np.isfinite(values)):
            summary[output_key] = np.nanmean(values)

    return summary, rows


def continuous_response_metrics(waveform, frequency_hz, result):
    time_values = result["time"]
    z_displacement = result["z_displacement"]
    power_factor = result["power_factor"]
    active = time_values >= driven_single_cycle_start_time
    asymmetry_summary, asymmetry_cycle_rows = (
        summarize_square_wave_cycle_asymmetry(
            time_values,
            z_displacement,
            frequency_hz,
        )
        if waveform == "square"
        else ({}, [])
    )
    result["asymmetry_cycle_rows"] = asymmetry_cycle_rows

    if np.count_nonzero(active) < 2:
        row = {
            "waveform": waveform,
            "drive_frequency_hz": frequency_hz,
            "pressure_pa": p,
            "z_rms_after_drive_um": np.nan,
            "coherent_z_amplitude_um": np.nan,
            "coherent_power_factor_amplitude": np.nan,
            "response_um_per_percent_power": np.nan,
            "phase_lag_deg": np.nan,
            "phase_delay_s": np.nan,
        }
        row.update(asymmetry_summary)
        return row

    z_component = coherent_component(
        time_values,
        z_displacement,
        frequency_hz,
        driven_single_cycle_start_time,
    )
    power_component = coherent_component(
        time_values,
        power_factor,
        frequency_hz,
        driven_single_cycle_start_time,
    )

    z_amplitude_um = 2.0 * abs(z_component) * 1e6
    power_factor_amplitude = 2.0 * abs(power_component)
    if power_factor_amplitude > 0.0:
        response_um_per_percent_power = z_amplitude_um / (
            100.0 * power_factor_amplitude
        )
        phase_lag_deg, phase_delay_s = phase_delay_from_complex_ratio(
            z_component / power_component,
            frequency_hz,
        )
    else:
        response_um_per_percent_power = np.nan
        phase_lag_deg = np.nan
        phase_delay_s = np.nan

    row = {
        "waveform": waveform,
        "drive_frequency_hz": frequency_hz,
        "pressure_pa": p,
        "z_rms_after_drive_um": np.std(z_displacement[active], ddof=1) * 1e6,
        "coherent_z_amplitude_um": z_amplitude_um,
        "coherent_power_factor_amplitude": power_factor_amplitude,
        "response_um_per_percent_power": response_um_per_percent_power,
        "phase_lag_deg": phase_lag_deg,
        "phase_delay_s": phase_delay_s,
    }
    row.update(linearized_response_time_metrics(frequency_hz))
    row.update(asymmetry_summary)
    return row


def save_rows_csv(output_path, rows):
    if not rows:
        return

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", output_path)


def print_square_wave_asymmetry_summary(metric_rows):
    asymmetry_rows = [
        row
        for row in metric_rows
        if row.get("waveform") == "square"
        and row.get("cycle_count_for_asymmetry", 0) > 0
    ]
    if not asymmetry_rows:
        return

    print("\nSquare-wave peak-shape asymmetry")
    print(
        f"{'f / Hz':>10}"
        f"{'cycles':>10}"
        f"{'rise 10-90 / ms':>18}"
        f"{'fall 90-10 / ms':>18}"
        f"{'rise/fall':>12}"
        f"{'A_time':>12}"
        f"{'A_area':>12}"
    )
    for row in asymmetry_rows:
        print(
            f"{row['drive_frequency_hz']:>10.5g}"
            f"{row['cycle_count_for_asymmetry']:>10d}"
            f"{row['mean_rise_time_10_90_s'] * 1e3:>18.5g}"
            f"{row['mean_fall_time_90_10_s'] * 1e3:>18.5g}"
            f"{row['mean_rise_fall_time_ratio']:>12.5g}"
            f"{row['mean_time_asymmetry']:>12.5g}"
            f"{row['mean_area_asymmetry']:>12.5g}"
        )


def plot_single_cycle_driven_summary(waveform):
    cycle_specs = (
        ("short", driven_short_cycle_duration),
        ("long", driven_long_cycle_duration),
    )
    cases = []

    for cycle_label, cycle_duration in cycle_specs:
        frequency_hz = 1.0 / cycle_duration
        cases.append(
            (
                cycle_label,
                cycle_duration,
                run_driven_case(
                    waveform,
                    frequency_hz,
                    single_cycle=True,
                    label=(
                        f"{waveform} single {cycle_label} cycle "
                        f"({cycle_duration:g} s)"
                    ),
                ),
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    for row_index, (cycle_label, cycle_duration, result) in enumerate(cases):
        trajectory_ax = axes[row_index, 0]
        psd_ax = axes[row_index, 1]
        plot_driven_trajectory(
            trajectory_ax,
            result["time"],
            result["z_displacement"],
        )
        trajectory_ax.axvline(
            driven_single_cycle_start_time,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        trajectory_ax.axvspan(
            driven_single_cycle_start_time,
            driven_single_cycle_start_time + cycle_duration,
            color="tab:orange",
            alpha=0.12,
        )
        trajectory_ax.set_title(
            f"{cycle_label.capitalize()} {waveform} cycle trajectory"
        )

        psd_ax.loglog(result["frequencies"], result["psd"], linewidth=1.0)
        psd_ax.set_title(f"{cycle_label.capitalize()} {waveform} cycle PSD")
        apply_driven_psd_axes(psd_ax)

    fig.suptitle(
        (
            f"Single-cycle {waveform} laser-power drive "
            f"{driven_modulation_fraction * 100:g}% amplitude, "
            f"starts at {driven_single_cycle_start_time:g} s)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_single_cycle_{waveform}.png",
    )


def plot_continuous_driven_summary(waveform, frequencies_hz, group_label):
    fig, axes = plt.subplots(
        len(frequencies_hz),
        2,
        figsize=(12, 2.6 * len(frequencies_hz)),
        squeeze=False,
    )
    results = []
    metric_rows = []
    noise_summary = (
        "Brownian + laser noise"
        if driven_use_brownian_motion and driven_include_laser_power_noise
        else "Brownian only"
        if driven_use_brownian_motion
        else "laser noise only"
        if driven_include_laser_power_noise
        else "no noise"
    )

    for row_index, frequency_hz in enumerate(frequencies_hz):
        trajectory_ax = axes[row_index, 0]
        psd_ax = axes[row_index, 1]
        result = run_driven_case(
            waveform,
            frequency_hz,
            single_cycle=False,
            label=f"{waveform} continuous {frequency_hz:g} Hz",
        )
        results.append((frequency_hz, result))
        metric_rows.append(continuous_response_metrics(waveform, frequency_hz, result))
        plot_driven_trajectory(
            trajectory_ax,
            result["time"],
            result["z_displacement"],
        )
        psd_ax.loglog(
            result["frequencies"],
            result["psd"],
            linewidth=1.0,
        )

        trajectory_ax.axvline(
            driven_single_cycle_start_time,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        trajectory_ax.set_title(f"{frequency_hz:g} Hz trajectory")

        psd_ax.set_title(f"{frequency_hz:g} Hz PSD")
        apply_driven_psd_axes(psd_ax)

    fig.suptitle(
        (
            f"Continuous {waveform} laser-power drive "
            f"({group_label} frequencies, "
            f"{driven_modulation_fraction * 100:g}% amplitude, "
            f"starts at {driven_single_cycle_start_time:g} s, {noise_summary})"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_continuous_{waveform}_{group_label}.png",
    )

    if driven_save_figures and driven_save_continuous_response_metrics:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_continuous_{waveform}_{group_label}_metrics.csv",
        )
        save_rows_csv(output_path, metric_rows)
        cycle_rows = []
        for frequency_hz, result in results:
            for cycle_row in result.get("asymmetry_cycle_rows", []):
                row = {
                    "waveform": waveform,
                    "drive_frequency_hz": frequency_hz,
                    "pressure_pa": p,
                }
                row.update(cycle_row)
                cycle_rows.append(row)

        if cycle_rows:
            cycle_output_path = os.path.join(
                driven_output_directory,
                (
                    f"{driven_output_prefix}_continuous_{waveform}_"
                    f"{group_label}_cycle_asymmetry.csv"
                ),
            )
            save_rows_csv(cycle_output_path, cycle_rows)
        print_square_wave_asymmetry_summary(metric_rows)

    return results


def plot_high_frequency_zoomed_trajectories(waveform, high_frequency_results):
    fig, axes = plt.subplots(
        len(high_frequency_results),
        1,
        figsize=(9, 2.4 * len(high_frequency_results)),
        squeeze=False,
    )
    zoom_start = driven_single_cycle_start_time

    for row_index, (frequency_hz, result) in enumerate(high_frequency_results):
        ax = axes[row_index, 0]
        zoom_end = zoom_start + 10.0 / frequency_hz
        zoom_mask = (result["time"] >= zoom_start) & (result["time"] <= zoom_end)
        plot_driven_trajectory(
            ax,
            result["time"][zoom_mask],
            result["z_displacement"][zoom_mask],
        )
        ax.axvline(zoom_start, color="black", linestyle=":", linewidth=1.0)
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_title(f"{frequency_hz:g} Hz trajectory, zoomed to 10 cycles")

    fig.suptitle(
        f"Zoomed high-frequency continuous {waveform} drive trajectories"
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_continuous_{waveform}_high_zoomed_trajectories.png",
    )


def driven_noise_description():
    enabled_noise = []
    if driven_use_brownian_motion:
        enabled_noise.append("Brownian motion")
    if driven_include_laser_power_noise:
        enabled_noise.append("laser-power noise")

    if enabled_noise:
        return "with " + " and ".join(enabled_noise)

    return "deterministic, no Brownian motion or laser-power noise"


def axial_drive_force_per_power_factor():
    h = driven_bode_power_factor_derivative_step
    _, Fz_plus = F_optical_2d(0.0, z_eq, power_factor=1.0 + h)
    _, Fz_minus = F_optical_2d(0.0, z_eq, power_factor=1.0 - h)
    return (Fz_plus - Fz_minus) / (2.0 * h)


def axial_bode_response(frequencies_hz):
    omega = 2.0 * np.pi * frequencies_hz
    drive_force = axial_drive_force_per_power_factor()
    denominator = kz - m * omega**2 + 1j * b * omega
    return drive_force / denominator


def plot_axial_bode_response():
    if not driven_plot_bode:
        return

    if driven_bode_start_hz <= 0 or driven_bode_end_hz <= driven_bode_start_hz:
        raise ValueError(
            "Bode frequency range must satisfy "
            "0 < driven_bode_start_hz < driven_bode_end_hz."
        )

    frequencies_hz = np.logspace(
        np.log10(driven_bode_start_hz),
        np.log10(driven_bode_end_hz),
        driven_bode_points,
    )
    response = axial_bode_response(frequencies_hz)
    magnitude_nm_per_percent = np.abs(response) * 0.01 * 1e9
    phase_degrees = np.unwrap(np.angle(response)) * 180.0 / np.pi
    natural_frequency_hz = omega_z / (2.0 * np.pi)
    damping_ratio_z = b / (2.0 * np.sqrt(m * kz))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].loglog(
        frequencies_hz,
        magnitude_nm_per_percent,
        color="tab:blue",
        linewidth=1.5,
    )
    axes[0].axvline(
        natural_frequency_hz,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"fz = {natural_frequency_hz:g} Hz",
    )
    axes[0].set_ylabel("Magnitude / nm per 1% power modulation")
    axes[0].set_title("Linearized axial magnitude response")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend()

    axes[1].semilogx(
        frequencies_hz,
        phase_degrees,
        color="tab:orange",
        linewidth=1.5,
    )
    axes[1].axvline(
        natural_frequency_hz,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set_xlabel("Drive frequency / Hz")
    axes[1].set_ylabel("Phase / degrees")
    axes[1].set_title("Linearized axial phase response")
    axes[1].grid(True, which="both", alpha=0.35)

    fig.suptitle(
        (
            "Bode plot for laser-power modulation to z displacement "
            f"(damping ratio {damping_ratio_z:.3g})"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_bode_axial_response.png",
    )


def steady_state_mask_for_drive(time_values, drive_frequency_hz):
    drive_start = driven_single_cycle_start_time
    final_time = time_values[-1]
    post_drive_duration = max(0.0, final_time - drive_start)
    settle_by_fraction = drive_start + (
        driven_amplitude_sweep_settle_fraction * post_drive_duration
    )
    settle_by_cycles = drive_start + (
        driven_amplitude_sweep_settle_cycles / drive_frequency_hz
    )
    steady_start = min(max(settle_by_fraction, settle_by_cycles), final_time)
    mask = time_values >= steady_start

    if np.count_nonzero(mask) < 4:
        mask = time_values >= (
            drive_start + 0.5 * post_drive_duration
        )

    return mask


def sinusoidal_peak_amplitude(signal, time_values, frequency_hz):
    signal = np.asarray(signal, dtype=float)
    time_values = np.asarray(time_values, dtype=float)

    if len(signal) < 4:
        return np.nan, np.nan

    omega = 2.0 * np.pi * frequency_hz
    shifted_time = time_values - time_values[0]
    design = np.column_stack(
        (
            np.sin(omega * shifted_time),
            np.cos(omega * shifted_time),
            np.ones_like(shifted_time),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    amplitude = np.hypot(coefficients[0], coefficients[1])
    phase_degrees = np.arctan2(coefficients[1], coefficients[0]) * 180.0 / np.pi
    return amplitude, phase_degrees


def natural_band_peak_amplitude(signal, time_values):
    frequencies, psd = positive_driven_psd(signal, time_values)
    natural_frequency_hz = omega_z / (2.0 * np.pi)
    half_width_hz = max(
        driven_amplitude_sweep_natural_band_min_hz,
        driven_amplitude_sweep_natural_band_fraction * natural_frequency_hz,
    )
    band_mask = (
        (frequencies >= natural_frequency_hz - half_width_hz)
        & (frequencies <= natural_frequency_hz + half_width_hz)
    )

    if np.count_nonzero(band_mask) < 2:
        return np.nan

    band_power = np.trapz(psd[band_mask], frequencies[band_mask])
    return np.sqrt(2.0 * max(band_power, 0.0))


def write_driven_amplitude_sweep_csv(rows, filename):
    if not driven_save_figures:
        return

    os.makedirs(driven_output_directory, exist_ok=True)
    output_path = os.path.join(driven_output_directory, filename)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "drive_frequency_hz",
                "drive_frequency_over_fz",
                "steady_total_rms_m",
                "steady_peak_to_peak_amplitude_m",
                "drive_frequency_peak_amplitude_m",
                "drive_frequency_phase_degrees",
                "natural_band_peak_amplitude_m",
                "lost_before_end",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote", output_path)


def plot_simulated_driven_amplitude_sweep():
    if not driven_plot_amplitude_sweep:
        return

    natural_frequency_hz = omega_z / (2.0 * np.pi)
    frequencies_hz = np.linspace(
        driven_amplitude_sweep_min_factor * natural_frequency_hz,
        driven_amplitude_sweep_max_factor * natural_frequency_hz,
        driven_amplitude_sweep_points,
    )
    rows = []

    for frequency_hz in frequencies_hz:
        result = run_driven_case(
            driven_amplitude_sweep_waveform,
            frequency_hz,
            single_cycle=False,
            label=(
                f"{driven_amplitude_sweep_waveform} amplitude sweep "
                f"{frequency_hz:g} Hz"
            ),
            use_brownian_motion=driven_use_brownian_motion,
            include_laser_noise=driven_include_laser_power_noise,
        )
        steady_mask = steady_state_mask_for_drive(result["time"], frequency_hz)
        steady_z = result["z_displacement"][steady_mask]
        steady_time = result["time"][steady_mask]
        lost_before_end = len(result["time"]) < len(t_baoab)

        if len(steady_z) >= 4:
            steady_total_rms = np.std(steady_z - np.mean(steady_z))
            steady_peak_to_peak_amplitude = 0.5 * (
                np.max(steady_z) - np.min(steady_z)
            )
            drive_amplitude, drive_phase_degrees = sinusoidal_peak_amplitude(
                steady_z,
                steady_time,
                frequency_hz,
            )
            natural_amplitude = natural_band_peak_amplitude(
                steady_z,
                steady_time,
            )
        else:
            steady_total_rms = np.nan
            steady_peak_to_peak_amplitude = np.nan
            drive_amplitude = np.nan
            drive_phase_degrees = np.nan
            natural_amplitude = np.nan

        rows.append(
            {
                "drive_frequency_hz": frequency_hz,
                "drive_frequency_over_fz": frequency_hz / natural_frequency_hz,
                "steady_total_rms_m": steady_total_rms,
                "steady_peak_to_peak_amplitude_m": steady_peak_to_peak_amplitude,
                "drive_frequency_peak_amplitude_m": drive_amplitude,
                "drive_frequency_phase_degrees": drive_phase_degrees,
                "natural_band_peak_amplitude_m": natural_amplitude,
                "lost_before_end": lost_before_end,
            }
        )

    frequency_ratio = np.array(
        [row["drive_frequency_over_fz"] for row in rows],
        dtype=float,
    )
    steady_peak_um = np.array(
        [row["steady_peak_to_peak_amplitude_m"] for row in rows],
        dtype=float,
    ) * 1e6
    drive_peak_um = np.array(
        [row["drive_frequency_peak_amplitude_m"] for row in rows],
        dtype=float,
    ) * 1e6
    natural_peak_um = np.array(
        [row["natural_band_peak_amplitude_m"] for row in rows],
        dtype=float,
    ) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(
        frequency_ratio,
        steady_peak_um,
        marker="o",
        markersize=3.5,
        linewidth=1.1,
        label="overall steady peak amplitude",
    )
    axes[0].semilogy(
        frequency_ratio,
        drive_peak_um,
        marker="s",
        markersize=3.2,
        linewidth=1.0,
        label="component at drive frequency",
    )
    axes[0].axvline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].axvline(2.0, color="tab:red", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("z amplitude / micrometres")
    axes[0].set_title("Simulation-derived driven amplitude response")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.35)

    axes[1].semilogy(
        frequency_ratio,
        natural_peak_um,
        marker="o",
        markersize=3.5,
        color="tab:orange",
        linewidth=1.1,
        label="component near fz",
    )
    axes[1].axvline(1.0, color="black", linestyle=":", linewidth=1.0, label="fz")
    axes[1].axvline(2.0, color="tab:red", linestyle=":", linewidth=1.0, label="2fz")
    axes[1].set_xlabel("Drive frequency / fz")
    axes[1].set_ylabel("near-fz amplitude / micrometres")
    axes[1].set_title("Natural-frequency response, useful for 2fz parametric drive")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.35)

    fig.suptitle(
        (
            f"{driven_amplitude_sweep_waveform.capitalize()} laser-power "
            f"amplitude sweep ({driven_modulation_fraction * 100:g}% modulation, "
            f"fz = {natural_frequency_hz:g} Hz, {driven_noise_description()})"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_simulated_amplitude_sweep.png",
    )
    write_driven_amplitude_sweep_csv(
        rows,
        f"{driven_output_prefix}_simulated_amplitude_sweep.csv",
    )


def plot_parametric_comparison_cases():
    if not driven_plot_parametric_comparison:
        return

    natural_frequency_hz = omega_z / (2.0 * np.pi)
    drive_frequency_hz = (
        driven_parametric_comparison_frequency_factor * natural_frequency_hz
    )
    cases = (
        (
            "sine, Brownian off, laser noise off",
            "sine",
            False,
            False,
            "tab:blue",
        ),
        (
            "square, Brownian off, laser noise off",
            "square",
            False,
            False,
            "tab:green",
        ),
        (
            "square, Brownian on, laser noise on",
            "square",
            True,
            True,
            "tab:orange",
        ),
    )

    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(12, 2.8 * len(cases)),
        squeeze=False,
    )
    summary_rows = []

    for row_index, (
        case_label,
        waveform,
        use_case_brownian,
        include_case_laser_noise,
        colour,
    ) in enumerate(cases):
        result = run_driven_case(
            waveform,
            drive_frequency_hz,
            single_cycle=False,
            label=f"{case_label} at {drive_frequency_hz:g} Hz",
            use_brownian_motion=use_case_brownian,
            include_laser_noise=include_case_laser_noise,
        )
        trajectory_ax = axes[row_index, 0]
        psd_ax = axes[row_index, 1]
        plot_time, plot_z = downsample_for_plot(
            result["time"],
            result["z_displacement"] * 1e6,
            max_plot_points,
        )
        trajectory_ax.plot(plot_time, plot_z, linewidth=0.75, color=colour)
        trajectory_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        trajectory_ax.axvline(
            driven_single_cycle_start_time,
            color="black",
            linestyle=":",
            linewidth=0.9,
        )
        trajectory_ax.set_title(f"{case_label} trajectory")
        trajectory_ax.set_xlabel("Time / s")
        trajectory_ax.set_ylabel("z displacement / micrometres")
        trajectory_ax.grid(True, alpha=0.35)

        psd_ax.loglog(
            result["frequencies"],
            result["psd"],
            linewidth=1.0,
            color=colour,
        )
        psd_ax.axvline(
            natural_frequency_hz,
            color="black",
            linestyle=":",
            linewidth=0.9,
            label="fz",
        )
        psd_ax.axvline(
            drive_frequency_hz,
            color="tab:red",
            linestyle=":",
            linewidth=0.9,
            label=(
                f"{driven_parametric_comparison_frequency_factor:g}fz drive"
            ),
        )
        psd_ax.set_title(f"{case_label} PSD")
        apply_driven_psd_axes(psd_ax)
        psd_ax.legend()

        steady_mask = steady_state_mask_for_drive(
            result["time"],
            drive_frequency_hz,
        )
        steady_z = result["z_displacement"][steady_mask]
        steady_time = result["time"][steady_mask]
        drive_amplitude, drive_phase_degrees = sinusoidal_peak_amplitude(
            steady_z,
            steady_time,
            drive_frequency_hz,
        )
        natural_amplitude = natural_band_peak_amplitude(steady_z, steady_time)
        summary_rows.append(
            {
                "case": case_label,
                "waveform": waveform,
                "drive_frequency_hz": drive_frequency_hz,
                "drive_frequency_over_fz": drive_frequency_hz / natural_frequency_hz,
                "brownian_motion": use_case_brownian,
                "laser_power_noise": include_case_laser_noise,
                "steady_peak_to_peak_amplitude_m": (
                    0.5 * (np.max(steady_z) - np.min(steady_z))
                    if len(steady_z) > 0
                    else np.nan
                ),
                "drive_frequency_peak_amplitude_m": drive_amplitude,
                "drive_frequency_phase_degrees": drive_phase_degrees,
                "natural_band_peak_amplitude_m": natural_amplitude,
                "lost_before_end": len(result["time"]) < len(t_baoab),
            }
        )

    fig.suptitle(
        (
            "Comparison of sine and square laser-power drive near "
            f"{driven_parametric_comparison_frequency_factor:g}fz "
            f"({drive_frequency_hz:g} Hz, fz = {natural_frequency_hz:g} Hz)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_parametric_comparison_"
        f"{driven_parametric_comparison_frequency_factor:g}fz.png",
    )

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            (
                f"{driven_output_prefix}_parametric_comparison_"
                f"{driven_parametric_comparison_frequency_factor:g}fz.csv"
            ),
        )
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=summary_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print("Wrote", output_path)


def plot_square_frequency_amplitude_sweep():
    if not driven_plot_square_frequency_amplitude_sweep:
        return

    natural_frequency_hz = omega_z / (2.0 * np.pi)
    frequencies_hz = np.linspace(
        driven_square_frequency_sweep_start_hz,
        driven_square_frequency_sweep_end_hz,
        driven_square_frequency_sweep_points,
    )
    rows = []

    for frequency_hz in frequencies_hz:
        result = run_driven_case(
            "square",
            frequency_hz,
            single_cycle=False,
            label=f"square frequency-amplitude sweep {frequency_hz:g} Hz",
            use_brownian_motion=driven_use_brownian_motion,
            include_laser_noise=driven_include_laser_power_noise,
        )
        steady_mask = steady_state_mask_for_drive(result["time"], frequency_hz)
        steady_z = result["z_displacement"][steady_mask]

        if len(steady_z) >= 4:
            steady_z_centred = steady_z - np.mean(steady_z)
            oscillation_amplitude = 0.5 * (
                np.max(steady_z_centred) - np.min(steady_z_centred)
            )
            steady_rms = np.std(steady_z_centred)
        else:
            oscillation_amplitude = np.nan
            steady_rms = np.nan

        rows.append(
            {
                "drive_frequency_hz": frequency_hz,
                "drive_frequency_over_fz": frequency_hz / natural_frequency_hz,
                "settled_oscillation_amplitude_m": oscillation_amplitude,
                "settled_rms_m": steady_rms,
                "lost_before_end": len(result["time"]) < len(t_baoab),
            }
        )

    drive_frequency_hz = np.array(
        [row["drive_frequency_hz"] for row in rows],
        dtype=float,
    )
    amplitude_um = np.array(
        [row["settled_oscillation_amplitude_m"] for row in rows],
        dtype=float,
    ) * 1e6
    lost_before_end = np.array(
        [row["lost_before_end"] for row in rows],
        dtype=bool,
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.semilogy(
        drive_frequency_hz[~lost_before_end],
        amplitude_um[~lost_before_end],
        marker="o",
        markersize=3.8,
        linewidth=1.2,
        label="settled oscillation amplitude",
    )

    if np.any(lost_before_end):
        ax.semilogy(
            drive_frequency_hz[lost_before_end],
            amplitude_um[lost_before_end],
            marker="x",
            linestyle="none",
            markersize=6.0,
            color="tab:red",
            label="trap loss before end",
        )

    ax.axvline(
        natural_frequency_hz,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"fz = {natural_frequency_hz:g} Hz",
    )
    ax.axvline(
        natural_frequency_hz / 3.0,
        color="0.45",
        linestyle=":",
        linewidth=1.0,
        label=f"fz/3 = {natural_frequency_hz / 3.0:g} Hz",
    )
    ax.axvline(
        2.0 * natural_frequency_hz,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        label=f"2fz = {2.0 * natural_frequency_hz:g} Hz",
    )
    ax.set_xlabel("Square-wave drive frequency / Hz")
    ax.set_ylabel("Settled z oscillation amplitude / micrometres")
    ax.set_title("Square-wave driven oscillation amplitude")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.suptitle(
        (
            f"Square drive frequency response "
            f"({driven_square_frequency_sweep_start_hz:g}-"
            f"{driven_square_frequency_sweep_end_hz:g} Hz, "
            f"{driven_modulation_fraction * 100:g}% modulation)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_square_frequency_amplitude_sweep.png",
    )

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_square_frequency_amplitude_sweep.csv",
        )
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote", output_path)


def stable_z_equilibrium_for_power_factor(power_factor):
    def Fz_net_on_axis_at_power(z):
        _, Fz = F_optical_2d_direct(0.0, z, power_factor)
        return Fz - m*g

    F_scan_power = Fz_net_on_axis_at_power(z_scan)
    roots_power = []

    for i in range(len(z_scan) - 1):
        if F_scan_power[i] * F_scan_power[i + 1] < 0:
            roots_power.append(
                brentq(
                    Fz_net_on_axis_at_power,
                    z_scan[i],
                    z_scan[i + 1],
                    xtol=root_finding_xtol,
                    rtol=root_finding_rtol,
                )
            )

    stable_roots_power = []
    for root in roots_power:
        slope = numerical_derivative_1d(Fz_net_on_axis_at_power, root)
        if slope < 0:
            stable_roots_power.append(root)

    if len(stable_roots_power) == 0:
        raise ValueError(
            "No stable on-axis equilibrium found for "
            f"laser power factor {power_factor:.5f}."
        )

    if not -len(stable_roots_power) <= equilibrium_root_index < len(stable_roots_power):
        raise IndexError(
            "equilibrium_root_index is outside the stable root list for "
            f"laser power factor {power_factor:.5f}. "
            f"Found {len(stable_roots_power)} stable root(s)."
        )

    return stable_roots_power[equilibrium_root_index]


def settling_time_to_equilibrium(
    time_values,
    position_values,
    target_equilibrium,
    reference_equilibrium,
    step_start_time,
    window_end_time=None,
):
    if window_end_time is None:
        window_mask = time_values >= step_start_time
    else:
        window_mask = (
            (time_values >= step_start_time)
            & (time_values < window_end_time)
        )
    after_step_indices = np.flatnonzero(window_mask)

    if len(after_step_indices) == 0:
        return np.nan, np.nan

    initial_error = abs(reference_equilibrium - target_equilibrium)
    tolerance = max(
        driven_step_response_settling_fraction * initial_error,
        driven_step_response_settling_abs_min_um * 1e-6,
    )
    window_error = np.abs(position_values[after_step_indices] - target_equilibrium)
    suffix_max_error = np.maximum.accumulate(window_error[::-1])[::-1]
    settled_offsets = np.flatnonzero(suffix_max_error <= tolerance)

    if len(settled_offsets) == 0:
        return np.nan, tolerance

    settled_index = after_step_indices[settled_offsets[0]]
    return time_values[settled_index] - step_start_time, tolerance


def first_fraction_crossing_time(
    time_values,
    position_values,
    initial_equilibrium,
    target_equilibrium,
    step_start_time,
    fraction,
):
    delta = target_equilibrium - initial_equilibrium
    if delta == 0:
        return np.nan

    after_step_indices = np.flatnonzero(time_values >= step_start_time)
    if len(after_step_indices) == 0:
        return np.nan

    progress = (
        position_values[after_step_indices] - initial_equilibrium
    ) / delta
    crossing_offsets = np.flatnonzero(progress >= fraction)
    if len(crossing_offsets) == 0:
        return np.nan

    crossing_index = after_step_indices[crossing_offsets[0]]
    return time_values[crossing_index] - step_start_time


def step_shape_metrics(
    time_values,
    position_values,
    velocity_values,
    initial_equilibrium,
    target_equilibrium,
    step_start_time,
):
    delta = target_equilibrium - initial_equilibrium
    after_step_indices = np.flatnonzero(time_values >= step_start_time)

    if delta == 0 or len(after_step_indices) == 0:
        return {
            "first_target_crossing_time_s": np.nan,
            "t50_s": np.nan,
            "t90_s": np.nan,
            "t98_s": np.nan,
            "peak_velocity_towards_target_um_s": np.nan,
            "peak_abs_velocity_um_s": np.nan,
            "overshoot_um": np.nan,
        }

    position_after_step = position_values[after_step_indices]
    velocity_after_step = velocity_values[after_step_indices]
    progress = (position_after_step - initial_equilibrium) / delta
    direction = np.sign(delta)

    overshoot_um = max(np.max(progress) - 1.0, 0.0) * abs(delta) * 1e6
    velocity_towards_target_um_s = direction * velocity_after_step * 1e6

    return {
        "first_target_crossing_time_s": first_fraction_crossing_time(
            time_values,
            position_values,
            initial_equilibrium,
            target_equilibrium,
            step_start_time,
            1.0,
        ),
        "t50_s": first_fraction_crossing_time(
            time_values,
            position_values,
            initial_equilibrium,
            target_equilibrium,
            step_start_time,
            0.50,
        ),
        "t90_s": first_fraction_crossing_time(
            time_values,
            position_values,
            initial_equilibrium,
            target_equilibrium,
            step_start_time,
            0.90,
        ),
        "t98_s": first_fraction_crossing_time(
            time_values,
            position_values,
            initial_equilibrium,
            target_equilibrium,
            step_start_time,
            0.98,
        ),
        "peak_velocity_towards_target_um_s": np.max(velocity_towards_target_um_s),
        "peak_abs_velocity_um_s": np.max(np.abs(velocity_after_step)) * 1e6,
        "overshoot_um": overshoot_um,
    }


def axial_frequency_at_equilibrium(power_factor, equilibrium):
    def Fz_net_at_power(z):
        _, Fz = F_optical_2d_direct(0.0, z, power_factor)
        return Fz - m*g

    k_local = -numerical_derivative_1d(Fz_net_at_power, equilibrium)
    if k_local <= 0:
        return np.nan

    return np.sqrt(k_local / m) / (2.0 * np.pi)


def plot_power_step_response():
    if not driven_plot_power_step_response:
        return

    upward_power_factor = 1.0 + driven_step_response_power_change_fraction
    downward_power_factor = 1.0
    upward_step_time = driven_step_response_start_time
    downward_step_time = upward_step_time + driven_step_response_hold_time

    power_factor = np.ones_like(t_baoab)
    power_factor[t_baoab >= upward_step_time] = upward_power_factor
    power_factor[t_baoab >= downward_step_time] = downward_power_factor

    upward_equilibrium = stable_z_equilibrium_for_power_factor(upward_power_factor)
    downward_equilibrium = stable_z_equilibrium_for_power_factor(downward_power_factor)

    x_out, y_out, z_out, vx_out, vy_out, vz_out = solve_baoab_3d_fast_with_power(
        power_factor,
        zero_brownian_normals,
        zero_brownian_normals,
        zero_brownian_normals,
        "combined laser-power step response",
    )
    sample_count = min(len(t_baoab), len(z_out))
    time_values = t_baoab[:sample_count]
    z_values = z_out[:sample_count]
    z_displacement = z_values - z_eq

    upward_settling_time, upward_tolerance = settling_time_to_equilibrium(
        time_values,
        z_values,
        upward_equilibrium,
        z_eq,
        upward_step_time,
        window_end_time=downward_step_time,
    )
    downward_settling_time, downward_tolerance = settling_time_to_equilibrium(
        time_values,
        z_values,
        downward_equilibrium,
        upward_equilibrium,
        downward_step_time,
    )

    upward_change = upward_equilibrium - z_eq
    downward_change = downward_equilibrium - z_eq

    fig, ax = plt.subplots(figsize=(9, 5.4))
    plot_time, plot_z = downsample_for_plot(
        time_values,
        z_displacement * 1e6,
        max_plot_points,
    )
    ax.plot(plot_time, plot_z, color="tab:blue", linewidth=0.85, label="z trajectory")

    upward_text = (
        f"upward settling time = {upward_settling_time:.3g} s"
        if np.isfinite(upward_settling_time)
        else "upward: not settled before return"
    )
    downward_text = (
        f"return settling time = {downward_settling_time:.3g} s"
        if np.isfinite(downward_settling_time)
        else "return: not settled by end"
    )
    fig.text(
        0.02,
        0.88,
        upward_text + "\n" + downward_text,
        ha="left",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel("Time / s")
    ax.set_ylabel("z displacement / micrometres")
    ax.set_title("Laser-power step up then return response")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")

    rows = [
        {
            "step_direction": "upward",
            "step_time_s": upward_step_time,
            "initial_power_factor": 1.0,
            "target_power_factor": upward_power_factor,
            "reference_equilibrium_um": z_eq * 1e6,
            "target_equilibrium_um": upward_equilibrium * 1e6,
            "target_displacement_from_initial_um": upward_change * 1e6,
            "settling_band_fraction": driven_step_response_settling_fraction,
            "settling_band_um": upward_tolerance * 1e6,
            "settling_time_s": upward_settling_time,
            "settled_before_next_step_or_end": bool(np.isfinite(upward_settling_time)),
        },
        {
            "step_direction": "downward",
            "step_time_s": downward_step_time,
            "initial_power_factor": upward_power_factor,
            "target_power_factor": downward_power_factor,
            "reference_equilibrium_um": upward_equilibrium * 1e6,
            "target_equilibrium_um": downward_equilibrium * 1e6,
            "target_displacement_from_initial_um": downward_change * 1e6,
            "settling_band_fraction": driven_step_response_settling_fraction,
            "settling_band_um": downward_tolerance * 1e6,
            "settling_time_s": downward_settling_time,
            "settled_before_next_step_or_end": bool(np.isfinite(downward_settling_time)),
        },
    ]

    fig.suptitle(
        (
            "Combined laser-power step response "
            f"(+{driven_step_response_power_change_fraction * 100:g}% then return, "
            f"{driven_step_response_hold_time:g} s hold)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_power_step_response.png",
        tight_layout_rect=(0.0, 0.0, 1.0, 0.78),
    )

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_power_step_response.csv",
        )
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote", output_path)


def plot_resonant_peak_trajectories():
    if not driven_plot_resonant_peak_trajectories:
        return

    natural_frequency_hz = omega_z / (2.0 * np.pi)
    peak_specs = (
        ("fz", natural_frequency_hz),
        ("2fz", 2.0 * natural_frequency_hz),
    )
    fig, axes = plt.subplots(
        len(peak_specs),
        1,
        figsize=(10, 3.0 * len(peak_specs)),
        sharex=True,
        squeeze=False,
    )
    rows = []

    for row_index, (peak_label, frequency_hz) in enumerate(peak_specs):
        ax = axes[row_index, 0]
        result = run_driven_case(
            "square",
            frequency_hz,
            single_cycle=False,
            label=f"square resonant trajectory at {peak_label}",
            use_brownian_motion=False,
            include_laser_noise=False,
        )
        time_values = result["time"]
        z_um = result["z_displacement"] * 1e6
        steady_mask = steady_state_mask_for_drive(time_values, frequency_hz)
        steady_time = time_values[steady_mask]
        steady_z_um = z_um[steady_mask]

        if len(steady_z_um) >= 4:
            steady_mean_um = np.mean(steady_z_um)
            steady_min_um = np.min(steady_z_um)
            steady_max_um = np.max(steady_z_um)
            half_peak_to_peak_um = 0.5 * (steady_max_um - steady_min_um)
            peak_to_peak_um = steady_max_um - steady_min_um
            steady_rms_um = np.std(steady_z_um - steady_mean_um, ddof=1)
            steady_start_time = steady_time[0]
        else:
            steady_mean_um = np.nan
            steady_min_um = np.nan
            steady_max_um = np.nan
            half_peak_to_peak_um = np.nan
            peak_to_peak_um = np.nan
            steady_rms_um = np.nan
            steady_start_time = np.nan

        plot_time, plot_z_um = downsample_for_plot(
            time_values,
            z_um,
            max_plot_points,
        )
        ax.plot(plot_time, plot_z_um, linewidth=0.85, color="tab:blue")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
        ax.axvline(
            driven_single_cycle_start_time,
            color="black",
            linestyle=":",
            linewidth=0.9,
            label="drive starts",
        )
        if np.isfinite(steady_start_time):
            ax.axvspan(
                steady_start_time,
                time_values[-1],
                color="tab:orange",
                alpha=0.10,
                label="amplitude window",
            )
            ax.axhline(
                steady_mean_um,
                color="tab:green",
                linestyle="--",
                linewidth=0.9,
                label="steady mean",
            )
            ax.axhline(
                steady_min_um,
                color="tab:red",
                linestyle=":",
                linewidth=0.9,
            )
            ax.axhline(
                steady_max_um,
                color="tab:red",
                linestyle=":",
                linewidth=0.9,
                label="steady min/max",
            )

        ax.set_title(
            (
                f"{peak_label} = {frequency_hz:g} Hz trajectory, "
                f"half peak-to-peak = {half_peak_to_peak_um:.3g} um"
            )
        )
        ax.set_ylabel("z displacement / micrometres")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")

        rows.append(
            {
                "peak_label": peak_label,
                "drive_frequency_hz": frequency_hz,
                "drive_frequency_over_fz": frequency_hz / natural_frequency_hz,
                "pressure_pa": p,
                "modulation_fraction": driven_modulation_fraction,
                "steady_window_start_s": steady_start_time,
                "steady_mean_um": steady_mean_um,
                "steady_min_um": steady_min_um,
                "steady_max_um": steady_max_um,
                "settled_half_peak_to_peak_amplitude_um": half_peak_to_peak_um,
                "settled_peak_to_peak_um": peak_to_peak_um,
                "settled_rms_um": steady_rms_um,
            }
        )

    axes[-1, 0].set_xlabel("Time / s")
    fig.suptitle(
        (
            "Square-drive resonant peak trajectories "
            f"({driven_modulation_fraction * 100:g}% modulation, "
            f"p = {p:g} Pa)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_square_resonant_peak_trajectories.png",
    )

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_square_resonant_peak_trajectories.csv",
        )
        save_rows_csv(output_path, rows)


def plot_modulation_amplitude_sweep():
    if not driven_plot_modulation_amplitude_sweep:
        return

    natural_frequency_hz = omega_z / (2.0 * np.pi)
    drive_frequency_hz = (
        driven_modulation_amplitude_sweep_frequency_factor
        * natural_frequency_hz
    )
    modulation_fractions = np.linspace(
        driven_modulation_amplitude_sweep_min_fraction,
        driven_modulation_amplitude_sweep_max_fraction,
        driven_modulation_amplitude_sweep_points,
    )
    rows = []

    for modulation_fraction in modulation_fractions:
        result = run_driven_case(
            "square",
            drive_frequency_hz,
            single_cycle=False,
            label=(
                "square modulation-amplitude sweep "
                f"{modulation_fraction * 100:g}%"
            ),
            use_brownian_motion=False,
            include_laser_noise=False,
            modulation_fraction=modulation_fraction,
        )
        steady_mask = steady_state_mask_for_drive(
            result["time"],
            drive_frequency_hz,
        )
        steady_z = result["z_displacement"][steady_mask]
        steady_time = result["time"][steady_mask]
        lost_before_end = len(result["time"]) < len(t_baoab)

        if len(steady_z) >= 4:
            steady_z_centred = steady_z - np.mean(steady_z)
            half_peak_to_peak = 0.5 * (
                np.max(steady_z_centred) - np.min(steady_z_centred)
            )
            steady_rms = np.std(steady_z_centred, ddof=1)
            drive_amplitude, drive_phase_degrees = sinusoidal_peak_amplitude(
                steady_z,
                steady_time,
                drive_frequency_hz,
            )
            natural_amplitude = natural_band_peak_amplitude(
                steady_z,
                steady_time,
            )
        else:
            half_peak_to_peak = np.nan
            steady_rms = np.nan
            drive_amplitude = np.nan
            drive_phase_degrees = np.nan
            natural_amplitude = np.nan

        rows.append(
            {
                "drive_frequency_hz": drive_frequency_hz,
                "drive_frequency_over_fz": (
                    drive_frequency_hz / natural_frequency_hz
                ),
                "pressure_pa": p,
                "modulation_fraction": modulation_fraction,
                "modulation_percent": modulation_fraction * 100.0,
                "settled_half_peak_to_peak_amplitude_um": (
                    half_peak_to_peak * 1e6
                ),
                "settled_peak_to_peak_um": 2.0 * half_peak_to_peak * 1e6,
                "settled_rms_um": steady_rms * 1e6,
                "drive_frequency_peak_amplitude_um": drive_amplitude * 1e6,
                "drive_frequency_phase_degrees": drive_phase_degrees,
                "natural_band_peak_amplitude_um": natural_amplitude * 1e6,
                "lost_before_end": lost_before_end,
            }
        )

    modulation_percent = np.array(
        [row["modulation_percent"] for row in rows],
        dtype=float,
    )
    half_peak_to_peak_um = np.array(
        [row["settled_half_peak_to_peak_amplitude_um"] for row in rows],
        dtype=float,
    )
    rms_um = np.array(
        [row["settled_rms_um"] for row in rows],
        dtype=float,
    )
    drive_component_um = np.array(
        [row["drive_frequency_peak_amplitude_um"] for row in rows],
        dtype=float,
    )
    natural_component_um = np.array(
        [row["natural_band_peak_amplitude_um"] for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.semilogy(
        modulation_percent,
        half_peak_to_peak_um,
        marker="o",
        markersize=3.8,
        linewidth=1.2,
        label="half peak-to-peak",
    )
    ax.semilogy(
        modulation_percent,
        rms_um,
        marker="s",
        markersize=3.2,
        linewidth=1.0,
        label="RMS",
    )
    ax.semilogy(
        modulation_percent,
        drive_component_um,
        marker="^",
        markersize=3.2,
        linewidth=1.0,
        label="component at drive frequency",
    )
    ax.semilogy(
        modulation_percent,
        natural_component_um,
        marker="d",
        markersize=3.0,
        linewidth=1.0,
        label="component near fz",
    )
    ax.set_xlabel("Laser-power modulation amplitude / percent")
    ax.set_ylabel("z amplitude / micrometres")
    ax.set_title(
        (
            "Square-drive response versus modulation amplitude "
            f"at {drive_frequency_hz:g} Hz"
        )
    )
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.suptitle(
        (
            f"Modulation-amplitude sweep at "
            f"{driven_modulation_amplitude_sweep_frequency_factor:g}fz "
            f"(fz = {natural_frequency_hz:g} Hz, p = {p:g} Pa)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_square_2fz_modulation_amplitude_sweep.png",
    )

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_square_2fz_modulation_amplitude_sweep.csv",
        )
        save_rows_csv(output_path, rows)


def run_pressure_step_settling_case(
    pressure,
    direction,
    initial_power_factor,
    target_power_factor,
    refresh_pressure=True,
    rebuild_lookup=True,
):
    global x0
    global y0
    global z0
    global vx0
    global vy0
    global vz0
    global terminate_on_trap_loss

    if refresh_pressure:
        update_baoab_damping_for_pressure(pressure)

    initial_equilibrium = stable_z_equilibrium_for_power_factor(initial_power_factor)
    target_equilibrium = stable_z_equilibrium_for_power_factor(target_power_factor)
    step_time = driven_step_response_start_time

    if rebuild_lookup and use_force_lookup_table:
        rebuild_force_lookup_table_for_pressure_step(
            initial_equilibrium,
            target_equilibrium,
        )

    power_factor = np.full_like(t_baoab, initial_power_factor, dtype=float)
    power_factor[t_baoab >= step_time] = target_power_factor

    saved_initial_state = (x0, y0, z0, vx0, vy0, vz0)
    saved_terminate_on_trap_loss = terminate_on_trap_loss

    x0 = x_eq
    y0 = y_eq
    z0 = initial_equilibrium
    vx0 = 0.0
    vy0 = 0.0
    vz0 = 0.0
    terminate_on_trap_loss = False

    try:
        _, _, z_out, _, _, vz_out = solve_baoab_3d_fast_with_power(
            power_factor,
            zero_brownian_normals,
            zero_brownian_normals,
            zero_brownian_normals,
            f"{pressure:g} Pa {direction} laser-power step",
        )
    finally:
        x0, y0, z0, vx0, vy0, vz0 = saved_initial_state
        terminate_on_trap_loss = saved_terminate_on_trap_loss

    sample_count = min(len(t_baoab), len(z_out))
    time_values = t_baoab[:sample_count]
    z_values = z_out[:sample_count]
    vz_values = vz_out[:sample_count]
    settling_time, tolerance = settling_time_to_equilibrium(
        time_values,
        z_values,
        target_equilibrium,
        initial_equilibrium,
        step_time,
    )
    final_error = (
        abs(z_values[-1] - target_equilibrium)
        if len(z_values) > 0
        else np.nan
    )
    transient_metrics = step_shape_metrics(
        time_values,
        z_values,
        vz_values,
        initial_equilibrium,
        target_equilibrium,
        step_time,
    )

    row = {
        "pressure_pa": pressure,
        "step_direction": direction,
        "drag_model": drag_model_used,
        "damping_rate_s^-1": gamma_baoab,
        "initial_power_factor": initial_power_factor,
        "target_power_factor": target_power_factor,
        "initial_equilibrium_um": initial_equilibrium * 1e6,
        "target_equilibrium_um": target_equilibrium * 1e6,
        "equilibrium_step_um": (target_equilibrium - initial_equilibrium) * 1e6,
        "initial_axial_frequency_hz": axial_frequency_at_equilibrium(
            initial_power_factor,
            initial_equilibrium,
        ),
        "target_axial_frequency_hz": axial_frequency_at_equilibrium(
            target_power_factor,
            target_equilibrium,
        ),
        "settling_band_fraction": driven_step_response_settling_fraction,
        "settling_band_um": tolerance * 1e6,
        "settling_time_s": settling_time,
        "settled_by_end": bool(np.isfinite(settling_time)),
        "final_abs_error_um": final_error * 1e6,
    }
    row.update(transient_metrics)
    return row


def rebuild_force_lookup_table_for_pressure_step(*equilibrium_values):
    z_min = min(equilibrium_values) - force_lookup_z_base_half_width
    z_max = max(equilibrium_values) + force_lookup_z_base_half_width
    r_max = max(force_lookup_r_base_max, force_lookup_r_thermal_factor * x_rms_thermal)

    build_force_lookup_table(
        force_lookup_r_min,
        r_max,
        z_min,
        z_max,
    )


def format_step_settling_markdown_table(rows):
    header = (
        "| Pressure (Pa) | Direction | Drag model | gamma (s^-1) | "
        "dz_eq (um) | fz start (Hz) | fz target (Hz) | t50 (s) | "
        "t90 (s) | t98 (s) | first crossing (s) | peak v (um/s) | "
        "overshoot (um) | settling (s) |"
    )
    separator = (
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header, separator]

    for row in rows:
        def format_number(value):
            return f"{value:.6g}" if np.isfinite(value) else "n/a"

        lines.append(
            "| "
            f"{row['pressure_pa']:g} | "
            f"{row['step_direction']} | "
            f"{row['drag_model']} | "
            f"{row['damping_rate_s^-1']:.6g} | "
            f"{row['equilibrium_step_um']:.6g} | "
            f"{format_number(row['initial_axial_frequency_hz'])} | "
            f"{format_number(row['target_axial_frequency_hz'])} | "
            f"{format_number(row['t50_s'])} | "
            f"{format_number(row['t90_s'])} | "
            f"{format_number(row['t98_s'])} | "
            f"{format_number(row['first_target_crossing_time_s'])} | "
            f"{format_number(row['peak_velocity_towards_target_um_s'])} | "
            f"{format_number(row['overshoot_um'])} | "
            f"{format_number(row['settling_time_s'])} |"
        )

    return "\n".join(lines)


def run_pressure_step_settling_table():
    global p
    global b_cunningham
    global b_epstein
    global b
    global drag_model_used
    global gamma_baoab
    global baoab_damping_factor
    global baoab_thermal_velocity_scale

    if not driven_run_pressure_step_settling_table:
        return []

    upward_initial_power_factor = 1.0
    upward_target_power_factor = 1.0 + driven_step_response_power_change_fraction
    downward_initial_power_factor = upward_target_power_factor
    downward_target_power_factor = 1.0

    saved_pressure_state = (
        p,
        b_cunningham,
        b_epstein,
        b,
        drag_model_used,
        gamma_baoab,
        baoab_damping_factor,
        baoab_thermal_velocity_scale,
    )

    rows = []
    try:
        for pressure in driven_pressure_step_pressures_pa:
            update_baoab_damping_for_pressure(float(pressure))
            nominal_equilibrium = stable_z_equilibrium_for_power_factor(1.0)
            stepped_equilibrium = stable_z_equilibrium_for_power_factor(
                upward_target_power_factor
            )
            if use_force_lookup_table:
                rebuild_force_lookup_table_for_pressure_step(
                    nominal_equilibrium,
                    stepped_equilibrium,
                )

            rows.append(
                run_pressure_step_settling_case(
                    float(pressure),
                    "up",
                    upward_initial_power_factor,
                    upward_target_power_factor,
                    refresh_pressure=False,
                    rebuild_lookup=False,
                )
            )
            rows.append(
                run_pressure_step_settling_case(
                    float(pressure),
                    "down",
                    downward_initial_power_factor,
                    downward_target_power_factor,
                    refresh_pressure=False,
                    rebuild_lookup=False,
                )
            )
    finally:
        (
            saved_p,
            saved_b_cunningham,
            saved_b_epstein,
            saved_b,
            saved_drag_model_used,
            saved_gamma_baoab,
            saved_baoab_damping_factor,
            saved_baoab_thermal_velocity_scale,
        ) = saved_pressure_state
        update_baoab_damping_for_pressure(saved_p)
        b_cunningham = saved_b_cunningham
        b_epstein = saved_b_epstein
        b = saved_b
        drag_model_used = saved_drag_model_used
        gamma_baoab = saved_gamma_baoab
        baoab_damping_factor = saved_baoab_damping_factor
        baoab_thermal_velocity_scale = saved_baoab_thermal_velocity_scale

    if driven_save_figures and rows:
        os.makedirs(driven_output_directory, exist_ok=True)
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_pressure_step_settling_table.csv",
        )
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote", output_path)

    print("\nPressure step settling table")
    print(format_step_settling_markdown_table(rows))

    return rows


def run_current_pressure_step_response_metrics():
    if not driven_save_current_pressure_step_response_metrics:
        return []

    target_power_factor = 1.0 + driven_response_time_step_fraction
    if use_force_lookup_table:
        nominal_equilibrium = stable_z_equilibrium_for_power_factor(1.0)
        stepped_equilibrium = stable_z_equilibrium_for_power_factor(
            target_power_factor
        )
        rebuild_force_lookup_table_for_pressure_step(
            nominal_equilibrium,
            stepped_equilibrium,
        )

    rows = [
        run_pressure_step_settling_case(
            float(p),
            "up",
            1.0,
            target_power_factor,
            refresh_pressure=False,
            rebuild_lookup=False,
        ),
        run_pressure_step_settling_case(
            float(p),
            "down",
            target_power_factor,
            1.0,
            refresh_pressure=False,
            rebuild_lookup=False,
        ),
    ]

    if driven_save_figures:
        output_path = os.path.join(
            driven_output_directory,
            f"{driven_output_prefix}_current_pressure_step_response_metrics.csv",
        )
        save_rows_csv(output_path, rows)

    print("\nCurrent-pressure deterministic step response metrics")
    print(format_step_settling_markdown_table(rows))

    return rows


def plot_power_sine_response():
    if not driven_plot_power_sine_response:
        return

    power_factor = np.ones_like(t_baoab)
    active_mask = t_baoab >= driven_power_sine_start_time
    phase_time = t_baoab[active_mask] - driven_power_sine_start_time
    power_factor[active_mask] = (
        1.0
        + driven_step_response_power_change_fraction
        * np.sin(2.0 * np.pi * phase_time / driven_power_sine_period)
    )

    x_out, y_out, z_out, vx_out, vy_out, vz_out = solve_baoab_3d_fast_with_power(
        power_factor,
        zero_brownian_normals,
        zero_brownian_normals,
        zero_brownian_normals,
        "sine laser-power response",
    )
    sample_count = min(len(t_baoab), len(z_out))
    time_values = t_baoab[:sample_count]
    z_displacement = z_out[:sample_count] - z_eq

    fig, ax = plt.subplots(figsize=(9, 5.4))
    plot_time, plot_z = downsample_for_plot(
        time_values,
        z_displacement * 1e6,
        max_plot_points,
    )
    ax.plot(plot_time, plot_z, color="tab:purple", linewidth=0.85)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("z displacement / micrometres")
    ax.set_title("Sine-modulated laser-power response")
    ax.grid(True, alpha=0.35)
    fig.suptitle(
        (
            "Sine laser-power modulation response "
            f"({driven_step_response_power_change_fraction * 100:g}% amplitude, "
            f"{driven_power_sine_period:g} s period)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_power_sine_response.png",
    )


def chirped_frequency_time(time_values):
    sweep_rate = (
        (driven_frequency_sweep_end_hz - driven_frequency_sweep_start_hz)
        / driven_frequency_sweep_duration
    )
    return driven_frequency_sweep_start_hz + sweep_rate * time_values


def chirped_laser_power_factor(time_values):
    sweep_rate = (
        (driven_frequency_sweep_end_hz - driven_frequency_sweep_start_hz)
        / driven_frequency_sweep_duration
    )
    phase_cycles = (
        driven_frequency_sweep_start_hz * time_values
        + 0.5 * sweep_rate * time_values**2
    )
    return 1.0 + driven_modulation_fraction * np.sin(2.0 * np.pi * phase_cycles)


def plot_chirped_laser_frequency_sweep():
    if not driven_plot_frequency_sweep:
        return

    overview_time = np.linspace(0.0, driven_frequency_sweep_duration, 5000)
    overview_frequency = chirped_frequency_time(overview_time)
    sweep_windows = (
        ("start", 0.0),
        ("middle", 0.5 * driven_frequency_sweep_duration),
        ("end", driven_frequency_sweep_duration - 10.0 / driven_frequency_sweep_end_hz),
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 9))
    axes[0].plot(overview_time, overview_frequency, color="tab:purple")
    axes[0].set_title("Laser-drive chirp frequency")
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Drive frequency / Hz")
    axes[0].grid(True, alpha=0.35)

    for ax, (window_label, window_start) in zip(axes[1:], sweep_windows):
        window_start = max(0.0, window_start)
        local_frequency = chirped_frequency_time(np.array([window_start]))[0]
        window_duration = 10.0 / local_frequency
        window_end = min(
            driven_frequency_sweep_duration,
            window_start + window_duration,
        )
        sample_rate = max(20000.0, 40.0 * local_frequency)
        sample_count = max(200, int(np.ceil((window_end - window_start) * sample_rate)))
        window_time = np.linspace(window_start, window_end, sample_count)
        power_mw = P_laser * chirped_laser_power_factor(window_time) * 1e3

        ax.plot(window_time, power_mw, color="tab:red", linewidth=0.9)
        ax.set_title(
            f"{window_label.capitalize()} zoom, "
            f"{local_frequency:g} Hz, 10 cycles"
        )
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Laser power / mW")
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        (
            "Laser-power chirp preview "
            f"({driven_frequency_sweep_start_hz:g}-"
            f"{driven_frequency_sweep_end_hz:g} Hz over "
            f"{driven_frequency_sweep_duration:g} s)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_laser_frequency_sweep_"
        f"{driven_frequency_sweep_start_hz:g}_to_"
        f"{driven_frequency_sweep_end_hz:g}_Hz.png",
    )


def chirped_laser_noise_factor(sample_count, dt_local):
    if not driven_include_laser_power_noise:
        return np.ones(sample_count, dtype=np.float64)

    local_rng = np.random.default_rng(seed=laser_noise_seed)
    laser_step_samples = max(1, int(round(laser_noise_step_duration / dt_local)))
    n_laser_steps = int(np.ceil(sample_count / laser_step_samples))
    laser_allowed_power_factors = np.array(
        [1 - laser_noise_fraction, 1.0, 1 + laser_noise_fraction]
    )
    laser_step_power_factors = np.zeros(n_laser_steps)
    laser_step_power_factors[0] = local_rng.choice(laser_allowed_power_factors)

    for j in range(1, n_laser_steps):
        previous_factor = laser_step_power_factors[j - 1]

        if np.isclose(previous_factor, 1 + laser_noise_fraction):
            possible_factors = np.array([1.0, 1 + laser_noise_fraction])
        elif np.isclose(previous_factor, 1 - laser_noise_fraction):
            possible_factors = np.array([1 - laser_noise_fraction, 1.0])
        else:
            possible_factors = laser_allowed_power_factors

        laser_step_power_factors[j] = local_rng.choice(possible_factors)

    return np.repeat(laser_step_power_factors, laser_step_samples)[:sample_count]


def solve_chirped_frequency_sweep_trajectory():
    if not driven_run_frequency_sweep_trajectory:
        return None

    if not (cython_baoab_available and use_force_lookup_table and force_lookup_ready):
        print(
            "Skipping chirped trajectory simulation because the compiled "
            "lookup Cython solver is not available."
        )
        return None

    dt_local = 1.0 / driven_frequency_sweep_sampling_frequency
    sample_count = int(round(driven_frequency_sweep_duration / dt_local)) + 1
    time_values = np.arange(sample_count, dtype=np.float64) * dt_local
    power_factor = chirped_laser_power_factor(time_values)
    power_factor *= chirped_laser_noise_factor(sample_count, dt_local)

    if driven_use_brownian_motion:
        local_rng = np.random.default_rng(seed=brownian_seed + 1000)
        brownian_x = local_rng.normal(size=sample_count - 1)
        brownian_y = local_rng.normal(size=sample_count - 1)
        brownian_z = local_rng.normal(size=sample_count - 1)
    else:
        brownian_x = np.zeros(sample_count - 1)
        brownian_y = np.zeros(sample_count - 1)
        brownian_z = np.zeros(sample_count - 1)

    damping_factor = np.exp(-gamma_baoab * dt_local)
    thermal_velocity_scale = np.sqrt(
        (kB * T / m)
        * (1.0 - damping_factor**2)
    )

    result = solve_baoab_3d_lookup_cython(
        np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
        np.ascontiguousarray(power_factor, dtype=np.float64),
        np.ascontiguousarray(brownian_x, dtype=np.float64),
        np.ascontiguousarray(brownian_y, dtype=np.float64),
        np.ascontiguousarray(brownian_z, dtype=np.float64),
        dt_local,
        m,
        m * g,
        damping_factor,
        thermal_velocity_scale,
        x0,
        y0,
        z0,
        vx0,
        vy0,
        vz0,
        radial_zero_tolerance,
        use_axial_energy_loss,
        np.ascontiguousarray(axial_energy_z_values, dtype=np.float64),
        np.ascontiguousarray(axial_potential_values, dtype=np.float64),
        z_eq if axial_lower_escape_z is None else axial_lower_escape_z,
        axial_lower_escape_energy,
        terminate_on_trap_loss,
        trap_loss_check_interval_steps,
        x_eq,
        y_eq,
        z_eq,
        trap_loss_x_outward_limit,
        trap_loss_below_equilibrium_limit,
    )

    (
        x_out,
        y_out,
        z_out,
        vx_out,
        vy_out,
        vz_out,
        out_of_bounds_count,
        stop_index,
        lost_flag,
    ) = result

    if lost_flag:
        reason = trap_loss_reason(
            x_out[stop_index],
            y_out[stop_index],
            z_out[stop_index],
            vx_out[stop_index],
            vz_out[stop_index],
            power_factor=power_factor[stop_index],
        )
        if reason is None:
            reason = "trap-loss boundary exceeded"
        record_trap_loss(
            "chirped frequency sweep",
            stop_index,
            x_out[stop_index],
            y_out[stop_index],
            z_out[stop_index],
            reason,
        )
        result_slice = slice(None, stop_index + 1)
    else:
        result_slice = slice(None)

    if out_of_bounds_count > 0:
        print(
            "Warning: chirped frequency sweep clamped",
            out_of_bounds_count,
            "force lookups to the lookup-table edge.",
        )

    time_values = time_values[result_slice]
    z_displacement = z_out[result_slice] - z_eq
    power_factor = power_factor[result_slice]
    frequencies, psd = positive_driven_psd(z_displacement, time_values)

    return {
        "time": time_values,
        "z_displacement": z_displacement,
        "power_factor": power_factor,
        "frequencies": frequencies,
        "psd": psd,
        "sampling_frequency": driven_frequency_sweep_sampling_frequency,
    }


def plot_chirped_frequency_sweep_trajectory():
    result = solve_chirped_frequency_sweep_trajectory()
    if result is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_time, plot_z = downsample_for_plot(
        result["time"],
        result["z_displacement"] * 1e6,
        max_plot_points,
    )
    axes[0].plot(plot_time, plot_z, linewidth=0.65)
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axes[0].set_title("Particle trajectory during chirped laser drive")
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("z displacement / micrometres")
    axes[0].grid(True, alpha=0.35)

    axes[1].loglog(result["frequencies"], result["psd"], linewidth=1.0)
    axes[1].set_title("PSD of chirped-drive trajectory")
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("z PSD / m^2 Hz^-1")
    axes[1].grid(True, which="both", alpha=0.35)

    fig.suptitle(
        (
            "Particle response to laser-drive frequency sweep "
            f"({driven_frequency_sweep_start_hz:g}-"
            f"{driven_frequency_sweep_end_hz:g} Hz over "
            f"{driven_frequency_sweep_duration:g} s, "
            f"sampled at {result['sampling_frequency']:g} Hz)"
        )
    )
    save_and_finish_driven_figure(
        fig,
        f"{driven_output_prefix}_laser_frequency_sweep_trajectory_psd.png",
    )


def run_driven_dynamics_study_plots():
    run_pressure_step_settling_table()
    plot_axial_bode_response()
    plot_simulated_driven_amplitude_sweep()
    plot_parametric_comparison_cases()
    plot_square_frequency_amplitude_sweep()
    plot_power_step_response()
    plot_power_sine_response()
    plot_resonant_peak_trajectories()
    plot_modulation_amplitude_sweep()
    plot_chirped_laser_frequency_sweep()
    plot_chirped_frequency_sweep_trajectory()

    if not driven_plot_standard_driven_cases:
        return

    for waveform in driven_standard_waveforms:
        if driven_plot_single_cycle_cases:
            plot_single_cycle_driven_summary(waveform)
        if len(driven_continuous_frequencies_hz) <= 3:
            plot_continuous_driven_summary(
                waveform,
                driven_continuous_frequencies_hz,
                "targeted",
            )
            run_current_pressure_step_response_metrics()
        else:
            low_frequencies = driven_continuous_frequencies_hz[:3]
            high_frequencies = driven_continuous_frequencies_hz[3:]
            plot_continuous_driven_summary(
                waveform,
                low_frequencies,
                "low",
            )
            high_frequency_results = plot_continuous_driven_summary(
                waveform,
                high_frequencies,
                "high",
            )
            plot_high_frequency_zoomed_trajectories(waveform, high_frequency_results)
            run_current_pressure_step_response_metrics()


brownian_normals_x = rng.normal(size=len(t_baoab) - 1)
brownian_normals_y = rng.normal(size=len(t_baoab) - 1)
brownian_normals_z = rng.normal(size=len(t_baoab) - 1)
zero_brownian_normals = np.zeros(len(t_baoab) - 1)
constant_power_factor = np.ones_like(t_baoab)

if run_driven_dynamics_study:
    run_driven_dynamics_study_plots()
    print(
        "Driven-dynamics study complete. Original random laser-noise plots "
        "were skipped for this script."
    )
    raise SystemExit(0)

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
    "BAOAB constant power",
)
baoab_constant_power_runtime = perf_counter() - baoab_constant_power_start_time

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
    "BAOAB laser noise",
)
baoab_laser_noise_runtime = perf_counter() - baoab_laser_noise_start_time

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
    "BAOAB laser noise without Brownian motion",
)
baoab_laser_noise_without_brownian_runtime = (
    perf_counter() - baoab_laser_noise_without_brownian_start_time
)

feedback_result = None
baoab_feedback_runtime = 0.0

if use_pd_feedback:
    baoab_feedback_start_time = perf_counter()
    feedback_result = solve_baoab_3d_fast_with_pd_feedback(
        laser_power_factor,
        brownian_normals_x,
        brownian_normals_y,
        brownian_normals_z,
        "BAOAB with PD feedback",
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

analysis_lengths = [
    len(x_baoab_constant_power),
    len(x_baoab),
    len(x_baoab_no_brownian),
]

if use_pd_feedback:
    analysis_lengths.append(len(x_baoab_feedback))

analysis_sample_count = min(analysis_lengths)

if analysis_sample_count < 2:
    raise SystemExit(
        "Simulation ended before two samples were available because the "
        "particle was already lost from the trap. Skipping trajectory and PSD plots."
    )

if analysis_sample_count < len(t_baoab):
    print(
        "Analysis truncated at t =",
        t_baoab[analysis_sample_count - 1],
        "s so all compared trajectories use the same time range."
    )


def trim_analysis_array(array):
    return array[:analysis_sample_count]


t_baoab = trim_analysis_array(t_baoab)
constant_power_factor = trim_analysis_array(constant_power_factor)
laser_power_factor = trim_analysis_array(laser_power_factor)
laser_power_time = trim_analysis_array(laser_power_time)
z_eq_laser_noise_time = trim_analysis_array(z_eq_laser_noise_time)

x_baoab_constant_power = trim_analysis_array(x_baoab_constant_power)
y_baoab_constant_power = trim_analysis_array(y_baoab_constant_power)
z_baoab_constant_power = trim_analysis_array(z_baoab_constant_power)
vx_baoab_constant_power = trim_analysis_array(vx_baoab_constant_power)
vy_baoab_constant_power = trim_analysis_array(vy_baoab_constant_power)
vz_baoab_constant_power = trim_analysis_array(vz_baoab_constant_power)

x_baoab = trim_analysis_array(x_baoab)
y_baoab = trim_analysis_array(y_baoab)
z_baoab = trim_analysis_array(z_baoab)
vx_baoab = trim_analysis_array(vx_baoab)
vy_baoab = trim_analysis_array(vy_baoab)
vz_baoab = trim_analysis_array(vz_baoab)

x_baoab_no_brownian = trim_analysis_array(x_baoab_no_brownian)
y_baoab_no_brownian = trim_analysis_array(y_baoab_no_brownian)
z_baoab_no_brownian = trim_analysis_array(z_baoab_no_brownian)
vx_baoab_no_brownian = trim_analysis_array(vx_baoab_no_brownian)
vy_baoab_no_brownian = trim_analysis_array(vy_baoab_no_brownian)
vz_baoab_no_brownian = trim_analysis_array(vz_baoab_no_brownian)

if use_pd_feedback:
    x_baoab_feedback = trim_analysis_array(x_baoab_feedback)
    y_baoab_feedback = trim_analysis_array(y_baoab_feedback)
    z_baoab_feedback = trim_analysis_array(z_baoab_feedback)
    vx_baoab_feedback = trim_analysis_array(vx_baoab_feedback)
    vy_baoab_feedback = trim_analysis_array(vy_baoab_feedback)
    vz_baoab_feedback = trim_analysis_array(vz_baoab_feedback)

    for feedback_key in (
        "t",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "command_power",
        "actual_power",
    ):
        feedback_result[feedback_key] = trim_analysis_array(feedback_result[feedback_key])

    feedback_time_limit = t_baoab[-1]
    feedback_control_mask = feedback_result["control_t"] <= feedback_time_limit
    for feedback_key in (
        "control_t",
        "measurement_t",
        "z_measured",
        "z_filtered",
        "requested_power",
    ):
        feedback_result[feedback_key] = feedback_result[feedback_key][feedback_control_mask]

baoab_total_runtime = (
    baoab_constant_power_runtime
    + baoab_laser_noise_runtime
    + baoab_laser_noise_without_brownian_runtime
    + baoab_feedback_runtime
)

x_laser_noise_difference = x_baoab - x_baoab_constant_power
y_laser_noise_difference = y_baoab - y_baoab_constant_power
z_laser_noise_difference = z_baoab - z_baoab_constant_power

x_brownian_difference = x_baoab - x_baoab_no_brownian
y_brownian_difference = y_baoab - y_baoab_no_brownian
z_brownian_difference = z_baoab - z_baoab_no_brownian

# print("BAOAB constant-power runtime =", baoab_constant_power_runtime, "s")
# print("BAOAB laser-noise runtime =", baoab_laser_noise_runtime, "s")
# print("BAOAB laser-noise without-Brownian runtime =", baoab_laser_noise_without_brownian_runtime, "s")
# print("BAOAB feedback runtime =", baoab_feedback_runtime, "s")
# print("BAOAB total runtime =", baoab_total_runtime, "s")
print("Total calculation runtime, excluding graph-viewing time =", perf_counter() - script_start_time, "s")

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


plot_experimental_simulation_first_graph()


















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
plt.step(t_plot, laser_power_time[plot_slice], where="post", label="laser power")
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
    signal = signal - np.mean(signal)

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
        detrend=False,
        scaling="density",
        return_onesided=True
    )

    positive_mask = (frequencies > 0) & (psd > 0)
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]

    return frequencies, psd


def positive_welch_averaged_psd(signal, time_values):
    dt = time_values[1] - time_values[0]
    fs = 1 / dt
    signal = signal - np.mean(signal)

    nperseg = min(
        max(2, int(round(welch_average_segment_duration_seconds * fs))),
        len(signal)
    )

    if nperseg < 2:
        raise ValueError("Welch PSD settings must leave at least two samples.")

    noverlap = min(int(round(psd_overlap_fraction * nperseg)), nperseg - 1)

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True
    )

    positive_mask = (frequencies > 0) & (psd > 0)
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]

    return frequencies, psd, nperseg, fs / nperseg


def analytic_langevin_displacement_psd(frequencies_hz, omega_0):
    omega = 2 * np.pi * frequencies_hz
    denominator = (omega_0**2 - omega**2)**2 + gamma_baoab**2 * omega**2
    return 4 * kB * T * gamma_baoab / (m * denominator)


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


baoab_x_freqs, baoab_x_psd = positive_psd(x_baoab - x_eq, t_baoab)
baoab_y_freqs, baoab_y_psd = positive_psd(y_baoab - y_eq, t_baoab)
baoab_z_freqs, baoab_z_psd = positive_psd(z_baoab - z_eq, t_baoab)

minimum_resolvable_frequency = baoab_x_freqs[0]
minimum_plot_frequency = psd_min_frequency_factor * minimum_resolvable_frequency

# print("Minimum resolvable non-zero frequency =", minimum_resolvable_frequency, "Hz")
# print("Lower frequency shown on plot =", minimum_plot_frequency, "Hz")

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
plt.loglog(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="BAOAB z PSD")
apply_psd_axes()
plt.title("PSD of BAOAB z motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

baoab_x_welch_freqs, baoab_x_welch_psd, welch_nperseg, welch_bin_size = (
    positive_welch_averaged_psd(x_baoab - x_eq, t_baoab)
)
baoab_y_welch_freqs, baoab_y_welch_psd, _, _ = positive_welch_averaged_psd(
    y_baoab - y_eq,
    t_baoab
)
baoab_z_welch_freqs, baoab_z_welch_psd, _, _ = positive_welch_averaged_psd(
    z_baoab - z_eq,
    t_baoab
)

constant_x_welch_freqs, constant_x_welch_psd, _, _ = positive_welch_averaged_psd(
    x_baoab_constant_power - x_eq,
    t_baoab
)
constant_y_welch_freqs, constant_y_welch_psd, _, _ = positive_welch_averaged_psd(
    y_baoab_constant_power - y_eq,
    t_baoab
)
constant_z_welch_freqs, constant_z_welch_psd, _, _ = positive_welch_averaged_psd(
    z_baoab_constant_power - z_eq,
    t_baoab
)

print("Welch-averaged PSD segment duration =", welch_average_segment_duration_seconds, "s")
print("Welch-averaged PSD nperseg =", welch_nperseg)
print("Welch-averaged PSD frequency bin size =", welch_bin_size, "Hz")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_welch_freqs, baoab_x_welch_psd, label="BAOAB x Welch-averaged PSD")
apply_psd_axes()
plt.title("Welch-averaged PSD of BAOAB x motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

plt.figure(figsize=spectrum_figsize)
plt.loglog(
    baoab_y_welch_freqs,
    baoab_y_welch_psd,
    color="tab:green",
    label="BAOAB y Welch-averaged PSD"
)
apply_psd_axes()
plt.title("Welch-averaged PSD of BAOAB y motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

plt.figure(figsize=spectrum_figsize)
plt.loglog(
    baoab_z_welch_freqs,
    baoab_z_welch_psd,
    color="tab:orange",
    label="BAOAB z Welch-averaged PSD"
)
apply_psd_axes()
plt.title("Welch-averaged PSD of BAOAB z motion")
plt.legend()
plt.grid(True, which="both")
finish_plot()

if include_analytic_langevin_validation_psd:
    fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)
    validation_psd_data = (
        (
            axes[0],
            constant_x_welch_freqs,
            constant_x_welch_psd,
            omega_x,
            "x",
            "tab:blue",
        ),
        (
            axes[1],
            constant_y_welch_freqs,
            constant_y_welch_psd,
            omega_y,
            "y",
            "tab:green",
        ),
        (
            axes[2],
            constant_z_welch_freqs,
            constant_z_welch_psd,
            omega_z,
            "z",
            "tab:orange",
        ),
    )

    for ax, frequencies, simulated_psd, omega_0, label, colour in validation_psd_data:
        analytic_psd = analytic_langevin_displacement_psd(frequencies, omega_0)
        ax.loglog(
            frequencies,
            simulated_psd,
            color=colour,
            linewidth=1.1,
            label=f"simulated {label} PSD, constant power"
        )
        ax.loglog(
            frequencies,
            analytic_psd,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"analytic {label} harmonic Langevin PSD"
        )
        ax.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
        ax.set_xlim(minimum_plot_frequency, psd_plot_max_frequency)
        if psd_max_value is None:
            ax.set_ylim(bottom=psd_min_value)
        else:
            ax.set_ylim(psd_min_value, psd_max_value)
        ax.set_ylabel("PSD / m^2 Hz^-1")
        ax.legend()
        ax.grid(True, which="both")

    axes[-1].set_xlabel("Frequency / Hz")
    fig.suptitle("Brownian PSD validation against analytic harmonic Langevin model")
    plt.tight_layout()
    finish_plot()

if include_z_detector_resolution_psd:
    detector_position_resolution_m = detector_position_resolution_nm * 1e-9
    z_displacement = z_baoab - z_eq
    z_displacement_measured = quantise_position_resolution(
        z_displacement,
        detector_position_resolution_m
    )
    measured_z_freqs, measured_z_psd = positive_psd(
        z_displacement_measured,
        t_baoab
    )
    measured_z_welch_freqs, measured_z_welch_psd, _, _ = (
        positive_welch_averaged_psd(
            z_displacement_measured,
            t_baoab
        )
    )

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
        label="true z PSD"
    )
    axes[1].loglog(
        measured_z_freqs,
        measured_z_psd,
        color="tab:green",
        alpha=0.45,
        label=f"z PSD with {detector_position_resolution_nm:g} nm resolution"
    )
    apply_psd_axes()
    axes[1].set_title("PSD of BAOAB z motion with detector resolution")
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.tight_layout()
    finish_plot()

    plt.figure(figsize=spectrum_figsize)
    plt.loglog(
        baoab_z_welch_freqs,
        baoab_z_welch_psd,
        color="tab:orange",
        alpha=0.45,
        label="true z Welch-averaged PSD"
    )
    plt.loglog(
        measured_z_welch_freqs,
        measured_z_welch_psd,
        color="tab:green",
        alpha=0.45,
        label=f"z Welch-averaged PSD with {detector_position_resolution_nm:g} nm resolution"
    )
    apply_psd_axes()
    plt.title("Welch-averaged PSD of BAOAB z motion with detector resolution")
    plt.legend()
    plt.grid(True, which="both")
    finish_plot()

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
laser_effect_x_welch_freqs, laser_effect_x_welch_psd, _, _ = (
    positive_welch_averaged_psd(
        x_laser_noise_difference,
        t_baoab
    )
)
laser_effect_y_welch_freqs, laser_effect_y_welch_psd, _, _ = (
    positive_welch_averaged_psd(
        y_laser_noise_difference,
        t_baoab
    )
)
laser_effect_z_welch_freqs, laser_effect_z_welch_psd, _, _ = (
    positive_welch_averaged_psd(
        z_laser_noise_difference,
        t_baoab
    )
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

plt.figure(figsize=spectrum_figsize)
plt.loglog(
    laser_effect_x_welch_freqs,
    laser_effect_x_welch_psd,
    label="x laser-noise effect"
)
plt.loglog(
    laser_effect_y_welch_freqs,
    laser_effect_y_welch_psd,
    color="tab:green",
    label="y laser-noise effect"
)
plt.loglog(
    laser_effect_z_welch_freqs,
    laser_effect_z_welch_psd,
    color="tab:orange",
    label="z laser-noise effect"
)
apply_psd_axes()
plt.title("Welch-averaged PSD of isolated laser-noise displacement effect")
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
