import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import welch
from time import perf_counter

from pathlib import Path
import sys
import os

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Publication style ---
fontsize = 7
mpl.rcParams.update({
    # Figure
    "figure.figsize": (4.0, 2.6),  # single column
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    # Font
    "font.family": "sans-serif",
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize - 1,
    "ytick.labelsize": fontsize - 1,
    "legend.fontsize": fontsize - 1,
    "legend.frameon": False,
    "legend.framealpha": 0.0,
    "legend.facecolor": "none",
    "legend.edgecolor": "none",
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
save_path = SCRIPT_DIR / "Phase_space"
os.makedirs(save_path, exist_ok=True)

CYTHON_DIR = Path("/Users/josephwhitfield/Masters/Summer Project/cython")
sys.path.insert(0, str(CYTHON_DIR))

try:
    import baoab_3d_cython_particle_loading as _baoab_3d_cython
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
    solve_baoab_3d_lookup_cython_with_loss = getattr(
        _baoab_3d_cython,
        "solve_baoab_3d_lookup_cython_with_loss",
        None
    )
    cython_baoab_available = solve_baoab_3d_lookup_cython is not None
    cython_feedback_baoab_available = solve_baoab_3d_feedback_lookup_cython is not None
except ImportError:
    solve_baoab_3d_lookup_cython = None
    solve_baoab_3d_feedback_lookup_cython = None
    solve_baoab_3d_lookup_cython_with_loss = None
    cython_baoab_available = False
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
radius = 6.3e-6          # m
density = 1100         # kg/m^3
n_particle = 1.555      # particle refractive index
n_medium = 1.00027     # surrounding medium refractive index, air

# Gas properties
p = 100        # pressure, Pa
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
use_m2_rayleigh_range = True
zR_manual = 100e-6
P_laser = 0.075             # W, example laser power
use_laser_power_noise = False
laser_noise_fraction = 0.01
laser_noise_frequency=300
laser_noise_step_duration = 1/laser_noise_frequency     # s

# Ashkin ray-optics sampling
ray_grid_points = 100

# Photophoretic force parameters
k_particle = 0.135       # W/(m K)
alpha_acc = 1.0
kappa_t = 1.14
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
x_displacement = 50e-6
y_displacement = 0e-6
z_displacement = 10000e-6
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0

# Time integration
t_start = 0
t_end =5.0
dt_baoab = 1 / 30000
brownian_seed = 900453457347589
laser_noise_seed = 1

# Batch particle-loading controls
batch_output_prefix = "particle_loading_batch"
batch_random_seed = 12345
batch_grid_points = 120
batch_t_end = t_end
batch_use_brownian = False
batch_use_laser_noise = False
display_plots = False
run_phase_space_maps = True
run_bottle_loading_monte_carlo_mode = False
phase_space_colormap = "orange_blue"
phase_space_save_grid_data = True

# Edit these ranges to explore a different loading region.
# z is measured relative to the beam focus, so z = 0 is the focus.
batch_z_range_um = np.linspace(-1000, 5000.0, batch_grid_points)
batch_vz_range_m_per_s = np.linspace(-7.5, 5, batch_grid_points)
batch_x_range_um = np.linspace(-50.0, 50.0, batch_grid_points)
batch_xz_x_range_um = np.linspace(-200.0, 200.0, batch_grid_points)
batch_vx_range_m_per_s = np.linspace(-0.2, 0.2, batch_grid_points)
batch_fixed_z_um = 1000.0
batch_force_diagnostic_x_points = 121
batch_force_diagnostic_z_points = 121
batch_force_diagnostic_quiver_stride = 8

# Bottle-loading Monte Carlo controls
monte_carlo_output_prefix = "bottle_loading_monte_carlo"
monte_carlo_particle_count = 1000000
monte_carlo_bottle_diameter_m = 0.025
monte_carlo_start_z_m = 0.05
monte_carlo_vx_mean_m_per_s = 0.0
monte_carlo_vy_mean_m_per_s = 0.0
monte_carlo_vz_mean_m_per_s = 0.0
monte_carlo_vx_std_m_per_s = 0.0
monte_carlo_vy_std_m_per_s = 0.0
monte_carlo_vz_std_m_per_s = 0.0
monte_carlo_entry_axial_limit_fraction = 0.95
monte_carlo_entry_radial_limit_fraction = 0.95
monte_carlo_max_freefall_time_s = 60.0
monte_carlo_max_captured_trajectories_to_plot = 12

# Trap-loss termination
# A particle is lost only after it remains outside the deterministic trapping
# region for a continuous time. Axially, the boundary is the lower unstable
# equilibrium z_u. Radially, the boundary is the outer zero crossing of
# F_r(r, z_eq), falling back to a deliberately large numerical safeguard if no
# clean restoring-region edge is found.
terminate_on_trap_loss = True
trap_loss_check_interval_steps = 1
trap_loss_radial_limit_manual = None
trap_loss_axial_boundary_manual = None
trap_loss_radial_beam_waists = 10.0
trap_loss_radial_boundary_scan_points = 2000
trap_loss_sustained_outside_time = 0.02
trap_loss_post_loss_plot_time = 0.0

# Successful trapping classification
trap_success_radial_limit_factor = 0.5
trap_success_axial_limit_factor = 0.5
trap_success_speed_limit = 0.02

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
time_trace_figsize = (4.0, 2.6)
single_diagnostic_figsize = (4.0, 2.6)
spectrum_figsize = (4.0, 2.6)
trajectory_projection_figsize = (4.0, 2.6)
force_check_figsize = (4.0, 2.6)
force_field_figsize = (4.0, 2.6)


plot_data_counter = 0


def save_plot_data_csv(fig, csv_path):
    rows = []

    for axis_index, axis in enumerate(fig.axes):
        axis_name = axis.get_ylabel() or axis.get_xlabel() or f"axis_{axis_index}"

        for line_index, line in enumerate(axis.get_lines()):
            x_data = np.asarray(line.get_xdata())
            y_data = np.asarray(line.get_ydata())
            label = line.get_label()

            if label.startswith("_"):
                label = f"line_{line_index}"

            for x_value, y_value in zip(x_data, y_data):
                rows.append((axis_index, axis_name, label, x_value, y_value))

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["axis_index", "axis_name", "line_label", "x", "y"])
        writer.writerows(rows)


def make_legends_transparent(fig):
    for axis in fig.axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            frame = legend.get_frame()
            frame.set_facecolor("none")
            frame.set_edgecolor("none")
            frame.set_alpha(0.0)
            frame.set_linewidth(0.0)

    for legend in fig.legends:
        legend.set_frame_on(False)
        frame = legend.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        frame.set_alpha(0.0)
        frame.set_linewidth(0.0)

def finish_plot(name=None, fig=None):
    target_fig = fig if fig is not None else plt.gcf()

    if name is not None:
        make_legends_transparent(target_fig)
        save_plot_data_csv(target_fig, os.path.join(save_path, f"{name}.csv"))
        target_fig.savefig(os.path.join(save_path, f"{name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
        target_fig.savefig(os.path.join(save_path, f"{name}.png"), bbox_inches="tight", facecolor="white", transparent=False)

    if display_plots:
        plt.show()
    else:
        plt.close()


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
equilibrium_root_info = []

for root in roots:
    slope = numerical_derivative_1d(Fz_net_on_axis, root)
    stability = "stable" if slope < 0 else "unstable"
    equilibrium_root_info.append((root, slope, stability))
    if slope < 0:
        stable_roots.append(root)

print("First two on-axis equilibrium positions:")
for equilibrium_index, (root, slope, stability) in enumerate(equilibrium_root_info[:2], start=1):
    print(
        f"  equilibrium {equilibrium_index}: "
        f"z = {root * 1e6:.6g} micrometres ({stability}, "
        f"dFz_net/dz = {slope:.6e} N/m)"
    )

if len(equilibrium_root_info) < 2:
    print(f"  only {len(equilibrium_root_info)} on-axis equilibrium position(s) found")

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

lower_unstable_roots = [
    root
    for root, _, stability in equilibrium_root_info
    if stability == "unstable" and root < z_eq
]

if trap_loss_axial_boundary_manual is None:
    if len(lower_unstable_roots) == 0:
        raise ValueError(
            "No lower unstable on-axis equilibrium was found for the axial "
            "loss boundary. Set trap_loss_axial_boundary_manual to override."
        )
    trap_loss_axial_boundary_z = max(lower_unstable_roots)
else:
    trap_loss_axial_boundary_z = trap_loss_axial_boundary_manual

if trap_loss_axial_boundary_z >= z_eq:
    raise ValueError(
        "The axial trap-loss boundary must be below the selected stable "
        f"equilibrium. z_u={trap_loss_axial_boundary_z:.6e} m, "
        f"z_eq={z_eq:.6e} m."
    )

trap_loss_axial_limit = z_eq - trap_loss_axial_boundary_z

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

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)

trap_loss_check_interval_steps = max(1, int(trap_loss_check_interval_steps))

def radial_force_at_stable_equilibrium(r):
    Fx, _ = F_optical_2d(r, z_eq)
    return Fx


def find_radial_restoring_boundary():
    if trap_loss_radial_limit_manual is not None:
        return trap_loss_radial_limit_manual, "manual"

    fallback_limit = trap_loss_radial_beam_waists * w0
    if use_force_lookup_table and force_lookup_ready:
        fallback_limit = max(fallback_limit, force_lookup_r_max)

    scan_points = max(3, int(trap_loss_radial_boundary_scan_points))
    r_min_scan = max(radial_zero_tolerance, fallback_limit * 1e-9)
    r_values = np.linspace(r_min_scan, fallback_limit, scan_points)
    Fr_values = np.asarray(radial_force_at_stable_equilibrium(r_values))

    finite = np.isfinite(Fr_values)
    if np.count_nonzero(finite) < 2:
        return fallback_limit, "fallback"

    r_values = r_values[finite]
    Fr_values = Fr_values[finite]

    for i in range(len(r_values) - 1):
        if Fr_values[i] < 0.0 and Fr_values[i + 1] >= 0.0:
            if Fr_values[i + 1] == 0.0:
                return r_values[i + 1], "radial force zero crossing"
            return brentq(
                radial_force_at_stable_equilibrium,
                r_values[i],
                r_values[i + 1],
                xtol=root_finding_xtol,
                rtol=root_finding_rtol
            ), "radial force zero crossing"

    return fallback_limit, "fallback"


trap_loss_radial_limit, trap_loss_radial_boundary_source = find_radial_restoring_boundary()

if trap_loss_radial_limit <= 0:
    raise ValueError("trap_loss_radial_limit must be positive.")

trap_loss_sustained_outside_steps = max(
    1,
    int(np.ceil(trap_loss_sustained_outside_time / dt_baoab))
)
trap_loss_post_loss_plot_steps = max(
    0,
    int(np.ceil(trap_loss_post_loss_plot_time / dt_baoab))
)
trap_loss_events = {}


def make_trap_loss_state():
    return {"outside_start_sample": None}


def trap_region_metrics(x, y, z):
    radial_displacement = np.sqrt((x - x_eq)**2 + (y - y_eq)**2)
    axial_displacement = z - z_eq
    outside_radial = radial_displacement > trap_loss_radial_limit
    outside_axial = z < trap_loss_axial_boundary_z
    return radial_displacement, axial_displacement, outside_radial, outside_axial


def is_inside_capture_region(x, y, z, tolerance=1.0):
    radial_displacement, axial_displacement, _, _ = trap_region_metrics(x, y, z)
    return (
        radial_displacement <= tolerance * trap_loss_radial_limit
        and z >= z_eq - tolerance * trap_loss_axial_limit
    )


def trap_loss_reason(x, y, z, vx, vy, vz, power_factor, sample_index, loss_state):
    if not terminate_on_trap_loss:
        return None

    if not np.all(np.isfinite([x, y, z, vx, vy, vz])):
        return "position or velocity became non-finite"

    radial_displacement, axial_displacement, outside_radial, outside_axial = (
        trap_region_metrics(x, y, z)
    )

    if not (outside_axial or outside_radial):
        loss_state["outside_start_sample"] = None
        return None

    if loss_state["outside_start_sample"] is None:
        loss_state["outside_start_sample"] = sample_index

    outside_steps = sample_index - loss_state["outside_start_sample"] + 1

    if outside_steps < trap_loss_sustained_outside_steps:
        return None

    outside_time = outside_steps * dt_baoab
    active_boundaries = []
    if outside_axial:
        active_boundaries.append("z < z_u")
    if outside_radial:
        active_boundaries.append("r > r_u")

    return (
        "outside deterministic trapping region continuously "
        f"for {outside_time:.3g} s ({', '.join(active_boundaries)}); "
        f"radial displacement = {radial_displacement * 1e6:.3g} micrometres, "
        f"axial displacement = {axial_displacement * 1e6:.3g} micrometres"
    )


def classify_trapping_result(x_final, y_final, z_final, vx_final, vy_final, vz_final, lost):
    radial_displacement = np.sqrt((x_final - x_eq)**2 + (y_final - y_eq)**2)
    axial_displacement = abs(z_final - z_eq)
    speed = np.sqrt(vx_final**2 + vy_final**2 + vz_final**2)

    captured = (
        not lost
        and radial_displacement <= trap_success_radial_limit_factor * trap_loss_radial_limit
        and axial_displacement <= trap_success_axial_limit_factor * trap_loss_axial_limit
        and speed <= trap_success_speed_limit
    )

    return captured, radial_displacement, axial_displacement, speed


def should_check_trap_loss(sample_index):
    return (
        terminate_on_trap_loss
        and (
            sample_index % trap_loss_check_interval_steps == 0
            or sample_index == len(t_baoab) - 1
        )
    )


def record_trap_loss(run_label, sample_index, x, y, z, reason, stop_sample_index=None):
    if stop_sample_index is None:
        stop_sample_index = min(
            len(t_baoab) - 1,
            sample_index + trap_loss_post_loss_plot_steps
        )
    trap_loss_events[run_label] = {
        "sample_index": sample_index,
        "time": t_baoab[sample_index],
        "post_loss_stop_sample_index": stop_sample_index,
        "post_loss_stop_time": t_baoab[stop_sample_index],
        "x": x,
        "y": y,
        "z": z,
        "reason": reason,
    }
    print("LOST")
    print(
        f"{run_label}: particle lost from trap at t = "
        f"{t_baoab[sample_index]:.6g} s; {reason}. Continuing to "
        f"t = {t_baoab[stop_sample_index]:.6g} s for post-loss plotting."
    )


def post_loss_stop_reached(run_label, sample_index):
    event = trap_loss_events.get(run_label)
    return (
        event is not None
        and sample_index >= event["post_loss_stop_sample_index"]
    )


def cython_loss_reason_from_code(reason_code, x, y, z):
    if reason_code == 1:
        return "position or velocity became non-finite"

    radial_displacement, axial_displacement, _, _ = trap_region_metrics(x, y, z)
    return (
        "outside deterministic trapping region continuously "
        f"for at least {trap_loss_sustained_outside_steps * dt_baoab:.3g} s; "
        f"radial displacement = {radial_displacement * 1e6:.3g} micrometres, "
        f"axial displacement = {axial_displacement * 1e6:.3g} micrometres"
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
        "Trap-loss radial boundary r_u =",
        trap_loss_radial_limit * 1e6,
        "micrometres",
        f"({trap_loss_radial_boundary_source})"
    )
    print(
        "Trap-loss axial boundary z_u =",
        trap_loss_axial_boundary_z * 1e6,
        "micrometres"
    )
    print(
        "Trap-loss axial distance z_eq - z_u =",
        trap_loss_axial_limit * 1e6,
        "micrometres"
    )
    print(
        "Trap-loss sustained outside time =",
        trap_loss_sustained_outside_steps * dt_baoab,
        "s"
    )
    print(
        "Trap-loss post-loss plotting time =",
        trap_loss_post_loss_plot_steps * dt_baoab,
        "s"
    )

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

    loss_state = make_trap_loss_state()

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
        if post_loss_stop_reached(run_label, sample_index):
            return truncate_outputs(
                sample_index,
                x_out,
                y_out,
                z_out,
                vx_out,
                vy_out,
                vz_out
            )

        if run_label not in trap_loss_events and should_check_trap_loss(sample_index):
            reason = trap_loss_reason(
                x_i,
                y_i,
                z_i,
                vx_i,
                vy_i,
                vz_i,
                power_factor_i,
                sample_index,
                loss_state
            )
            if reason is not None:
                record_trap_loss(run_label, sample_index, x_i, y_i, z_i, reason)

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
    if (
        terminate_on_trap_loss
        and solve_baoab_3d_lookup_cython_with_loss is not None
        and use_force_lookup_table
        and force_lookup_ready
    ):
        result = solve_baoab_3d_lookup_cython_with_loss(
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
            x_eq,
            y_eq,
            z_eq,
            trap_loss_radial_limit,
            trap_loss_axial_boundary_z,
            trap_loss_check_interval_steps,
            trap_loss_sustained_outside_steps,
            trap_loss_post_loss_plot_steps,
            radial_zero_tolerance,
        )

        (
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out,
            out_of_bounds_count,
            loss_sample_index,
            post_loss_stop_sample_index,
            loss_reason_code,
        ) = result

        if loss_sample_index >= 0:
            reason = cython_loss_reason_from_code(
                loss_reason_code,
                x_out[loss_sample_index],
                y_out[loss_sample_index],
                z_out[loss_sample_index]
            )
            record_trap_loss(
                run_label,
                loss_sample_index,
                x_out[loss_sample_index],
                y_out[loss_sample_index],
                z_out[loss_sample_index],
                reason,
                post_loss_stop_sample_index
            )

        if out_of_bounds_count > 0:
            pass

        return x_out, y_out, z_out, vx_out, vy_out, vz_out

    if (
        not terminate_on_trap_loss
        and cython_baoab_available
        and use_force_lookup_table
        and force_lookup_ready
    ):
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
        )

        (
            x_out,
            y_out,
            z_out,
            vx_out,
            vy_out,
            vz_out,
            out_of_bounds_count,
        ) = result

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
    loss_state = make_trap_loss_state()

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
        if post_loss_stop_reached(run_label, sample_index):
            return build_feedback_result(sample_index)

        if run_label not in trap_loss_events and should_check_trap_loss(sample_index):
            reason = trap_loss_reason(
                x_i,
                y_i,
                z_i,
                vx_i,
                vy_i,
                vz_i,
                actual_power_factor,
                sample_index,
                loss_state
            )
            if reason is not None:
                record_trap_loss(run_label, sample_index, x_i, y_i, z_i, reason)

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
    if (
        not terminate_on_trap_loss
        and cython_feedback_baoab_available
        and use_force_lookup_table
        and force_lookup_ready
    ):
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
        brownian_normals_z,
        run_label
    )



# ****************************************************************************************************************************************************
# Batch particle-loading phase-space sweeps
# ****************************************************************************************************************************************************
# This script runs three 2D maps. In each map only the two named initial-condition
# variables vary; all other initial offsets/velocities are set to zero.
if solve_baoab_3d_lookup_cython_with_loss is None:
    raise ImportError(
        "Could not import solve_baoab_3d_lookup_cython_with_loss from "
        "baoab_3d_cython_particle_loading. Rebuild with "
        "python3 setup_baoab_3d_particle_loading.py build_ext --inplace."
    )

# Rebuild the integration time array for batch runs in case batch_t_end is changed.
t_baoab = np.arange(t_start, batch_t_end + 0.5 * dt_baoab, dt_baoab)
if batch_use_laser_noise:
    batch_laser_rng = np.random.default_rng(seed=laser_noise_seed)
    batch_laser_step_samples = max(1, int(round(laser_noise_step_duration / dt_baoab)))
    batch_n_laser_steps = int(np.ceil(len(t_baoab) / batch_laser_step_samples))
    batch_laser_step_power_factors = batch_laser_rng.choice(
        [1 - laser_noise_fraction, 1 + laser_noise_fraction],
        size=batch_n_laser_steps
    )
    batch_power_factor_time = np.repeat(
        batch_laser_step_power_factors,
        batch_laser_step_samples
    )[:len(t_baoab)]
else:
    batch_power_factor_time = np.ones_like(t_baoab)

batch_rng = np.random.default_rng(seed=batch_random_seed)


def batch_brownian_normals():
    if batch_use_brownian:
        return (
            batch_rng.normal(size=len(t_baoab) - 1),
            batch_rng.normal(size=len(t_baoab) - 1),
            batch_rng.normal(size=len(t_baoab) - 1),
        )

    zeros = np.zeros(len(t_baoab) - 1)
    return zeros, zeros, zeros


def run_batch_particle(sweep_name, grid_i, grid_j, x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial):
    bx, by, bz = batch_brownian_normals()
    result = solve_baoab_3d_lookup_cython_with_loss(
        np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
        np.ascontiguousarray(batch_power_factor_time, dtype=np.float64),
        np.ascontiguousarray(bx, dtype=np.float64),
        np.ascontiguousarray(by, dtype=np.float64),
        np.ascontiguousarray(bz, dtype=np.float64),
        dt_baoab,
        m,
        m * g,
        baoab_damping_factor,
        baoab_thermal_velocity_scale,
        x_initial,
        y_initial,
        z_initial,
        vx_initial,
        vy_initial,
        vz_initial,
        x_eq,
        y_eq,
        z_eq,
        trap_loss_radial_limit,
        trap_loss_axial_boundary_z,
        trap_loss_check_interval_steps,
        trap_loss_sustained_outside_steps,
        trap_loss_post_loss_plot_steps,
        radial_zero_tolerance,
    )

    (
        x_out,
        y_out,
        z_out,
        vx_out,
        vy_out,
        vz_out,
        out_of_bounds_count,
        loss_sample_index,
        post_loss_stop_sample_index,
        loss_reason_code,
    ) = result

    lost = loss_sample_index >= 0
    captured, final_radial_displacement, final_axial_displacement, final_speed = classify_trapping_result(
        x_out[-1],
        y_out[-1],
        z_out[-1],
        vx_out[-1],
        vy_out[-1],
        vz_out[-1],
        lost,
    )

    initial_radial = np.sqrt((x_initial - x_eq)**2 + (y_initial - y_eq)**2)
    initial_radial_velocity = 0.0
    if initial_radial > radial_zero_tolerance:
        initial_radial_velocity = (
            (x_initial - x_eq) * vx_initial + (y_initial - y_eq) * vy_initial
        ) / initial_radial

    return {
        "sweep": sweep_name,
        "grid_i": grid_i,
        "grid_j": grid_j,
        "initial_x_m": x_initial,
        "initial_y_m": y_initial,
        "initial_z_m": z_initial,
        "initial_x_focus_relative_m": x_initial,
        "initial_z_focus_relative_m": z_initial,
        "initial_x_relative_m": x_initial - x_eq,
        "initial_r_m": initial_radial,
        "initial_z_relative_m": z_initial - z_eq,
        "initial_vx_m_per_s": vx_initial,
        "initial_vy_m_per_s": vy_initial,
        "initial_vz_m_per_s": vz_initial,
        "initial_vr_m_per_s": initial_radial_velocity,
        "scanned_velocity_m_per_s": vx_initial,
        "captured": bool(captured),
        "lost": bool(lost),
        "loss_time_s": t_baoab[loss_sample_index] if lost else np.nan,
        "loss_reason_code": int(loss_reason_code),
        "post_loss_stop_time_s": t_baoab[post_loss_stop_sample_index] if lost else np.nan,
        "final_x_m": x_out[-1],
        "final_y_m": y_out[-1],
        "final_z_m": z_out[-1],
        "final_z_focus_relative_m": z_out[-1],
        "final_r_m": final_radial_displacement,
        "final_z_relative_m": z_out[-1] - z_eq,
        "final_vx_m_per_s": vx_out[-1],
        "final_vy_m_per_s": vy_out[-1],
        "final_vz_m_per_s": vz_out[-1],
        "final_speed_m_per_s": final_speed,
        "samples_returned": len(x_out),
        "out_of_bounds_count": int(out_of_bounds_count),
    }


def run_phase_space_sweeps():
    rows = []

    total_runs = 3 * batch_grid_points * batch_grid_points
    completed_runs = 0

    for i, z_initial_um in enumerate(batch_z_range_um):
        for j, vz_initial in enumerate(batch_vz_range_m_per_s):
            rows.append(run_batch_particle(
                "z0_vs_vz0",
                i,
                j,
                0.0,
                0.0,
                z_initial_um * 1e-6,
                0.0,
                0.0,
                vz_initial,
            ))
            completed_runs += 1
            if completed_runs % max(1, total_runs // 10) == 0:
                print(f"Completed {completed_runs}/{total_runs} batch runs")

    for i, x_initial_um in enumerate(batch_x_range_um):
        for j, vx_initial in enumerate(batch_vx_range_m_per_s):
            rows.append(run_batch_particle(
                "x0_vs_vx0",
                i,
                j,
                x_initial_um * 1e-6,
                0.0,
                batch_fixed_z_um * 1e-6,
                vx_initial,
                0.0,
                0.0,
            ))
            completed_runs += 1
            if completed_runs % max(1, total_runs // 10) == 0:
                print(f"Completed {completed_runs}/{total_runs} batch runs")

    for i, x_initial_um in enumerate(batch_xz_x_range_um):
        for j, z_initial_um in enumerate(batch_z_range_um):
            rows.append(run_batch_particle(
                "x0_vs_z0",
                i,
                j,
                x_initial_um * 1e-6,
                0.0,
                z_initial_um * 1e-6,
                0.0,
                0.0,
                0.0,
            ))
            completed_runs += 1
            if completed_runs % max(1, total_runs // 10) == 0:
                print(f"Completed {completed_runs}/{total_runs} batch runs")

    return rows


def save_batch_csv(rows, csv_path):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_equilibrium_markers(ax, markers_um, axis):
    if not markers_um:
        return

    axis_min, axis_max = ax.get_xlim() if axis == "x" else ax.get_ylim()
    marker_styles = {
        "stable": ("tab:blue", "-"),
        "unstable": ("tab:orange", "--"),
    }

    visible_markers = [
        marker for marker in markers_um
        if axis_min <= marker[0] <= axis_max
    ]

    for marker_index, (position_um, label, stability) in enumerate(visible_markers):
        color, linestyle = marker_styles.get(stability, ("0.25", ":"))
        if axis == "x":
            ax.axvline(
                position_um,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.9,
                zorder=0,
            )
            ax.text(
                position_um,
                0.98 - 0.10 * (marker_index % 3),
                label,
                color=color,
                rotation=90,
                ha="left",
                va="top",
                fontsize=fontsize - 1,
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            )
        else:
            ax.axhline(
                position_um,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.9,
                zorder=0,
            )
            ax.text(
                0.99,
                position_um,
                label,
                color=color,
                ha="right",
                va="center",
                fontsize=fontsize - 1,
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            )

    if axis == "x":
        ax.set_xlim(axis_min, axis_max)
    else:
        ax.set_ylim(axis_min, axis_max)


def equilibrium_markers_for_axis(axis_key):
    if axis_key in ("initial_x_focus_relative_um", "initial_x_relative_um"):
        x_stability = "stable" if kx > 0 else "unstable"
        return [(0.0, f"{x_stability} x eq", x_stability)]

    if axis_key in ("initial_z_focus_relative_um", "initial_z_relative_um"):
        return [
            (
                root * 1e6,
                f"{stability} z eq",
                stability,
            )
            for root, _, stability in equilibrium_root_info
        ]

    return []


def grid_edges(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step])

    midpoints = 0.5 * (values[:-1] + values[1:])
    first_edge = values[0] - 0.5 * (values[1] - values[0])
    last_edge = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def phase_space_title(output_name):
    title = output_name.replace("phase_space_", "").replace("_", " ")
    return f"Phase space: {title}"


def phase_space_status_colours():
    if phase_space_colormap == "orange_blue":
        return {
            0: "#1f77b4",
            1: "#ff7f0e",
            2: "#b0b0b0",
        }

    cmap = plt.get_cmap(phase_space_colormap)
    return {
        0: cmap(0.45),
        1: cmap(1.0),
        2: cmap(0.0),
    }


def save_phase_space_grid_data(
    output_name,
    sweep_name,
    x_values,
    y_values,
    status_grid,
    x_key,
    y_key,
    x_label,
    y_label,
):
    if not phase_space_save_grid_data:
        return

    npz_path = os.path.join(save_path, f"{output_name}_grid_data.npz")
    np.savez_compressed(
        npz_path,
        sweep_name=sweep_name,
        x_key=x_key,
        y_key=y_key,
        x_label=x_label,
        y_label=y_label,
        x_values=x_values,
        y_values=y_values,
        x_edges=grid_edges(x_values),
        y_edges=grid_edges(y_values),
        status_grid=status_grid,
        status_code_captured=0,
        status_code_lost=1,
        status_code_not_trapped=2,
        colormap=phase_space_colormap,
    )
    print("Saved phase-space grid data to", npz_path)


def plot_phase_space(rows, sweep_name, x_key, y_key, x_label, y_label, output_name):
    sweep_rows = [row for row in rows if row["sweep"] == sweep_name]
    fig, ax = plt.subplots(figsize=force_check_figsize)

    x_values = np.array(sorted({row[x_key] for row in sweep_rows}), dtype=float)
    y_values = np.array(sorted({row[y_key] for row in sweep_rows}), dtype=float)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    status_grid = np.full((len(y_values), len(x_values)), np.nan)

    for row in sweep_rows:
        if row["captured"]:
            status = 0
        elif row["lost"]:
            status = 1
        else:
            status = 2

        status_grid[y_index[row[y_key]], x_index[row[x_key]]] = status

    phase_colours = phase_space_status_colours()
    phase_cmap = mpl.colors.ListedColormap(
        [phase_colours[0], phase_colours[1], phase_colours[2]]
    )
    phase_norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], phase_cmap.N)
    ax.pcolormesh(
        grid_edges(x_values),
        grid_edges(y_values),
        status_grid,
        cmap=phase_cmap,
        norm=phase_norm,
        shading="flat",
        alpha=0.88,
        rasterized=True,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(phase_space_title(output_name), pad=8)
    ax.grid(color="white", linestyle="-", linewidth=0.35, alpha=0.45)
    add_equilibrium_markers(
        ax,
        equilibrium_markers_for_axis(x_key),
        "x",
    )
    add_equilibrium_markers(
        ax,
        equilibrium_markers_for_axis(y_key),
        "y",
    )
    status_legend = {
        0: (phase_colours[0], "successfully trapped"),
        1: (phase_colours[1], "lost"),
        2: (phase_colours[2], "not trapped by final time"),
    }
    present_statuses = sorted({
        int(status)
        for status in status_grid[np.isfinite(status_grid)]
    })
    legend_handles = [
        mpl.patches.Patch(
            facecolor=status_legend[status][0],
            edgecolor="none",
            alpha=0.88,
            label=status_legend[status][1],
        )
        for status in present_statuses
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        borderpad=0.4,
        handlelength=1.2,
        handletextpad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.85")
    legend.get_frame().set_alpha(0.88)
    plt.tight_layout()
    save_phase_space_grid_data(
        output_name,
        sweep_name,
        x_values,
        y_values,
        status_grid,
        x_key,
        y_key,
        x_label,
        y_label,
    )
    fig.savefig(os.path.join(save_path, f"{output_name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(os.path.join(save_path, f"{output_name}.png"), bbox_inches="tight", facecolor="white", transparent=False)
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def row_capture_status(row):
    if row["captured"]:
        return 0
    if row["lost"]:
        return 1
    return 2


def find_boundary_validation_pair(rows, sweep_name, x_key, y_key):
    sweep_rows = [row for row in rows if row["sweep"] == sweep_name]
    row_by_grid = {
        (row["grid_i"], row["grid_j"]): row
        for row in sweep_rows
    }
    x_values = np.array([row[x_key] for row in sweep_rows], dtype=float)
    y_values = np.array([row[y_key] for row in sweep_rows], dtype=float)
    x_center = 0.5 * (np.min(x_values) + np.max(x_values))
    y_center = 0.5 * (np.min(y_values) + np.max(y_values))
    x_span = max(np.ptp(x_values), 1.0)
    y_span = max(np.ptp(y_values), 1.0)

    best_pair = None
    best_score = np.inf

    for row in sweep_rows:
        for offset_i, offset_j in ((1, 0), (0, 1)):
            neighbour = row_by_grid.get((row["grid_i"] + offset_i, row["grid_j"] + offset_j))
            if neighbour is None:
                continue

            statuses = {row_capture_status(row), row_capture_status(neighbour)}
            if statuses != {0, 1}:
                continue

            midpoint_x = 0.5 * (row[x_key] + neighbour[x_key])
            midpoint_y = 0.5 * (row[y_key] + neighbour[y_key])
            score = (
                ((midpoint_x - x_center) / x_span)**2
                + ((midpoint_y - y_center) / y_span)**2
            )

            if score < best_score:
                best_score = score
                best_pair = (row, neighbour)

    if best_pair is None:
        return None

    first, second = best_pair
    if row_capture_status(first) == 0:
        return first, second
    return second, first


def run_boundary_validation_trajectory(row):
    bx, by, bz = batch_brownian_normals()
    result = solve_baoab_3d_lookup_cython_with_loss(
        np.ascontiguousarray(force_lookup_r_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_z_values, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fr_table, dtype=np.float64),
        np.ascontiguousarray(force_lookup_Fz_table, dtype=np.float64),
        np.ascontiguousarray(batch_power_factor_time, dtype=np.float64),
        np.ascontiguousarray(bx, dtype=np.float64),
        np.ascontiguousarray(by, dtype=np.float64),
        np.ascontiguousarray(bz, dtype=np.float64),
        dt_baoab,
        m,
        m * g,
        baoab_damping_factor,
        baoab_thermal_velocity_scale,
        row["initial_x_m"],
        row["initial_y_m"],
        row["initial_z_m"],
        row["initial_vx_m_per_s"],
        row["initial_vy_m_per_s"],
        row["initial_vz_m_per_s"],
        x_eq,
        y_eq,
        z_eq,
        trap_loss_radial_limit,
        trap_loss_axial_boundary_z,
        trap_loss_check_interval_steps,
        trap_loss_sustained_outside_steps,
        trap_loss_post_loss_plot_steps,
        radial_zero_tolerance,
    )

    (
        x_out,
        y_out,
        z_out,
        vx_out,
        vy_out,
        vz_out,
        out_of_bounds_count,
        loss_sample_index,
        post_loss_stop_sample_index,
        loss_reason_code,
    ) = result

    return {
        "t": t_baoab[:len(x_out)],
        "x": x_out,
        "y": y_out,
        "z": z_out,
        "vx": vx_out,
        "vy": vy_out,
        "vz": vz_out,
        "lost": loss_sample_index >= 0,
        "loss_sample_index": loss_sample_index,
        "out_of_bounds_count": out_of_bounds_count,
    }


def format_initial_condition(row):
    return (
        f"x0={row['initial_x_focus_relative_m'] * 1e6:.4g} um, "
        f"z0={row['initial_z_focus_relative_m'] * 1e6:.4g} um, "
        f"vx0={row['initial_vx_m_per_s']:.4g} m/s, "
        f"vz0={row['initial_vz_m_per_s']:.4g} m/s"
    )


def add_time_trace_equilibrium_lines(ax_x, ax_z):
    ax_x.axhline(
        x_eq * 1e6,
        color="tab:blue",
        linestyle="-",
        linewidth=0.9,
        alpha=0.75,
        label="stable x eq",
    )

    for root, _, stability in equilibrium_root_info:
        color = "tab:blue" if stability == "stable" else "tab:orange"
        linestyle = "-" if stability == "stable" else "--"
        ax_z.axhline(
            root * 1e6,
            color=color,
            linestyle=linestyle,
            linewidth=0.9,
            alpha=0.75,
            label=f"{stability} z eq",
        )


def plot_boundary_validation_trajectories(rows, sweep_name, x_key, y_key, output_name):
    pair = find_boundary_validation_pair(rows, sweep_name, x_key, y_key)
    if pair is None:
        print(f"No captured/lost boundary pair found for {sweep_name}; skipping validation trajectory plot.")
        return

    captured_row, lost_row = pair
    captured_traj = run_boundary_validation_trajectory(captured_row)
    lost_traj = run_boundary_validation_trajectory(lost_row)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.4, 3.4),
        sharex=True,
        constrained_layout=True,
    )
    ax_x, ax_z = axes

    trajectory_specs = (
        ("captured side", captured_row, captured_traj, "#2ca25f"),
        ("lost side", lost_row, lost_traj, "#d94841"),
    )

    for label, row, trajectory, color in trajectory_specs:
        ax_x.plot(
            trajectory["t"],
            trajectory["x"] * 1e6,
            color=color,
            linewidth=1.2,
            label=label,
        )
        ax_z.plot(
            trajectory["t"],
            trajectory["z"] * 1e6,
            color=color,
            linewidth=1.2,
            label=label,
        )

        if trajectory["lost"]:
            loss_time = trajectory["t"][trajectory["loss_sample_index"]]
            ax_x.axvline(loss_time, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
            ax_z.axvline(loss_time, color=color, linestyle=":", linewidth=0.8, alpha=0.7)

    add_time_trace_equilibrium_lines(ax_x, ax_z)

    ax_x.set_ylabel(r"$x$ / $\mu$m")
    ax_z.set_ylabel(r"$z$ / $\mu$m")
    ax_z.set_xlabel(r"$t$ / s")
    ax_x.grid(color="0.85", linestyle="-", linewidth=0.4)
    ax_z.grid(color="0.85", linestyle="-", linewidth=0.4)
    fig.suptitle(
        "Boundary trajectory check: "
        f"{output_name.replace('boundary_check_', '').replace('_', ' ')}\n"
        f"captured start: {format_initial_condition(captured_row)}\n"
        f"lost start: {format_initial_condition(lost_row)}",
        fontsize=fontsize - 1,
    )
    ax_x.legend(loc="best", fontsize=fontsize - 2, frameon=True)
    ax_z.legend(loc="best", fontsize=fontsize - 2, frameon=True)

    fig.savefig(os.path.join(save_path, f"{output_name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(os.path.join(save_path, f"{output_name}.png"), bbox_inches="tight", facecolor="white", transparent=False)
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_batch_force_diagnostic():
    x_um = np.linspace(
        np.min(batch_x_range_um),
        np.max(batch_x_range_um),
        batch_force_diagnostic_x_points,
    )
    z_um = np.linspace(
        np.min(batch_z_range_um),
        np.max(batch_z_range_um),
        batch_force_diagnostic_z_points,
    )
    X_um, Z_um = np.meshgrid(x_um, z_um, indexing="xy")
    X = X_um * 1e-6
    Z = Z_um * 1e-6

    Fx, _, Fz = F_optical_3d(X, np.zeros_like(X), Z)
    ax_transverse = Fx / m
    az_net = (Fz - m * g) / m

    restoring_acceleration = np.zeros_like(ax_transverse)
    nonzero_x = np.abs(X) > radial_zero_tolerance
    restoring_acceleration[nonzero_x] = (
        -np.sign(X[nonzero_x]) * ax_transverse[nonzero_x]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    heatmaps = (
        (
            axes[0],
            ax_transverse,
            r"$a_x = F_x/m$ / m s$^{-2}$",
            "Transverse acceleration",
            "coolwarm",
        ),
        (
            axes[1],
            restoring_acceleration,
            r"inward $x$ acceleration / m s$^{-2}$",
            "Restoring acceleration",
            "BrBG",
        ),
    )

    for axis, values, colorbar_label, title, cmap in heatmaps:
        finite_abs_max = np.nanmax(np.abs(values))
        if finite_abs_max <= 0:
            finite_abs_max = 1.0

        mesh = axis.pcolormesh(
            X_um,
            Z_um,
            values,
            cmap=cmap,
            norm=mpl.colors.TwoSlopeNorm(
                vmin=-finite_abs_max,
                vcenter=0.0,
                vmax=finite_abs_max,
            ),
            shading="auto",
            rasterized=True,
        )
        fig.colorbar(mesh, ax=axis, label=colorbar_label)

        try:
            axis.contour(
                X_um,
                Z_um,
                values,
                levels=[0.0],
                colors="black",
                linewidths=0.7,
                alpha=0.85,
            )
        except ValueError:
            pass

        q_slice = (
            slice(None, None, batch_force_diagnostic_quiver_stride),
            slice(None, None, batch_force_diagnostic_quiver_stride),
        )
        vector_norm = np.sqrt(ax_transverse[q_slice]**2 + az_net[q_slice]**2)
        vector_norm = np.where(vector_norm > 0, vector_norm, 1.0)
        axis.quiver(
            X_um[q_slice],
            Z_um[q_slice],
            ax_transverse[q_slice] / vector_norm,
            az_net[q_slice] / vector_norm,
            color="0.15",
            angles="xy",
            scale_units="xy",
            scale=0.035,
            width=0.0025,
            alpha=0.65,
        )

        add_equilibrium_markers(
            axis,
            equilibrium_markers_for_axis("initial_x_focus_relative_um"),
            "x",
        )
        add_equilibrium_markers(
            axis,
            equilibrium_markers_for_axis("initial_z_focus_relative_um"),
            "y",
        )
        axis.set_title(title, pad=6)
        axis.set_xlabel(r"$x$ / $\mu$m")
        axis.grid(color="white", linewidth=0.35, alpha=0.35)

    axes[0].set_ylabel(r"$z$ / $\mu$m")
    fig.suptitle(
        "Force-field diagnostic over the batch x0-z0 region",
        fontsize=fontsize,
    )
    output_name = "force_field_batch_x0_z0_diagnostic"
    fig.savefig(os.path.join(save_path, f"{output_name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(os.path.join(save_path, f"{output_name}.png"), bbox_inches="tight", facecolor="white", transparent=False)
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def add_micrometre_columns(row):
    row["initial_x_focus_relative_um"] = row["initial_x_focus_relative_m"] * 1e6
    row["initial_z_focus_relative_um"] = row["initial_z_focus_relative_m"] * 1e6
    row["initial_x_relative_um"] = row["initial_x_relative_m"] * 1e6
    row["initial_r_um"] = row["initial_r_m"] * 1e6
    row["initial_z_relative_um"] = row["initial_z_relative_m"] * 1e6
    row["final_z_focus_relative_um"] = row["final_z_focus_relative_m"] * 1e6
    row["final_r_um"] = row["final_r_m"] * 1e6
    row["final_z_relative_um"] = row["final_z_relative_m"] * 1e6


def sample_uniform_bottle_disk(rng, particle_count):
    bottle_radius = 0.5 * monte_carlo_bottle_diameter_m
    disk_radius = bottle_radius * np.sqrt(rng.random(particle_count))
    disk_angle = rng.uniform(0.0, 2 * np.pi, particle_count)
    x_initial = disk_radius * np.cos(disk_angle)
    y_initial = disk_radius * np.sin(disk_angle)
    return x_initial, y_initial


def freefall_to_z_plane(x0_free, y0_free, z0_free, vx0_free, vy0_free, vz0_free, target_z):
    if z0_free <= target_z:
        return 0.0, x0_free, y0_free, z0_free, vx0_free, vy0_free, vz0_free

    gamma = gamma_baoab

    if gamma <= 0:
        fall_distance = z0_free - target_z
        discriminant = vz0_free**2 + 2 * g * fall_distance
        if discriminant < 0:
            return None
        fall_time = (vz0_free + np.sqrt(discriminant)) / g
        return (
            fall_time,
            x0_free + vx0_free * fall_time,
            y0_free + vy0_free * fall_time,
            target_z,
            vx0_free,
            vy0_free,
            vz0_free - g * fall_time,
        )

    def z_position_after_time(t_value):
        return (
            z0_free
            + (vz0_free + g / gamma) * (1 - np.exp(-gamma * t_value)) / gamma
            - (g / gamma) * t_value
        )

    def crossing_function(t_value):
        return z_position_after_time(t_value) - target_z

    high_time = min(1.0, monte_carlo_max_freefall_time_s)
    while crossing_function(high_time) > 0 and high_time < monte_carlo_max_freefall_time_s:
        high_time = min(2 * high_time, monte_carlo_max_freefall_time_s)

    if crossing_function(high_time) > 0:
        return None

    fall_time = brentq(
        crossing_function,
        0.0,
        high_time,
        xtol=root_finding_xtol,
        rtol=root_finding_rtol,
    )
    damping = np.exp(-gamma * fall_time)
    x_entry = x0_free + vx0_free * (1 - damping) / gamma
    y_entry = y0_free + vy0_free * (1 - damping) / gamma
    vx_entry = vx0_free * damping
    vy_entry = vy0_free * damping
    vz_entry = (vz0_free + g / gamma) * damping - g / gamma

    return fall_time, x_entry, y_entry, target_z, vx_entry, vy_entry, vz_entry


def make_monte_carlo_miss_row(
    particle_index,
    x_initial,
    y_initial,
    z_initial,
    vx_initial,
    vy_initial,
    vz_initial,
    miss_reason,
):
    initial_radial = np.sqrt((x_initial - x_eq)**2 + (y_initial - y_eq)**2)
    row = {
        "sweep": "bottle_loading_monte_carlo",
        "grid_i": particle_index,
        "grid_j": 0,
        "particle_index": particle_index,
        "simulated_from_entry": False,
        "miss_reason": miss_reason,
        "bottle_initial_x_m": x_initial,
        "bottle_initial_y_m": y_initial,
        "bottle_initial_z_m": z_initial,
        "bottle_initial_r_m": initial_radial,
        "bottle_initial_vx_m_per_s": vx_initial,
        "bottle_initial_vy_m_per_s": vy_initial,
        "bottle_initial_vz_m_per_s": vz_initial,
        "entry_time_s": np.nan,
        "initial_x_m": x_initial,
        "initial_y_m": y_initial,
        "initial_z_m": z_initial,
        "initial_x_focus_relative_m": x_initial,
        "initial_z_focus_relative_m": z_initial,
        "initial_x_relative_m": x_initial - x_eq,
        "initial_r_m": initial_radial,
        "initial_z_relative_m": z_initial - z_eq,
        "initial_vx_m_per_s": vx_initial,
        "initial_vy_m_per_s": vy_initial,
        "initial_vz_m_per_s": vz_initial,
        "initial_vr_m_per_s": np.nan,
        "scanned_velocity_m_per_s": np.nan,
        "captured": False,
        "lost": True,
        "loss_time_s": np.nan,
        "loss_reason_code": -1,
        "post_loss_stop_time_s": np.nan,
        "final_x_m": np.nan,
        "final_y_m": np.nan,
        "final_z_m": np.nan,
        "final_z_focus_relative_m": np.nan,
        "final_r_m": np.nan,
        "final_z_relative_m": np.nan,
        "final_vx_m_per_s": np.nan,
        "final_vy_m_per_s": np.nan,
        "final_vz_m_per_s": np.nan,
        "final_speed_m_per_s": np.nan,
        "samples_returned": 0,
        "out_of_bounds_count": 0,
    }
    add_micrometre_columns(row)
    row["bottle_initial_x_mm"] = row["bottle_initial_x_m"] * 1e3
    row["bottle_initial_y_mm"] = row["bottle_initial_y_m"] * 1e3
    row["bottle_initial_r_mm"] = row["bottle_initial_r_m"] * 1e3
    return row


def run_bottle_loading_monte_carlo():
    rng = np.random.default_rng(seed=batch_random_seed)
    x_samples, y_samples = sample_uniform_bottle_disk(
        rng,
        monte_carlo_particle_count,
    )
    vx_samples = rng.normal(
        monte_carlo_vx_mean_m_per_s,
        monte_carlo_vx_std_m_per_s,
        monte_carlo_particle_count,
    )
    vy_samples = rng.normal(
        monte_carlo_vy_mean_m_per_s,
        monte_carlo_vy_std_m_per_s,
        monte_carlo_particle_count,
    )
    vz_samples = rng.normal(
        monte_carlo_vz_mean_m_per_s,
        monte_carlo_vz_std_m_per_s,
        monte_carlo_particle_count,
    )

    entry_z = min(
        monte_carlo_start_z_m,
        z_eq + monte_carlo_entry_axial_limit_fraction * trap_loss_axial_limit,
        force_lookup_z_values[-1],
    )
    entry_radial_limit = min(
        monte_carlo_entry_radial_limit_fraction * trap_loss_radial_limit,
        monte_carlo_entry_radial_limit_fraction * force_lookup_r_values[-1],
    )

    rows = []
    captured_trajectory_rows = []
    completed_runs = 0

    for particle_index in range(monte_carlo_particle_count):
        x_initial = x_samples[particle_index]
        y_initial = y_samples[particle_index]
        z_initial = monte_carlo_start_z_m
        vx_initial = vx_samples[particle_index]
        vy_initial = vy_samples[particle_index]
        vz_initial = vz_samples[particle_index]

        initial_radial = np.sqrt((x_initial - x_eq)**2 + (y_initial - y_eq)**2)
        if initial_radial > entry_radial_limit:
            rows.append(make_monte_carlo_miss_row(
                particle_index,
                x_initial,
                y_initial,
                z_initial,
                vx_initial,
                vy_initial,
                vz_initial,
                "outside near-axis simulation radius",
            ))
            continue

        entry_state = freefall_to_z_plane(
            x_initial,
            y_initial,
            z_initial,
            vx_initial,
            vy_initial,
            vz_initial,
            entry_z,
        )

        if entry_state is None:
            rows.append(make_monte_carlo_miss_row(
                particle_index,
                x_initial,
                y_initial,
                z_initial,
                vx_initial,
                vy_initial,
                vz_initial,
                "did not reach entry plane within max freefall time",
            ))
            continue

        (
            entry_time,
            x_entry,
            y_entry,
            z_entry,
            vx_entry,
            vy_entry,
            vz_entry,
        ) = entry_state

        entry_radial = np.sqrt((x_entry - x_eq)**2 + (y_entry - y_eq)**2)
        if entry_radial > entry_radial_limit:
            rows.append(make_monte_carlo_miss_row(
                particle_index,
                x_initial,
                y_initial,
                z_initial,
                vx_initial,
                vy_initial,
                vz_initial,
                "outside near-axis simulation radius at entry plane",
            ))
            continue

        row = run_batch_particle(
            "bottle_loading_monte_carlo",
            particle_index,
            0,
            x_entry,
            y_entry,
            z_entry,
            vx_entry,
            vy_entry,
            vz_entry,
        )
        row["particle_index"] = particle_index
        row["simulated_from_entry"] = True
        row["miss_reason"] = ""
        row["bottle_initial_x_m"] = x_initial
        row["bottle_initial_y_m"] = y_initial
        row["bottle_initial_z_m"] = z_initial
        row["bottle_initial_r_m"] = initial_radial
        row["bottle_initial_vx_m_per_s"] = vx_initial
        row["bottle_initial_vy_m_per_s"] = vy_initial
        row["bottle_initial_vz_m_per_s"] = vz_initial
        row["entry_time_s"] = entry_time
        add_micrometre_columns(row)
        row["bottle_initial_x_mm"] = row["bottle_initial_x_m"] * 1e3
        row["bottle_initial_y_mm"] = row["bottle_initial_y_m"] * 1e3
        row["bottle_initial_r_mm"] = row["bottle_initial_r_m"] * 1e3
        rows.append(row)

        if (
            row["captured"]
            and len(captured_trajectory_rows) < monte_carlo_max_captured_trajectories_to_plot
        ):
            captured_trajectory_rows.append(row)

        completed_runs += 1
        if completed_runs % max(1, monte_carlo_particle_count // 10) == 0:
            print(
                "Completed",
                completed_runs,
                "near-axis Monte Carlo trajectory runs",
            )

    return rows, captured_trajectory_rows, entry_z, entry_radial_limit


def plot_bottle_loading_initial_positions(rows, entry_radial_limit):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    bottle_radius_mm = 0.5 * monte_carlo_bottle_diameter_m * 1e3
    near_axis_radius_mm = entry_radial_limit * 1e3

    statuses = (
        ("captured", lambda row: row["captured"], "#2ca25f", 18, 0.95),
        (
            "simulated, not trapped",
            lambda row: row["simulated_from_entry"] and not row["captured"],
            "#d94841",
            12,
            0.85,
        ),
        (
            "outside near-axis region",
            lambda row: not row["simulated_from_entry"],
            "0.65",
            8,
            0.45,
        ),
    )

    for label, selector, color, size, alpha in statuses:
        selected = [row for row in rows if selector(row)]
        if not selected:
            continue
        ax.scatter(
            [row["bottle_initial_x_mm"] for row in selected],
            [row["bottle_initial_y_mm"] for row in selected],
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="none",
            label=label,
        )

    bottle_circle = plt.Circle(
        (0.0, 0.0),
        bottle_radius_mm,
        fill=False,
        color="black",
        linewidth=0.9,
        label="2.5 cm bottle",
    )
    near_axis_circle = plt.Circle(
        (0.0, 0.0),
        near_axis_radius_mm,
        fill=False,
        color="tab:blue",
        linestyle="--",
        linewidth=0.9,
        label="simulated near-axis region",
    )
    ax.add_patch(bottle_circle)
    ax.add_patch(near_axis_circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("initial x / mm")
    ax.set_ylabel("initial y / mm")
    ax.set_title(
        "Bottle-loading Monte Carlo initial positions\n"
        f"N={len(rows)}, z0={monte_carlo_start_z_m * 1e2:.3g} cm"
    )
    ax.grid(color="0.88", linewidth=0.4)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    output_name = f"{monte_carlo_output_prefix}_initial_xy"
    fig.savefig(os.path.join(save_path, f"{output_name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(os.path.join(save_path, f"{output_name}.png"), bbox_inches="tight", facecolor="white", transparent=False)
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_bottle_loading_captured_trajectories(captured_rows):
    if not captured_rows:
        print("No captured Monte Carlo particles; skipping captured trajectory plot.")
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.8, 3.5),
        sharex=True,
        constrained_layout=True,
    )
    ax_r, ax_z = axes

    for row in captured_rows:
        trajectory = run_boundary_validation_trajectory(row)
        radial_um = np.sqrt(
            (trajectory["x"] - x_eq)**2
            + (trajectory["y"] - y_eq)**2
        ) * 1e6
        label = (
            f"particle {row['particle_index']}: "
            f"x0={row['bottle_initial_x_mm']:.3g} mm, "
            f"y0={row['bottle_initial_y_mm']:.3g} mm"
        )
        ax_r.plot(trajectory["t"], radial_um, linewidth=0.9, label=label)
        ax_z.plot(trajectory["t"], trajectory["z"] * 1e6, linewidth=0.9)

    ax_r.axhline(trap_loss_radial_limit * 1e6, color="tab:red", linestyle=":", linewidth=0.8)
    for root, _, stability in equilibrium_root_info:
        color = "tab:blue" if stability == "stable" else "tab:orange"
        linestyle = "-" if stability == "stable" else "--"
        ax_z.axhline(root * 1e6, color=color, linestyle=linestyle, linewidth=0.8)

    ax_r.set_ylabel(r"radial displacement / $\mu$m")
    ax_z.set_ylabel(r"$z$ / $\mu$m")
    ax_z.set_xlabel("time / s")
    ax_r.set_title("Captured Monte Carlo trajectories")
    ax_r.grid(color="0.85", linewidth=0.4)
    ax_z.grid(color="0.85", linewidth=0.4)
    ax_r.legend(loc="best", fontsize=fontsize - 2, frameon=True)
    output_name = f"{monte_carlo_output_prefix}_captured_trajectories"
    fig.savefig(os.path.join(save_path, f"{output_name}.pdf"), bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(os.path.join(save_path, f"{output_name}.png"), bbox_inches="tight", facecolor="white", transparent=False)
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


batch_start_time = perf_counter()
print("Batch Brownian noise enabled =", batch_use_brownian)
print("Batch laser noise enabled =", batch_use_laser_noise)

if run_phase_space_maps:
    print("Starting particle-loading batch phase-space sweeps")
    print("Grid points per axis =", batch_grid_points)
    print("Runs =", 3 * batch_grid_points * batch_grid_points)

    batch_rows = run_phase_space_sweeps()
    for row in batch_rows:
        add_micrometre_columns(row)

    batch_csv_path = os.path.join(save_path, f"{batch_output_prefix}.csv")
    save_batch_csv(batch_rows, batch_csv_path)

    plot_phase_space(
        batch_rows,
        "z0_vs_vz0",
        "initial_z_focus_relative_um",
        "initial_vz_m_per_s",
        r"$z_0$ / $\mu$m",
        r"$v_{z0}$ / m s$^{-1}$",
        "phase_space_z0_vs_vz0",
    )
    plot_phase_space(
        batch_rows,
        "x0_vs_vx0",
        "initial_x_focus_relative_um",
        "scanned_velocity_m_per_s",
        r"$x_0$ / $\mu$m",
        r"$v_{x0}$ / m s$^{-1}$",
        "phase_space_x0_vs_vx0",
    )
    plot_phase_space(
        batch_rows,
        "x0_vs_z0",
        "initial_x_focus_relative_um",
        "initial_z_focus_relative_um",
        r"$x_0$ / $\mu$m",
        r"$z_0$ / $\mu$m",
        "phase_space_x0_vs_z0",
    )

    plot_batch_force_diagnostic()

    plot_boundary_validation_trajectories(
        batch_rows,
        "z0_vs_vz0",
        "initial_z_focus_relative_um",
        "initial_vz_m_per_s",
        "boundary_check_z0_vs_vz0",
    )
    plot_boundary_validation_trajectories(
        batch_rows,
        "x0_vs_vx0",
        "initial_x_focus_relative_um",
        "scanned_velocity_m_per_s",
        "boundary_check_x0_vs_vx0",
    )
    plot_boundary_validation_trajectories(
        batch_rows,
        "x0_vs_z0",
        "initial_x_focus_relative_um",
        "initial_z_focus_relative_um",
        "boundary_check_x0_vs_z0",
    )

    captured_count = sum(row["captured"] for row in batch_rows)
    lost_count = sum(row["lost"] for row in batch_rows)
    print("Saved batch data to", batch_csv_path)
    print("Successfully trapped particles =", captured_count)
    print("Lost particles =", lost_count)
    print("Not trapped by final time =", len(batch_rows) - captured_count - lost_count)

if run_bottle_loading_monte_carlo_mode:
    print("Starting bottle-loading Monte Carlo")
    print("Monte Carlo particles =", monte_carlo_particle_count)
    print("Bottle diameter =", monte_carlo_bottle_diameter_m * 1e2, "cm")
    print("Bottle start z =", monte_carlo_start_z_m * 1e2, "cm")

    (
        monte_carlo_rows,
        captured_trajectory_rows,
        monte_carlo_entry_z,
        monte_carlo_entry_radial_limit,
    ) = run_bottle_loading_monte_carlo()
    monte_carlo_csv_path = os.path.join(save_path, f"{monte_carlo_output_prefix}.csv")
    save_batch_csv(monte_carlo_rows, monte_carlo_csv_path)
    plot_bottle_loading_initial_positions(
        monte_carlo_rows,
        monte_carlo_entry_radial_limit,
    )
    plot_bottle_loading_captured_trajectories(captured_trajectory_rows)

    monte_carlo_captured_count = sum(row["captured"] for row in monte_carlo_rows)
    monte_carlo_simulated_count = sum(row["simulated_from_entry"] for row in monte_carlo_rows)
    monte_carlo_missed_count = len(monte_carlo_rows) - monte_carlo_simulated_count
    monte_carlo_lost_count = sum(row["lost"] for row in monte_carlo_rows)
    print("Saved Monte Carlo data to", monte_carlo_csv_path)
    print("Monte Carlo entry z =", monte_carlo_entry_z * 1e2, "cm")
    print("Near-axis simulation radius =", monte_carlo_entry_radial_limit * 1e6, "micrometres")
    print("Particles outside near-axis region =", monte_carlo_missed_count)
    print("Particles simulated through trap =", monte_carlo_simulated_count)
    print(
        "MONTE CARLO CAPTURED PARTICLES =",
        monte_carlo_captured_count,
        "out of",
        len(monte_carlo_rows),
    )
    print(
        "Monte Carlo capture fraction =",
        monte_carlo_captured_count / len(monte_carlo_rows),
    )
    print("Lost/missed particles =", monte_carlo_lost_count)
print("Batch runtime =", perf_counter() - batch_start_time, "s")
