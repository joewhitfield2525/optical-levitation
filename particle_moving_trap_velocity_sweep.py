import csv
import os
import sys
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


# ======================================================================
# Output and plotting
# ======================================================================
save_path = SCRIPT_DIR / "Moving_trap_velocity"
os.makedirs(save_path, exist_ok=True)

fontsize = 9
axis_label_fontsize = 9
legend_fontsize = 8
mpl.rcParams.update({
    "figure.figsize": (4.0, 2.6),
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": fontsize,
    "axes.labelsize": axis_label_fontsize,
    "axes.titlesize": axis_label_fontsize,
    "xtick.labelsize": axis_label_fontsize,
    "ytick.labelsize": axis_label_fontsize,
    "legend.fontsize": legend_fontsize,
    "legend.frameon": False,
    "legend.framealpha": 0.0,
    "legend.facecolor": "none",
    "legend.edgecolor": "none",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
})

_unit_parentheses_pattern = re.compile(r"^(?P<label>.*?)\s+\((?P<unit>[^()]*)\)$")


def capitalise_first_word(label):
    if not label:
        return label

    for index, character in enumerate(label):
        if character.isalpha():
            return label[:index] + character.upper() + label[index + 1:]

    return label


def format_axis_label(label):
    if not isinstance(label, str):
        return label

    match = _unit_parentheses_pattern.match(label)
    if match is not None:
        label = f"{match.group('label')} / {match.group('unit')}"

    return capitalise_first_word(label)


_original_axes_set_xlabel = Axes.set_xlabel
_original_axes_set_ylabel = Axes.set_ylabel
_original_axes3d_set_zlabel = Axes3D.set_zlabel
_original_colorbar_set_label = Colorbar.set_label
_original_figure_savefig = Figure.savefig


def set_xlabel_with_plot_style(self, xlabel, *args, **kwargs):
    return _original_axes_set_xlabel(self, format_axis_label(xlabel), *args, **kwargs)


def set_ylabel_with_plot_style(self, ylabel, *args, **kwargs):
    return _original_axes_set_ylabel(self, format_axis_label(ylabel), *args, **kwargs)


def set_zlabel_with_plot_style(self, zlabel, *args, **kwargs):
    return _original_axes3d_set_zlabel(self, format_axis_label(zlabel), *args, **kwargs)


def set_colorbar_label_with_plot_style(self, label, *args, **kwargs):
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


Axes.set_xlabel = set_xlabel_with_plot_style
Axes.set_ylabel = set_ylabel_with_plot_style
Axes.set_title = remove_plot_title
Axes3D.set_zlabel = set_zlabel_with_plot_style
Colorbar.set_label = set_colorbar_label_with_plot_style
Figure.suptitle = remove_plot_title
Figure.savefig = savefig_with_plot_text_style

display_plots = False
max_plot_points = 120000


# ======================================================================
# Physical constants
# ======================================================================
g = 9.81
c_light = 299792458
kB = 1.38e-23


# ======================================================================
# User-adjustable parameters
# ======================================================================
# Particle properties
radius = 6.3e-6
density = 1100
n_particle = 1.555
n_medium = 1.00027

# Gas properties
p = 100
T = 300
eta = 1.8e-5
M_air = 0.029
R = 8.314
d_air = 3.7e-10

# Laser / force parameters
w0 = 7e-6
wavelength = 532e-9
M2 = 1.2
use_m2_rayleigh_range = True
zR_manual = 100e-6
P_laser = 0.075

# Optional laser-power noise
use_laser_power_noise = False
laser_noise_fraction = 0.01
laser_noise_frequency = 300
laser_noise_step_duration = 1 / laser_noise_frequency
laser_noise_seed = 789

# Ashkin ray-optics sampling
ray_grid_points = 100

# Photophoretic force parameters
k_particle = 0.135
alpha_acc = 1.0
kappa_t = 1.14
extinction_coefficient = 3e-6
effective_absorption_path_length = 2 * radius
absorption_coefficient = 4 * np.pi * extinction_coefficient / wavelength
absorption_fraction = 1 - np.exp(
    -absorption_coefficient * effective_absorption_path_length
)

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
cunningham_A = 1.257
cunningham_B = 0.4
cunningham_C = 1.1
epstein_accommodation_alpha = 1.0
drag_transition_half_width_decades = 0.5

# Time integration
t_start = 0.0
t_end = 0.2
dt_baoab = 1 / 350000
brownian_seed = 900

# Brownian noise is off by default for the clean deterministic threshold.
# Turn this on to test a noisy limit using the same gas damping model.
use_brownian_noise = False

# Moving trap controls
moving_trap_output_prefix = "moving_trap_velocity"
moving_trap_start_x = 0.0
moving_trap_start_y = 0.0
moving_trap_start_z = 0.0
moving_trap_direction = "x"
initial_velocity_mode = "stationary_particle"
# Choose one of:
#   "stationary_particle"  particle starts at rest in the lab while the trap moves
#   "comoving_particle"    particle initially has the same velocity as the trap

# A coarse scan is saved first. The bisection search then refines the speed
# limit between a survived and lost run, if such a bracket is found.
run_velocity_scan = False
velocity_scan_m_per_s = np.linspace(0.0, 0.50, 11)

run_threshold_search = False
threshold_initial_high_velocity = 0.10
threshold_max_velocity = 5.0
threshold_bisection_iterations = 20
threshold_tolerance_m_per_s = 1e-6

# Set to a number to force one diagnostic speed. If None, the script plots the
# highest survived and lowest lost speeds found during the threshold search.
diagnostic_velocity_m_per_s = None

# Multi-pressure plot of loss time versus trap velocity. The velocity grid is
# logarithmic, so it must not include zero.
run_pressure_comparison_plot = False
pressure_comparison_pressures_pa = (100000.0, 1000.0, 10.0)
pressure_comparison_velocity_m_per_s = np.geomspace(1e-4, 0.50, 36)
pressure_comparison_xscale = "log"
pressure_comparison_yscale = "log"

# Multi-pressure threshold plot of maximum survived trap velocity.
run_pressure_threshold_plot = True
pressure_threshold_pressures_pa = np.geomspace(1.0, 100000.0, 30)

# Pressure dependence of the radial trap frequency.
run_pressure_radial_frequency_plot = False
pressure_radial_frequency_pressures_pa = np.geomspace(1.0, 100000.0, 30)

# 3D force lookup table. The moving-trap solver evaluates the optical force in
# the trap frame, so the radial lookup range must cover the particle lag.
use_force_lookup_table = True
force_lookup_grid_points_r = 101
force_lookup_grid_points_z = 801
force_lookup_r_min = 0.0
force_lookup_r_base_max = 80e-6
force_lookup_z_base_half_width = 3000e-6

# Trap-loss termination, measured relative to the moving trap equilibrium.
terminate_on_trap_loss = True
trap_loss_check_interval_steps = 1000
trap_loss_radial_limit_manual = None
trap_loss_axial_limit_manual = None
trap_loss_radial_beam_waists = 10.0
trap_loss_axial_rayleigh_ranges = 3.0
trap_loss_radial_boundary_search_max = 80e-6
trap_loss_radial_boundary_scan_points = 900
trap_loss_sustained_outside_time = 0.1
trap_loss_post_loss_plot_time = 3.0


# ======================================================================
# Derived quantities
# ======================================================================
volume = (4 / 3) * np.pi * radius**3
m = density * volume

rho_g = p * M_air / (R * T)
lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

zR_m2 = np.pi * w0**2 / (M2 * wavelength)
zR = zR_m2 if use_m2_rayleigh_range else zR_manual
I0 = 2 * P_laser / (np.pi * w0**2)


def buoyancy_force(pressure=None):
    if pressure is None:
        pressure = p

    rho_g_local = pressure * M_air / (R * T)
    return rho_g_local * volume * g


def effective_weight_force(pressure=None):
    return m * g - buoyancy_force(pressure)

force_lookup_ready = False
force_lookup_r_values = None
force_lookup_z_values = None
force_lookup_Fr_interpolator = None
force_lookup_Fz_interpolator = None
force_lookup_Fr_table = None
force_lookup_Fz_table = None
force_lookup_r_max = None
force_lookup_z_half_width = None

x_eq = None
y_eq = None
z_eq = None
equilibrium_root_info = []
stable_roots = []
trap_loss_axial_boundary_z = None
trap_loss_axial_limit = None
trap_loss_radial_limit = None
trap_loss_radial_boundary_source = None
trap_loss_sustained_outside_steps = None
trap_loss_post_loss_plot_steps = None
kx = None
ky = None
kz = None
omega_x = None
omega_y = None
omega_z = None
b = None
drag_model_used = None
gamma_baoab = None
baoab_damping_factor = None
baoab_thermal_velocity_scale = None


# ======================================================================
# Ray-optics force model
# ======================================================================
u_values = np.linspace(-radius, radius, ray_grid_points)
v_values = np.linspace(-radius, radius, ray_grid_points)
du = u_values[1] - u_values[0]
dv = v_values[1] - v_values[0]
dA = du * dv

U, V = np.meshgrid(u_values, v_values, indexing="ij")
rho = np.sqrt(U**2 + V**2)
ray_mask = rho < radius

U_hit = U[ray_mask]
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
Q_s = 0.5 * (Q_s_s_pol + Q_s_p_pol)
Q_g = 0.5 * (Q_g_s_pol + Q_g_p_pol)

rho_safe = np.where(rho_hit > 0, rho_hit, 1.0)
u_hat = U_hit / rho_safe

focus_zero_tolerance = 1e-30
radial_zero_tolerance = 1e-30


def beam_width(z):
    return w0 * np.sqrt(1 + (z / zR)**2)


def intensity(x, z):
    s = 1 + (z / zR)**2
    return (1 / s) * np.exp(-2 * x**2 / (w0**2 * s))


def physical_intensity(x, z, power_factor=1.0):
    return power_factor * I0 * intensity(x, z)


def wavefront_radius(z):
    if abs(z) < focus_zero_tolerance:
        return np.inf
    return z * (1 + (zR / z)**2)


def F_ray_optics_2d_components_scalar(x, z, power_factor=1.0):
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

    e_perp_x = s_z
    e_perp_z = -s_x

    Fx_scat = prefactor * np.sum(dP * Q_s * s_x)
    Fz_scat = prefactor * np.sum(dP * Q_s * s_z)
    Fx_grad = prefactor * np.sum(dP * (-Q_g * u_hat * e_perp_x))
    Fz_grad = prefactor * np.sum(dP * (-Q_g * u_hat * e_perp_z))

    return Fx_scat, Fz_scat, Fx_grad, Fz_grad


def F_ray_optics_2d_scalar(x, z, power_factor=1.0):
    Fx_scat, Fz_scat, Fx_grad, Fz_grad = F_ray_optics_2d_components_scalar(
        x,
        z,
        power_factor,
    )
    return Fx_scat + Fx_grad, Fz_scat + Fz_grad


def F_ray_optics_2d(x, z, power_factor=1.0):
    x_arr, z_arr, power_factor_arr = np.broadcast_arrays(x, z, power_factor)

    if x_arr.shape == ():
        return F_ray_optics_2d_scalar(
            float(x_arr),
            float(z_arr),
            float(power_factor_arr),
        )

    Fx = np.zeros_like(x_arr, dtype=float)
    Fz = np.zeros_like(z_arr, dtype=float)

    for index in np.ndindex(x_arr.shape):
        Fx[index], Fz[index] = F_ray_optics_2d_scalar(
            x_arr[index],
            z_arr[index],
            power_factor_arr[index],
        )

    return Fx, Fz


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
    return 0.0, photophoretic_force_magnitude(x, z, power_factor)


def F_optical_2d_direct(x, z, power_factor=1.0):
    Fx_ray, Fz_ray = F_ray_optics_2d(x, z, power_factor)
    Fx_photo, Fz_photo = F_photo_2d(x, z, power_factor)
    return Fx_ray + Fx_photo, Fz_ray + Fz_photo


def F_optical_3d_direct(x, y, z, power_factor=1.0):
    x_arr, y_arr, z_arr, power_factor_arr = np.broadcast_arrays(x, y, z, power_factor)
    r_arr = np.sqrt(x_arr**2 + y_arr**2)

    if x_arr.shape == ():
        Fr, Fz = F_optical_2d_direct(float(r_arr), float(z_arr), float(power_factor_arr))
        if r_arr < radial_zero_tolerance:
            return 0.0, 0.0, Fz
        return Fr * float(x_arr) / float(r_arr), Fr * float(y_arr) / float(r_arr), Fz

    Fx = np.zeros_like(x_arr, dtype=float)
    Fy = np.zeros_like(y_arr, dtype=float)
    Fz = np.zeros_like(z_arr, dtype=float)

    for index in np.ndindex(x_arr.shape):
        r_value = r_arr[index]
        Fr_value, Fz_value = F_optical_2d_direct(
            r_value,
            z_arr[index],
            power_factor_arr[index],
        )
        if r_value > radial_zero_tolerance:
            Fx[index] = Fr_value * x_arr[index] / r_value
            Fy[index] = Fr_value * y_arr[index] / r_value
        Fz[index] = Fz_value

    return Fx, Fy, Fz


def build_force_lookup_table(r_min, r_max, z_min, z_max):
    global force_lookup_ready
    global force_lookup_r_values
    global force_lookup_z_values
    global force_lookup_Fr_interpolator
    global force_lookup_Fz_interpolator
    global force_lookup_Fr_table
    global force_lookup_Fz_table

    start = perf_counter()
    force_lookup_r_values = np.linspace(r_min, r_max, force_lookup_grid_points_r)
    force_lookup_z_values = np.linspace(z_min, z_max, force_lookup_grid_points_z)

    Fr_table = np.zeros((force_lookup_grid_points_r, force_lookup_grid_points_z))
    Fz_table = np.zeros_like(Fr_table)

    for ir, r_value in enumerate(force_lookup_r_values):
        for iz, z_value in enumerate(force_lookup_z_values):
            Fr_table[ir, iz], Fz_table[ir, iz] = F_optical_2d_direct(
                r_value,
                z_value,
                power_factor=1.0,
            )

    force_lookup_Fr_interpolator = RegularGridInterpolator(
        (force_lookup_r_values, force_lookup_z_values),
        Fr_table,
        bounds_error=True,
    )
    force_lookup_Fz_interpolator = RegularGridInterpolator(
        (force_lookup_r_values, force_lookup_z_values),
        Fz_table,
        bounds_error=True,
    )
    force_lookup_Fr_table = np.ascontiguousarray(Fr_table, dtype=np.float64)
    force_lookup_Fz_table = np.ascontiguousarray(Fz_table, dtype=np.float64)
    force_lookup_ready = True

    print(
        "Built force lookup table:",
        force_lookup_grid_points_r,
        "x",
        force_lookup_grid_points_z,
        "points in",
        f"{perf_counter() - start:.3g}",
        "s",
    )


def F_optical_3d_lookup_scalar_clamped(x, y, z, power_factor=1.0):
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
        return F_optical_3d_lookup_scalar_clamped(x, y, z, power_factor)[:3]
    return F_optical_3d_direct(x, y, z, power_factor)


def F_optical_2d(x, z, power_factor=1.0):
    Fx, _, Fz = F_optical_3d(x, 0.0, z, power_factor)
    return Fx, Fz


def Fz_net_on_axis(z):
    _, Fz = F_optical_2d_direct(0.0, z)
    return Fz - effective_weight_force()


def numerical_derivative_1d(func, z, h=numerical_derivative_step):
    return (func(z + h) - func(z - h)) / (2 * h)


# ======================================================================
# Damping
# ======================================================================
def gas_density(pressure):
    return pressure * M_air / (R * T)


def mean_free_path(pressure):
    return kB * T / (np.sqrt(2) * np.pi * d_air**2 * pressure)


def knudsen_number(pressure):
    return mean_free_path(pressure) / radius


def cunningham_correction(kn_value):
    if kn_value <= 0:
        return 1.0
    return 1 + kn_value * (
        cunningham_A
        + cunningham_B * np.exp(-cunningham_C / kn_value)
    )


def damping_coefficient_stokes():
    return 6 * np.pi * eta * radius


def damping_coefficient_stokes_cunningham(pressure):
    return damping_coefficient_stokes() / cunningham_correction(knudsen_number(pressure))


def damping_coefficient_epstein(pressure):
    rho = gas_density(pressure)
    c_bar = mean_thermal_speed()
    accommodation_factor = 1 + np.pi * epstein_accommodation_alpha / 8
    return (4 / 3) * np.pi * radius**2 * rho * c_bar * accommodation_factor


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def transition_blend_weight(
    kn_value,
    transition_kn,
    half_width_decades=drag_transition_half_width_decades,
):
    if kn_value <= 0 or transition_kn <= 0:
        return 0.0

    log_kn = np.log10(kn_value)
    log_transition = np.log10(transition_kn)
    u = (
        log_kn
        - (log_transition - half_width_decades)
    ) / (2.0 * half_width_decades)

    return smoothstep(u)


def choose_drag_model(pressure):
    kn_value = knudsen_number(pressure)
    if kn_value < 0.1:
        return "stokes"
    if kn_value < 10:
        return "cunningham"
    return "epstein"


def damping_coefficient_auto_sharp(pressure):
    model = choose_drag_model(pressure)
    if model == "stokes":
        return damping_coefficient_stokes(), model
    if model == "cunningham":
        return damping_coefficient_stokes_cunningham(pressure), model
    if model == "epstein":
        return damping_coefficient_epstein(pressure), model
    raise ValueError("Unknown automatically selected drag model.")


def damping_coefficient_auto_smooth(pressure):
    kn_value = knudsen_number(pressure)

    b_stokes_local = damping_coefficient_stokes()
    b_cunningham_local = damping_coefficient_stokes_cunningham(pressure)
    b_epstein_local = damping_coefficient_epstein(pressure)

    stokes_cunningham_weight = transition_blend_weight(kn_value, 0.1)
    cunningham_epstein_weight = transition_blend_weight(kn_value, 10.0)

    b_low = (
        (1.0 - stokes_cunningham_weight) * b_stokes_local
        + stokes_cunningham_weight * b_cunningham_local
    )

    return (
        (1.0 - cunningham_epstein_weight) * b_low
        + cunningham_epstein_weight * b_epstein_local
    )


def damping_coefficient(pressure, model="auto"):
    if model == "auto":
        return damping_coefficient_auto_smooth(pressure), "auto_smooth"
    if model == "auto_sharp":
        return damping_coefficient_auto_sharp(pressure)
    if model == "stokes":
        return damping_coefficient_stokes(), model
    if model == "cunningham":
        return damping_coefficient_stokes_cunningham(pressure), model
    if model == "epstein":
        return damping_coefficient_epstein(pressure), model
    raise ValueError(
        "Unknown drag_model. Use 'stokes', 'cunningham', 'epstein', "
        "'auto_sharp', or 'auto'."
    )


# ======================================================================
# Model setup and trap-loss boundaries
# ======================================================================
def radial_force_at_stable_equilibrium(r):
    Fx, _ = F_optical_2d_direct(r, z_eq)
    return Fx


def find_radial_restoring_boundary():
    if trap_loss_radial_limit_manual is not None:
        return trap_loss_radial_limit_manual, "manual"

    return max(
        trap_loss_radial_beam_waists * w0,
        trap_loss_radial_boundary_search_max,
    ), "auto displacement limit"


def initialise_trap_model(pressure_pa=None):
    global p
    global x_eq, y_eq, z_eq
    global equilibrium_root_info, stable_roots
    global trap_loss_axial_boundary_z, trap_loss_axial_limit
    global trap_loss_radial_limit, trap_loss_radial_boundary_source
    global trap_loss_sustained_outside_steps, trap_loss_post_loss_plot_steps
    global kx, ky, kz, omega_x, omega_y, omega_z
    global b, drag_model_used, gamma_baoab
    global baoab_damping_factor, baoab_thermal_velocity_scale
    global force_lookup_ready
    global force_lookup_r_max, force_lookup_z_half_width

    if pressure_pa is not None:
        p = float(pressure_pa)

    force_lookup_ready = False

    z_scan = np.linspace(equilibrium_z_min, equilibrium_z_max, equilibrium_scan_points)
    F_scan = Fz_net_on_axis(z_scan)
    roots = []

    for i in range(len(z_scan) - 1):
        if F_scan[i] * F_scan[i + 1] < 0:
            roots.append(brentq(
                Fz_net_on_axis,
                z_scan[i],
                z_scan[i + 1],
                xtol=root_finding_xtol,
                rtol=root_finding_rtol,
            ))

    equilibrium_root_info = []
    stable_roots = []
    for root in roots:
        slope = numerical_derivative_1d(Fz_net_on_axis, root)
        stability = "stable" if slope < 0 else "unstable"
        equilibrium_root_info.append((root, slope, stability))
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

    def Fx_at_x(x):
        Fx, _ = F_optical_2d_direct(x, z_eq)
        return Fx

    def Fz_net_at_z(z):
        _, Fz = F_optical_2d_direct(x_eq, z)
        return Fz - effective_weight_force()

    kx = -numerical_derivative_1d(Fx_at_x, x_eq)
    ky = kx
    kz = -numerical_derivative_1d(Fz_net_at_z, z_eq)

    if kx <= 0 or kz <= 0:
        raise ValueError(
            "The selected equilibrium is not stable. "
            f"kx={kx:.3e} N/m, kz={kz:.3e} N/m."
        )

    omega_x = np.sqrt(kx / m)
    omega_y = np.sqrt(ky / m)
    omega_z = np.sqrt(kz / m)

    b, drag_model_used = damping_coefficient(p, drag_model)
    gamma_baoab = b / m
    baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
    baoab_thermal_velocity_scale = np.sqrt(
        (kB * T / m) * (1 - baoab_damping_factor**2)
    )

    trap_loss_radial_limit, trap_loss_radial_boundary_source = find_radial_restoring_boundary()
    if trap_loss_radial_limit <= 0:
        raise ValueError("trap_loss_radial_limit must be positive.")

    force_lookup_r_max = max(
        force_lookup_r_base_max,
        trap_loss_radial_limit,
    )

    axial_limit_auto = trap_loss_axial_rayleigh_ranges * zR
    axial_limit_for_lookup = (
        axial_limit_auto
        if trap_loss_axial_limit_manual is None
        else trap_loss_axial_limit_manual
    )
    force_lookup_z_half_width = max(
        force_lookup_z_base_half_width,
        axial_limit_for_lookup,
    )

    if use_force_lookup_table:
        build_force_lookup_table(
            force_lookup_r_min,
            force_lookup_r_max,
            z_eq - force_lookup_z_half_width,
            z_eq + force_lookup_z_half_width,
        )

    trap_loss_axial_limit = (
        max(axial_limit_auto, force_lookup_z_half_width)
        if trap_loss_axial_limit_manual is None
        else trap_loss_axial_limit_manual
    )
    if trap_loss_axial_limit <= 0:
        raise ValueError("trap_loss_axial_limit must be positive.")

    trap_loss_axial_boundary_z = z_eq - trap_loss_axial_limit

    trap_loss_sustained_outside_steps = max(
        1,
        int(np.ceil(trap_loss_sustained_outside_time / dt_baoab)),
    )
    trap_loss_post_loss_plot_steps = max(
        0,
        int(np.ceil(trap_loss_post_loss_plot_time / dt_baoab)),
    )

    print("Stable equilibrium z =", f"{z_eq * 1e6:.6g}", "um")
    print("Pressure =", f"{p:.6g}", "Pa")
    print("Trap frequencies fx, fy, fz =", f"{omega_x / (2 * np.pi):.6g}", f"{omega_y / (2 * np.pi):.6g}", f"{omega_z / (2 * np.pi):.6g}", "Hz")
    print("Drag model used =", drag_model_used)
    print("Damping rate gamma =", f"{gamma_baoab:.6g}", "s^-1")
    print("Radial loss boundary =", f"{trap_loss_radial_limit * 1e6:.6g}", "um", f"({trap_loss_radial_boundary_source})")
    print("Axial loss boundary =", f"{trap_loss_axial_limit * 1e6:.6g}", "um")


# ======================================================================
# Moving-trap simulation
# ======================================================================
def time_array(sim_t_end=t_end):
    return np.arange(t_start, sim_t_end + 0.5 * dt_baoab, dt_baoab)


def moving_trap_position(t, velocity):
    dx = velocity * t if moving_trap_direction == "x" else 0.0
    dy = velocity * t if moving_trap_direction == "y" else 0.0
    return (
        moving_trap_start_x + dx,
        moving_trap_start_y + dy,
        moving_trap_start_z,
    )


def make_laser_power_factor_time(t_values):
    if not use_laser_power_noise:
        return np.ones_like(t_values)

    laser_rng = np.random.default_rng(seed=laser_noise_seed)
    laser_step_samples = max(1, int(round(laser_noise_step_duration / dt_baoab)))
    n_laser_steps = int(np.ceil(len(t_values) / laser_step_samples))
    allowed_power_factors = np.array([
        1 - laser_noise_fraction,
        1.0,
        1 + laser_noise_fraction,
    ])

    step_power_factors = np.zeros(n_laser_steps)
    step_power_factors[0] = laser_rng.choice(allowed_power_factors)

    for j in range(1, n_laser_steps):
        previous_factor = step_power_factors[j - 1]
        if np.isclose(previous_factor, 1 + laser_noise_fraction):
            possible_factors = np.array([1.0, 1 + laser_noise_fraction])
        elif np.isclose(previous_factor, 1 - laser_noise_fraction):
            possible_factors = np.array([1 - laser_noise_fraction, 1.0])
        else:
            possible_factors = allowed_power_factors
        step_power_factors[j] = laser_rng.choice(possible_factors)

    return np.repeat(step_power_factors, laser_step_samples)[:len(t_values)]


def deterministic_force_lab_frame(x, y, z, t, trap_velocity, power_factor=1.0):
    trap_x, trap_y, trap_z = moving_trap_position(t, trap_velocity)
    x_rel = x - trap_x
    y_rel = y - trap_y
    z_rel = z - trap_z

    if use_force_lookup_table and force_lookup_ready:
        Fx, Fy, Fz, out_of_bounds = F_optical_3d_lookup_scalar_clamped(
            x_rel,
            y_rel,
            z_rel,
            power_factor,
        )
    else:
        Fx, Fy, Fz = F_optical_3d_direct(x_rel, y_rel, z_rel, power_factor)
        out_of_bounds = False

    return Fx, Fy, Fz - effective_weight_force(), out_of_bounds


def trap_frame_state(x, y, z, vx, vy, vz, t, trap_velocity):
    trap_x, trap_y, trap_z = moving_trap_position(t, trap_velocity)
    x_rel = x - trap_x - x_eq
    y_rel = y - trap_y - y_eq
    z_rel = z - trap_z - z_eq

    vx_rel = vx - (trap_velocity if moving_trap_direction == "x" else 0.0)
    vy_rel = vy - (trap_velocity if moving_trap_direction == "y" else 0.0)
    vz_rel = vz

    radial_rel = np.sqrt(x_rel**2 + y_rel**2)
    speed_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
    return x_rel, y_rel, z_rel, radial_rel, vx_rel, vy_rel, vz_rel, speed_rel


def loss_reason(x, y, z, vx, vy, vz, t, trap_velocity, outside_steps):
    x_rel, y_rel, z_rel, radial_rel, _, _, _, _ = trap_frame_state(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        t,
        trap_velocity,
    )
    outside_radial = radial_rel > trap_loss_radial_limit
    outside_axial = abs(z_rel) > trap_loss_axial_limit
    active_boundaries = []
    if outside_radial:
        active_boundaries.append("radial lag exceeded r_u")
    if outside_axial:
        active_boundaries.append("axial displacement exceeded z_u")
    outside_time = outside_steps * dt_baoab
    return (
        "outside moving-trap capture region continuously "
        f"for {outside_time:.3g} s ({', '.join(active_boundaries)}); "
        f"x_rel={x_rel * 1e6:.3g} um, y_rel={y_rel * 1e6:.3g} um, "
        f"z_rel={z_rel * 1e6:.3g} um, "
        f"trap_x={moving_trap_position(t, trap_velocity)[0] * 1e6:.3g} um, "
        f"trap_y={moving_trap_position(t, trap_velocity)[1] * 1e6:.3g} um"
    )


def simulate_moving_trap(
    trap_velocity,
    sim_t_end=t_end,
    store_trajectory=True,
    seed_offset=0,
):
    t_values = time_array(sim_t_end)
    power_factor_time = make_laser_power_factor_time(t_values)

    if use_brownian_noise:
        rng = np.random.default_rng(seed=brownian_seed + seed_offset)
        normals_x = rng.normal(size=len(t_values) - 1)
        normals_y = rng.normal(size=len(t_values) - 1)
        normals_z = rng.normal(size=len(t_values) - 1)
        thermal_velocity_scale = baoab_thermal_velocity_scale
    else:
        normals_x = np.zeros(len(t_values) - 1)
        normals_y = np.zeros(len(t_values) - 1)
        normals_z = np.zeros(len(t_values) - 1)
        thermal_velocity_scale = 0.0

    trap_x0, trap_y0, trap_z0 = moving_trap_position(t_values[0], trap_velocity)
    x_i = trap_x0 + x_eq
    y_i = trap_y0 + y_eq
    z_i = trap_z0 + z_eq

    if initial_velocity_mode == "stationary_particle":
        vx_i = 0.0
        vy_i = 0.0
    elif initial_velocity_mode == "comoving_particle":
        vx_i = trap_velocity if moving_trap_direction == "x" else 0.0
        vy_i = trap_velocity if moving_trap_direction == "y" else 0.0
    else:
        raise ValueError("initial_velocity_mode must be 'stationary_particle' or 'comoving_particle'.")
    vz_i = 0.0

    if store_trajectory:
        x_out = np.zeros_like(t_values)
        y_out = np.zeros_like(t_values)
        z_out = np.zeros_like(t_values)
        vx_out = np.zeros_like(t_values)
        vy_out = np.zeros_like(t_values)
        vz_out = np.zeros_like(t_values)
        x_out[0] = x_i
        y_out[0] = y_i
        z_out[0] = z_i
        vx_out[0] = vx_i
        vy_out[0] = vy_i
        vz_out[0] = vz_i
    else:
        x_out = y_out = z_out = vx_out = vy_out = vz_out = None

    lost = False
    loss_sample_index = -1
    post_loss_stop_sample_index = -1
    reason = ""
    outside_start_sample = None
    out_of_bounds_count = 0
    max_radial_lag = 0.0
    max_abs_x_lag = 0.0
    max_abs_z_lag = 0.0

    for i in range(len(t_values) - 1):
        t_i = t_values[i]
        t_next = t_values[i + 1]
        power_factor_i = power_factor_time[i]
        power_factor_next = power_factor_time[i + 1]

        Fx_i, Fy_i, Fz_i, was_clamped = deterministic_force_lab_frame(
            x_i,
            y_i,
            z_i,
            t_i,
            trap_velocity,
            power_factor_i,
        )
        out_of_bounds_count += int(was_clamped)
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        vx_i = baoab_damping_factor * vx_i + thermal_velocity_scale * normals_x[i]
        vy_i = baoab_damping_factor * vy_i + thermal_velocity_scale * normals_y[i]
        vz_i = baoab_damping_factor * vz_i + thermal_velocity_scale * normals_z[i]

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        Fx_i, Fy_i, Fz_i, was_clamped = deterministic_force_lab_frame(
            x_i,
            y_i,
            z_i,
            t_next,
            trap_velocity,
            power_factor_next,
        )
        out_of_bounds_count += int(was_clamped)
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        sample_index = i + 1
        (
            x_lag,
            _,
            z_lag,
            radial_lag,
            _,
            _,
            _,
            _,
        ) = trap_frame_state(x_i, y_i, z_i, vx_i, vy_i, vz_i, t_next, trap_velocity)
        max_radial_lag = max(max_radial_lag, radial_lag)
        max_abs_x_lag = max(max_abs_x_lag, abs(x_lag))
        max_abs_z_lag = max(max_abs_z_lag, abs(z_lag))

        if store_trajectory:
            x_out[sample_index] = x_i
            y_out[sample_index] = y_i
            z_out[sample_index] = z_i
            vx_out[sample_index] = vx_i
            vy_out[sample_index] = vy_i
            vz_out[sample_index] = vz_i

        if terminate_on_trap_loss and (
            sample_index % trap_loss_check_interval_steps == 0
            or sample_index == len(t_values) - 1
        ):
            outside = (
                radial_lag > trap_loss_radial_limit
                or abs(z_lag) > trap_loss_axial_limit
                or not np.all(np.isfinite([x_i, y_i, z_i, vx_i, vy_i, vz_i]))
            )

            if outside:
                if outside_start_sample is None:
                    outside_start_sample = sample_index
                outside_steps = sample_index - outside_start_sample + 1
            else:
                outside_start_sample = None
                outside_steps = 0

            if (
                not lost
                and outside_start_sample is not None
                and outside_steps >= trap_loss_sustained_outside_steps
            ):
                lost = True
                loss_sample_index = sample_index
                post_loss_stop_sample_index = min(
                    len(t_values) - 1,
                    sample_index + trap_loss_post_loss_plot_steps,
                )
                if not np.all(np.isfinite([x_i, y_i, z_i, vx_i, vy_i, vz_i])):
                    reason = "position or velocity became non-finite"
                else:
                    reason = loss_reason(
                        x_i,
                        y_i,
                        z_i,
                        vx_i,
                        vy_i,
                        vz_i,
                        t_next,
                        trap_velocity,
                        outside_steps,
                    )

            if lost and sample_index >= post_loss_stop_sample_index:
                break

    stop_index = sample_index
    (
        final_x_rel,
        final_y_rel,
        final_z_rel,
        final_radial_lag,
        final_vx_rel,
        final_vy_rel,
        final_vz_rel,
        final_relative_speed,
    ) = trap_frame_state(x_i, y_i, z_i, vx_i, vy_i, vz_i, t_values[stop_index], trap_velocity)

    summary = {
        "pressure_pa": p,
        "pressure_mbar": p / 100.0,
        "trap_velocity_m_per_s": trap_velocity,
        "lost": bool(lost),
        "loss_time_s": t_values[loss_sample_index] if lost else np.nan,
        "loss_reason": reason,
        "final_time_s": t_values[stop_index],
        "final_x_m": x_i,
        "final_y_m": y_i,
        "final_z_m": z_i,
        "final_x_rel_m": final_x_rel,
        "final_y_rel_m": final_y_rel,
        "final_z_rel_m": final_z_rel,
        "final_radial_lag_m": final_radial_lag,
        "final_vx_rel_m_per_s": final_vx_rel,
        "final_vy_rel_m_per_s": final_vy_rel,
        "final_vz_rel_m_per_s": final_vz_rel,
        "final_relative_speed_m_per_s": final_relative_speed,
        "max_radial_lag_m": max_radial_lag,
        "max_abs_x_lag_m": max_abs_x_lag,
        "max_abs_z_lag_m": max_abs_z_lag,
        "samples_returned": stop_index + 1,
        "out_of_bounds_count": int(out_of_bounds_count),
    }

    if not store_trajectory:
        return summary

    result_slice = slice(None, stop_index + 1)
    trap_x = np.array([moving_trap_position(t, trap_velocity)[0] for t in t_values[result_slice]])
    trap_y = np.array([moving_trap_position(t, trap_velocity)[1] for t in t_values[result_slice]])
    trap_z = np.array([moving_trap_position(t, trap_velocity)[2] for t in t_values[result_slice]])
    result = {
        "summary": summary,
        "t": t_values[result_slice],
        "x": x_out[result_slice],
        "y": y_out[result_slice],
        "z": z_out[result_slice],
        "vx": vx_out[result_slice],
        "vy": vy_out[result_slice],
        "vz": vz_out[result_slice],
        "trap_x": trap_x,
        "trap_y": trap_y,
        "trap_z": trap_z,
        "power_factor": power_factor_time[result_slice],
        "loss_sample_index": loss_sample_index,
    }
    result["x_rel"] = result["x"] - result["trap_x"] - x_eq
    result["y_rel"] = result["y"] - result["trap_y"] - y_eq
    result["z_rel"] = result["z"] - result["trap_z"] - z_eq
    result["radial_lag"] = np.sqrt(result["x_rel"]**2 + result["y_rel"]**2)
    return result


# ======================================================================
# Sweeps, threshold search, and diagnostics
# ======================================================================
def write_csv(rows, csv_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_micrometre_summary_columns(row):
    row["final_x_rel_um"] = row["final_x_rel_m"] * 1e6
    row["final_y_rel_um"] = row["final_y_rel_m"] * 1e6
    row["final_z_rel_um"] = row["final_z_rel_m"] * 1e6
    row["final_radial_lag_um"] = row["final_radial_lag_m"] * 1e6
    row["max_radial_lag_um"] = row["max_radial_lag_m"] * 1e6
    row["max_abs_x_lag_um"] = row["max_abs_x_lag_m"] * 1e6
    row["max_abs_z_lag_um"] = row["max_abs_z_lag_m"] * 1e6


def run_scan():
    rows = []
    for index, velocity in enumerate(velocity_scan_m_per_s):
        summary = simulate_moving_trap(
            float(velocity),
            store_trajectory=False,
            seed_offset=index,
        )
        add_micrometre_summary_columns(summary)
        rows.append(summary)
        status = "LOST" if summary["lost"] else "survived"
        print(
            f"scan {index + 1}/{len(velocity_scan_m_per_s)}:",
            f"v={velocity:.6g} m/s -> {status},",
            f"max lag={summary['max_radial_lag_um']:.4g} um",
        )
    return rows


def find_velocity_threshold():
    low_v = 0.0
    low_summary = simulate_moving_trap(low_v, store_trajectory=False, seed_offset=100000)
    add_micrometre_summary_columns(low_summary)
    if low_summary["lost"]:
        return {
            "status": "zero_velocity_lost",
            "highest_survived_velocity_m_per_s": np.nan,
            "lowest_lost_velocity_m_per_s": 0.0,
            "highest_survived_summary": None,
            "lowest_lost_summary": low_summary,
            "rows": [low_summary],
        }

    high_v = threshold_initial_high_velocity
    high_summary = None
    rows = [low_summary]
    expansion_index = 0
    while high_v <= threshold_max_velocity:
        high_summary = simulate_moving_trap(
            high_v,
            store_trajectory=False,
            seed_offset=200000 + expansion_index,
        )
        add_micrometre_summary_columns(high_summary)
        rows.append(high_summary)
        status = "LOST" if high_summary["lost"] else "survived"
        print(f"bracket: v={high_v:.6g} m/s -> {status}")

        if high_summary["lost"]:
            break

        low_v = high_v
        low_summary = high_summary
        high_v *= 2
        expansion_index += 1

    if high_summary is None or not high_summary["lost"]:
        return {
            "status": "no_loss_up_to_max_velocity",
            "highest_survived_velocity_m_per_s": low_v,
            "lowest_lost_velocity_m_per_s": np.nan,
            "highest_survived_summary": low_summary,
            "lowest_lost_summary": None,
            "rows": rows,
        }

    lost_v = high_v
    lost_summary = high_summary

    for iteration in range(threshold_bisection_iterations):
        mid_v = 0.5 * (low_v + lost_v)
        mid_summary = simulate_moving_trap(
            mid_v,
            store_trajectory=False,
            seed_offset=300000 + iteration,
        )
        add_micrometre_summary_columns(mid_summary)
        rows.append(mid_summary)
        status = "LOST" if mid_summary["lost"] else "survived"
        print(f"bisect {iteration + 1}: v={mid_v:.8g} m/s -> {status}")

        if mid_summary["lost"]:
            lost_v = mid_v
            lost_summary = mid_summary
        else:
            low_v = mid_v
            low_summary = mid_summary

        if lost_v - low_v <= threshold_tolerance_m_per_s:
            break

    return {
        "status": "bracketed",
        "highest_survived_velocity_m_per_s": low_v,
        "lowest_lost_velocity_m_per_s": lost_v,
        "highest_survived_summary": low_summary,
        "lowest_lost_summary": lost_summary,
        "rows": rows,
    }


def decimation_indices(length):
    step = max(1, int(np.ceil(length / max_plot_points)))
    return slice(None, None, step)


def save_trajectory_csv(result, csv_path):
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "t_s",
            "particle_x_m",
            "particle_y_m",
            "particle_z_m",
            "trap_x_m",
            "trap_y_m",
            "trap_z_m",
            "x_lag_m",
            "y_lag_m",
            "z_lag_m",
            "radial_lag_m",
            "vx_m_per_s",
            "vy_m_per_s",
            "vz_m_per_s",
            "laser_power_factor",
        ])
        for values in zip(
            result["t"],
            result["x"],
            result["y"],
            result["z"],
            result["trap_x"],
            result["trap_y"],
            result["trap_z"],
            result["x_rel"],
            result["y_rel"],
            result["z_rel"],
            result["radial_lag"],
            result["vx"],
            result["vy"],
            result["vz"],
            result["power_factor"],
        ):
            writer.writerow(values)


def plot_velocity_sweep(rows, threshold_result=None):
    if not rows:
        return

    velocities = np.array([row["trap_velocity_m_per_s"] for row in rows])
    lost = np.array([row["lost"] for row in rows], dtype=bool)
    max_lag_um = np.array([row["max_radial_lag_um"] for row in rows])
    loss_time = np.array([row["loss_time_s"] for row in rows])
    final_time = np.array([row["final_time_s"] for row in rows])
    plotted_time = np.where(lost, loss_time, final_time)
    order = np.argsort(velocities)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.6, 3.4),
        sharex=True,
        constrained_layout=True,
    )
    ax_lag, ax_loss = axes

    ax_lag.plot(
        velocities[order],
        max_lag_um[order],
        "-",
        color="#1f77b4",
        linewidth=1.4,
        label="max radial lag",
    )
    ax_lag.axhline(trap_loss_radial_limit * 1e6, color="black", linestyle=":", linewidth=0.9, label="radial loss boundary")
    ax_lag.set_ylabel("max radial lag / um")
    ax_lag.grid(color="0.85", linewidth=0.4)
    ax_lag.legend(loc="best", frameon=True)

    ax_loss.plot(
        velocities[order],
        plotted_time[order],
        "-",
        color="#d94841",
        linewidth=1.4,
        label="loss time; survived runs shown at t_end",
    )
    ax_loss.set_xlabel("trap velocity / m s$^{-1}$")
    ax_loss.set_ylabel("loss time / s")
    ax_loss.grid(color="0.85", linewidth=0.4)
    ax_loss.legend(loc="best", frameon=True)

    positive_velocities = velocities[velocities > 0]
    if len(positive_velocities) > 0:
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlim(np.min(positive_velocities), np.max(positive_velocities))

    if threshold_result is not None and threshold_result["status"] == "bracketed":
        low_v = threshold_result["highest_survived_velocity_m_per_s"]
        high_v = threshold_result["lowest_lost_velocity_m_per_s"]
        if len(positive_velocities) > 0:
            low_v = max(low_v, np.min(positive_velocities))
        for axis in axes:
            if high_v > low_v:
                axis.axvspan(low_v, high_v, color="0.2", alpha=0.12, linewidth=0)

    fig.savefig(save_path / f"{moving_trap_output_prefix}_summary.png", bbox_inches="tight")
    fig.savefig(save_path / f"{moving_trap_output_prefix}_summary.pdf", bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_pressure_comparison_loss_times(rows):
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    colours = ["#d94841", "#1f77b4", "#238b45", "#6a51a3", "#8c6d31"]
    pressures = sorted({row["pressure_pa"] for row in rows}, reverse=True)

    for pressure_index, pressure_value in enumerate(pressures):
        pressure_rows = [
            row for row in rows
            if np.isclose(row["pressure_pa"], pressure_value)
        ]
        pressure_rows.sort(key=lambda row: row["trap_velocity_m_per_s"])
        velocities = np.array(
            [row["trap_velocity_m_per_s"] for row in pressure_rows],
            dtype=float,
        )
        lost = np.array([row["lost"] for row in pressure_rows], dtype=bool)
        loss_time = np.array([row["loss_time_s"] for row in pressure_rows], dtype=float)
        final_time = np.array([row["final_time_s"] for row in pressure_rows], dtype=float)
        plotted_time = np.where(lost, loss_time, final_time)
        positive = velocities > 0

        if np.count_nonzero(positive) < 2:
            continue

        ax.plot(
            velocities[positive],
            plotted_time[positive],
            "-",
            linewidth=1.6,
            color=colours[pressure_index % len(colours)],
            label=f"{pressure_value:g} Pa",
        )

    if pressure_comparison_xscale == "log":
        ax.set_xscale("log")
    if pressure_comparison_yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel("trap velocity / m s$^{-1}$")
    ax.set_ylabel("loss time / s")
    ax.set_title("Moving-trap loss time versus velocity")
    ax.grid(True, which="both", color="0.88", linewidth=0.45)
    ax.legend(title="pressure", loc="best", frameon=True)

    output_stem = f"{moving_trap_output_prefix}_pressure_comparison_loss_time"
    fig.savefig(save_path / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(save_path / f"{output_stem}.pdf", bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def run_pressure_comparison_loss_time_scan():
    rows = []
    original_pressure = p

    for pressure_index, pressure_value in enumerate(pressure_comparison_pressures_pa):
        print("Starting pressure comparison line for", f"{pressure_value:g}", "Pa")
        initialise_trap_model(pressure_value)

        for velocity_index, velocity in enumerate(pressure_comparison_velocity_m_per_s):
            summary = simulate_moving_trap(
                float(velocity),
                store_trajectory=False,
                seed_offset=700000 + 10000 * pressure_index + velocity_index,
            )
            add_micrometre_summary_columns(summary)
            summary["pressure_index"] = pressure_index
            summary["velocity_index"] = velocity_index
            rows.append(summary)
            status = "LOST" if summary["lost"] else "survived"
            plotted_time = summary["loss_time_s"] if summary["lost"] else summary["final_time_s"]
            print(
                f"pressure {pressure_value:g} Pa, "
                f"v={velocity:.6g} m/s -> {status}, "
                f"plotted time={plotted_time:.6g} s",
            )

    comparison_csv = save_path / f"{moving_trap_output_prefix}_pressure_comparison_loss_time.csv"
    write_csv(rows, comparison_csv)
    plot_pressure_comparison_loss_times(rows)
    print("Saved pressure-comparison loss-time data to", comparison_csv)

    initialise_trap_model(original_pressure)
    return rows


def plot_pressure_threshold_velocity(rows):
    if not rows:
        return

    rows = sorted(rows, key=lambda row: row["pressure_pa"])
    pressures = np.array([row["pressure_pa"] for row in rows], dtype=float)
    survived_velocity = np.array(
        [row["highest_survived_velocity_m_per_s"] for row in rows],
        dtype=float,
    )
    finite_velocity = np.isfinite(survived_velocity) & (survived_velocity > 0)
    if not np.any(finite_velocity):
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.plot(
        pressures,
        survived_velocity,
        "-",
        color="#1f77b4",
        linewidth=1.6,
        label="highest survived velocity",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pressure / Pa")
    ax.set_ylabel("maximum trap velocity / m s$^{-1}$")
    ax.set_title("Moving-trap velocity threshold versus pressure")
    ax.grid(True, which="both", color="0.88", linewidth=0.45)
    ax.legend(loc="best", frameon=True)

    output_stem = f"{moving_trap_output_prefix}_maximum_velocity_vs_pressure"
    fig.savefig(save_path / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(save_path / f"{output_stem}.pdf", bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def run_pressure_threshold_scan():
    rows = []
    all_threshold_rows = []
    original_pressure = p

    for pressure_index, pressure_value in enumerate(pressure_threshold_pressures_pa):
        print("Starting velocity-threshold search for", f"{pressure_value:g}", "Pa")
        try:
            initialise_trap_model(pressure_value)
            threshold_result = find_velocity_threshold()
        except ValueError as error:
            rows.append({
                "pressure_pa": pressure_value,
                "pressure_mbar": pressure_value / 100.0,
                "status": "radially unstable or unavailable",
                "highest_survived_velocity_m_per_s": np.nan,
                "lowest_lost_velocity_m_per_s": np.nan,
                "threshold_bracket_width_m_per_s": np.nan,
                "error_message": str(error),
            })
            print(
                "Skipping",
                f"{pressure_value:g}",
                "Pa because the moving-trap threshold is undefined:",
                error,
            )
            continue

        high_survived = threshold_result["highest_survived_velocity_m_per_s"]
        low_lost = threshold_result["lowest_lost_velocity_m_per_s"]
        bracket_width = (
            low_lost - high_survived
            if np.isfinite(high_survived) and np.isfinite(low_lost)
            else np.nan
        )

        rows.append({
            "pressure_pa": pressure_value,
            "pressure_mbar": pressure_value / 100.0,
            "status": threshold_result["status"],
            "highest_survived_velocity_m_per_s": high_survived,
            "lowest_lost_velocity_m_per_s": low_lost,
            "threshold_bracket_width_m_per_s": bracket_width,
            "error_message": "",
        })

        for threshold_row in threshold_result["rows"]:
            threshold_row = dict(threshold_row)
            threshold_row["pressure_threshold_index"] = pressure_index
            all_threshold_rows.append(threshold_row)

        if threshold_result["status"] == "bracketed":
            print(
                "Pressure",
                f"{pressure_value:g}",
                "Pa speed limit:",
                f"{high_survived:.8g}",
                "to",
                f"{low_lost:.8g}",
                "m/s",
            )
        elif threshold_result["status"] == "no_loss_up_to_max_velocity":
            print(
                "Pressure",
                f"{pressure_value:g}",
                "Pa: no loss up to",
                f"{high_survived:.8g}",
                "m/s",
            )
        else:
            print("Pressure", f"{pressure_value:g}", "Pa: particle lost at zero velocity.")

    summary_csv = save_path / f"{moving_trap_output_prefix}_maximum_velocity_vs_pressure.csv"
    detail_csv = save_path / f"{moving_trap_output_prefix}_pressure_threshold_search_details.csv"
    write_csv(rows, summary_csv)
    write_csv(all_threshold_rows, detail_csv)
    plot_pressure_threshold_velocity(rows)
    print("Saved maximum-velocity pressure summary to", summary_csv)
    print("Saved pressure-threshold search details to", detail_csv)

    initialise_trap_model(original_pressure)
    return rows


def plot_pressure_radial_frequency(rows):
    if not rows:
        return

    rows = sorted(rows, key=lambda row: row["pressure_pa"])
    pressures = np.array([row["pressure_pa"] for row in rows], dtype=float)
    radial_frequency = np.array([row["radial_trap_frequency_hz"] for row in rows], dtype=float)
    finite_frequency = np.isfinite(radial_frequency) & (radial_frequency > 0)

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    if np.any(finite_frequency):
        ax.plot(
            pressures[finite_frequency],
            radial_frequency[finite_frequency],
            "o-",
            color="#1f77b4",
            linewidth=1.4,
            label="radial trap frequency",
        )

    undefined_frequency = ~finite_frequency
    if np.any(undefined_frequency) and np.any(finite_frequency):
        marker_y = np.nanmin(radial_frequency[finite_frequency]) * 0.8
        ax.plot(
            pressures[undefined_frequency],
            np.full(np.count_nonzero(undefined_frequency), marker_y),
            "x",
            color="#d94841",
            markersize=4,
            label="radially unstable",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pressure / Pa")
    ax.set_ylabel("radial trap frequency / Hz")
    ax.set_title("Radial trap frequency versus pressure")
    ax.grid(True, which="both", color="0.88", linewidth=0.45)
    ax.legend(loc="best", frameon=True)

    output_stem = f"{moving_trap_output_prefix}_radial_frequency_vs_pressure"
    fig.savefig(save_path / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(save_path / f"{output_stem}.pdf", bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def run_pressure_radial_frequency_scan():
    rows = []
    original_pressure = p

    for pressure_index, pressure_value in enumerate(pressure_radial_frequency_pressures_pa):
        print("Computing radial trap frequency for", f"{pressure_value:g}", "Pa")
        try:
            initialise_trap_model(pressure_value)
            rows.append({
                "pressure_pa": pressure_value,
                "pressure_mbar": pressure_value / 100.0,
                "radial_trap_frequency_hz": omega_x / (2 * np.pi),
                "axial_trap_frequency_hz": omega_z / (2 * np.pi),
                "damping_rate_s_inv": gamma_baoab,
                "stable_equilibrium_z_um": z_eq * 1e6,
                "drag_model": drag_model_used,
                "radial_loss_boundary_um": trap_loss_radial_limit * 1e6,
                "radial_loss_boundary_source": trap_loss_radial_boundary_source,
                "status": "stable",
                "error_message": "",
            })
        except ValueError as error:
            rows.append({
                "pressure_pa": pressure_value,
                "pressure_mbar": pressure_value / 100.0,
                "radial_trap_frequency_hz": np.nan,
                "axial_trap_frequency_hz": np.nan,
                "damping_rate_s_inv": np.nan,
                "stable_equilibrium_z_um": np.nan,
                "drag_model": "",
                "radial_loss_boundary_um": np.nan,
                "radial_loss_boundary_source": "",
                "status": "radially unstable or unavailable",
                "error_message": str(error),
            })
            print(
                "Skipping",
                f"{pressure_value:g}",
                "Pa because radial trap frequency is undefined:",
                error,
            )

    frequency_csv = save_path / f"{moving_trap_output_prefix}_radial_frequency_vs_pressure.csv"
    write_csv(rows, frequency_csv)
    plot_pressure_radial_frequency(rows)
    print("Saved radial-frequency pressure data to", frequency_csv)

    initialise_trap_model(original_pressure)
    return rows


def plot_moving_trap_trajectory(result, output_name):
    idx = decimation_indices(len(result["t"]))
    t_plot = result["t"][idx]
    lost = result["summary"]["lost"]
    loss_index = result["loss_sample_index"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.7),
        constrained_layout=True,
    )
    ax_x, ax_lag, ax_z, ax_phase = axes.ravel()

    ax_x.plot(t_plot, result["x"][idx] * 1e6, color="#1f77b4", label="particle x")
    ax_x.plot(t_plot, (result["trap_x"][idx] + x_eq) * 1e6, color="black", linestyle="--", label="trap equilibrium x")
    ax_x.set_ylabel("lab x / um")
    ax_x.set_xlabel("time / s")
    ax_x.grid(color="0.85", linewidth=0.4)
    ax_x.legend(loc="best", frameon=True)

    ax_lag.plot(t_plot, result["x_rel"][idx] * 1e6, color="#6a51a3", label="x lag")
    ax_lag.plot(t_plot, result["radial_lag"][idx] * 1e6, color="#238b45", label="radial lag")
    ax_lag.axhline(trap_loss_radial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_lag.axhline(-trap_loss_radial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_lag.set_ylabel("trap-frame lag / um")
    ax_lag.set_xlabel("time / s")
    ax_lag.grid(color="0.85", linewidth=0.4)
    ax_lag.legend(loc="best", frameon=True)

    ax_z.plot(t_plot, result["z_rel"][idx] * 1e6, color="#08519c", label="z - z_eq")
    ax_z.axhline(trap_loss_axial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9, label="axial loss boundary")
    ax_z.axhline(-trap_loss_axial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_z.set_ylabel("axial offset / um")
    ax_z.set_xlabel("time / s")
    ax_z.grid(color="0.85", linewidth=0.4)
    ax_z.legend(loc="best", frameon=True)

    ax_phase.plot(result["x_rel"][idx] * 1e6, (result["z"][idx] - result["trap_z"][idx]) * 1e6, color="#1f77b4", label="particle")
    ax_phase.axvline(trap_loss_radial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_phase.axvline(-trap_loss_radial_limit * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_phase.axhline(z_eq * 1e6, color="black", linestyle="--", linewidth=0.8, label="stable z_eq")
    ax_phase.axhline((z_eq + trap_loss_axial_limit) * 1e6, color="#d94841", linestyle=":", linewidth=0.9, label="z_u")
    ax_phase.axhline((z_eq - trap_loss_axial_limit) * 1e6, color="#d94841", linestyle=":", linewidth=0.9)
    ax_phase.set_xlabel("trap-frame x / um")
    ax_phase.set_ylabel("trap-frame z / um")
    ax_phase.grid(color="0.85", linewidth=0.4)
    ax_phase.legend(loc="best", frameon=True)

    if lost and 0 <= loss_index < len(result["t"]):
        loss_time = result["t"][loss_index]
        for axis in (ax_x, ax_lag, ax_z):
            axis.axvline(loss_time, color="#d94841", linestyle="-", linewidth=0.8, alpha=0.75)
        ax_phase.plot(
            result["x_rel"][loss_index] * 1e6,
            (result["z"][loss_index] - result["trap_z"][loss_index]) * 1e6,
            "x",
            color="#d94841",
            markersize=5,
            label="loss",
        )

    title_status = "lost" if lost else "survived"
    fig.suptitle(
        f"Moving trap v={result['summary']['trap_velocity_m_per_s']:.6g} m/s ({title_status})",
        fontsize=fontsize + 1,
    )
    fig.savefig(save_path / f"{output_name}.png", bbox_inches="tight")
    fig.savefig(save_path / f"{output_name}.pdf", bbox_inches="tight")
    if display_plots:
        plt.show()
    else:
        plt.close(fig)


def run_diagnostics(threshold_result):
    velocities_to_plot = []

    if diagnostic_velocity_m_per_s is not None:
        velocities_to_plot.append(float(diagnostic_velocity_m_per_s))
    elif threshold_result is not None:
        survived_v = threshold_result.get("highest_survived_velocity_m_per_s")
        lost_v = threshold_result.get("lowest_lost_velocity_m_per_s")
        if np.isfinite(survived_v):
            velocities_to_plot.append(float(survived_v))
        if np.isfinite(lost_v) and lost_v not in velocities_to_plot:
            velocities_to_plot.append(float(lost_v))

    if not velocities_to_plot:
        velocities_to_plot.append(float(velocity_scan_m_per_s[-1]))

    for index, velocity in enumerate(velocities_to_plot):
        result = simulate_moving_trap(
            velocity,
            store_trajectory=True,
            seed_offset=500000 + index,
        )
        label = "lost" if result["summary"]["lost"] else "survived"
        safe_velocity = f"{velocity:.6g}".replace(".", "p").replace("-", "minus")
        output_name = f"{moving_trap_output_prefix}_trajectory_{safe_velocity}_{label}"
        save_trajectory_csv(result, save_path / f"{output_name}.csv")
        plot_moving_trap_trajectory(result, output_name)
        print(
            "diagnostic:",
            f"v={velocity:.6g} m/s -> {label},",
            f"saved {output_name}.png/.pdf/.csv",
        )


def main():
    script_start = perf_counter()
    initialise_trap_model()

    all_rows = []
    scan_rows = []
    threshold_result = None

    print("Brownian noise enabled =", use_brownian_noise)
    print("Laser power noise enabled =", use_laser_power_noise)
    print("Initial velocity mode =", initial_velocity_mode)
    print("Moving trap direction =", moving_trap_direction)

    if run_velocity_scan:
        print("Starting coarse velocity scan")
        scan_rows = run_scan()
        all_rows.extend(scan_rows)
        scan_csv = save_path / f"{moving_trap_output_prefix}_coarse_scan.csv"
        write_csv(scan_rows, scan_csv)
        print("Saved coarse scan to", scan_csv)

    if run_threshold_search:
        print("Starting threshold search")
        threshold_result = find_velocity_threshold()
        threshold_rows = threshold_result["rows"]
        all_rows.extend(threshold_rows)
        threshold_csv = save_path / f"{moving_trap_output_prefix}_threshold_search.csv"
        write_csv(threshold_rows, threshold_csv)
        print("Saved threshold search to", threshold_csv)

        if threshold_result["status"] == "bracketed":
            low_v = threshold_result["highest_survived_velocity_m_per_s"]
            high_v = threshold_result["lowest_lost_velocity_m_per_s"]
            print(
                "Estimated speed limit:",
                f"{low_v:.8g}",
                "to",
                f"{high_v:.8g}",
                "m/s",
                f"(width {high_v - low_v:.3g} m/s)",
            )
        elif threshold_result["status"] == "no_loss_up_to_max_velocity":
            print(
                "No loss observed up to",
                f"{threshold_result['highest_survived_velocity_m_per_s']:.6g}",
                "m/s over",
                f"{t_end:.6g}",
                "s",
            )
        else:
            print("Particle was lost even at zero trap velocity.")

    if all_rows:
        plot_velocity_sweep(all_rows, threshold_result)

    if diagnostic_velocity_m_per_s is not None or run_velocity_scan or run_threshold_search:
        run_diagnostics(threshold_result)

    if run_pressure_comparison_plot:
        print("Starting multi-pressure loss-time plot")
        run_pressure_comparison_loss_time_scan()

    if run_pressure_threshold_plot:
        print("Starting multi-pressure velocity-threshold plot")
        run_pressure_threshold_scan()

    if run_pressure_radial_frequency_plot:
        print("Starting radial-frequency pressure plot")
        run_pressure_radial_frequency_scan()

    print("Outputs saved in", save_path)
    print("Runtime =", f"{perf_counter() - script_start:.3g}", "s")


if __name__ == "__main__":
    main()
