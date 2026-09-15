"""
3D optical levitation simulation with Brownian motion and optional laser-power noise.

Run:
    python3 project_3d_cython.py

Outputs:
    Figures and CSV summaries are written to cython_outputs/ beside this script
    by default. Set PROJECT_3D_SAVE_PATH to override the output folder.

Normal handover workflow:
    1. Edit the values in USER SETTINGS.
    2. Run the script.
    3. Inspect the generated plots and CSV files in the output folder.

Optional inputs:
    - The Cython solver is used if it can be imported from CYTHON_DIR.
    - The measured laser-power CSV is only required when laser_noise_model is
      set to "psd_matched".
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import welch
from time import perf_counter

from pathlib import Path
import sys
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# ****************************************************************************************************************************************************
# USER SETTINGS - edit these values for normal runs
# ****************************************************************************************************************************************************

# Output folder. Relative paths are resolved beside this script so another user
# can run the code from any working directory and still find the outputs.
save_path = Path(os.environ.get("PROJECT_3D_SAVE_PATH", SCRIPT_DIR / "cython_outputs")).expanduser()
if not save_path.is_absolute():
    save_path = SCRIPT_DIR / save_path

# Optional Cython extension location. The script falls back to the pure-Python
# solver if the compiled module is not available here.
CYTHON_DIR = PROJECT_DIR / "cython"

# Optional measured laser-power trace used only with laser_noise_model =
# "psd_matched". Keep this as a project-relative path for portability.
laser_noise_power_csv_path = PROJECT_DIR / "Power_30min.csv"

# Set True to print extra diagnostics while the simulation runs.
verbose_output = False



# ****************************************************************************************************************************************************
# INTERNAL SETUP
# ****************************************************************************************************************************************************

verbose_output = verbose_output or os.environ.get("PROJECT_3D_VERBOSE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Use Matplotlib's default plotting style.  Individual figures below still set
# their own sizes where a specific output layout is useful.
plt.style.use("default")
default_label_fontsize = plt.rcParams["axes.labelsize"]


def log(message="", *, verbose=False):
    if verbose and not verbose_output:
        return
    print(message)


save_path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CYTHON_DIR))

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
    cython_baoab_available = solve_baoab_3d_lookup_cython is not None
    cython_feedback_baoab_available = solve_baoab_3d_feedback_lookup_cython is not None
except ImportError:
    solve_baoab_3d_lookup_cython = None
    solve_baoab_3d_feedback_lookup_cython = None
    cython_baoab_available = False
    cython_feedback_baoab_available = False

log(f"Cython BAOAB solver available = {cython_baoab_available}", verbose=True)
log(
    f"Cython feedback BAOAB solver available = {cython_feedback_baoab_available}",
    verbose=True,
)

script_start_time = perf_counter()


# ****************************************************************************************************************************************************
# Main True/False switches for common run modes.
# ****************************************************************************************************************************************************

use_brownian_motion = True                        # Include random thermal kicks from gas collisions.
use_laser_power_noise = True                      # Add laser-power fluctuations during the trajectory.
damping = True                                    # Include gas damping/drag.
use_pd_feedback = False                           # Enable delayed PD feedback control of laser power.
terminate_on_trap_loss = False                    # Stop early if the particle leaves the trap region.
use_force_lookup_table = True                     # Use a precomputed force table for faster force calls.
display_plots = False                             # Show plots interactively instead of only saving them.
include_z_detector_resolution_psd = False         # Take into account 100nm resolution on detector


# Optional diagnostic plots. These are useful while validating the model, but
# leave them off for a clean default run.

include_uncoupled_nonlinear_psd = False
plot_position_histogram_linear_prediction = False
plot_uncoupled_vertical_histogram = False
plot_vertical_expected_distribution_comparison = False
plot_aligned_summary_figure = False



# ****************************************************************************************************************************************************
# PHYSICAL CONSTANTS 
# ****************************************************************************************************************************************************
g = 9.81
c_light = 299792458

# ****************************************************************************************************************************************************
# USER SETTINGS CONTINUED - simulation parameters
# ****************************************************************************************************************************************************
# Particle properties
radius = 6.3e-6       # m
density = 1100        # kg/m^3
n_particle = 1.555    # particle refractive index
n_medium = 1.00027    # surrounding medium refractive index, air

# Gas properties
p = 100           # pressure, Pa
T = 300           # impinging/ambient gas temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
kB = 1.38e-23
d_air = 3.7e-10

# Laser / force parameters
w0 = 2.97e-6      # beam waist, m
wavelength = 532e-9
M2 = 1.2
use_m2_rayleigh_range = True
zR_manual = 100e-6
P_laser = 0.1     # W, example laser power
laser_noise_model = "square"
# Choose one of:
#   "square"       original bounded pseudo-square-wave power noise
#   "psd_matched"  synthetic noise generated from the measured laser-power PSD
laser_noise_fraction = 0.01
laser_noise_frequency=300
laser_noise_step_duration = 1/laser_noise_frequency    # s
laser_noise_psd_matched_seed = 12345
laser_noise_equilibrium_grid_points = 21

# Ashkin ray-optics sampling
ray_grid_points = 100

# Photophoretic force parameters
k_particle = 0.135    # W/(m K)
alpha_acc = 1.0
kappa_t = 1.14
extinction_coefficient = 3e-6
effective_absorption_path_length = 2 * radius
absorption_coefficient = 4 * np.pi * extinction_coefficient / wavelength
absorption_fraction = 1 - np.exp(
    -absorption_coefficient * effective_absorption_path_length
)

# 3D force lookup table
force_lookup_grid_points_r = 101
force_lookup_grid_points_z = 801
force_lookup_r_min = 0.0
force_lookup_r_base_max = 80e-6
force_lookup_r_displacement_factor = 4
force_lookup_r_thermal_factor = 8
force_lookup_z_base_half_width = 3000e-6
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
#   "auto"        smoothly blend between models from Knudsen number
cunningham_A = 1.257
cunningham_B = 0.4
cunningham_C = 1.1
epstein_accommodation_alpha = 1.0
drag_transition_half_width_decades = 0.5

# Initial conditions relative to the chosen equilibrium
x_displacement = 0e-6
y_displacement = 0e-6
z_displacement = 0.0e-6
vx0 = 0.0e-6
vy0 = 0.0
vz0 = 0.0

# Time integration
t_start = 0
t_end =20
dt_baoab = 1 / 350000

# Brownian motion
# Set to False to remove the random thermal kicks while keeping gas damping.
brownian_seed = 90089
laser_noise_seed = 789

# Trap-loss termination
# The automatic limits are intentionally wider than the thermal motion but
# local to the trapped region. Set either manual limit to a number in metres to
# override the corresponding automatic value.
trap_loss_check_interval_steps = 1000
trap_loss_radial_limit_manual = None
trap_loss_axial_limit_manual = None
trap_loss_radial_beam_waists = 10.0
trap_loss_axial_rayleigh_ranges = 3.0
trap_loss_sustained_outside_time = 0.1
trap_loss_prediction_time = 0.1
trap_loss_prediction_steps = 200
trap_loss_reentry_tolerance = 0.98
trap_loss_post_loss_plot_time = 3.0

# Optional laser feedback loop
# Set this to True to add a delayed PD loop that reads z and varies laser power.
feedback_update_frequency = 20000                              # Hz, 50 microsecond updates
feedback_noise_seed = 17
use_feedback_position_noise = True
feedback_position_noise_rms = 0e-9                             # m RMS
feedback_total_loop_frequency=20000
feedback_total_loop_delay = 1/feedback_total_loop_frequency    # s
feedback_kp_multiplier = 0.2                                   # Kp = this number * axial spring constant kz
feedback_kd_multiplier = 8.0                                   # Kd = this number * gas damping coefficient b
feedback_power_min_factor = 0.80                               # minimum command = this * nominal power
feedback_power_max_factor = 1.20                               # maximum command = this * nominal power
feedback_velocity_filter_alpha = 0.25                          # lower values smooth velocity more

# Numerical tolerances
focus_zero_tolerance = 1e-30
radial_zero_tolerance = 1e-30
near_axis_tolerance = 1e-12

# Plotting and diagnostic ranges
max_plot_points = 200000
psd_plot_max_frequency_override = None
psd_min_frequency_factor = 0.95
psd_min_value = 1e-30
laser_power_psd_min_value = 1e-21
psd_max_value = None
psd_segment_duration_seconds = 2.0
welch_average_segment_duration_seconds = 2.0
psd_segment_samples = None
psd_overlap_fraction = 0.5
psd_overlap_samples = None
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
fx_heatmap_x_min = -100e-6
fx_heatmap_x_max = 100e-6
fx_heatmap_z_min = -1000e-6
fx_heatmap_z_max = 3000e-6
fx_heatmap_x_points = 121
fx_heatmap_z_points = 181
on_axis_check_z_min = -200e-6
on_axis_check_z_max = 300e-6
on_axis_check_points = 1000
force_comparison_x_half_width = 50e-6
force_comparison_z_half_width = 200e-6
force_comparison_points = 1000
poster_single_figsize = (3.4, 2.6)
poster_stacked_figsize = (3.4, 3.6)
poster_wide_figsize = (6.8, 2.6)
time_trace_figsize = poster_stacked_figsize
single_diagnostic_figsize = poster_single_figsize
spectrum_figsize = poster_single_figsize
trajectory_projection_figsize = poster_wide_figsize
force_check_figsize = poster_single_figsize
force_field_figsize = poster_single_figsize

fontsize=11


# ****************************************************************************************************************************************************
# OUTPUT HELPERS - all generated files should go through save_path
# ****************************************************************************************************************************************************

plot_data_counter = 0


def output_path(filename):
    return save_path / filename


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
    return


def apply_plot_text_style(fig):
    return


def finish_plot(name=None, fig=None):
    target_fig = fig if fig is not None else plt.gcf()

    if name is not None:
        make_legends_transparent(target_fig)
        apply_plot_text_style(target_fig)
        target_fig.savefig(
            output_path(f"{name}.png"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

    if display_plots:
        plt.show()
    else:
        plt.close()


# ****************************************************************************************************************************************************
# DERIVED QUANTITIES - calculated from the settings above
# ****************************************************************************************************************************************************
volume = (4/3) * np.pi * radius**3
m = density * volume

rho_g = p * M_air / (R * T)
lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

zR_m2 = np.pi * w0**2 / (M2 * wavelength)
zR = zR_m2 if use_m2_rayleigh_range else zR_manual

I0 = 2 * P_laser / (np.pi * w0**2)    # Gaussian peak intensity, W/m^2


def buoyancy_force(pressure=None):
    """
    Upward buoyancy force from displaced gas.

    F_b = rho_g V g, with rho_g evaluated using the ideal-gas relation at
    the current simulation pressure unless another pressure is supplied.
    """
    if pressure is None:
        pressure = p

    rho_g_local = pressure * M_air / (R * T)
    return rho_g_local * volume * g


def effective_weight_force(pressure=None):
    """
    Downward body force after buoyancy correction.
    """
    return m * g - buoyancy_force(pressure)

# print("Particle mass =", m, "kg")
# print("Weight mg =", m*g, "N")
# print("Buoyancy force =", buoyancy_force(), "N")
# print("Effective weight =", effective_weight_force(), "N")
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

    cos_2theta_i = np.cos(2 * theta_i)
    sin_2theta_i = np.sin(2 * theta_i)
    cos_refracted_angle = np.cos(2 * theta_i - 2 * theta_r)
    sin_refracted_angle = np.sin(2 * theta_i - 2 * theta_r)

    Q_scat = 1 + fresnel_R * cos_2theta_i - fresnel_T**2 * (cos_refracted_angle + fresnel_R * cos_2theta_i) / denom
    Q_grad = fresnel_R * sin_2theta_i - fresnel_T**2 * (sin_refracted_angle + fresnel_R * sin_2theta_i) / denom
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
    """
    Locally absorbed laser intensity.

    I_abs(x,z) = f_abs I(x,z), where f_abs is estimated from

        f_abs = 1 - exp(-alpha_abs L_eff),
        alpha_abs = 4 pi kappa / lambda.

    Here kappa is the extinction coefficient, taken as the imaginary part of
    the complex refractive index, lambda is the vacuum wavelength, and L_eff is
    the effective optical path length through the particle.
    """
    return absorption_fraction * physical_intensity(x, z, power_factor)


def photophoretic_force_magnitude(x, z, power_factor=1.0):
    """
    Photophoretic force magnitude using the Rohatschek-style pressure scaling.

    The maximum photophoretic force is

        F_max = (a^2 / 2) D sqrt(alpha / 2) I_abs(x,z) / k_p,

    where a is the particle radius, D is the photophoretic coefficient, alpha
    is the thermal accommodation coefficient, k_p is the particle thermal
    conductivity, and I_abs(x,z) is the locally absorbed laser intensity.
    """
    D = photophoretic_D()
    p_max_ph = photophoretic_p_max()

    I_abs = absorbed_intensity(x, z, power_factor)

    F_max = 0.5 * radius**2 * D * np.sqrt(alpha_acc / 2) * I_abs / k_particle

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

    r_inside = np.min(r_arr) >= force_lookup_r_values[0] and np.max(r_arr) <= force_lookup_r_values[-1]
    z_inside = np.min(z_arr) >= force_lookup_z_values[0] and np.max(z_arr) <= force_lookup_z_values[-1]

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

    Fr_nominal = (1 - r_weight) * (1 - z_weight) * Fr00 + r_weight * (1 - z_weight) * Fr10 + (1 - r_weight) * z_weight * Fr01 + r_weight * z_weight * Fr11
    Fz_nominal = (1 - r_weight) * (1 - z_weight) * Fz00 + r_weight * (1 - z_weight) * Fz10 + (1 - r_weight) * z_weight * Fz01 + r_weight * z_weight * Fz11

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
    return Fz - effective_weight_force()


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

equilibrium_roots = stable_roots

if len(equilibrium_roots) == 0:
    if len(roots) == 0:
        raise ValueError("No on-axis equilibrium found.")

    print(
        "No stable on-axis equilibrium found. "
        "Continuing with an unstable on-axis equilibrium so the trajectory can be plotted."
    )
    equilibrium_roots = roots

if not -len(equilibrium_roots) <= equilibrium_root_index < len(equilibrium_roots):
    raise IndexError(
        "equilibrium_root_index is outside the available equilibrium root list. "
        f"Found {len(equilibrium_roots)} available root(s)."
    )

x_eq = x_equilibrium
y_eq = y_equilibrium
z_eq = equilibrium_roots[equilibrium_root_index]

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
    return Fz - effective_weight_force()


kx = -numerical_derivative_1d(Fx_at_x, x_eq)
ky = kx
kz = -numerical_derivative_1d(Fz_net_at_z, z_eq)

selected_equilibrium_is_stable = kx > 0 and kz > 0

if not selected_equilibrium_is_stable:
    print(
        "The selected equilibrium is not stable. "
        f"kx={kx:.3e} N/m, kz={kz:.3e} N/m. "
        "Check the ray-optics force signs, laser power, search range, or chosen root."
    )


def angular_frequency_from_stiffness(stiffness):
    if stiffness <= 0 or not np.isfinite(stiffness):
        return np.nan
    return np.sqrt(stiffness / m)


omega_x = angular_frequency_from_stiffness(kx)
omega_y = angular_frequency_from_stiffness(ky)
omega_z = angular_frequency_from_stiffness(kz)
expected_fx = omega_x / (2 * np.pi)
expected_fy = omega_y / (2 * np.pi)
expected_fz = omega_z / (2 * np.pi)

log(f"Effective spring constant kx = {kx} N/m", verbose=True)
log(f"Effective spring constant ky = {ky} N/m", verbose=True)
log(f"Effective spring constant kz = {kz} N/m", verbose=True)
log(f"Expected x resonant frequency = {expected_fx} Hz", verbose=True)
log(f"Expected y resonant frequency = {expected_fy} Hz", verbose=True)
log(f"Expected z resonant frequency = {expected_fz} Hz", verbose=True)

# ****************************************************************************************************************************************************
# Nonlinear force compared with the local linear approximation
# ****************************************************************************************************************************************************
x_force_values = np.linspace(
    x_eq - force_comparison_x_half_width,
    x_eq + force_comparison_x_half_width,
    force_comparison_points
)
x_nonlinear_net_force = np.array([
    F_optical_3d(x_i, y_eq, z_eq)[0]
    for x_i in x_force_values
])
x_linear_net_force = -kx * (x_force_values - x_eq)

plt.figure(figsize=force_check_figsize)
plt.plot(
    x_force_values * 1e6,
    x_nonlinear_net_force,
    label="Nonlinear net force"
)
plt.plot(
    x_force_values * 1e6,
    x_linear_net_force,
    linestyle="--",
    label="Linearised net force"
)
plt.axhline(0.0, color="black", linewidth=0.8)
plt.axvline(
    x_eq * 1e6,
    linestyle=":",
    color="tab:blue",
    label="Equilibrium"
)
plt.xlabel("x position (μm)")
plt.ylabel("Net force (N)")
plt.title("Nonlinear x force compared with linear approximation")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="nonlinear_vs_linear_force_x")

z_force_values = np.linspace(
    z_eq - force_comparison_z_half_width,
    z_eq + force_comparison_z_half_width,
    force_comparison_points
)
z_nonlinear_net_force = np.array([
    F_optical_3d(x_eq, y_eq, z_i)[2] - effective_weight_force()
    for z_i in z_force_values
])
z_linear_net_force = -kz * (z_force_values - z_eq)

plt.figure(figsize=force_check_figsize)
plt.plot(
    z_force_values * 1e6,
    z_nonlinear_net_force,
    label="Nonlinear net force"
)
plt.plot(
    z_force_values * 1e6,
    z_linear_net_force,
    linestyle="--",
    label="Linearised net force"
)
plt.axhline(0.0, color="black", linewidth=0.8)
plt.axvline(
    z_eq * 1e6,
    linestyle=":",
    color="tab:blue",
    label="Equilibrium"
)
plt.xlabel("z position (μm)")
plt.ylabel("Net force (N)")
plt.title("Nonlinear z force compared with linear approximation")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="nonlinear_vs_linear_force_z")


def percentage_difference_from_linear(nonlinear_force, linear_force, displacement_um=None):
    if displacement_um is not None and np.isclose(displacement_um, 0.0):
        return 0.0

    if not np.isfinite(linear_force) or abs(linear_force) < 1e-30:
        return None
    return 100.0 * abs(nonlinear_force - linear_force) / abs(linear_force)


def percentage_difference_array(displacement_offsets_um, nonlinear_forces, linear_forces):
    return np.array([
        percentage_difference_from_linear(
            nonlinear_force,
            linear_force,
            displacement_um
        )
        for displacement_um, nonlinear_force, linear_force in zip(
            displacement_offsets_um,
            nonlinear_forces,
            linear_forces
        )
    ], dtype=float)


def print_force_difference_table(axis_label, displacement_offsets_um, nonlinear_forces, linear_forces):
    print()
    print(f"{axis_label} force percentage difference from linear approximation")
    print("Displacement from equilibrium (um) | Nonlinear force (N) | Linear force (N) | Difference (%)")
    print("-" * 88)

    for displacement_um, nonlinear_force, linear_force in zip(
        displacement_offsets_um,
        nonlinear_forces,
        linear_forces
    ):
        percentage_difference = percentage_difference_from_linear(
            nonlinear_force,
            linear_force,
            displacement_um
        )
        percentage_text = "undefined" if percentage_difference is None else f"{percentage_difference:.3g}"
        print(
            f"{displacement_um:>31.1f} | "
            f"{nonlinear_force:>18.6e} | "
            f"{linear_force:>16.6e} | "
            f"{percentage_text:>14}"
        )


x_table_offsets_um = np.arange(-10.0, 10.0 + 0.5, 2.0)
x_table_positions = x_eq + x_table_offsets_um * 1e-6
x_table_nonlinear_forces = np.array([
    F_optical_3d(x_i, y_eq, z_eq)[0]
    for x_i in x_table_positions
])
x_table_linear_forces = -kx * (x_table_positions - x_eq)
x_table_percentage_differences = percentage_difference_array(
    x_table_offsets_um,
    x_table_nonlinear_forces,
    x_table_linear_forces
)
if verbose_output:
    print_force_difference_table(
        "x",
        x_table_offsets_um,
        x_table_nonlinear_forces,
        x_table_linear_forces
    )

z_table_offsets_um = np.arange(-100.0, 100.0 + 0.5, 20.0)
z_table_positions = z_eq + z_table_offsets_um * 1e-6
z_table_nonlinear_forces = np.array([
    F_optical_3d(x_eq, y_eq, z_i)[2] - effective_weight_force()
    for z_i in z_table_positions
])
z_table_linear_forces = -kz * (z_table_positions - z_eq)
z_table_percentage_differences = percentage_difference_array(
    z_table_offsets_um,
    z_table_nonlinear_forces,
    z_table_linear_forces
)
if verbose_output:
    print_force_difference_table(
        "z",
        z_table_offsets_um,
        z_table_nonlinear_forces,
        z_table_linear_forces
    )

plt.figure(figsize=force_check_figsize)
plt.plot(
    x_table_offsets_um,
    x_table_percentage_differences,
    marker="o",
    label="Difference from linear"
)
plt.axvline(0.0, color="tab:blue", linestyle=":", label="Equilibrium")
plt.xlabel("x displacement from equilibrium (μm)")
plt.ylabel("Difference from linear (%)")
plt.title("x force nonlinearity")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="force_percentage_difference_x")

plt.figure(figsize=force_check_figsize)
plt.plot(
    z_table_offsets_um,
    z_table_percentage_differences,
    marker="o",
    label="Difference from linear"
)
plt.axvline(0.0, color="tab:blue", linestyle=":", label="Equilibrium")
plt.xlabel("z displacement from equilibrium (μm)")
plt.ylabel("Difference from linear (%)")
plt.title("z force nonlinearity")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="force_percentage_difference_z")

x_effective_stiffness = -np.gradient(x_nonlinear_net_force, x_force_values)
z_effective_stiffness = -np.gradient(z_nonlinear_net_force, z_force_values)

plt.figure(figsize=force_check_figsize)
plt.plot(
    (x_force_values - x_eq) * 1e6,
    x_effective_stiffness,
    label="Nonlinear effective stiffness"
)
plt.axhline(kx, color="tab:orange", linestyle="--", label="Linearised stiffness")
plt.axvline(0.0, color="tab:blue", linestyle=":", label="Equilibrium")
plt.xlabel("x displacement from equilibrium (μm)")
plt.ylabel("Effective stiffness (N m^-1)")
plt.title("x effective stiffness")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="effective_stiffness_x")

plt.figure(figsize=force_check_figsize)
plt.plot(
    (z_force_values - z_eq) * 1e6,
    z_effective_stiffness,
    label="Nonlinear effective stiffness"
)
plt.axhline(kz, color="tab:orange", linestyle="--", label="Linearised stiffness")
plt.axvline(0.0, color="tab:blue", linestyle=":", label="Equilibrium")
plt.xlabel("z displacement from equilibrium (μm)")
plt.ylabel("Effective stiffness (N m^-1)")
plt.title("z effective stiffness")
plt.legend()
plt.grid()
plt.tight_layout()
finish_plot(name="effective_stiffness_z")

def thermal_rms_from_stiffness(stiffness):
    if stiffness <= 0 or not np.isfinite(stiffness):
        return np.nan
    return np.sqrt(kB * T / stiffness)


x_rms_thermal = thermal_rms_from_stiffness(kx)
y_rms_thermal = thermal_rms_from_stiffness(ky)
z_rms_thermal = thermal_rms_from_stiffness(kz)

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

    return 1 + Kn * (cunningham_A + cunningham_B * np.exp(-cunningham_C / Kn))


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


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def transition_blend_weight(
    Kn,
    transition_Kn,
    half_width_decades=drag_transition_half_width_decades,
):
    """
    Smoothly changes from 0 to 1 around a Knudsen-regime boundary.
    The blend is performed in log10(Kn), because the regimes span decades.
    """
    if Kn <= 0 or transition_Kn <= 0:
        return 0.0

    log_Kn = np.log10(Kn)
    log_transition = np.log10(transition_Kn)
    u = (log_Kn - (log_transition - half_width_decades)) / (2.0 * half_width_decades)

    return smoothstep(u)


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


def damping_coefficient_auto_sharp(pressure):
    """
    Original automatic model selection with sharp boundaries.
    """
    model = choose_drag_model(pressure)

    if model == "stokes":
        return damping_coefficient_stokes(), model
    if model == "cunningham":
        return damping_coefficient_stokes_cunningham(pressure), model
    if model == "epstein":
        return damping_coefficient_epstein(pressure), model

    raise ValueError("Unknown automatically selected drag model.")


def damping_coefficient_auto_smooth(pressure):
    """
    Automatic pressure-dependent damping with smooth weighted transitions.

    Stokes and Cunningham drag are blended around Kn = 0.1, and
    Cunningham and Epstein drag are blended around Kn = 10.
    """
    Kn = knudsen_number(pressure)

    b_stokes_local = damping_coefficient_stokes()
    b_cunningham_local = damping_coefficient_stokes_cunningham(pressure)
    b_epstein_local = damping_coefficient_epstein(pressure)

    stokes_cunningham_weight = transition_blend_weight(Kn, 0.1)
    cunningham_epstein_weight = transition_blend_weight(Kn, 10.0)

    b_low = (1.0 - stokes_cunningham_weight) * b_stokes_local + stokes_cunningham_weight * b_cunningham_local

    return (1.0 - cunningham_epstein_weight) * b_low + cunningham_epstein_weight * b_epstein_local


def damping_coefficient(pressure, model="auto"):
    """
    Return damping coefficient b for the chosen drag model.
    """
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


b_stokes = damping_coefficient_stokes()
b_cunningham = damping_coefficient_stokes_cunningham(p)
b_epstein = damping_coefficient_epstein(p)

b, drag_model_used = damping_coefficient(p, drag_model)

if damping == False:
    b=0

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
az_initial = (Fz_initial - effective_weight_force()) / m

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
    force_lookup_r_thermal_limit = force_lookup_r_thermal_factor * x_rms_thermal if np.isfinite(x_rms_thermal) else 0.0
    force_lookup_z_thermal_limit = force_lookup_z_thermal_factor * z_rms_thermal if np.isfinite(z_rms_thermal) else 0.0
    force_lookup_r_max = max(force_lookup_r_base_max, force_lookup_r_displacement_factor * np.sqrt(x_displacement**2 + y_displacement**2), force_lookup_r_thermal_limit)
    force_lookup_z_half_width = max(
        force_lookup_z_base_half_width,
        force_lookup_z_displacement_factor * abs(z_displacement),
        force_lookup_z_thermal_limit
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

trap_loss_radial_limit_auto = trap_loss_radial_beam_waists * w0
trap_loss_axial_limit_auto = trap_loss_axial_rayleigh_ranges * zR

if use_force_lookup_table:
    trap_loss_radial_limit_auto = max(
        trap_loss_radial_limit_auto,
        force_lookup_r_max
    )
    trap_loss_axial_limit_auto = max(
        trap_loss_axial_limit_auto,
        force_lookup_z_half_width
    )

trap_loss_radial_limit = trap_loss_radial_limit_auto if trap_loss_radial_limit_manual is None else trap_loss_radial_limit_manual
trap_loss_axial_limit = trap_loss_axial_limit_auto if trap_loss_axial_limit_manual is None else trap_loss_axial_limit_manual

if trap_loss_radial_limit <= 0:
    raise ValueError("trap_loss_radial_limit must be positive.")

if trap_loss_axial_limit <= 0:
    raise ValueError("trap_loss_axial_limit must be positive.")

trap_loss_sustained_outside_steps = max(
    1,
    int(np.ceil(trap_loss_sustained_outside_time / dt_baoab))
)
trap_loss_prediction_steps = max(1, int(trap_loss_prediction_steps))
trap_loss_prediction_dt = trap_loss_prediction_time / trap_loss_prediction_steps
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
    outside_axial = abs(axial_displacement) > trap_loss_axial_limit
    return radial_displacement, axial_displacement, outside_radial, outside_axial


def is_inside_capture_region(x, y, z, tolerance=1.0):
    radial_displacement, axial_displacement, _, _ = trap_region_metrics(x, y, z)
    return radial_displacement <= tolerance * trap_loss_radial_limit and abs(axial_displacement) <= tolerance * trap_loss_axial_limit


def is_reentering_capture_region(x, y, z, vx, vy, vz):
    radial_displacement, axial_displacement, outside_radial, outside_axial = trap_region_metrics(x, y, z)

    if not (outside_radial or outside_axial):
        return True

    reentering_radially = True
    if outside_radial and radial_displacement > radial_zero_tolerance:
        radial_velocity = ((x - x_eq) * vx + (y - y_eq) * vy) / radial_displacement
        reentering_radially = radial_velocity < 0

    reentering_axially = True
    if outside_axial:
        reentering_axially = axial_displacement * vz < 0

    return reentering_radially and reentering_axially


def predicted_to_reenter_capture_region(x, y, z, vx, vy, vz, power_factor):
    x_pred = x
    y_pred = y
    z_pred = z
    vx_pred = vx
    vy_pred = vy
    vz_pred = vz

    for _ in range(trap_loss_prediction_steps):
        Fx_pred, Fy_pred, Fz_pred = deterministic_force_no_drag_3d(
            x_pred,
            y_pred,
            z_pred,
            power_factor
        )
        vx_pred += trap_loss_prediction_dt * Fx_pred / m
        vy_pred += trap_loss_prediction_dt * Fy_pred / m
        vz_pred += trap_loss_prediction_dt * Fz_pred / m
        x_pred += trap_loss_prediction_dt * vx_pred
        y_pred += trap_loss_prediction_dt * vy_pred
        z_pred += trap_loss_prediction_dt * vz_pred

        if not np.all(np.isfinite([x_pred, y_pred, z_pred])):
            return False

        if is_inside_capture_region(
            x_pred,
            y_pred,
            z_pred,
            trap_loss_reentry_tolerance
        ):
            return True

    return False


def trap_loss_reason(x, y, z, vx, vy, vz, power_factor, sample_index, loss_state):
    if not terminate_on_trap_loss:
        return None

    if not np.all(np.isfinite([x, y, z, vx, vy, vz])):
        return "position or velocity became non-finite"

    radial_displacement, axial_displacement, outside_radial, outside_axial = trap_region_metrics(x, y, z)

    if not (outside_radial or outside_axial):
        loss_state["outside_start_sample"] = None
        return None

    if loss_state["outside_start_sample"] is None:
        loss_state["outside_start_sample"] = sample_index

    outside_steps = sample_index - loss_state["outside_start_sample"] + 1

    if outside_steps < trap_loss_sustained_outside_steps:
        return None

    if is_reentering_capture_region(x, y, z, vx, vy, vz):
        return None

    if predicted_to_reenter_capture_region(x, y, z, vx, vy, vz, power_factor):
        return None

    outside_time = outside_steps * dt_baoab
    return (
        "outside capture region for "
        f"{outside_time:.3g} s without re-entering; "
        f"radial displacement = {radial_displacement * 1e6:.3g} micrometres, "
        f"axial displacement = {axial_displacement * 1e6:.3g} micrometres"
    )


def should_check_trap_loss(sample_index):
    return (
        terminate_on_trap_loss
        and (
            sample_index % trap_loss_check_interval_steps == 0
            or sample_index == len(t_baoab) - 1
        )
    )


def record_trap_loss(run_label, sample_index, x, y, z, reason):
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
        "Trap-loss radial displacement limit =",
        trap_loss_radial_limit * 1e6,
        "micrometres"
    )
    print(
        "Trap-loss axial displacement limit =",
        trap_loss_axial_limit * 1e6,
        "micrometres"
    )
    print(
        "Trap-loss sustained outside time =",
        trap_loss_sustained_outside_steps * dt_baoab,
        "s"
    )
    print(
        "Trap-loss deterministic prediction time =",
        trap_loss_prediction_time,
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
# Laser-power noise helpers
# ****************************************************************************************************************************************************
def load_laser_power_csv(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    try:
        graph_data_index = next(
            index for index, row in enumerate(rows) if row and row[0] == "--Graph Data--"
        )
    except StopIteration as exc:
        raise ValueError(
            "Could not find '--Graph Data--' section in the laser-power CSV file."
        ) from exc

    data_rows = rows[graph_data_index + 2:]
    time_ms = []
    power_w = []

    for row in data_rows:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue

        try:
            time_ms.append(float(row[0]))
            power_w.append(float(row[1]))
        except ValueError:
            continue

    if not time_ms:
        raise ValueError("No numeric laser-power data was found in the CSV file.")

    return np.array(time_ms) / 1000.0, np.array(power_w)


def resample_to_uniform_time_grid(time_values, signal):
    order = np.argsort(time_values)
    time_values = time_values[order]
    signal = signal[order]

    finite = np.isfinite(time_values) & np.isfinite(signal)
    time_values = time_values[finite]
    signal = signal[finite]

    time_values = time_values - time_values[0]
    dt_uniform = np.median(np.diff(time_values))
    uniform_time = np.arange(0.0, time_values[-1], dt_uniform)
    uniform_signal = np.interp(uniform_time, time_values, signal)
    fs_uniform = 1.0 / dt_uniform

    return uniform_time, uniform_signal, fs_uniform


def relative_power_fluctuation(power_values):
    return (power_values - np.mean(power_values)) / np.mean(power_values)


def generate_square_laser_power_factor(time_values):
    laser_rng = np.random.default_rng(seed=laser_noise_seed)

    laser_step_samples_local = max(1, int(round(laser_noise_step_duration / dt_baoab)))
    laser_step_duration_actual_local = laser_step_samples_local * dt_baoab
    allowed_power_factors = np.array(
        [1 - laser_noise_fraction, 1.0, 1 + laser_noise_fraction]
    )
    n_laser_steps = int(np.ceil(len(time_values) / laser_step_samples_local))

    laser_step_power_factors_local = np.zeros(n_laser_steps)
    laser_step_power_factors_local[0] = laser_rng.choice(allowed_power_factors)

    for j in range(1, n_laser_steps):
        previous_factor = laser_step_power_factors_local[j - 1]

        if np.isclose(previous_factor, 1 + laser_noise_fraction):
            possible_factors = np.array([1.0, 1 + laser_noise_fraction])
        elif np.isclose(previous_factor, 1 - laser_noise_fraction):
            possible_factors = np.array([1 - laser_noise_fraction, 1.0])
        else:
            possible_factors = allowed_power_factors

        laser_step_power_factors_local[j] = laser_rng.choice(possible_factors)

    power_factor = np.repeat(
        laser_step_power_factors_local,
        laser_step_samples_local
    )[:len(time_values)]

    return (
        power_factor,
        laser_step_samples_local,
        laser_step_duration_actual_local,
        allowed_power_factors,
        laser_step_power_factors_local,
    )


def generate_psd_matched_laser_power_factor(time_values):
    if not laser_noise_power_csv_path.exists():
        raise FileNotFoundError(
            "PSD-matched laser noise requires a measured power CSV. "
            f"Expected: {laser_noise_power_csv_path}. "
            "Either place the file there, update laser_noise_power_csv_path in "
            "USER SETTINGS, or set laser_noise_model = 'square'."
        )

    measured_time, measured_power = load_laser_power_csv(laser_noise_power_csv_path)
    _, measured_power_uniform, measured_fs = resample_to_uniform_time_grid(
        measured_time,
        measured_power,
    )
    measured_relative = relative_power_fluctuation(measured_power_uniform)
    measured_relative = measured_relative - np.mean(measured_relative)
    measured_relative_rms = np.std(measured_relative, ddof=1)

    target_duration = time_values[-1] - time_values[0] + dt_baoab
    synthetic_dt = 1.0 / measured_fs
    synthetic_time = np.arange(0.0, target_duration + synthetic_dt, synthetic_dt)
    synthetic_n = len(synthetic_time)

    welch_nperseg = min(
        len(measured_relative),
        max(8, int(round(2.0 * measured_fs))),
    )
    measured_freqs, measured_relative_psd = welch(
        measured_relative,
        fs=measured_fs,
        window="hann",
        nperseg=welch_nperseg,
        noverlap=welch_nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    synthetic_freqs = np.fft.rfftfreq(synthetic_n, synthetic_dt)
    target_psd = np.interp(
        synthetic_freqs,
        measured_freqs,
        measured_relative_psd,
        left=measured_relative_psd[0],
        right=0.0,
    )

    phase_rng = np.random.default_rng(seed=laser_noise_psd_matched_seed)
    phases = phase_rng.uniform(0.0, 2.0 * np.pi, len(synthetic_freqs))
    spectrum_magnitude = np.sqrt(target_psd * measured_fs * synthetic_n / 2.0)
    synthetic_spectrum = spectrum_magnitude * np.exp(1j * phases)
    synthetic_spectrum[0] = 0.0

    if synthetic_n % 2 == 0:
        synthetic_spectrum[-1] = spectrum_magnitude[-1] * phase_rng.choice([-1.0, 1.0])

    synthetic_relative = np.fft.irfft(synthetic_spectrum, n=synthetic_n)
    synthetic_relative = synthetic_relative - np.mean(synthetic_relative)
    synthetic_relative_rms = np.std(synthetic_relative, ddof=1)

    if synthetic_relative_rms > 0:
        synthetic_relative *= measured_relative_rms / synthetic_relative_rms

    interpolated_relative = np.interp(
        time_values - time_values[0],
        synthetic_time,
        synthetic_relative,
    )
    power_factor = 1.0 + interpolated_relative

    metadata = {
        "measured_samples": len(measured_power),
        "measured_mean_power": np.mean(measured_power),
        "measured_relative_rms": measured_relative_rms,
        "measured_sampling_frequency": measured_fs,
        "measured_nyquist_frequency": measured_fs / 2.0,
        "welch_nperseg": welch_nperseg,
        "synthetic_relative_rms": np.std(interpolated_relative, ddof=1),
    }

    return power_factor, metadata


# ****************************************************************************************************************************************************
# BAOAB solution with Brownian motion and optional laser-power noise
# ****************************************************************************************************************************************************
rng = np.random.default_rng(seed=brownian_seed)

laser_noise_model = laser_noise_model.lower().strip()

if laser_noise_model == "psd_matched":
    if plot_vertical_expected_distribution_comparison:
        print(
            "Disabling nonlinear expected-distribution plot for PSD-matched "
            "laser noise because the continuous power trace makes this "
            "diagnostic expensive."
        )
        plot_vertical_expected_distribution_comparison = False

    if plot_uncoupled_vertical_histogram:
        print(
            "Disabling uncoupled vertical histogram for PSD-matched laser "
            "noise because it requires an expensive nonlinear distribution "
            "calculation."
        )
        plot_uncoupled_vertical_histogram = False

if use_laser_power_noise:
    if laser_noise_model == "square":
        (
            laser_power_factor,
            laser_step_samples,
            laser_step_duration_actual,
            laser_allowed_power_factors,
            laser_step_power_factors,
        ) = generate_square_laser_power_factor(t_baoab)
        laser_noise_metadata = {}
    elif laser_noise_model == "psd_matched":
        laser_power_factor, laser_noise_metadata = (
            generate_psd_matched_laser_power_factor(t_baoab)
        )
        laser_step_samples = 1
        laser_step_duration_actual = dt_baoab
        laser_allowed_power_factors = np.linspace(
            np.min(laser_power_factor),
            np.max(laser_power_factor),
            laser_noise_equilibrium_grid_points,
        )
        laser_step_power_factors = laser_power_factor
    else:
        raise ValueError(
            "Unknown laser_noise_model. Choose 'square' or 'psd_matched'."
        )
else:
    laser_noise_metadata = {}
    laser_step_samples = len(t_baoab)
    laser_step_duration_actual = t_baoab[-1] - t_baoab[0]
    laser_allowed_power_factors = np.array([1.0])
    laser_step_power_factors = np.array([1.0])
    laser_power_factor = np.ones_like(t_baoab)

laser_power_time = P_laser * laser_power_factor

instantaneous_z_equilibrium_by_power_factor = {}

if laser_noise_model == "psd_matched" and use_laser_power_noise:
    # For the continuous PSD-matched trace, repeatedly root-finding the exact
    # equilibrium for many power factors is only a diagnostic cost. Since the
    # optical force scales linearly with power, the small equilibrium shift can
    # be estimated from the local axial stiffness.
    z_equilibrium_power_sensitivity = effective_weight_force() / kz
    z_eq_laser_noise_time = z_eq + z_equilibrium_power_sensitivity * (laser_power_factor - 1.0)
    laser_equilibrium_power_factors = np.array(
        [np.min(laser_power_factor), 1.0, np.max(laser_power_factor)]
    )
    laser_equilibrium_z_values = z_eq + z_equilibrium_power_sensitivity * (laser_equilibrium_power_factors - 1.0)
    instantaneous_z_equilibrium_by_power_factor = dict(
        zip(laser_equilibrium_power_factors, laser_equilibrium_z_values)
    )
else:
    laser_equilibrium_power_factors = np.unique(laser_allowed_power_factors)
    laser_equilibrium_z_values = []

    for power_factor_value in laser_equilibrium_power_factors:
        def Fz_net_on_axis_at_power(z):
            _, Fz = F_optical_2d_direct(0.0, z, power_factor_value)
            return Fz - effective_weight_force()

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

        z_eq_power = stable_roots_power[equilibrium_root_index]
        instantaneous_z_equilibrium_by_power_factor[power_factor_value] = z_eq_power
        laser_equilibrium_z_values.append(z_eq_power)

    laser_equilibrium_z_values = np.array(laser_equilibrium_z_values)
    z_eq_laser_noise_time = np.interp(
        laser_power_factor,
        laser_equilibrium_power_factors,
        laser_equilibrium_z_values,
    )

gamma_baoab = b / m
baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
baoab_thermal_velocity_scale = np.sqrt((kB * T / m) * (1 - baoab_damping_factor**2))

log(f"Brownian bath temperature = {T} K", verbose=True)
log(f"Trajectory damping rate gamma = {gamma_baoab} s^-1", verbose=True)
log(f"Laser power noise enabled = {use_laser_power_noise}", verbose=True)
log(f"Laser power noise model = {laser_noise_model}", verbose=True)
log(
    "Laser power factor range = "
    f"{np.min(laser_power_factor)} to {np.max(laser_power_factor)}",
    verbose=True,
)
log(
    f"Laser power factor RMS = {np.std(laser_power_factor - 1.0, ddof=1)}",
    verbose=True,
)

if laser_noise_model == "psd_matched" and use_laser_power_noise:
    log(f"PSD-matched laser noise CSV = {laser_noise_power_csv_path}", verbose=True)
    log(
        "Measured laser-power sampling frequency = "
        f"{laser_noise_metadata['measured_sampling_frequency']} Hz",
        verbose=True,
    )
    log(
        "Measured laser-power PSD Nyquist limit = "
        f"{laser_noise_metadata['measured_nyquist_frequency']} Hz",
        verbose=True,
    )
    log(
        "Measured laser-power relative RMS = "
        f"{laser_noise_metadata['measured_relative_rms']}",
        verbose=True,
    )
    log(
        "Synthetic laser-power relative RMS = "
        f"{laser_noise_metadata['synthetic_relative_rms']}",
        verbose=True,
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
    return Fx, Fy, Fz - effective_weight_force()


def solve_baoab_uncoupled_nonlinear_1d(
    force_function,
    power_factor_time,
    initial_position,
    initial_velocity,
    brownian_normals
):
    position_out = np.zeros_like(t_baoab)
    velocity_out = np.zeros_like(t_baoab)

    position_out[0] = initial_position
    velocity_out[0] = initial_velocity

    for i in range(len(t_baoab) - 1):
        position_i = position_out[i]
        velocity_i = velocity_out[i]
        power_factor_i = power_factor_time[i]

        force_i = force_function(position_i, power_factor_i)
        velocity_i += 0.5 * dt_baoab * force_i / m
        position_i += 0.5 * dt_baoab * velocity_i

        velocity_i = baoab_damping_factor * velocity_i + baoab_thermal_velocity_scale * brownian_normals[i]

        position_i += 0.5 * dt_baoab * velocity_i

        force_i = force_function(position_i, power_factor_i)
        velocity_i += 0.5 * dt_baoab * force_i / m

        position_out[i + 1] = position_i
        velocity_out[i + 1] = velocity_i

    return position_out, velocity_out


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

        vx_i = baoab_damping_factor * vx_i + baoab_thermal_velocity_scale * brownian_normals_x[i]
        vy_i = baoab_damping_factor * vy_i + baoab_thermal_velocity_scale * brownian_normals_y[i]
        vz_i = baoab_damping_factor * vz_i + baoab_thermal_velocity_scale * brownian_normals_z[i]

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
        not terminate_on_trap_loss
        and cython_baoab_available
        and use_force_lookup_table
        and force_lookup_ready
    ):
        dummy_axial_energy_z_values = np.ascontiguousarray([0.0], dtype=np.float64)
        dummy_axial_potential_values = np.ascontiguousarray([0.0], dtype=np.float64)
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
            effective_weight_force(),
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
            dummy_axial_energy_z_values,
            dummy_axial_potential_values,
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
            _stop_index,
            _lost_flag,
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
                z_filtered = feedback_velocity_filter_alpha * z_measured + (1 - feedback_velocity_filter_alpha) * filtered_z_previous
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
        Fz_i -= effective_weight_force()
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vy_i += 0.5 * dt_baoab * Fy_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_i += 0.5 * dt_baoab * vx_i
        y_i += 0.5 * dt_baoab * vy_i
        z_i += 0.5 * dt_baoab * vz_i

        vx_i = baoab_damping_factor * vx_i + baoab_thermal_velocity_scale * brownian_normals_x[i]
        vy_i = baoab_damping_factor * vy_i + baoab_thermal_velocity_scale * brownian_normals_y[i]
        vz_i = baoab_damping_factor * vz_i + baoab_thermal_velocity_scale * brownian_normals_z[i]

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
        Fz_i -= effective_weight_force()
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
            effective_weight_force(),
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


zero_brownian_normals = np.zeros(len(t_baoab) - 1)
if use_brownian_motion:
    brownian_normals_x = rng.normal(size=len(t_baoab) - 1)
    brownian_normals_y = rng.normal(size=len(t_baoab) - 1)
    brownian_normals_z = rng.normal(size=len(t_baoab) - 1)
else:
    brownian_normals_x = zero_brownian_normals
    brownian_normals_y = zero_brownian_normals
    brownian_normals_z = zero_brownian_normals
constant_power_factor = np.ones_like(t_baoab)

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
baoab_laser_noise_without_brownian_runtime = perf_counter() - baoab_laser_noise_without_brownian_start_time

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

uncoupled_nonlinear_runtime = 0.0
x_uncoupled_nonlinear = None
z_uncoupled_nonlinear = None
vx_uncoupled_nonlinear = None
vz_uncoupled_nonlinear = None
run_uncoupled_nonlinear = include_uncoupled_nonlinear_psd or plot_uncoupled_vertical_histogram or plot_aligned_summary_figure

if run_uncoupled_nonlinear:
    uncoupled_nonlinear_start_time = perf_counter()
    x_uncoupled_nonlinear, vx_uncoupled_nonlinear = solve_baoab_uncoupled_nonlinear_1d(
        lambda x_position, power_factor: F_optical_3d(
            x_position,
            y_eq,
            z_eq,
            power_factor
        )[0],
        laser_power_factor,
        x0,
        vx0,
        brownian_normals_x
    )
    z_uncoupled_nonlinear, vz_uncoupled_nonlinear = solve_baoab_uncoupled_nonlinear_1d(
        lambda z_position, power_factor: F_optical_3d(x_eq, y_eq, z_position, power_factor)[2] - effective_weight_force(),
        laser_power_factor,
        z0,
        vz0,
        brownian_normals_z
    )
    uncoupled_nonlinear_runtime = perf_counter() - uncoupled_nonlinear_start_time

analysis_lengths = [
    len(x_baoab_constant_power),
    len(x_baoab),
    len(x_baoab_no_brownian),
]

if use_pd_feedback:
    analysis_lengths.append(len(x_baoab_feedback))

if run_uncoupled_nonlinear:
    analysis_lengths.append(len(x_uncoupled_nonlinear))

analysis_sample_count = min(analysis_lengths)

if analysis_sample_count < 2:
    raise RuntimeError("Trap loss occurred before enough samples were available for analysis.")

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

if run_uncoupled_nonlinear:
    x_uncoupled_nonlinear = trim_analysis_array(x_uncoupled_nonlinear)
    z_uncoupled_nonlinear = trim_analysis_array(z_uncoupled_nonlinear)
    vx_uncoupled_nonlinear = trim_analysis_array(vx_uncoupled_nonlinear)
    vz_uncoupled_nonlinear = trim_analysis_array(vz_uncoupled_nonlinear)

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

baoab_total_runtime = baoab_constant_power_runtime + baoab_laser_noise_runtime + baoab_laser_noise_without_brownian_runtime + baoab_feedback_runtime + uncoupled_nonlinear_runtime

x_laser_noise_difference = x_baoab - x_baoab_constant_power
y_laser_noise_difference = y_baoab - y_baoab_constant_power
z_laser_noise_difference = z_baoab - z_baoab_constant_power
radial_laser_noise_difference = np.sqrt(x_laser_noise_difference**2 + y_laser_noise_difference**2)

x_brownian_difference = x_baoab - x_baoab_no_brownian
y_brownian_difference = y_baoab - y_baoab_no_brownian
z_brownian_difference = z_baoab - z_baoab_no_brownian


def finite_rms(values):
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.nan
    return np.sqrt(np.mean(finite_values**2))


def finite_abs_max(values):
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.nan
    return np.max(np.abs(finite_values))


detector_position_resolution_m = detector_position_resolution_nm * 1e-9
radial_laser_noise_rms_nm = finite_rms(radial_laser_noise_difference) * 1e9
radial_laser_noise_max_nm = finite_abs_max(radial_laser_noise_difference) * 1e9
z_laser_noise_rms_nm = finite_rms(z_laser_noise_difference) * 1e9
z_laser_noise_max_nm = finite_abs_max(z_laser_noise_difference) * 1e9

if verbose_output:
    log(f"BAOAB constant-power runtime = {baoab_constant_power_runtime} s")
    log(f"BAOAB laser-noise runtime = {baoab_laser_noise_runtime} s")
    log(
        "BAOAB laser-noise without-Brownian runtime = "
        f"{baoab_laser_noise_without_brownian_runtime} s"
    )
    log(f"BAOAB feedback runtime = {baoab_feedback_runtime} s")
    log(f"Uncoupled nonlinear reference runtime = {uncoupled_nonlinear_runtime} s")
    log(f"BAOAB total runtime = {baoab_total_runtime} s")
    log(
        "RMS isolated laser-noise effect in x = "
        f"{finite_rms(x_laser_noise_difference) * 1e9} nm"
    )
    log(
        "RMS isolated laser-noise effect in y = "
        f"{finite_rms(y_laser_noise_difference) * 1e9} nm"
    )
    log(
        "RMS isolated laser-noise effect in radial position = "
        f"{radial_laser_noise_rms_nm} nm"
    )
    log(f"RMS isolated laser-noise effect in z = {z_laser_noise_rms_nm} nm")
    log(
        "Max isolated laser-noise effect in x = "
        f"{finite_abs_max(x_laser_noise_difference) * 1e9} nm"
    )
    log(
        "Max isolated laser-noise effect in y = "
        f"{finite_abs_max(y_laser_noise_difference) * 1e9} nm"
    )
    log(
        "Max isolated laser-noise effect in radial position = "
        f"{radial_laser_noise_max_nm} nm"
    )
    log(f"Max isolated laser-noise effect in z = {z_laser_noise_max_nm} nm")
    if detector_position_resolution_nm > 0:
        log(
            "Max isolated radial laser-noise effect / detector resolution = "
            f"{radial_laser_noise_max_nm / detector_position_resolution_nm}"
        )
        log(
            "Max isolated z laser-noise effect / detector resolution = "
            f"{z_laser_noise_max_nm / detector_position_resolution_nm}"
        )

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
trajectory_color = "tab:orange"

axes[0].plot(
    t_plot,
    x_baoab[plot_slice] * 1e6,
    color=trajectory_color,
    linewidth=0.9,
    alpha=0.75,
    label="trajectory"
)
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[0].set_ylabel("x (μm)")
axes[0].grid()

axes[1].plot(
    t_plot,
    y_baoab[plot_slice] * 1e6,
    color=trajectory_color,
    linewidth=0.9,
    alpha=0.75,
    label="trajectory"
)
axes[1].axhline(y_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[1].set_ylabel("y (μm)")
axes[1].grid()

axes[2].plot(
    t_plot,
    z_baoab[plot_slice] * 1e6,
    color=trajectory_color,
    linewidth=0.9,
    alpha=0.75,
    label="trajectory"
)
axes[2].axhline(z_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("z (μm)")
axes[2].grid()

fig.suptitle("3D BAOAB ray-optics motion")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    ["Trajectory", "Equilibrium"],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
    ncol=2
)
plt.tight_layout(rect=[0, 0, 1, 0.88])
finish_plot(name="xyz_time_traces")

# ****************************************************************************************************************************************************
# Position histograms
# ****************************************************************************************************************************************************
position_histogram_data = [
    ("x", x_baoab, x_eq, x_rms_thermal),
    ("y", y_baoab, y_eq, y_rms_thermal),
    ("z", z_baoab, z_eq, z_rms_thermal),
]

for axis_label, position_values, equilibrium_position, linear_rms in position_histogram_data:
    plt.figure(figsize=single_diagnostic_figsize)
    position_values_um = position_values * 1e6
    finite_position_values_um = position_values_um[np.isfinite(position_values_um)]
    counts, bin_edges, _ = plt.hist(
        finite_position_values_um,
        bins=80,
        color=trajectory_color,
        alpha=0.75,
        label="Trajectory"
    )

    if plot_position_histogram_linear_prediction:
        linear_rms_um = linear_rms * 1e6
        if np.isfinite(linear_rms_um) and linear_rms_um > 0 and len(finite_position_values_um) > 0:
            bin_width_um = bin_edges[1] - bin_edges[0]
            distribution_min_um = min(
                bin_edges[0],
                equilibrium_position * 1e6 - 4 * linear_rms_um
            )
            distribution_max_um = max(
                bin_edges[-1],
                equilibrium_position * 1e6 + 4 * linear_rms_um
            )
            distribution_positions_um = np.linspace(
                distribution_min_um,
                distribution_max_um,
                1000
            )
            linear_pdf_per_um = np.exp(-0.5 * ((distribution_positions_um - equilibrium_position * 1e6) / linear_rms_um)**2) / (linear_rms_um * np.sqrt(2 * np.pi))
            linear_expected_counts = linear_pdf_per_um * len(finite_position_values_um) * bin_width_um
            plt.plot(
                distribution_positions_um,
                linear_expected_counts,
                color="tab:blue",
                linewidth=1.2,
                label="Linear prediction"
            )
        else:
            print(
                f"Skipping {axis_label} linear histogram prediction because the "
                "linear RMS is not finite and positive."
            )

    plt.axvline(
        equilibrium_position * 1e6,
        linestyle=":",
        color="black",
        label="Equilibrium"
    )
    plt.xlabel(f"{axis_label} position (μm)")
    plt.ylabel("Count")
    plt.title(f"{axis_label} position histogram")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot(name=f"position_histogram_{axis_label}")


def nonlinear_vertical_probability_density_um(z_grid_m, power_factor):
    force_values = np.array([
        F_optical_3d(x_eq, y_eq, z_i, power_factor)[2] - effective_weight_force()
        for z_i in z_grid_m
    ])
    potential_values = np.zeros_like(z_grid_m)
    potential_values[1:] = np.cumsum(-0.5 * (force_values[:-1] + force_values[1:]) * np.diff(z_grid_m))

    exponent = -(potential_values - np.min(potential_values)) / (kB * T)
    probability_density = np.exp(exponent)
    z_grid_um = z_grid_m * 1e6
    normalization = np.trapz(probability_density, z_grid_um)

    if not np.isfinite(normalization) or normalization <= 0:
        return None

    return probability_density / normalization


def linear_vertical_probability_density_um(z_grid_m):
    linear_rms_um = z_rms_thermal * 1e6

    if not np.isfinite(linear_rms_um) or linear_rms_um <= 0:
        return None

    z_grid_um = z_grid_m * 1e6
    equilibrium_z_um = z_eq * 1e6
    return np.exp(-0.5 * ((z_grid_um - equilibrium_z_um) / linear_rms_um)**2) / (linear_rms_um * np.sqrt(2 * np.pi))


def laser_power_weighted_uncoupled_z_density_um(z_grid_m):
    if laser_noise_model == "psd_matched" and use_laser_power_noise:
        bin_edges = np.linspace(
            np.min(laser_power_factor),
            np.max(laser_power_factor),
            laser_noise_equilibrium_grid_points + 1
        )
        power_factor_counts, _ = np.histogram(laser_power_factor, bins=bin_edges)
        power_factors = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        nonempty_bins = power_factor_counts > 0
        power_factors = power_factors[nonempty_bins]
        power_factor_counts = power_factor_counts[nonempty_bins]
    else:
        power_factors, power_factor_counts = np.unique(
            laser_power_factor,
            return_counts=True
        )

    weighted_density = np.zeros_like(z_grid_m)
    total_count = np.sum(power_factor_counts)

    for power_factor_value, power_factor_count in zip(
        power_factors,
        power_factor_counts
    ):
        density = nonlinear_vertical_probability_density_um(
            z_grid_m,
            power_factor_value
        )
        if density is None:
            continue

        weighted_density += (power_factor_count / total_count) * density

    normalization = np.trapz(weighted_density, z_grid_m * 1e6)

    if not np.isfinite(normalization) or normalization <= 0:
        return None

    return weighted_density / normalization


if plot_vertical_expected_distribution_comparison:
    equilibrium_z_um = z_eq * 1e6
    finite_laser_equilibria_um = z_eq_laser_noise_time[
        np.isfinite(z_eq_laser_noise_time)
    ] * 1e6

    if len(finite_laser_equilibria_um) > 0:
        centre_min_um = min(equilibrium_z_um, np.min(finite_laser_equilibria_um))
        centre_max_um = max(equilibrium_z_um, np.max(finite_laser_equilibria_um))
    else:
        centre_min_um = equilibrium_z_um
        centre_max_um = equilibrium_z_um

    if np.isfinite(z_rms_thermal) and z_rms_thermal > 0:
        distribution_padding_um = max(4 * z_rms_thermal * 1e6, 0.5)
    else:
        distribution_padding_um = force_comparison_z_half_width * 1e6

    distribution_min_um = centre_min_um - distribution_padding_um
    distribution_max_um = centre_max_um + distribution_padding_um

    if np.isclose(distribution_min_um, distribution_max_um):
        distribution_min_um -= 0.5
        distribution_max_um += 0.5

    distribution_positions_um = np.linspace(
        distribution_min_um,
        distribution_max_um,
        2000
    )
    distribution_positions_m = distribution_positions_um * 1e-6
    linear_density = linear_vertical_probability_density_um(
        distribution_positions_m
    )
    nonlinear_density = laser_power_weighted_uncoupled_z_density_um(
        distribution_positions_m
    )

    plt.figure(figsize=single_diagnostic_figsize)

    if linear_density is not None:
        plt.plot(
            distribution_positions_um,
            linear_density,
            color="tab:blue",
            linewidth=1.2,
            label="Linear prediction"
        )
    else:
        print(
            "Skipping linear expected distribution because z_rms_thermal is "
            "not finite and positive."
        )

    if nonlinear_density is not None:
        plt.plot(
            distribution_positions_um,
            nonlinear_density,
            color="tab:orange",
            linewidth=1.2,
            label="Nonlinear expected distribution"
        )
    else:
        print(
            "Skipping nonlinear expected distribution because the "
            "normalization failed."
        )

    plt.axvline(
        equilibrium_z_um,
        linestyle=":",
        color="black",
        label="Equilibrium"
    )
    plt.xlabel("z position (μm)")
    plt.ylabel("Probability density (μm^-1)")
    plt.title("Linear vs nonlinear expected z distribution")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot(name="z_expected_distribution_linear_vs_nonlinear")


if plot_uncoupled_vertical_histogram:
    z_uncoupled_values_um = z_uncoupled_nonlinear * 1e6
    finite_z_uncoupled_values_um = z_uncoupled_values_um[
        np.isfinite(z_uncoupled_values_um)
    ]

    if len(finite_z_uncoupled_values_um) > 0:
        equilibrium_z_um = z_eq * 1e6
        distribution_min_um = min(
            np.min(finite_z_uncoupled_values_um),
            equilibrium_z_um - 4 * z_rms_thermal * 1e6
            if np.isfinite(z_rms_thermal)
            else np.inf
        )
        distribution_max_um = max(
            np.max(finite_z_uncoupled_values_um),
            equilibrium_z_um + 4 * z_rms_thermal * 1e6
            if np.isfinite(z_rms_thermal)
            else -np.inf
        )

        if not np.isfinite(distribution_min_um) or not np.isfinite(distribution_max_um):
            distribution_min_um = np.min(finite_z_uncoupled_values_um)
            distribution_max_um = np.max(finite_z_uncoupled_values_um)

        if np.isclose(distribution_min_um, distribution_max_um):
            distribution_min_um -= 0.5
            distribution_max_um += 0.5

        distribution_positions_um = np.linspace(
            distribution_min_um,
            distribution_max_um,
            2000
        )
        expected_density = laser_power_weighted_uncoupled_z_density_um(
            distribution_positions_um * 1e-6
        )

        plt.figure(figsize=single_diagnostic_figsize)
        plt.hist(
            finite_z_uncoupled_values_um,
            bins=80,
            density=True,
            color=trajectory_color,
            alpha=0.75,
            label="Uncoupled trajectory"
        )

        if expected_density is not None:
            plt.plot(
                distribution_positions_um,
                expected_density,
                color="tab:blue",
                linewidth=1.2,
                label="Expected distribution"
            )
        else:
            print(
                "Skipping uncoupled z expected distribution because the "
                "normalization failed."
            )

        plt.axvline(
            equilibrium_z_um,
            linestyle=":",
            color="black",
            label="Equilibrium"
        )
        plt.xlabel("z position (μm)")
        plt.ylabel("Probability density (μm^-1)")
        plt.title("Uncoupled z position distribution")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        finish_plot(name="position_histogram_z_uncoupled_expected")
    else:
        print(
            "Skipping uncoupled z histogram because no finite uncoupled "
            "trajectory samples are available."
        )

# ****************************************************************************************************************************************************
# Laser-power noise diagnostic
# ****************************************************************************************************************************************************
plt.figure(figsize=single_diagnostic_figsize)
if laser_noise_model == "square":
    plt.step(t_plot, laser_power_time[plot_slice], where="post", label="laser power")
else:
    plt.plot(t_plot, laser_power_time[plot_slice], label="laser power")
plt.axhline(P_laser, linestyle="--", color="black", label="nominal power")
if laser_noise_model == "square":
    plt.axhline(
        P_laser * (1 + laser_noise_fraction),
        linestyle=":",
        color="tab:red",
        label="+1% nominal laser power",
    )
    plt.axhline(
        P_laser * (1 - laser_noise_fraction),
        linestyle=":",
        color="tab:red",
        label="−1% nominal laser power",
    )
plt.xlabel("Time / s")
plt.ylabel("Laser power / W")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
plt.grid()
plt.tight_layout()
finish_plot(name="laser_power_noise")


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
    plt.xlabel("Time (s)")
    plt.ylabel("Laser power (mW)")
    plt.title("PD feedback laser power")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot(name="pd_feedback_power")

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
    plt.xlabel("Time (s)")
    plt.ylabel("Measured z displacement (nm)")
    plt.title("PD feedback measurement signal")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    finish_plot(name="pd_feedback_measurement")

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
axes[0].set_ylabel("x (micrometres)")
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
axes[1].set_ylabel("y (micrometres)")
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
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("z (micrometres)")
axes[2].legend()
axes[2].grid()

fig.suptitle("BAOAB motion with and without laser-power noise")
plt.tight_layout()
finish_plot(name="xyz_laser_noise_comparison")


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
    axes[0].set_ylabel("x (micrometres)")
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
    axes[1].set_ylabel("y (micrometres)")
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
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("z (micrometres)")
    axes[2].legend()
    axes[2].grid()

    fig.suptitle("BAOAB motion with and without PD feedback")
    plt.tight_layout()
    finish_plot(name="xyz_pd_feedback_comparison")

# ****************************************************************************************************************************************************
# PD feedback vertical-axis-only comparison
# ****************************************************************************************************************************************************
if use_pd_feedback:
    plt.figure(figsize=single_diagnostic_figsize)
    plt.plot(
        t_plot,
        (z_baoab[plot_slice] - z_eq) * 1e6,
        label="without feedback"
    )
    plt.plot(
        t_plot,
        (z_baoab_feedback[plot_slice] - z_eq) * 1e6,
        linewidth=0.9,
        label="with PD feedback"
    )
    plt.axhline(0, linestyle=":", color="black", label="equilibrium")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (μm)")
    plt.title("Vertical (z) motion with and without PD feedback")
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.12)
    finish_plot(name="z_pd_feedback_comparison")

# ****************************************************************************************************************************************************
# Isolated laser-noise effect
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

axes[0].plot(t_plot, x_laser_noise_difference[plot_slice] * 1e9)
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x difference (nm)")
axes[0].grid()

axes[1].plot(t_plot, y_laser_noise_difference[plot_slice] * 1e9)
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_ylabel("y difference (nm)")
axes[1].grid()

axes[2].plot(t_plot, z_laser_noise_difference[plot_slice] * 1e9)
axes[2].axhline(0, linestyle=":", color="black")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("z difference (nm)")
axes[2].grid()

fig.suptitle("Isolated displacement effect from laser-power noise")
plt.tight_layout()
finish_plot(name="laser_noise_effect")

# ****************************************************************************************************************************************************
# Isolated Brownian-motion effect
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(3, 1, figsize=time_trace_figsize, sharex=True)

axes[0].plot(t_plot, x_brownian_difference[plot_slice] * 1e9)
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x difference (nm)")
axes[0].grid()

axes[1].plot(t_plot, y_brownian_difference[plot_slice] * 1e9)
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_ylabel("y difference (nm)")
axes[1].grid()

axes[2].plot(t_plot, z_brownian_difference[plot_slice] * 1e9)
axes[2].axhline(0, linestyle=":", color="black")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("z difference (nm)")
axes[2].grid()

fig.suptitle("Isolated displacement effect from Brownian motion")
plt.tight_layout()
finish_plot(name="brownian_effect")

#****************************************************************************************************************************************************
# Power spectral density
#****************************************************************************************************************************************************
sampling_frequency = 1 / dt_baoab
nyquist_frequency = sampling_frequency / 2
frequency_bin_size = 1 / (len(t_baoab) * dt_baoab)
psd_plot_max_frequency = nyquist_frequency if psd_plot_max_frequency_override is None else psd_plot_max_frequency_override

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

    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]
    psd = np.where(np.isfinite(psd) & (psd > 0), psd, psd_min_value)

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

    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]
    psd = np.where(np.isfinite(psd) & (psd > 0), psd, psd_min_value)

    return frequencies, psd, nperseg, fs / nperseg


def apply_psd_axes():
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(minimum_plot_frequency, psd_plot_max_frequency)

    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(psd_min_value, psd_max_value)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m^2 Hz^-1)")


def apply_psd_axes_without_nyquist():
    plt.xlim(minimum_plot_frequency, psd_plot_max_frequency)

    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(psd_min_value, psd_max_value)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m^2 Hz^-1)")


def apply_linear_psd_axes(max_frequency=120.0):
    plt.xlim(0.0, max_frequency)

    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(0.0, psd_max_value)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m^2 Hz^-1)")


def save_laser_power_psd_csv(
    csv_path,
    frequencies,
    power_psd,
    relative_power_psd,
):
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frequency_hz",
                "laser_power_psd_w2_per_hz",
                "relative_power_psd_1_per_hz",
            ]
        )
        writer.writerows(
            zip(frequencies, power_psd, relative_power_psd)
        )


def apply_laser_power_psd_axes():
    plt.xlim(minimum_plot_frequency, psd_plot_max_frequency)
    plt.ylim(bottom=laser_power_psd_min_value)
    plt.xlabel("Frequency / Hz")
    plt.ylabel(r"Laser power PSD / W$^2$ Hz$^{-1}$")


def add_resonance_harmonic_lines(max_frequency=120.0):
    resonances = [
        ("transverse", expected_fx, "tab:blue"),
        ("axial", expected_fz, "tab:orange"),
    ]

    for resonance_label, resonance_frequency, color in resonances:
        if not np.isfinite(resonance_frequency) or resonance_frequency <= 0:
            continue

        harmonic_count = int(np.floor(max_frequency / resonance_frequency))
        for harmonic_number in range(1, harmonic_count + 1):
            plt.axvline(
                harmonic_number * resonance_frequency,
                color=color,
                linestyle="--" if harmonic_number == 1 else ":",
                linewidth=0.8,
                alpha=0.75,
                label=(
                    f"{resonance_label} resonance"
                    if harmonic_number == 1
                    else (
                        f"{resonance_label} harmonics"
                        if harmonic_number == 2
                        else "_nolegend_"
                    )
                )
            )


def add_trap_resonance_lines():
    resonance_lines = [
        ("x trap resonance", expected_fx, "tab:blue"),
        ("z trap resonance", expected_fz, "tab:orange"),
    ]

    for label, frequency, color in resonance_lines:
        if not np.isfinite(frequency) or frequency <= 0:
            continue
        plt.axvline(
            frequency,
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.85,
            label=label,
        )


def quantise_position_resolution(position_m, resolution_m):
    if resolution_m <= 0.0:
        return np.array(position_m, copy=True)
    return np.round(position_m / resolution_m) * resolution_m


def finite_axis_limits_with_padding(values, centre=None, minimum_span=1e-6, padding_fraction=0.08):
    values = np.asarray(values)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        if centre is None or not np.isfinite(centre):
            centre = 0.0
        half_span = 0.5 * minimum_span
        return centre - half_span, centre + half_span, minimum_span

    lower = min(np.min(finite_values), centre) if centre is not None else np.min(finite_values)
    upper = max(np.max(finite_values), centre) if centre is not None else np.max(finite_values)
    span = upper - lower

    if not np.isfinite(span) or span < minimum_span:
        mid = 0.5 * (lower + upper)
        span = minimum_span
        lower = mid - 0.5 * span
        upper = mid + 0.5 * span
    else:
        padding = padding_fraction * span
        lower -= padding
        upper += padding
        span = upper - lower

    return lower, upper, span


baoab_x_freqs, baoab_x_psd = positive_psd(x_baoab - x_eq, t_baoab)
baoab_y_freqs, baoab_y_psd = positive_psd(y_baoab - y_eq, t_baoab)
baoab_z_freqs, baoab_z_psd = positive_psd(z_baoab - z_eq, t_baoab)

if len(baoab_x_freqs) == 0:
    raise RuntimeError(
        "PSD frequency array is empty. Check that the simulation has at least "
        "two time samples and a non-zero duration."
    )

minimum_resolvable_frequency = baoab_x_freqs[0]
minimum_plot_frequency = psd_min_frequency_factor * minimum_resolvable_frequency

# print("Minimum resolvable non-zero frequency =", minimum_resolvable_frequency, "Hz")
# print("Lower frequency shown on plot =", minimum_plot_frequency, "Hz")

laser_power_freqs, laser_power_psd = positive_psd(laser_power_time, t_baoab)
laser_relative_power_freqs, laser_relative_power_psd = positive_psd(
    laser_power_factor,
    t_baoab,
)

if not np.array_equal(laser_power_freqs, laser_relative_power_freqs):
    laser_relative_power_psd = np.interp(
        laser_power_freqs,
        laser_relative_power_freqs,
        laser_relative_power_psd,
    )

plt.figure(figsize=spectrum_figsize)
plt.loglog(
    laser_power_freqs,
    laser_power_psd,
    color="tab:blue",
    label="laser power PSD",
)
apply_laser_power_psd_axes()
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_laser_power")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))

if laser_noise_model == "square":
    axes[0].step(
        t_plot,
        laser_power_time[plot_slice],
        where="post",
        label="laser power",
    )
else:
    axes[0].plot(
        t_plot,
        laser_power_time[plot_slice],
        label="laser power",
    )
axes[0].axhline(P_laser, linestyle="--", color="black", label="nominal power")
if laser_noise_model == "square":
    axes[0].axhline(
        P_laser * (1 + laser_noise_fraction),
        linestyle=":",
        color="tab:red",
        label="+1% nominal laser power",
    )
    axes[0].axhline(
        P_laser * (1 - laser_noise_fraction),
        linestyle=":",
        color="tab:red",
        label="−1% nominal laser power",
    )
axes[0].set_xlabel("Time / s")
axes[0].set_ylabel("Laser power / W")
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=2, columnspacing=1.4, handletextpad=0.7)
axes[0].grid()

axes[1].loglog(
    laser_power_freqs,
    laser_power_psd,
    color="tab:blue",
    label="laser power PSD",
)
axes[1].set_xlim(minimum_plot_frequency, psd_plot_max_frequency)
axes[1].set_ylim(bottom=laser_power_psd_min_value)
axes[1].set_xlabel("Frequency / Hz")
axes[1].set_ylabel(r"Laser power PSD / W$^2$ Hz$^{-1}$")
axes[1].legend()
axes[1].grid(True, which="both")

axes[0].text(
    -0.18,
    1.03,
    "(a)",
    transform=axes[0].transAxes,
    fontsize=default_label_fontsize,
    ha="left",
    va="bottom",
)
axes[1].text(
    -0.18,
    1.03,
    "(b)",
    transform=axes[1].transAxes,
    fontsize=default_label_fontsize,
    ha="left",
    va="bottom",
)

fig.subplots_adjust(left=0.095, right=0.985, bottom=0.21, top=0.70, wspace=0.62)
finish_plot(name="laser_power_noise_and_psd", fig=fig)

laser_power_psd_csv_path = output_path("psd_laser_power.csv")
save_laser_power_psd_csv(
    laser_power_psd_csv_path,
    laser_power_freqs,
    laser_power_psd,
    laser_relative_power_psd,
)
log(f"Saved laser-power PSD CSV: {laser_power_psd_csv_path}", verbose=True)

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_freqs, baoab_x_psd, label="BAOAB x PSD")
apply_psd_axes()
plt.title("PSD of BAOAB x motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_x")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_y_freqs, baoab_y_psd, color="tab:green", label="BAOAB y PSD")
apply_psd_axes()
plt.title("PSD of BAOAB y motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_y")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="BAOAB z PSD")
apply_psd_axes()
plt.title("PSD of BAOAB z motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_z")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_freqs, baoab_x_psd, label="x PSD")
plt.loglog(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="z PSD")
apply_psd_axes()
plt.title("Raw x and z PSDs of BAOAB motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_xz_raw")

expected_resonant_frequencies = np.array([expected_fx, expected_fy, expected_fz])
positive_expected_resonant_frequencies = expected_resonant_frequencies[
    np.isfinite(expected_resonant_frequencies)
    & (expected_resonant_frequencies > 0)
]

if len(positive_expected_resonant_frequencies) > 0:
    harmonic_plot_min_frequency = 0.9 * 2 * np.min(positive_expected_resonant_frequencies)
    harmonic_plot_max_frequency = 1.1 * 4 * np.max(positive_expected_resonant_frequencies)

    plt.figure(figsize=spectrum_figsize)
    plt.semilogy(baoab_x_freqs, baoab_x_psd, label="x PSD")
    plt.semilogy(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="z PSD")

    for harmonic_number in range(2, 5):
        if np.isfinite(expected_fx) and expected_fx > 0:
            plt.axvline(
                harmonic_number * expected_fx,
                color="tab:blue",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
                label="x harmonics" if harmonic_number == 2 else "_nolegend_"
            )
        if np.isfinite(expected_fz) and expected_fz > 0:
            plt.axvline(
                harmonic_number * expected_fz,
                color="tab:orange",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
                label="z harmonics" if harmonic_number == 2 else "_nolegend_"
            )

    plt.xlim(harmonic_plot_min_frequency, harmonic_plot_max_frequency)
    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(psd_min_value, psd_max_value)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m^2 Hz^-1)")
    plt.title("x and z PSD harmonic zoom")
    plt.legend()
    plt.grid(True, which="both")
    finish_plot(name="psd_xz_harmonic_zoom")
else:
    print(
        "Skipping PSD harmonic zoom because no positive stable resonant "
        "frequencies are available."
    )

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

log(
    "Welch-averaged PSD segment duration = "
    f"{welch_average_segment_duration_seconds} s",
    verbose=True,
)
log(f"Welch-averaged PSD nperseg = {welch_nperseg}", verbose=True)
log(f"Welch-averaged PSD frequency bin size = {welch_bin_size} Hz", verbose=True)


def simulated_frequency_from_psd_peak(frequencies, psd, expected_frequency):
    if not np.isfinite(expected_frequency) or expected_frequency <= 0:
        return np.nan

    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd)
    peak_mask = np.isfinite(frequencies) & np.isfinite(psd) & (frequencies >= 0.5 * expected_frequency) & (frequencies <= 1.5 * expected_frequency)

    if not np.any(peak_mask):
        return np.nan

    local_frequencies = frequencies[peak_mask]
    local_psd = psd[peak_mask]
    return local_frequencies[np.argmax(local_psd)]


def percentage_difference(simulated_value, theoretical_value):
    if not (
        np.isfinite(simulated_value)
        and np.isfinite(theoretical_value)
        and theoretical_value != 0
    ):
        return np.nan

    return 100 * (simulated_value - theoretical_value) / theoretical_value


simulated_fx = simulated_frequency_from_psd_peak(
    baoab_x_welch_freqs,
    baoab_x_welch_psd,
    expected_fx,
)
simulated_fz = simulated_frequency_from_psd_peak(
    baoab_z_welch_freqs,
    baoab_z_welch_psd,
    expected_fz,
)

trap_frequency_rows = [
    (
        "x",
        expected_fx,
        simulated_fx,
        percentage_difference(simulated_fx, expected_fx),
        x_rms_thermal * 1e6,
        np.std(x_baoab - x_eq) * 1e6,
    ),
    (
        "z",
        expected_fz,
        simulated_fz,
        percentage_difference(simulated_fz, expected_fz),
        z_rms_thermal * 1e6,
        np.std(z_baoab - z_eq) * 1e6,
    ),
]

trap_frequency_rows = [
    row + (percentage_difference(row[5], row[4]),)
    for row in trap_frequency_rows
]

trap_frequency_csv_path = output_path("trap_frequency_comparison.csv")
with open(trap_frequency_csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "axis",
            "theoretical_trap_frequency_hz",
            "simulated_trap_frequency_hz",
            "trap_frequency_percentage_difference",
            "theoretical_rms_um",
            "simulated_rms_um",
            "rms_percentage_difference",
        ]
    )
    writer.writerows(trap_frequency_rows)

if verbose_output:
    print()
    print("Trap frequency and RMS comparison")
    print(
        "Axis | Theoretical f / Hz | Simulated f / Hz | f difference / % | "
        "Theoretical RMS / um | Simulated RMS / um | RMS difference / %"
    )
    print("-" * 125)
    for (
        axis_label,
        theoretical_frequency,
        simulated_frequency,
        frequency_difference,
        theoretical_rms_um,
        simulated_rms_um,
        rms_difference,
    ) in trap_frequency_rows:
        print(
            f"{axis_label:>4} | "
            f"{theoretical_frequency:18.2f} | "
            f"{simulated_frequency:16.2f} | "
            f"{frequency_difference:16.2f} | "
            f"{theoretical_rms_um:20.3f} | "
            f"{simulated_rms_um:18.3f} | "
            f"{rms_difference:18.2f}"
        )

log(f"Saved trap-frequency comparison CSV: {trap_frequency_csv_path}", verbose=True)

if include_uncoupled_nonlinear_psd:
    uncoupled_x_welch_freqs, uncoupled_x_welch_psd, _, _ = (
        positive_welch_averaged_psd(x_uncoupled_nonlinear - x_eq, t_baoab)
    )
    uncoupled_z_welch_freqs, uncoupled_z_welch_psd, _, _ = (
        positive_welch_averaged_psd(z_uncoupled_nonlinear - z_eq, t_baoab)
    )

    plt.figure(figsize=spectrum_figsize)
    plt.loglog(
        baoab_x_welch_freqs,
        baoab_x_welch_psd,
        label="Coupled x PSD"
    )
    plt.loglog(
        uncoupled_x_welch_freqs,
        uncoupled_x_welch_psd,
        linestyle="--",
        color="tab:orange",
        label="Uncoupled x PSD"
    )
    plt.axvline(
        expected_fx,
        color="tab:blue",
        linestyle=":",
        linewidth=0.8,
        alpha=0.8,
        label="x trap resonance"
    )
    apply_psd_axes_without_nyquist()
    plt.title("x coupled vs uncoupled PSD")
    plt.legend()
    plt.grid(True, which="both")
    finish_plot(name="psd_coupled_vs_uncoupled_x")

    plt.figure(figsize=spectrum_figsize)
    plt.loglog(
        baoab_z_welch_freqs,
        baoab_z_welch_psd,
        color="tab:blue",
        label="Coupled z PSD"
    )
    plt.loglog(
        uncoupled_z_welch_freqs,
        uncoupled_z_welch_psd,
        linestyle="--",
        color="tab:orange",
        label="Uncoupled z PSD"
    )
    plt.axvline(
        expected_fz,
        color="tab:blue",
        linestyle=":",
        linewidth=0.8,
        alpha=0.8,
        label="z trap resonance"
    )
    apply_psd_axes_without_nyquist()
    plt.title("z coupled vs uncoupled PSD")
    plt.legend()
    plt.grid(True, which="both")
    finish_plot(name="psd_coupled_vs_uncoupled_z")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_welch_freqs, baoab_x_welch_psd, label="BAOAB x Welch-averaged PSD")
apply_psd_axes()
plt.title("Welch-averaged PSD of BAOAB x motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_x_welch")

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
finish_plot(name="psd_y_welch")

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
finish_plot(name="psd_z_welch")

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_x_welch_freqs, baoab_x_welch_psd, label="x Welch-averaged PSD")
plt.loglog(
    baoab_z_welch_freqs,
    baoab_z_welch_psd,
    color="tab:orange",
    label="z Welch-averaged PSD"
)
add_trap_resonance_lines()
apply_psd_axes_without_nyquist()
plt.title("Welch-averaged x and z PSDs of BAOAB motion")
plt.legend()
plt.grid(True, which="both")
finish_plot(name="psd_xz_welch")


def add_panel_label(axis, label, x_offset=-0.16, y_offset=1.14):
    axis.text(
        x_offset,
        y_offset,
        label,
        transform=axis.transAxes,
        fontsize=default_label_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_xyz_summary_panel(
    figure,
    grid_slot,
    panel_label,
    time_values,
    x_values,
    y_values,
    z_values,
    x_limits=None,
):
    subgrid = grid_slot.subgridspec(3, 1, hspace=0.10)
    panel_axes = [
        figure.add_subplot(subgrid[0, 0]),
        figure.add_subplot(subgrid[1, 0]),
        figure.add_subplot(subgrid[2, 0]),
    ]

    trajectory_arrays = [
        (x_values, x_eq, "x / μm"),
        (y_values, y_eq, "y / μm"),
        (z_values, z_eq, "z / μm"),
    ]

    for axis_index, (axis, (positions, equilibrium_position, label)) in enumerate(
        zip(panel_axes, trajectory_arrays)
    ):
        axis.plot(
            time_values,
            positions,
            color=trajectory_color,
            linewidth=0.9,
            alpha=0.75,
            label="Trajectory",
        )
        axis.axhline(
            equilibrium_position * 1e6,
            linestyle=":",
            color="black",
            label="Equilibrium",
        )
        axis.set_ylabel(label)
        axis.grid(True)
        if x_limits is not None:
            axis.set_xlim(*x_limits)
        if axis_index < 2:
            axis.tick_params(labelbottom=False)

    panel_axes[-1].set_xlabel("Time / s")
    handles, labels = panel_axes[0].get_legend_handles_labels()
    panel_axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.42),
        ncol=2,
        borderaxespad=0.0,
    )
    add_panel_label(panel_axes[0], panel_label, y_offset=1.42)
    return panel_axes


def plot_histogram_summary_panel(
    axis,
    values_um,
    equilibrium_um,
    linear_rms_um,
    axis_label,
    panel_label,
    density=False,
    trajectory_label="Trajectory",
):
    finite_values_um = values_um[np.isfinite(values_um)]
    if len(finite_values_um) == 0:
        axis.text(0.5, 0.5, "No finite samples", transform=axis.transAxes, ha="center")
        add_panel_label(axis, panel_label)
        return

    counts, bin_edges, _ = axis.hist(
        finite_values_um,
        bins=80,
        density=density,
        color=trajectory_color,
        alpha=0.75,
        label=trajectory_label,
    )

    if np.isfinite(linear_rms_um) and linear_rms_um > 0:
        bin_width_um = bin_edges[1] - bin_edges[0]
        distribution_min_um = min(bin_edges[0], equilibrium_um - 4 * linear_rms_um)
        distribution_max_um = max(bin_edges[-1], equilibrium_um + 4 * linear_rms_um)
        distribution_positions_um = np.linspace(
            distribution_min_um,
            distribution_max_um,
            1000,
        )
        linear_pdf_per_um = np.exp(-0.5 * ((distribution_positions_um - equilibrium_um) / linear_rms_um) ** 2) / (linear_rms_um * np.sqrt(2 * np.pi))
        linear_values = linear_pdf_per_um if density else linear_pdf_per_um * len(finite_values_um) * bin_width_um
        axis.plot(
            distribution_positions_um,
            linear_values,
            color="tab:blue",
            linewidth=1.2,
            label="Linear prediction",
        )

    axis.axvline(equilibrium_um, linestyle=":", color="black", label="Equilibrium")
    axis.set_xlabel(axis_label)
    axis.set_ylabel("Count")
    axis.legend()
    axis.grid(True)
    add_panel_label(axis, panel_label)


if plot_aligned_summary_figure:
    summary_fig = plt.figure(figsize=(7.4, 8.8))
    summary_grid = summary_fig.add_gridspec(
        3,
        2,
        left=0.095,
        right=0.985,
        bottom=0.075,
        top=0.945,
        wspace=0.25,
        hspace=0.35,
    )

    plot_xyz_summary_panel(
        summary_fig,
        summary_grid[0, 0],
        "(a)",
        t_plot,
        x_baoab[plot_slice] * 1e6,
        y_baoab[plot_slice] * 1e6,
        z_baoab[plot_slice] * 1e6,
    )

    zoom_time_limit = min(0.2, t_baoab[-1])
    zoom_mask = t_baoab <= zoom_time_limit
    plot_xyz_summary_panel(
        summary_fig,
        summary_grid[0, 1],
        "(b)",
        t_baoab[zoom_mask],
        x_baoab[zoom_mask] * 1e6,
        y_baoab[zoom_mask] * 1e6,
        z_baoab[zoom_mask] * 1e6,
        x_limits=(0.0, zoom_time_limit),
    )

    psd_axis = summary_fig.add_subplot(summary_grid[1, 0])
    psd_axis.loglog(baoab_x_welch_freqs, baoab_x_welch_psd, label="x Welch-averaged PSD")
    psd_axis.loglog(
        baoab_z_welch_freqs,
        baoab_z_welch_psd,
        color="tab:orange",
        label="z Welch-averaged PSD",
    )
    for label, frequency, color in [
        ("x trap resonance", expected_fx, "tab:blue"),
        ("z trap resonance", expected_fz, "tab:orange"),
    ]:
        if np.isfinite(frequency) and frequency > 0:
            psd_axis.axvline(
                frequency,
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
                label=label,
            )
    psd_axis.set_xlim(minimum_plot_frequency, psd_plot_max_frequency)
    psd_axis.set_ylim(bottom=psd_min_value if psd_max_value is None else psd_min_value)
    if psd_max_value is not None:
        psd_axis.set_ylim(psd_min_value, psd_max_value)
    psd_axis.set_xlabel("Frequency / Hz")
    psd_axis.set_ylabel("PSD / m$^2$ Hz$^{-1}$")
    psd_axis.legend()
    psd_axis.grid(True, which="both")
    add_panel_label(psd_axis, "(c)")

    x_hist_axis = summary_fig.add_subplot(summary_grid[1, 1])
    plot_histogram_summary_panel(
        x_hist_axis,
        x_baoab * 1e6,
        x_eq * 1e6,
        x_rms_thermal * 1e6,
        "x position / μm",
        "(d)",
    )

    z_hist_axis = summary_fig.add_subplot(summary_grid[2, 0])
    plot_histogram_summary_panel(
        z_hist_axis,
        z_baoab * 1e6,
        z_eq * 1e6,
        z_rms_thermal * 1e6,
        "z position / μm",
        "(e)",
    )

    uncoupled_hist_axis = summary_fig.add_subplot(summary_grid[2, 1])
    uncoupled_values_um = z_uncoupled_nonlinear * 1e6 if z_uncoupled_nonlinear is not None else z_baoab * 1e6
    plot_histogram_summary_panel(
        uncoupled_hist_axis,
        uncoupled_values_um,
        z_eq * 1e6,
        z_rms_thermal * 1e6,
        "z position / μm",
        "(f)",
        density=True,
        trajectory_label="Uncoupled trajectory",
    )

    finish_plot(name="aligned_summary_figure", fig=summary_fig)

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
        figsize=poster_stacked_figsize,
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
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("z displacement (nm)")
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
    finish_plot(name="z_detector_resolution")

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
    finish_plot(name="psd_z_welch_detector_resolution")

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
finish_plot(name="psd_laser_noise_effect")

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
finish_plot(name="psd_laser_noise_effect_welch")


# ****************************************************************************************************************************************************
# Plot 3D trajectory
# ****************************************************************************************************************************************************
fig = plt.figure(figsize=poster_single_figsize)
ax = fig.add_subplot(111, projection="3d")
x_3d_um_all = x_baoab[plot_slice] * 1e6
y_3d_um_all = y_baoab[plot_slice] * 1e6
z_3d_um_all = z_baoab[plot_slice] * 1e6
finite_3d_mask = np.isfinite(x_3d_um_all) & np.isfinite(y_3d_um_all) & np.isfinite(z_3d_um_all)
x_3d_um = x_3d_um_all[finite_3d_mask]
y_3d_um = y_3d_um_all[finite_3d_mask]
z_3d_um = z_3d_um_all[finite_3d_mask]
if len(x_3d_um) > 0:
    ax.plot(
        x_3d_um,
        y_3d_um,
        z_3d_um,
        color=trajectory_color,
        linewidth=0.8,
        label="trajectory"
    )
else:
    print("Skipping 3D trajectory line because no finite trajectory samples are available.")
ax.scatter(
    [x_eq * 1e6],
    [y_eq * 1e6],
    [z_eq * 1e6],
    color="black",
    s=20,
    label="equilibrium"
)
ax.set_xlabel("x (micrometres)")
ax.set_ylabel("y (micrometres)")
ax.set_zlabel("z (micrometres)")
ax.set_title("3D trajectory")
ax.legend()
ax.grid(True)

x_lower_um, x_upper_um, x_span_um = finite_axis_limits_with_padding(
    x_3d_um,
    centre=x_eq * 1e6,
    minimum_span=1e-3
)
y_lower_um, y_upper_um, y_span_um = finite_axis_limits_with_padding(
    y_3d_um,
    centre=y_eq * 1e6,
    minimum_span=1e-3
)
z_lower_um, z_upper_um, z_span_um = finite_axis_limits_with_padding(
    z_3d_um,
    centre=z_eq * 1e6,
    minimum_span=1e-3
)

ax.set_xlim(x_lower_um, x_upper_um)
ax.set_ylim(y_lower_um, y_upper_um)
ax.set_zlim(z_lower_um, z_upper_um)
ax.set_box_aspect((x_span_um, y_span_um, z_span_um))

plt.tight_layout()
finish_plot(name="trajectory_3d", fig=fig)


# ****************************************************************************************************************************************************
# Plot 3D trajectory projections
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(1, 3, figsize=trajectory_projection_figsize)

axes[0].plot(
    x_3d_um,
    y_3d_um,
    linewidth=0.8
)
axes[0].scatter([x_eq * 1e6], [y_eq * 1e6], color="black", s=20)
axes[0].set_xlabel("x (micrometres)")
axes[0].set_ylabel("y (micrometres)")
axes[0].set_title("x-y projection")
axes[0].axis("equal")
axes[0].grid()

axes[1].plot(
    x_3d_um,
    z_3d_um,
    linewidth=0.8
)
axes[1].scatter([x_eq * 1e6], [z_eq * 1e6], color="black", s=20)
axes[1].set_xlabel("x (micrometres)")
axes[1].set_ylabel("z (micrometres)")
axes[1].set_title("x-z projection")
axes[1].axis("equal")
axes[1].grid()

axes[2].plot(
    y_3d_um,
    z_3d_um,
    linewidth=0.8
)
axes[2].scatter([y_eq * 1e6], [z_eq * 1e6], color="black", s=20)
axes[2].set_xlabel("y (micrometres)")
axes[2].set_ylabel("z (micrometres)")
axes[2].set_title("y-z projection")
axes[2].axis("equal")
axes[2].grid()

fig.suptitle("3D trajectory projections")
plt.tight_layout()
finish_plot(name="trajectory_projections")

total_runtime = perf_counter() - script_start_time
print("Simulation complete.")
print(f"Plots and CSV outputs saved to: {save_path}")
print(
    "Equilibrium position: "
    f"x={x_eq * 1e6:.3g} um, "
    f"y={y_eq * 1e6:.3g} um, "
    f"z={z_eq * 1e6:.3g} um"
)
print(
    "Expected trap frequencies: "
    f"fx={expected_fx:.3g} Hz, "
    f"fy={expected_fy:.3g} Hz, "
    f"fz={expected_fz:.3g} Hz"
)
print(f"Total runtime: {total_runtime:.3g} s")

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
# plt.xlabel("x (micrometres)")
# plt.ylabel("Fx (mg)")
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
# plt.xlabel("z (micrometres)")
# plt.ylabel("Force (mg)")
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

# plt.xlabel("x (micrometres)")
# plt.ylabel("z (micrometres)")
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

# # plt.xlabel("z (micrometres)")
# # plt.ylabel("Force (N)")
# # plt.title("On-axis vertical force check")
# # plt.legend()
# # plt.grid()
# # plt.show()
