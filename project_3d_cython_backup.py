import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, least_squares
from scipy.signal import welch
from time import perf_counter

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
radius = 10e-6          # m
density = 2200         # kg/m^3
n_particle = 1.46      # particle refractive index
n_medium = 1.00027     # surrounding medium refractive index, air

# Gas properties
p = 1         # pressure, Pa
T = 300           # temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
kB = 1.38e-23
d_air = 3.7e-10

# Laser / force parameters
w0 = 6e-6        # beam waist, m
wavelength = 532e-9
M2 = 1.2
use_m2_rayleigh_range = False
zR_manual = 100e-6
P_laser = 0.041             # W, example laser power
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
t_end = 120.0
dt_baoab = 1 / 200000
brownian_seed = 92635
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
max_plot_points = 200000
psd_plot_max_frequency_override = None
psd_min_frequency_factor = 0.95
psd_min_value = 1e-30
psd_max_value = None
psd_segment_duration_seconds = 30.0
psd_segment_samples = None
psd_overlap_fraction = 0.5
psd_overlap_samples = None
include_z_detector_resolution_psd = True
detector_position_resolution_nm = 100.0
fit_z_detector_resolution_psd = True
calculate_surface_temperature_from_psd = True
surface_temperature_fit_source = "both"  # "true", "measured", or "both"
surface_temperature_impinging_gas_temperature = None  # None uses gas temperature T
surface_temperature_accommodation_alpha = 0.777
use_physical_psd_c_initial_guess = True
psd_fit_frequency_min_hz = None
psd_fit_frequency_max_hz = None
psd_fit_peak_band_half_width_hz = 1.0
psd_fit_peak_weight = 25.0
psd_fit_max_points = None
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

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)

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
    (kB * T / m) * (1 - baoab_damping_factor**2)
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
    brownian_normals_z
):
    """
    Use the compiled Cython BAOAB loop when available.

    If the Cython extension has not been built yet, this falls back to the
    original Python implementation so the script remains runnable.
    """
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


brownian_normals_x = rng.normal(size=len(t_baoab) - 1)
brownian_normals_y = rng.normal(size=len(t_baoab) - 1)
brownian_normals_z = rng.normal(size=len(t_baoab) - 1)
zero_brownian_normals = np.zeros(len(t_baoab) - 1)
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
plt.show()

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
plt.show()


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
    plt.show()

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
    plt.show()

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
plt.show()


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
    plt.show()

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
plt.show()

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
plt.show()

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
    gamma_0_rad_per_s = gamma_baoab
    alpha = surface_temperature_accommodation_alpha

    if temperature_impinging <= 0.0 or gamma_0_rad_per_s <= 0.0 or alpha <= 0.0:
        return None

    gamma_cm_rad_per_s = quantities["gamma_cm_rad_per_s"]
    temperature_cm = quantities["temperature_cm"]
    temperature_ratio_term = (
        (temperature_cm / temperature_impinging)
        * (gamma_cm_rad_per_s / gamma_0_rad_per_s)
        * (1.0 + np.pi / 8.0)
    )
    temperature_emerging = temperature_impinging * (8.0 / np.pi) * (
        temperature_ratio_term - 1.0
    )
    temperature_surface = temperature_impinging + (
        temperature_emerging - temperature_impinging
    ) / alpha

    quantities.update(
        {
            "temperature_impinging": temperature_impinging,
            "gamma_0_rad_per_s": gamma_0_rad_per_s,
            "temperature_emerging": temperature_emerging,
            "temperature_surface": temperature_surface,
            "thermal_accommodation_alpha": alpha,
        }
    )
    return quantities


def print_surface_temperature_from_psd_fit(label, fit_parameters):
    result = surface_temperature_from_psd_fit(fit_parameters)
    if result is None:
        print(f"Could not calculate surface temperature from {label}.")
        return

    print(f"\nSurface-temperature estimate from {label}:")
    print("  f0 =", result["f0_hz"], "Hz")
    print("  fitted Gamma_CM =", result["gamma_cm_rad_per_s"], "s^-1")
    print("  cold Gamma_0 =", result["gamma_0_rad_per_s"], "s^-1")
    print("  PSD RMS displacement =", result["rms_nm"], "nm")
    print("  T_CM =", result["temperature_cm"], "K")
    print("  T_imp =", result["temperature_impinging"], "K")
    print("  T_em =", result["temperature_emerging"], "K")
    print("  alpha =", result["thermal_accommodation_alpha"])
    print("  T_surface =", result["temperature_surface"], "K")

    if result["temperature_emerging"] <= 0.0 or result["temperature_surface"] <= 0.0:
        print(
            "  Warning: this fit gives a non-physical gas/surface temperature; "
            "check the fit range, damping model, and PSD calibration."
        )


def percentage_error(measured, reference):
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return 100.0 * (measured - reference) / reference


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
plt.show()

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_y_freqs, baoab_y_psd, color="tab:green", label="BAOAB y PSD")
apply_psd_axes()
plt.title("PSD of BAOAB y motion")
plt.legend()
plt.grid(True, which="both")
plt.show()

plt.figure(figsize=spectrum_figsize)
plt.loglog(baoab_z_freqs, baoab_z_psd, color="tab:orange", label="BAOAB z Welch-averaged PSD")
apply_psd_axes()
plt.title("PSD of BAOAB z motion")
plt.legend()
plt.grid(True, which="both")
plt.show()

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

    if fit_z_detector_resolution_psd:
        true_z_peak_frequency = baoab_z_freqs[int(np.argmax(baoab_z_psd))]
        active_z_fit_frequency_min_hz = (
            psd_fit_frequency_min_hz
            if psd_fit_frequency_min_hz is not None
            else max(
                baoab_z_freqs[0],
                true_z_peak_frequency - psd_fit_peak_band_half_width_hz,
            )
        )
        active_z_fit_frequency_max_hz = (
            psd_fit_frequency_max_hz
            if psd_fit_frequency_max_hz is not None
            else min(
                psd_plot_max_frequency,
                true_z_peak_frequency + psd_fit_peak_band_half_width_hz,
            )
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

        if calculate_surface_temperature_from_psd:
            if surface_temperature_fit_source in ("true", "both"):
                print_surface_temperature_from_psd_fit(
                    "true z PSD fit",
                    true_z_fit_parameters,
                )
            if surface_temperature_fit_source in ("measured", "both"):
                print_surface_temperature_from_psd_fit(
                    f"{detector_position_resolution_nm:g} nm z PSD fit",
                    measured_z_fit_parameters,
                )
            if surface_temperature_fit_source not in ("true", "measured", "both"):
                print(
                    "surface_temperature_fit_source must be 'true', 'measured', "
                    "or 'both' to calculate surface temperature."
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
            (baoab_z_freqs >= active_z_fit_frequency_min_hz)
            & (baoab_z_freqs <= active_z_fit_frequency_max_hz)
        )
        measured_fit_plot_mask = (
            (measured_z_freqs >= active_z_fit_frequency_min_hz)
            & (measured_z_freqs <= active_z_fit_frequency_max_hz)
        )
        axes[1].loglog(
            baoab_z_freqs[true_fit_plot_mask],
            true_z_fit_curve[true_fit_plot_mask],
            color="tab:orange",
            linewidth=2.0,
            label="true z fitted curve over fit range"
        )
        axes[1].loglog(
            measured_z_freqs[measured_fit_plot_mask],
            measured_z_fit_curve[measured_fit_plot_mask],
            color="tab:green",
            linewidth=2.0,
            label=f"{detector_position_resolution_nm:g} nm fitted curve over fit range"
        )
        axes[1].axvline(
            active_z_fit_frequency_min_hz,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="fit range"
        )
        axes[1].axvline(
            active_z_fit_frequency_max_hz,
            color="black",
            linestyle="--",
            linewidth=0.9
        )
    apply_psd_axes()
    axes[1].set_title("Welch-averaged PSD of BAOAB z motion with detector resolution")
    axes[1].legend()
    axes[1].grid(True, which="both")

    fig.tight_layout()
    plt.show()

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
plt.show()


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
plt.show()

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
plt.show()

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
