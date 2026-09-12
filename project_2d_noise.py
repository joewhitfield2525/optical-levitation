import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import welch
from time import perf_counter

script_start_time = perf_counter()

# ****************************************************************************************************************************************************
# Constants
# ****************************************************************************************************************************************************
g = 9.81
c_light = 299792458

# ****************************************************************************************************************************************************
# Particle properties
# ****************************************************************************************************************************************************
radius = 5e-6          # m
density = 1000         # kg/m^3
n_particle = 1.46      # particle refractive index
n_medium = 1.00027     # surrounding medium refractive index, air

volume = (4/3) * np.pi * radius**3
m = density * volume

print("Particle mass =", m, "kg")
print("Weight mg =", m*g, "N")

# ****************************************************************************************************************************************************
# Gas properties
# ****************************************************************************************************************************************************
p = 3         # pressure, Pa
T = 293           # temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
kB = 1.38e-23
d_air = 3.7e-10

rho_g = p * M_air / (R * T)
lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

print("Gas density =", rho_g, "kg/m^3")
print("Mean free path =", lambda_mfp, "m")
print("Knudsen number =", Kn)

# ****************************************************************************************************************************************************
# Laser / force parameters
# ****************************************************************************************************************************************************
w0 = 20e-6        # beam waist, m
wavelength = 1064e-9
M2 = 1.2

# If this is False, the code keeps the manually chosen zR below.
# If this is True, zR is calculated from the Gaussian beam-quality relation:
#
#     zR = pi w0^2 / (M2 wavelength)
#
# Increasing M2 then shortens zR and increases the beam divergence.
use_m2_rayleigh_range = False
zR_manual = 100e-6
zR_m2 = np.pi * w0**2 / (M2 * wavelength)
zR = zR_m2 if use_m2_rayleigh_range else zR_manual

P_laser = 0.08             # W, example laser power
I0 = 2 * P_laser / (np.pi * w0**2)   # Gaussian peak intensity, W/m^2
laser_noise_fraction = 0.01
laser_noise_step_duration = 0.01     # s

# If True, the time integration uses a precomputed/interpolated force table
# instead of recalculating the full ray-optics force at every timestep.
use_force_lookup_table = True
force_lookup_grid_points_x = 101
force_lookup_grid_points_z = 201

print("Laser power =", P_laser, "W")
print("Wavelength =", wavelength, "m")
print("Beam quality M2 =", M2)
print("Using M2-derived Rayleigh range =", use_m2_rayleigh_range)
print("Manual Rayleigh range zR_manual =", zR_manual, "m")
print("M2-derived Rayleigh range zR_m2 =", zR_m2, "m")
print("Rayleigh range used zR =", zR, "m")
print("Peak intensity I0 =", I0, "W/m^2")
print("Paraxial divergence half-angle w0/zR =", w0 / zR, "rad")


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
    if abs(z) < 1e-30:
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

ray_grid_points = 81
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
fresnel_R = 0.5 * (Rs + Rp)
fresnel_T = 1 - fresnel_R

denom = 1 + fresnel_R**2 + 2 * fresnel_R * np.cos(2 * theta_r)

Q_s = (
    1
    + fresnel_R * np.cos(2 * theta_i)
    - (
        fresnel_T**2
        * (np.cos(2 * theta_i - 2 * theta_r) + fresnel_R * np.cos(2 * theta_i))
        / denom
    )
)

Q_g = (
    fresnel_R * np.sin(2 * theta_i)
    - (
        fresnel_T**2
        * (np.sin(2 * theta_i - 2 * theta_r) + fresnel_R * np.sin(2 * theta_i))
        / denom
    )
)

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
print("Ray-optics force at focus Fx =", Fx_focus, "N")
print("Ray-optics force at focus Fz =", Fz_focus, "N")
print("Ray-optics Fz at focus / mg =", Fz_focus / (m*g))


# ****************************************************************************************************************************************************
# Photophoretic force parameters
# ****************************************************************************************************************************************************
# -----------------------------
# Photophoretic force parameters
# -----------------------------
k_particle = 1.4          # W/(m K), approximate silica
alpha_acc = 1.0           # thermal accommodation coefficient
kappa_t = 1.14            # thermal creep coefficient
absorption_fraction = 1e-3

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
force_lookup_x_values = None
force_lookup_z_values = None
force_lookup_Fx_interpolator = None
force_lookup_Fz_interpolator = None


def build_force_lookup_table(x_min, x_max, z_min, z_max):
    """
    Precompute the optical + photophoretic force on an x-z grid.

    The table is built at nominal laser power. This is valid because both the
    ray-optics force and the current photophoretic force scale linearly with
    laser power in this model, so laser noise is applied by multiplying the
    interpolated nominal force by power_factor.
    """
    global force_lookup_ready
    global force_lookup_x_values
    global force_lookup_z_values
    global force_lookup_Fx_interpolator
    global force_lookup_Fz_interpolator

    force_lookup_start_time = perf_counter()

    force_lookup_x_values = np.linspace(
        x_min,
        x_max,
        force_lookup_grid_points_x
    )
    force_lookup_z_values = np.linspace(
        z_min,
        z_max,
        force_lookup_grid_points_z
    )

    Fx_table = np.zeros(
        (force_lookup_grid_points_x, force_lookup_grid_points_z)
    )
    Fz_table = np.zeros_like(Fx_table)

    for ix, x_value in enumerate(force_lookup_x_values):
        for iz, z_value in enumerate(force_lookup_z_values):
            Fx_table[ix, iz], Fz_table[ix, iz] = F_optical_2d_direct(
                x_value,
                z_value,
                power_factor=1.0
            )

    force_lookup_Fx_interpolator = RegularGridInterpolator(
        (force_lookup_x_values, force_lookup_z_values),
        Fx_table,
        bounds_error=True
    )
    force_lookup_Fz_interpolator = RegularGridInterpolator(
        (force_lookup_x_values, force_lookup_z_values),
        Fz_table,
        bounds_error=True
    )

    force_lookup_ready = True

    print("Force lookup table enabled =", use_force_lookup_table)
    print("Force lookup grid x points =", force_lookup_grid_points_x)
    print("Force lookup grid z points =", force_lookup_grid_points_z)
    print(
        "Force lookup x range =",
        force_lookup_x_values[0] * 1e6,
        "to",
        force_lookup_x_values[-1] * 1e6,
        "micrometres"
    )
    print(
        "Force lookup z range =",
        force_lookup_z_values[0] * 1e6,
        "to",
        force_lookup_z_values[-1] * 1e6,
        "micrometres"
    )
    print(
        "Force lookup table build runtime =",
        perf_counter() - force_lookup_start_time,
        "s"
    )


def F_optical_2d_lookup(x, z, power_factor=1.0):
    x_arr, z_arr, power_factor_arr = np.broadcast_arrays(x, z, power_factor)

    x_inside = (
        np.min(x_arr) >= force_lookup_x_values[0]
        and np.max(x_arr) <= force_lookup_x_values[-1]
    )
    z_inside = (
        np.min(z_arr) >= force_lookup_z_values[0]
        and np.max(z_arr) <= force_lookup_z_values[-1]
    )

    if not (x_inside and z_inside):
        return F_optical_2d_direct(x, z, power_factor)

    points = np.column_stack((x_arr.ravel(), z_arr.ravel()))

    Fx_nominal = force_lookup_Fx_interpolator(points).reshape(x_arr.shape)
    Fz_nominal = force_lookup_Fz_interpolator(points).reshape(z_arr.shape)

    Fx = power_factor_arr * Fx_nominal
    Fz = power_factor_arr * Fz_nominal

    if Fx.shape == ():
        return float(Fx), float(Fz)

    return Fx, Fz


def F_optical_2d(x, z, power_factor=1.0):
    if use_force_lookup_table and force_lookup_ready:
        return F_optical_2d_lookup(x, z, power_factor)

    return F_optical_2d_direct(x, z, power_factor)


def Fz_net_on_axis(z):
    
    _, Fz = F_optical_2d(0.0, z)
    return Fz - m*g


# ****************************************************************************************************************************************************
# Find on-axis equilibrium
# ****************************************************************************************************************************************************
z_min = -10e-3
z_max = 10e-3

z_scan = np.linspace(z_min, z_max, 3000)
F_scan = Fz_net_on_axis(z_scan)

roots = []

for i in range(len(z_scan) - 1):
    if F_scan[i] * F_scan[i + 1] < 0:
        root = brentq(Fz_net_on_axis,z_scan[i],z_scan[i + 1],xtol=1e-15,rtol=1e-15)
        roots.append(root)

print("On-axis equilibrium positions / micrometres:")
for root in roots:
    print(root * 1e6)


def numerical_derivative_1d(func, z, h=1e-9):
    return (func(z + h) - func(z - h)) / (2*h)


stable_roots = []

for root in roots:
    slope = numerical_derivative_1d(Fz_net_on_axis, root)
    if slope < 0:
        stable_roots.append(root)

if len(stable_roots) == 0:
    raise ValueError("No stable on-axis equilibrium found.")

x_eq = 0.0
z_eq = stable_roots[0]

print("Chosen equilibrium x =", x_eq * 1e6, "micrometres")
print("Chosen equilibrium z =", z_eq * 1e6, "micrometres")
print("Fz_net_on_axis(z_eq) =", Fz_net_on_axis(z_eq), "N")

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
kz = -numerical_derivative_1d(Fz_net_at_z, z_eq)

if kx <= 0 or kz <= 0:
    raise ValueError(
        "The selected equilibrium is not stable. "
        f"kx={kx:.3e} N/m, kz={kz:.3e} N/m. "
        "Check the ray-optics force signs, laser power, search range, or chosen root."
    )

omega_x = np.sqrt(kx / m)
omega_z = np.sqrt(kz / m)

print("kx =", kx, "N/m")
print("kz =", kz, "N/m")
print("fx =", omega_x / (2*np.pi), "Hz")
print("fz =", omega_z / (2*np.pi), "Hz")

x_rms_thermal = np.sqrt(kB * T / kx)
z_rms_thermal = np.sqrt(kB * T / kz)

print("Expected x thermal RMS =", x_rms_thermal * 1e6, "micrometres")
print("Expected z thermal RMS =", z_rms_thermal * 1e6, "micrometres")

# ****************************************************************************************************************************************************
# Harmonic approximation range in x and z
# ****************************************************************************************************************************************************
# Compare the full nonlinear restoring force with the local harmonic model:
#
#     Fx ~= -kx (x - x_eq)
#     Fz_net ~= -kz (z - z_eq)
#
# The percentage error is normalised by max(|F|, mg), matching the 1D
# validation plot and avoiding artificial spikes at the force-balance root.
def centred_valid_half_width(displacements, percent_error, threshold_percent):
    # Return the largest symmetric half-width around zero displacement for
    # which every sampled point remains below the requested threshold.
    centre_index = int(np.argmin(np.abs(displacements)))

    if percent_error[centre_index] > threshold_percent:
        return 0.0

    left_index = centre_index
    while left_index > 0 and percent_error[left_index - 1] <= threshold_percent:
        left_index -= 1

    right_index = centre_index
    while right_index < len(displacements) - 1 and percent_error[right_index + 1] <= threshold_percent:
        right_index += 1

    return min(abs(displacements[left_index]), abs(displacements[right_index]))


x_harmonic_half_width = max(80e-6, 12 * x_rms_thermal)
z_harmonic_half_width = max(150e-6, 12 * z_rms_thermal)

x_displacements = np.linspace(-x_harmonic_half_width, x_harmonic_half_width, 1201)
z_displacements = np.linspace(-z_harmonic_half_width, z_harmonic_half_width, 1201)

Fx_nonlinear_values = np.array([
    F_optical_2d_direct(x_eq + x_i, z_eq)[0]
    for x_i in x_displacements
])
Fx_harmonic_values = -kx * x_displacements

Fz_net_nonlinear_values = np.array([
    F_optical_2d_direct(x_eq, z_eq + z_i)[1] - m * g
    for z_i in z_displacements
])
Fz_harmonic_values = -kz * z_displacements

Fx_percent_error = 100 * np.abs(Fx_nonlinear_values - Fx_harmonic_values) / np.maximum(np.abs(Fx_nonlinear_values), m * g)
Fz_percent_error = 100 * np.abs(Fz_net_nonlinear_values - Fz_harmonic_values) / np.maximum(np.abs(Fz_net_nonlinear_values), m * g)

error_thresholds = (1, 5, 10)
x_valid_ranges = {
    threshold: centred_valid_half_width(x_displacements, Fx_percent_error, threshold)
    for threshold in error_thresholds
}
z_valid_ranges = {
    threshold: centred_valid_half_width(z_displacements, Fz_percent_error, threshold)
    for threshold in error_thresholds
}

print("Harmonic approximation valid ranges:")
print("| Direction | 1% valid range | 5% valid range | 10% valid range | Theoretical Brownian RMS displacement |")
print("|---|---:|---:|---:|---:|")
print(
    f"| x | +/- {x_valid_ranges[1] * 1e6:.2f} um | "
    f"+/- {x_valid_ranges[5] * 1e6:.2f} um | "
    f"+/- {x_valid_ranges[10] * 1e6:.2f} um | "
    f"{x_rms_thermal * 1e6:.3f} um |"
)
print(
    f"| z | +/- {z_valid_ranges[1] * 1e6:.2f} um | "
    f"+/- {z_valid_ranges[5] * 1e6:.2f} um | "
    f"+/- {z_valid_ranges[10] * 1e6:.2f} um | "
    f"{z_rms_thermal * 1e6:.3f} um |"
)

fig, axes = plt.subplots(2, 1, figsize=(8, 7))
axes[0].plot(x_displacements * 1e6, Fx_nonlinear_values, label="Nonlinear Fx")
axes[0].plot(x_displacements * 1e6, Fx_harmonic_values, "--", label="-kx x")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axvline(0, color="black", linestyle=":", linewidth=0.9, label="equilibrium")
axes[0].set_xlabel("Horizontal displacement, x - x_eq / micrometres")
axes[0].set_ylabel("Fx / N")
axes[0].set_title("Transverse nonlinear force compared with harmonic approximation")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(z_displacements * 1e6, Fz_net_nonlinear_values, label="Nonlinear Fz_net")
axes[1].plot(z_displacements * 1e6, Fz_harmonic_values, "--", label="-kz dz")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].axvline(0, color="black", linestyle=":", linewidth=0.9, label="equilibrium")
axes[1].set_xlabel("Axial displacement, z - z_eq / micrometres")
axes[1].set_ylabel("Fz_net / N")
axes[1].set_title("Axial nonlinear force compared with harmonic approximation")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharey=True)

axes[0].plot(x_displacements * 1e6, Fx_percent_error, color="tab:purple", label="Fx error")
# for threshold, colour in [(1, "0.4"), (5, "tab:orange"), (10, "tab:red")]:
#     axes[0].axhline(threshold, color=colour, linestyle="-", linewidth=0.9, label=f"{threshold}%")
axes[0].axvline(0, color="black", linestyle=":", linewidth=0.9)
axes[0].axvspan(
    -x_valid_ranges[5] * 1e6,
    x_valid_ranges[5] * 1e6,
    color="tab:green",
    alpha=0.12,
    label=f"<=5% for |x| <= {x_valid_ranges[5] * 1e6:.1f} um"
)
axes[0].set_xlabel("Horizontal displacement, x - x_eq / micrometres")
axes[0].set_ylabel("Force error / %")
axes[0].set_title("Transverse harmonic-approximation error")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid(True, which="both")

axes[1].plot(z_displacements * 1e6, Fz_percent_error, color="tab:purple", label="Fz_net error")
# for threshold, colour in [(1, "0.4"), (5, "tab:orange"), (10, "tab:red")]:
#     axes[1].axhline(threshold, color=colour, linestyle="-", linewidth=0.9, label=f"{threshold}%")
axes[1].axvline(0, color="black", linestyle=":", linewidth=0.9)
axes[1].axvspan(
    -z_valid_ranges[5] * 1e6,
    z_valid_ranges[5] * 1e6,
    color="tab:green",
    alpha=0.12,
    label=f"<=5% for |dz| <= {z_valid_ranges[5] * 1e6:.1f} um"
)
axes[1].set_xlabel("Axial displacement, z - z_eq / micrometres")
axes[1].set_ylabel("Force error / %")
axes[1].set_title("Axial harmonic-approximation error")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(True, which="both")

fig.tight_layout()
plt.show()

# ****************************************************************************************************************************************************
# Damping
# ****************************************************************************************************************************************************
drag_model = "auto"
# Choose one of:
#   "stokes"       continuum Stokes drag, best for Kn << 1
#   "cunningham"  Stokes drag with slip correction, useful in transition regime
#   "epstein"     free-molecular Epstein drag, best for Kn >> 1
#   "auto"        pick a model from Knudsen number


def gas_density(pressure):
    return pressure * M_air / (R * T)


def mean_free_path(pressure):
    return kB * T / (np.sqrt(2) * np.pi * d_air**2 * pressure)


def knudsen_number(pressure):
    return mean_free_path(pressure) / radius


def cunningham_correction(Kn):
    A = 1.257
    B = 0.4
    C = 1.1

    if Kn <= 0:
        return 1.0

    return 1 + Kn * (A + B * np.exp(-C / Kn))


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
    alpha_E = 1.0
    rho = gas_density(pressure)
    c_bar = mean_thermal_speed()
    accommodation_factor = 1 + np.pi * alpha_E / 8

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

print("Drag model requested =", drag_model)
print("Drag model used =", drag_model_used)
print("Stokes damping b =", b_stokes, "kg/s")
print("Cunningham-corrected Stokes damping b =", b_cunningham, "kg/s")
print("Epstein damping b =", b_epstein, "kg/s")
print("Selected pressure-dependent damping b =", b, "kg/s")
print("Damping ratio x =", b / (2 * np.sqrt(m * kx)))
print("Damping ratio z =", b / (2 * np.sqrt(m * kz)))

# ****************************************************************************************************************************************************
# Initial conditions
# ****************************************************************************************************************************************************
x_displacement = 6e-6
z_displacement = 30e-6

x0 = x_eq + x_displacement
z0 = z_eq + z_displacement
vx0 = 0.0
vz0 = 0.0

Fx_initial, Fz_initial = F_optical_2d(x0, z0)
ax_initial = Fx_initial / m
az_initial = (Fz_initial - m*g) / m

print("Initial x displacement =", x_displacement * 1e6, "micrometres")
print("Initial z displacement =", z_displacement * 1e6, "micrometres")
print("Initial Fx / mg =", Fx_initial / (m*g))
print("Initial Fz_net / mg =", (Fz_initial - m*g) / (m*g))
print("Initial ax =", ax_initial, "m/s^2")
print("Initial az =", az_initial, "m/s^2")

if use_force_lookup_table:
    # Keep the lookup region local to the trap. If the particle leaves this
    # region, F_optical_2d automatically falls back to the direct ray sum.
    force_lookup_x_half_width = max(
        80e-6,
        4 * abs(x_displacement),
        8 * x_rms_thermal
    )
    force_lookup_z_half_width = max(
        150e-6,
        4 * abs(z_displacement),
        8 * z_rms_thermal
    )

    build_force_lookup_table(
        x_eq - force_lookup_x_half_width,
        x_eq + force_lookup_x_half_width,
        z_eq - force_lookup_z_half_width,
        z_eq + force_lookup_z_half_width
    )

    Fx_initial_direct, Fz_initial_direct = F_optical_2d_direct(x0, z0)
    Fx_initial_lookup, Fz_initial_lookup = F_optical_2d_lookup(x0, z0)

    print(
        "Lookup check at initial position: |Fx error| / mg =",
        abs(Fx_initial_lookup - Fx_initial_direct) / (m*g)
    )
    print(
        "Lookup check at initial position: |Fz error| / mg =",
        abs(Fz_initial_lookup - Fz_initial_direct) / (m*g)
    )
else:
    print("Force lookup table enabled = False")

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_start = 0
t_end = 2.0
dt_baoab = 1 / 200000
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)

print("BAOAB requested timestep =", dt_baoab, "s")
print("BAOAB sampling frequency =", 1 / dt_baoab, "Hz")
print("BAOAB number of samples =", len(t_baoab))

# ****************************************************************************************************************************************************
# BAOAB solution with Brownian motion and stepwise laser-power noise
# ****************************************************************************************************************************************************
rng = np.random.default_rng(seed=4)
laser_rng = np.random.default_rng(seed=12)

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
                xtol=1e-15,
                rtol=1e-15
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

    instantaneous_z_equilibrium_by_power_factor[power_factor_value] = stable_roots_power[0]

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

print("BAOAB timestep =", dt_baoab, "s")
print("BAOAB damping factor =", baoab_damping_factor)
print("BAOAB thermal velocity kick scale =", baoab_thermal_velocity_scale, "m/s")
print("Laser noise fraction =", laser_noise_fraction)
print("Requested laser noise step duration =", laser_noise_step_duration, "s")
print("Actual laser noise step duration =", laser_step_duration_actual, "s")

print("Instantaneous stable z equilibria from laser noise:")
for power_factor_value, z_eq_power in instantaneous_z_equilibrium_by_power_factor.items():
    print(
        "  power factor =",
        power_factor_value,
        ", z_eq =",
        z_eq_power * 1e6,
        "micrometres"
    )


def deterministic_force_no_drag_2d(x, z, power_factor=1.0):
    Fx, Fz = F_optical_2d(x, z, power_factor)
    return Fx, Fz - m*g


def solve_baoab_2d_with_power(
    power_factor_time,
    brownian_normals_x,
    brownian_normals_z
):
    x_out = np.zeros_like(t_baoab)
    z_out = np.zeros_like(t_baoab)
    vx_out = np.zeros_like(t_baoab)
    vz_out = np.zeros_like(t_baoab)

    x_out[0] = x0
    z_out[0] = z0
    vx_out[0] = vx0
    vz_out[0] = vz0

    for i in range(len(t_baoab) - 1):
        x_i = x_out[i]
        z_i = z_out[i]
        vx_i = vx_out[i]
        vz_i = vz_out[i]
        power_factor_i = power_factor_time[i]

        Fx_i, Fz_i = deterministic_force_no_drag_2d(x_i, z_i, power_factor_i)
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_i += 0.5 * dt_baoab * vx_i
        z_i += 0.5 * dt_baoab * vz_i

        vx_i = (
            baoab_damping_factor * vx_i
            + baoab_thermal_velocity_scale * brownian_normals_x[i]
        )
        vz_i = (
            baoab_damping_factor * vz_i
            + baoab_thermal_velocity_scale * brownian_normals_z[i]
        )

        x_i += 0.5 * dt_baoab * vx_i
        z_i += 0.5 * dt_baoab * vz_i

        Fx_i, Fz_i = deterministic_force_no_drag_2d(x_i, z_i, power_factor_i)
        vx_i += 0.5 * dt_baoab * Fx_i / m
        vz_i += 0.5 * dt_baoab * Fz_i / m

        x_out[i + 1] = x_i
        z_out[i + 1] = z_i
        vx_out[i + 1] = vx_i
        vz_out[i + 1] = vz_i

    return x_out, z_out, vx_out, vz_out


brownian_normals_x = rng.normal(size=len(t_baoab) - 1)
brownian_normals_z = rng.normal(size=len(t_baoab) - 1)
constant_power_factor = np.ones_like(t_baoab)

baoab_constant_power_start_time = perf_counter()
x_baoab_constant_power, z_baoab_constant_power, vx_baoab_constant_power, vz_baoab_constant_power = (
    solve_baoab_2d_with_power(
        constant_power_factor,
        brownian_normals_x,
        brownian_normals_z
    )
)
baoab_constant_power_runtime = perf_counter() - baoab_constant_power_start_time

baoab_laser_noise_start_time = perf_counter()
x_baoab, z_baoab, vx_baoab, vz_baoab = solve_baoab_2d_with_power(
    laser_power_factor,
    brownian_normals_x,
    brownian_normals_z
)
baoab_laser_noise_runtime = perf_counter() - baoab_laser_noise_start_time
baoab_total_runtime = baoab_constant_power_runtime + baoab_laser_noise_runtime

x_laser_noise_difference = x_baoab - x_baoab_constant_power
z_laser_noise_difference = z_baoab - z_baoab_constant_power

x_constant_power_rms = np.std(x_baoab_constant_power - x_eq)
z_constant_power_rms = np.std(z_baoab_constant_power - z_eq)
x_laser_noise_rms = np.std(x_laser_noise_difference)
z_laser_noise_rms = np.std(z_laser_noise_difference)
x_laser_noise_max = np.max(np.abs(x_laser_noise_difference))
z_laser_noise_max = np.max(np.abs(z_laser_noise_difference))

print("BAOAB constant-power runtime =", baoab_constant_power_runtime, "s")
print("BAOAB laser-noise runtime =", baoab_laser_noise_runtime, "s")
print("BAOAB total runtime =", baoab_total_runtime, "s")
print("Total calculation runtime, excluding graph-viewing time =", perf_counter() - script_start_time, "s")

print(
    "BAOAB constant-power RMS x from equilibrium =",
    x_constant_power_rms * 1e6,
    "micrometres"
)
print(
    "BAOAB constant-power RMS z from equilibrium =",
    z_constant_power_rms * 1e6,
    "micrometres"
)
print(
    "RMS isolated laser-noise effect in x =",
    x_laser_noise_rms * 1e9,
    "nm"
)
print(
    "RMS isolated laser-noise effect in z =",
    z_laser_noise_rms * 1e9,
    "nm"
)
print(
    "Max isolated laser-noise effect in x =",
    x_laser_noise_max * 1e9,
    "nm"
)
print(
    "Max isolated laser-noise effect in z =",
    z_laser_noise_max * 1e9,
    "nm"
)

print("Harmonic approximation valid ranges with stochastic displacement scales:")
print(
    "| Direction | 1% valid range | 5% valid range | 10% valid range | "
    "Theoretical Brownian RMS displacement | Laser-noise-only RMS displacement |"
)
print("|---|---:|---:|---:|---:|---:|")
print(
    f"| x | +/- {x_valid_ranges[1] * 1e6:.2f} um | "
    f"+/- {x_valid_ranges[5] * 1e6:.2f} um | "
    f"+/- {x_valid_ranges[10] * 1e6:.2f} um | "
    f"{x_rms_thermal * 1e6:.3f} um | "
    f"{x_laser_noise_rms * 1e6:.3f} um |"
)
print(
    f"| z | +/- {z_valid_ranges[1] * 1e6:.2f} um | "
    f"+/- {z_valid_ranges[5] * 1e6:.2f} um | "
    f"+/- {z_valid_ranges[10] * 1e6:.2f} um | "
    f"{z_rms_thermal * 1e6:.3f} um | "
    f"{z_laser_noise_rms * 1e6:.3f} um |"
)

# ****************************************************************************************************************************************************
# Plot x(t) and z(t)
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(
    t_baoab,
    x_baoab * 1e6,
    linewidth=0.9,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[0].set_ylabel("x / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    t_baoab,
    z_baoab * 1e6,
    linewidth=0.9,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
axes[1].axhline(z_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[1].set_xlabel("Time / s")
axes[1].set_ylabel("z / micrometres")
axes[1].legend()
axes[1].grid()

fig.suptitle("2D BAOAB ray-optics motion")
plt.tight_layout()
plt.show()

# ****************************************************************************************************************************************************
# Laser-power noise diagnostic
# ****************************************************************************************************************************************************
plt.figure(figsize=(8, 4))
plt.step(t_baoab, laser_power_time, where="post", label="laser power")
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
# Instantaneous equilibrium position from laser-power noise
# ****************************************************************************************************************************************************
plt.figure(figsize=(8, 4))
plt.step(
    t_baoab,
    z_eq_laser_noise_time * 1e6,
    where="post",
    label="instantaneous z equilibrium"
)
plt.axhline(z_eq * 1e6, linestyle="--", color="black", label="nominal equilibrium")
plt.xlabel("Time / s")
plt.ylabel("z equilibrium / micrometres")
plt.title("Equilibrium height shift from laser-power noise")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ****************************************************************************************************************************************************
# Compare Brownian trajectories with and without laser-power noise
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(
    t_baoab,
    x_baoab_constant_power * 1e6,
    label="BAOAB Brownian, constant laser power"
)
axes[0].plot(
    t_baoab,
    x_baoab * 1e6,
    linewidth=0.9,
    label="BAOAB Brownian + laser noise"
)
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="equilibrium")
axes[0].set_ylabel("x / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    t_baoab,
    z_baoab_constant_power * 1e6,
    label="BAOAB Brownian, constant laser power"
)
axes[1].plot(
    t_baoab,
    z_baoab * 1e6,
    linewidth=0.9,
    label="BAOAB Brownian + laser noise"
)
axes[1].axhline(z_eq * 1e6, linestyle=":", color="black", label="equilibrium")
axes[1].set_xlabel("Time / s")
axes[1].set_ylabel("z / micrometres")
axes[1].legend()
axes[1].grid()

fig.suptitle("BAOAB motion with and without laser-power noise")
plt.tight_layout()
plt.show()

# ****************************************************************************************************************************************************
# Isolated laser-noise effect
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(t_baoab, x_laser_noise_difference * 1e9)
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x difference / nm")
axes[0].grid()

axes[1].plot(t_baoab, z_laser_noise_difference * 1e9)
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_xlabel("Time / s")
axes[1].set_ylabel("z difference / nm")
axes[1].grid()

fig.suptitle("Isolated displacement effect from laser-power noise")
plt.tight_layout()
plt.show()

#****************************************************************************************************************************************************
# Power spectral density
#****************************************************************************************************************************************************
sampling_frequency = 1 / dt_baoab
nyquist_frequency = sampling_frequency / 2
raw_frequency_bin_size = 1 / (len(t_baoab) * dt_baoab)
psd_plot_max = nyquist_frequency

print("PSD sampling frequency =", sampling_frequency, "Hz")
print("PSD Nyquist frequency =", nyquist_frequency, "Hz")
print("Raw PSD frequency bin size =", raw_frequency_bin_size, "Hz")


def positive_raw_periodogram_psd(signal, time_values):
    """
    Return a full-record, non-Welch-averaged one-sided PSD in m^2/Hz.
    """
    dt = time_values[1] - time_values[0]
    fs = 1 / dt
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    window_power = np.sum(window**2)

    fft_values = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(len(signal), dt)
    psd = np.abs(fft_values)**2 / (fs * window_power)

    if len(psd) > 2:
        psd[1:-1] *= 2

    positive_mask = (
        (frequencies > 0)
        & np.isfinite(psd)
        & (psd > 0)
    )

    return frequencies[positive_mask], psd[positive_mask]


def positive_welch_psd(signal, time_values, target_bin_width=None):
    """
    Return a Welch-averaged one-sided PSD in m^2/Hz.
    """
    dt = time_values[1] - time_values[0]
    fs = 1 / dt
    signal = signal - np.mean(signal)

    if target_bin_width is None:
        target_bin_width = max(1.0, min(omega_x, omega_z) / (2 * np.pi) / 20)

    nperseg = int(np.ceil(fs / target_bin_width))
    nperseg = min(len(signal), max(256, nperseg))

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    positive_mask = (
        (frequencies > 0)
        & np.isfinite(psd)
        & (psd > 0)
    )

    return frequencies[positive_mask], psd[positive_mask], nperseg, fs / nperseg


def plot_single_psd(freqs, psd, title, label, color, linear_frequency):
    plt.figure(figsize=(8, 5))
    plt.loglog(freqs, psd * 1e12, color=color, label=label)
    plt.axvline(linear_frequency, linestyle="--", label="linear frequency")
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(max(freqs[0] * 0.95, 1e-12), psd_plot_max)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / micrometre$^2$/Hz")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


def plot_laser_effect_psd(x_freqs, x_psd, z_freqs, z_psd, title):
    plt.figure(figsize=(8, 5))
    plt.loglog(x_freqs, x_psd * 1e18, label="x laser-noise effect")
    plt.loglog(
        z_freqs,
        z_psd * 1e18,
        color="tab:orange",
        label="z laser-noise effect"
    )
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(max(min(x_freqs[0], z_freqs[0]) * 0.95, 1e-12), psd_plot_max)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / nm$^2$/Hz")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


raw_baoab_x_freqs, raw_baoab_x_psd = positive_raw_periodogram_psd(
    x_baoab - x_eq,
    t_baoab
)
raw_baoab_z_freqs, raw_baoab_z_psd = positive_raw_periodogram_psd(
    z_baoab - z_eq,
    t_baoab
)
raw_laser_x_freqs, raw_laser_x_psd = positive_raw_periodogram_psd(
    x_laser_noise_difference,
    t_baoab
)
raw_laser_z_freqs, raw_laser_z_psd = positive_raw_periodogram_psd(
    z_laser_noise_difference,
    t_baoab
)

minimum_resolvable_frequency = raw_baoab_x_freqs[0]
minimum_plot_frequency = 0.95 * minimum_resolvable_frequency

print("Minimum resolvable non-zero frequency =", minimum_resolvable_frequency, "Hz")
print("Lower frequency shown on plot =", minimum_plot_frequency, "Hz")

plot_single_psd(
    raw_baoab_x_freqs,
    raw_baoab_x_psd,
    "Raw PSD of BAOAB x motion",
    "BAOAB x raw PSD",
    "tab:blue",
    omega_x / (2*np.pi),
)

plot_single_psd(
    raw_baoab_z_freqs,
    raw_baoab_z_psd,
    "Raw PSD of BAOAB z motion",
    "BAOAB z raw PSD",
    "tab:orange",
    omega_z / (2*np.pi),
)

plot_laser_effect_psd(
    raw_laser_x_freqs,
    raw_laser_x_psd,
    raw_laser_z_freqs,
    raw_laser_z_psd,
    "Raw PSD of isolated laser-noise displacement effect",
)

welch_baoab_x_freqs, welch_baoab_x_psd, welch_nperseg, welch_bin_size = (
    positive_welch_psd(
        x_baoab - x_eq,
        t_baoab
    )
)
welch_baoab_z_freqs, welch_baoab_z_psd, _, _ = positive_welch_psd(
    z_baoab - z_eq,
    t_baoab
)
welch_laser_x_freqs, welch_laser_x_psd, _, _ = positive_welch_psd(
    x_laser_noise_difference,
    t_baoab
)
welch_laser_z_freqs, welch_laser_z_psd, _, _ = positive_welch_psd(
    z_laser_noise_difference,
    t_baoab
)

print("Welch PSD nperseg =", welch_nperseg)
print("Welch PSD frequency bin size =", welch_bin_size, "Hz")

plot_single_psd(
    welch_baoab_x_freqs,
    welch_baoab_x_psd,
    "Welch-averaged PSD of BAOAB x motion",
    "BAOAB x Welch PSD",
    "tab:blue",
    omega_x / (2*np.pi),
)

plot_single_psd(
    welch_baoab_z_freqs,
    welch_baoab_z_psd,
    "Welch-averaged PSD of BAOAB z motion",
    "BAOAB z Welch PSD",
    "tab:orange",
    omega_z / (2*np.pi),
)

plot_laser_effect_psd(
    welch_laser_x_freqs,
    welch_laser_x_psd,
    welch_laser_z_freqs,
    welch_laser_z_psd,
    "Welch-averaged PSD of isolated laser-noise displacement effect",
)


# ****************************************************************************************************************************************************
# Plot 2D trajectory
# ****************************************************************************************************************************************************
plt.figure(figsize=(7, 6))

plt.plot(
    x_baoab * 1e6,
    z_baoab * 1e6,
    linewidth=0.8,
    alpha=0.75,
    label="BAOAB Brownian + laser noise"
)
plt.scatter([x_eq * 1e6], [z_eq * 1e6], color="black", s=30, label="equilibrium")

plt.xlabel("x / micrometres")
plt.ylabel("z / micrometres")
plt.title("2D trajectory in the x-z plane")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()

# ****************************************************************************************************************************************************
# Transverse restoring-force check
# ****************************************************************************************************************************************************
x_scan = np.linspace(-40e-6, 40e-6, 401)
Fx_scan = np.array([F_optical_2d(x_i, z_eq)[0] for x_i in x_scan])

restoring = Fx_scan * x_scan < 0
near_axis = np.abs(x_scan) < 1e-12
restoring_or_axis = restoring | near_axis

plt.figure(figsize=(8, 5))
plt.plot(x_scan * 1e6, Fx_scan / (m*g), label="Fx at z_eq")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, linestyle=":", color="black", label="beam axis")
plt.axvline(x_displacement * 1e6, linestyle="--", label="initial x displacement")
plt.axvline(-x_displacement * 1e6, linestyle="--")
plt.xlabel("x / micrometres")
plt.ylabel("Fx / mg")
plt.title("Transverse force check at z = z_eq")
plt.legend()
plt.grid()
plt.show()

if not np.interp(abs(x_displacement), x_scan[x_scan >= 0], restoring_or_axis[x_scan >= 0].astype(float)) > 0.5:
    print(
        "Warning: the chosen x displacement may be outside the local restoring region. "
        "Try reducing x_displacement."
    )

# ****************************************************************************************************************************************************
# Vertical force breakdown along the beam axis
# ****************************************************************************************************************************************************
z_force_values = np.linspace(z_eq - 500e-6, z_eq + 500e-6, 1000)

Fz_ray_scat_values = []
Fz_ray_grad_values = []

for z_i in z_force_values:
    _, Fz_scat, _, Fz_grad = F_ray_optics_2d_components_scalar(0.0, z_i)
    Fz_ray_scat_values.append(Fz_scat)
    Fz_ray_grad_values.append(Fz_grad)

Fz_ray_scat_values = np.array(Fz_ray_scat_values)
Fz_ray_grad_values = np.array(Fz_ray_grad_values)
Fz_ray_values = Fz_ray_scat_values + Fz_ray_grad_values
Fz_photo_values = np.array([F_photo_2d(0.0, z_i)[1] for z_i in z_force_values])
Fz_gravity_values = -m * g * np.ones_like(z_force_values)
Fz_total_net_values = Fz_ray_values + Fz_photo_values + Fz_gravity_values

plt.figure(figsize=(8, 5))
plt.plot(
    z_force_values * 1e6,
    Fz_ray_scat_values / (m*g),
    label="Ray scattering-like axial force"
)
plt.plot(
    z_force_values * 1e6,
    Fz_ray_grad_values / (m*g),
    label="Ray gradient-like axial force"
)
plt.plot(
    z_force_values * 1e6,
    Fz_ray_values / (m*g),
    linestyle="--",
    label="Total ray-optics axial force"
)
plt.plot(z_force_values * 1e6, Fz_photo_values / (m*g), label="Photophoretic force")
plt.plot(z_force_values * 1e6, Fz_gravity_values / (m*g), label="Gravity")
plt.plot(
    z_force_values * 1e6,
    Fz_total_net_values / (m*g),
    linewidth=2,
    label="Total net vertical force"
)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(z_eq * 1e6, linestyle=":", color="black", label="equilibrium")
plt.xlabel("z / micrometres")
plt.ylabel("Force / mg")
plt.title("Vertical force breakdown along beam axis")
plt.legend()
plt.grid()
plt.show()

# ****************************************************************************************************************************************************
# Plot force field
# ****************************************************************************************************************************************************
x_values = np.linspace(-60e-6, 60e-6, 31)
z_values = np.linspace(z_eq - 100e-6, z_eq + 100e-6, 31)
X, Z = np.meshgrid(x_values, z_values)

Fx_field, Fz_field = F_optical_2d(X, Z)
Fz_net_field = Fz_field - m*g

force_scale = m*g

plt.figure(figsize=(8, 6))
plt.contourf(X * 1e6,Z * 1e6,intensity(X, Z),levels=30,cmap="viridis",alpha=0.75)
plt.colorbar(label="Normalized intensity")
plt.quiver(X * 1e6,Z * 1e6,Fx_field / force_scale,Fz_net_field / force_scale,color="white",pivot="mid",scale=55)
plt.scatter([x_eq * 1e6], [z_eq * 1e6], color="red", s=35, label="equilibrium")

plt.xlabel("x / micrometres")
plt.ylabel("z / micrometres")
plt.title("2D net force field over intensity")
plt.legend()
plt.grid()
plt.show()

# ****************************************************************************************************************************************************
# On-axis force check
# ****************************************************************************************************************************************************
# z_axis = np.linspace(-200e-6, 300e-6, 1000)
# Fx_axis, Fz_axis = F_optical_2d(0.0, z_axis)

# plt.figure(figsize=(8, 5))
# plt.plot(z_axis * 1e6, Fz_axis, label="upward optical + photophoretic force")
# plt.plot(z_axis * 1e6, Fz_axis - m*g, label="net vertical force")
# plt.axhline(m*g, linestyle="--", label="gravity mg")
# plt.axhline(0, linewidth=0.8, color="black")
# plt.axvline(z_eq * 1e6, linestyle=":", label="equilibrium")

# plt.xlabel("z / micrometres")
# plt.ylabel("Force / N")
# plt.title("On-axis vertical force check")
# plt.legend()
# plt.grid()
# plt.show()
