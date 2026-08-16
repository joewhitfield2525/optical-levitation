import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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
zR = 100e-6       # Rayleigh-like axial length scale, m
w0 = 20e-6        # beam waist, m

P_laser = 0.08             # W, example laser power
I0 = 2 * P_laser / (np.pi * w0**2)   # Gaussian peak intensity, W/m^2

print("Laser power =", P_laser, "W")
print("Peak intensity I0 =", I0, "W/m^2")
print("Paraxial divergence half-angle w0/zR =", w0 / zR, "rad")


def beam_width(z):
   
    return w0 * np.sqrt(1 + (z / zR)**2)


def intensity(x, z):
   
    s = 1 + (z / zR)**2
    return (1 / s) * np.exp(-2 * x**2 / (w0**2 * s))


def physical_intensity(x, z):
    return I0 * intensity(x, z)


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


def F_ray_optics_2d_scalar(x, z):
    """
    Ashkin-style ray-optics force for the x-z model.

    The full projected disk is integrated, but only Fx and Fz are returned.
    """
    Fx_scat, Fz_scat, Fx_grad, Fz_grad = F_ray_optics_2d_components_scalar(x, z)
    return Fx_scat + Fx_grad, Fz_scat + Fz_grad


def F_ray_optics_2d_components_scalar(x, z):
    """
    Ray-optics force split into scattering-like and gradient-like parts.

    Returns:
        Fx_scat, Fz_scat, Fx_grad, Fz_grad

    In this Ashkin ray model, Q_s is treated as the scattering-like component
    along the local ray direction, while Q_g is treated as the gradient-like
    component perpendicular to the local ray direction.
    """
    ray_x = x + U_hit
    dP = physical_intensity(ray_x, z) * dA
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


def F_ray_optics_2d(x, z):
    """
    Vectorized wrapper around the scalar ray-optics force.
    """
    x_arr, z_arr = np.broadcast_arrays(x, z)

    if x_arr.shape == ():
        return F_ray_optics_2d_scalar(float(x_arr), float(z_arr))

    Fx = np.zeros_like(x_arr, dtype=float)
    Fz = np.zeros_like(z_arr, dtype=float)

    for index in np.ndindex(x_arr.shape):
        Fx[index], Fz[index] = F_ray_optics_2d_scalar(x_arr[index], z_arr[index])

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


def absorbed_intensity(x, z):
    return absorption_fraction * I0 * intensity(x, z)


def photophoretic_force_magnitude(x, z):
    D = photophoretic_D()
    p_max_ph = photophoretic_p_max()

    I_abs = absorbed_intensity(x, z)

    F_max = (
        0.5
        * radius**2
        * D
        * np.sqrt(alpha_acc / 2)
        * I_abs
        / k_particle
    )

    return 2 * F_max / ((p / p_max_ph) + (p_max_ph / p))


def F_photo_2d(x, z):
    """
    Improved photophoretic force.
    Positive sign means force points upward, in +z.
    """
    return 0.0, photophoretic_force_magnitude(x, z)


def F_optical_2d(x, z):
    
    Fx_ray, Fz_ray = F_ray_optics_2d(x, z)
    Fx_photo, Fz_photo = F_photo_2d(x, z)

    Fx = Fx_ray + Fx_photo
    Fz = Fz_ray + Fz_photo

    return Fx, Fz


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
# Deterministic 2D nonlinear equation
# ****************************************************************************************************************************************************


def nonlinear_system_2d(t,Y):
    x, z, vx, vz = Y

    Fx, Fz = F_optical_2d(x, z)

    Fx_drag = -b * vx
    Fz_drag = -b * vz

    ax = (Fx + Fx_drag) / m
    az = (Fz - m*g + Fz_drag) / m

    return [vx, vz, ax, az]


# ****************************************************************************************************************************************************
# Linear 2D SHM approximation
# ****************************************************************************************************************************************************
def linear_system_2d(t, Y):
    x = Y[0]   # displacement from equilibrium
    z = Y[1]   # displacement from equilibrium
    vx = Y[2]
    vz = Y[3]

    dxdt = vx
    dzdt = vz
    dvxdt = -(b/m)*vx - (kx/m)*x
    dvzdt = -(b/m)*vz - (kz/m)*z

    return [dxdt, dzdt, dvxdt, dvzdt]


# ****************************************************************************************************************************************************
# Initial conditions
# ****************************************************************************************************************************************************
x_displacement = 6e-6
z_displacement = 10e-6

x0 = x_eq + x_displacement
z0 = z_eq + z_displacement
vx0 = 0.0
vz0 = 0.0

Y0 = [x0, z0, vx0, vz0]
Y0_linear = [x_displacement, z_displacement, vx0, vz0]

Fx_initial, Fz_initial = F_optical_2d(x0, z0)
ax_initial = Fx_initial / m
az_initial = (Fz_initial - m*g) / m

print("Initial x displacement =", x_displacement * 1e6, "micrometres")
print("Initial z displacement =", z_displacement * 1e6, "micrometres")
print("Initial Fx / mg =", Fx_initial / (m*g))
print("Initial Fz_net / mg =", (Fz_initial - m*g) / (m*g))
print("Initial ax =", ax_initial, "m/s^2")
print("Initial az =", az_initial, "m/s^2")

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_start = 0
t_end = 3
t_eval = np.linspace(t_start, t_end, 10000)

# ****************************************************************************************************************************************************
# Solve deterministic 2D model
# ****************************************************************************************************************************************************
sol_2d = solve_ivp(
    nonlinear_system_2d,
    [t_start, t_end],
    Y0,
    t_eval=t_eval,
    rtol=1e-10,
    atol=[1e-14, 1e-14, 1e-12, 1e-12],
    max_step=1e-4
)

# ****************************************************************************************************************************************************
# Solve linear 2D SHM approximation
# ****************************************************************************************************************************************************
sol_linear_2d = solve_ivp(
    linear_system_2d,
    [t_start, t_end],
    Y0_linear,
    t_eval=t_eval,
    rtol=1e-10,
    atol=[1e-14, 1e-14, 1e-12, 1e-12],
    max_step=1e-4
)

# ****************************************************************************************************************************************************
# Extract solutions
# ****************************************************************************************************************************************************
t = sol_2d.t
x_det = sol_2d.y[0]
z_det = sol_2d.y[1]
vx_det = sol_2d.y[2]
vz_det = sol_2d.y[3]

x_linear_disp = sol_linear_2d.y[0]
z_linear_disp = sol_linear_2d.y[1]
vx_linear = sol_linear_2d.y[2]
vz_linear = sol_linear_2d.y[3]

x_linear = x_eq + x_linear_disp
z_linear = z_eq + z_linear_disp

# ****************************************************************************************************************************************************
# Plot x(t) and z(t)
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(t, x_det * 1e6, label="Nonlinear ray-optics model")
axes[0].axhline(x_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[0].set_ylabel("x / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(t, z_det * 1e6, label="Nonlinear ray-optics model")
axes[1].axhline(z_eq * 1e6, linestyle=":", color="black", label="Equilibrium")
axes[1].set_xlabel("Time / s")
axes[1].set_ylabel("z / micrometres")
axes[1].legend()
axes[1].grid()

fig.suptitle("2D ray-optics nonlinear motion")
plt.tight_layout()
plt.show()

#****************************************************************************************************************************************************
# Power spectral density
#****************************************************************************************************************************************************
x_signal = x_det - x_eq
z_signal = z_det - z_eq

dt_psd = t[1] - t[0]
sampling_frequency = 1 / dt_psd
nyquist_frequency = sampling_frequency / 2
frequency_bin_size = sampling_frequency / len(t)
psd_plot_max = 10_000

def raw_periodogram_psd(signal):
    """
    Return a full-record, non-Welch-averaged one-sided PSD in m^2/Hz.
    """
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    window_power = np.sum(window**2)

    fft_values = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(len(signal), dt_psd)
    psd = np.abs(fft_values)**2 / (sampling_frequency * window_power)

    if len(psd) > 2:
        psd[1:-1] *= 2

    return freqs, psd


x_psd_freqs, x_psd = raw_periodogram_psd(x_signal)
z_psd_freqs, z_psd = raw_periodogram_psd(z_signal)

x_positive_psd_mask = (
    (x_psd_freqs > 0)
    & np.isfinite(x_psd)
    & (x_psd > 0)
)
z_positive_psd_mask = (
    (z_psd_freqs > 0)
    & np.isfinite(z_psd)
    & (z_psd > 0)
)

print("Raw PSD sampling frequency =", sampling_frequency, "Hz")
print("Raw PSD Nyquist frequency =", nyquist_frequency, "Hz")
print("Raw PSD frequency bin size =", frequency_bin_size, "Hz")

if np.any(x_positive_psd_mask):
    plt.figure(figsize=(8, 5))
    plt.loglog(
        x_psd_freqs[x_positive_psd_mask],
        x_psd[x_positive_psd_mask] * 1e12,
        label="x PSD"
    )
    plt.axvline(omega_x / (2*np.pi), linestyle="--", label="linear fx")
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(
        max(x_psd_freqs[x_positive_psd_mask][0] * 0.95, 1e-12),
        min(psd_plot_max / 10, nyquist_frequency)
    )
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / micrometre$^2$/Hz")
    plt.title("Raw PSD of x motion")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
else:
    print("Skipping x PSD plot because the x signal has no positive PSD values.")

if np.any(z_positive_psd_mask):
    plt.figure(figsize=(8, 5))
    plt.loglog(
        z_psd_freqs[z_positive_psd_mask],
        z_psd[z_positive_psd_mask] * 1e12,
        color="tab:orange",
        label="z PSD"
    )
    plt.axvline(omega_z / (2*np.pi), linestyle="--", label="linear fz")
    plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(
        max(z_psd_freqs[z_positive_psd_mask][0] * 0.95, 1e-12),
        min(psd_plot_max / 10, nyquist_frequency)
    )
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / micrometre$^2$/Hz")
    plt.title("Raw PSD of z motion")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
else:
    print("Skipping z PSD plot because the z signal has no positive PSD values.")


# ****************************************************************************************************************************************************
# Plot 2D trajectory
# ****************************************************************************************************************************************************
plt.figure(figsize=(7, 6))

plt.plot(x_det * 1e6,z_det * 1e6,label="Nonlinear ray-optics trajectory")
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
