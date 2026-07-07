import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import welch

# ****************************************************************************************************************************************************
# Constants
# ****************************************************************************************************************************************************

g = 9.81

# ****************************************************************************************************************************************************
# Particle properties
# ****************************************************************************************************************************************************

radius = 10e-6          # m
density = 2200         # kg/m^3

volume = (4/3) * np.pi * radius**3
m = density * volume

print("Particle mass =", m, "kg")
F_weight = m * g
print("Bare particle weight mg =", F_weight, "N")

# ****************************************************************************************************************************************************
# Gas properties
# ****************************************************************************************************************************************************
# Use one pressure value throughout the model. This pressure is used for
# gas density, mean free path, photophoresis, and damping.
p = 30          # pressure, Pa
T = 130           # temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
rho_g = p * M_air / (R * T)
kB = 1.38e-23
d_air = 3.7e-10

F_buoyancy = rho_g * volume * g
F_weight_effective = F_weight - F_buoyancy

lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

print("Gas density =", rho_g, "kg/m^3")
print("Buoyancy force =", F_buoyancy, "N")
print("Effective downward weight mg - F_buoyancy =", F_weight_effective, "N")
print("Buoyancy / bare weight =", F_buoyancy / F_weight)
print("Mean free path =", lambda_mfp, "m")
print("Knudsen number =", Kn)

# ****************************************************************************************************************************************************
# Laser / force parameters
# ****************************************************************************************************************************************************
wavelength=532e-9                                           # wavelength, m    
w0 = 3.5e-6                                                 # beam waist, 2 * (1/e^2) , m
zR=(np.pi*w0**2)/wavelength                                 #rayleigh length, m


P_laser = 0.1                                               # Laser power W
I0 = 2 * P_laser / (np.pi * w0**2)                          # peak Gaussian intensity.

# Stepwise laser-power noise used in the BAOAB Brownian simulation.
# The power is held constant for laser_noise_step_duration, then randomly
# moves between 0.99 P_laser, P_laser, and 1.01 P_laser.
laser_noise_fraction = 0.01
laser_noise_step_duration = 0.01                             # seconds




c_light = 299792458                                         # m/s
Q_pr = 0.1                                                  #estimate, has not been calcualted 
sigma_geom = np.pi * radius**2                              #
Fs0 = Q_pr * sigma_geom * I0 / c_light                      #phenenological scattering force


Fg0 = 5.0 * F_weight                                        #phenomenological gradient force.



def intensity_shape(z):                                     #Simple 1D laser intensity profile along z. Maximum at the focus
    
    z_rel = z - z_focus
    return 1 / (1 + (z_rel / zR)**2)


def F_scat(z, power_factor=1.0):                             #Upward scattering/radiation-pressure force.

    return power_factor * Fs0 * intensity_shape(z)


def F_grad(z, power_factor=1.0):                             #  Axial gradient force.
                                                            #  For z > 0, this is negative.
                                                            #  For z < 0, this is positive.
    return -power_factor * Fg0 * (z / zR) / (1 + (z / zR)**2)**2


# ******************************************************************************************************************
# Photophoretic force parameters
# ******************************************************************************************************************

z_focus = 0.0                                            # laser focus position, m
k_particle = 1.4                                         # thermal conductivity W/(m K), approximate silica
alpha_acc = 1.0                                          # thermal accommodation coefficient
kappa_t = 1.14                                           # thermal creep coefficient
absorption_fraction = 1e-4




def mean_thermal_speed():
    return np.sqrt(8 * R * T / (np.pi * M_air))


def photophoretic_D():
    c_bar = mean_thermal_speed()
    return (np.pi * c_bar * eta / (2 * T)) * np.sqrt(np.pi * kappa_t / 3)


def photophoretic_p_max():
    D = photophoretic_D()
    return (3 * T / (np.pi * radius)) * D * np.sqrt(2 / alpha_acc)


def absorbed_intensity(z, power_factor=1.0):
    return power_factor * absorption_fraction * I0 * intensity_shape(z)


def photophoretic_force_magnitude(z, p, power_factor=1.0):               # Rohatschek-style all-pressure photophoretic force magnitude.

    
    D = photophoretic_D()
    p_max_ph = photophoretic_p_max()
    I_abs = absorbed_intensity(z, power_factor)

    F_max = (0.5* radius**2* D* np.sqrt(alpha_acc / 2)* I_abs/ k_particle)

    return 2 * F_max / ((p / p_max_ph) + (p_max_ph / p))


def F_photo(z, p, power_factor=1.0):
    """
    Photophoretic force.

    Positive sign means the force points upward in +z.
    """
    return photophoretic_force_magnitude(z, p, power_factor)


print("Photophoretic p_max =", photophoretic_p_max(), "Pa")
print("Photophoretic force at focus =", F_photo(0.0, p), "N")
print("Photophoretic force at focus / bare weight =", F_photo(0.0, p) / F_weight)
print("Photophoretic force at focus / effective weight =", F_photo(0.0, p) / F_weight_effective)


def F_total(z, power_factor=1.0):
    """
    Total vertical optical force.
    """
    return F_scat(z, power_factor) + F_grad(z, power_factor) + F_photo(z, p, power_factor)

def F_net(z):
    """
    Net force excluding damping.
    Equilibrium occurs when F_net = 0.
    """
    return F_total(z) - F_weight_effective


# ******************************************************************************************************************
# Find equilibrium position
# ******************************************************************************************************************
# We look for z_eq where:
#
# F_scat(z_eq) + F_grad(z_eq) + F_photo(z_eq) = mg - F_buoyancy

z_min = -10e-3
z_max = 10e-3

# Scan to find possible equilibrium points
z_scan = np.linspace(z_min, z_max, 2000)
F_scan = F_net(z_scan)

roots = []

for i in range(len(z_scan) - 1):
    if F_scan[i] * F_scan[i + 1] < 0:
        root = brentq(F_net,z_scan[i],z_scan[i + 1],xtol=1e-15,rtol=1e-15)
        roots.append(root)

print("Equilibrium positions / micrometres:")
for root in roots:
    print(root * 1e6)

# Choose a stable equilibrium
# Stability condition:
#
# dF_net/dz < 0
#
# because if z increases above equilibrium, force should become more negative.

def numerical_derivative(func, z, h=1e-9):
    return (func(z + h) - func(z - h)) / (2*h)

stable_roots = []

for root in roots:
    slope = numerical_derivative(F_net, root)
    if slope < 0:
        stable_roots.append(root)

if len(stable_roots) == 0:
    raise ValueError("No stable equilibrium found. Try changing Fs0 or Fg0.")

z_eq = stable_roots[0]

print("Chosen stable equilibrium =", z_eq * 1e6, "micrometres")

# ******************************************************************************************************************
# Linear SHM approximation
# ******************************************************************************************************************
# Near equilibrium:
#
# F_total(z) - F_weight_effective ≈ -k (z - z_eq)
#
# so:
#
# k = -dF_total/dz at z_eq

dFdz_eq = numerical_derivative(F_total, z_eq)
k = -dFdz_eq

omega = np.sqrt(k / m)
frequency = omega / (2*np.pi)

print("Effective spring constant k =", k, "N/m")
print("Small-oscillation angular frequency omega =", omega, "rad/s")
print("Small-oscillation frequency =", frequency, "Hz")

x_rms_thermal = np.sqrt(kB * T / k)
print("Expected thermal RMS displacement =", x_rms_thermal * 1e6, "micrometres")

# ******************************************************************************************************************
# Damping
# ******************************************************************************************************************
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
#b=0
damping_ratio = b / (2 * np.sqrt(m * k))

print("Drag model requested =", drag_model)
print("Drag model used =", drag_model_used)
print("Stokes damping b =", b_stokes, "kg/s")
print("Cunningham-corrected Stokes damping b =", b_cunningham, "kg/s")
print("Epstein damping b =", b_epstein, "kg/s")
print("Selected pressure-dependent damping b =", b, "kg/s")
print("Damping ratio =", damping_ratio)

# ******************************************************************************************************************
# Nonlinear separated-force equation
# ******************************************************************************************************************

def nonlinear_system(t, Y):
    z = Y[0]
    v = Y[1]

    dzdt = v

    Fopt = F_scat(z) + F_grad(z)
    Fph =F_photo(z, p)
    Fdrag = -b * v

    dvdt = (Fopt + Fph - F_weight_effective + Fdrag) / m

    return [dzdt, dvdt]
# ******************************************************************************************************************
# Linear SHM equation
# ******************************************************************************************************************
def linear_system(t, Y):
    x = Y[0]   # displacement from equilibrium
    v = Y[1]

    dxdt = v
    dvdt = -(b/m)*v - (k/m)*x

    return [dxdt, dvdt]

# ******************************************************************************************************************
# Initial conditions
# ******************************************************************************************************************
# Start slightly away from equilibrium
displacement = 50e-6

z0 = z_eq + displacement
v0 = 0.0

x0 = z0 - z_eq

Y0_nonlinear = [z0, v0]
Y0_linear = [x0, v0]

# ******************************************************************************************************************
# Time range
# ******************************************************************************************************************
t_start = 0
t_end = 10.0
t_eval = np.linspace(t_start, t_end, 10000)

# ******************************************************************************************************************
# Solve nonlinear model
# ******************************************************************************************************************
sol_nonlinear = solve_ivp(
    nonlinear_system,
    [t_start, t_end],
    Y0_nonlinear,
    t_eval=t_eval,
    rtol=1e-10,
    atol=[1e-14, 1e-12],
    max_step=1e-4
)

# ******************************************************************************************************************
# Solve linear SHM approximation
# ******************************************************************************************************************
sol_linear = solve_ivp(
    linear_system,
    [t_start, t_end],
    Y0_linear,
    t_eval=t_eval,
    rtol=1e-10,
    atol=[1e-14, 1e-12],
    max_step=1e-4
)

# ******************************************************************************************************************
# Solve nonlinear model with BAOAB Brownian dynamics
# ******************************************************************************************************************
# BAOAB is used for the Langevin equation:
#
#     dz/dt = v
#     m dv/dt = F_det(z) - b v + thermal noise
#
# where:
#
#     F_det(z) = F_scat(z) + F_grad(z) + F_photo(z,p) - F_weight_effective
#
# The deterministic force and the damping/noise are split into:
#
#     B: half deterministic-force kick
#     A: half drift
#     O: exact damping + Brownian velocity kick
#     A: half drift
#     B: half deterministic-force kick

rng = np.random.default_rng(seed=4)
laser_rng = np.random.default_rng(seed=12)

dt_baoab = t_eval[1] - t_eval[0]
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)

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

laser_power_factor = np.repeat(
    laser_step_power_factors,
    laser_step_samples
)[:len(t_baoab)]
laser_power_time = P_laser * laser_power_factor

gamma_baoab = b / m
baoab_damping_factor = np.exp(-gamma_baoab * dt_baoab)
baoab_thermal_velocity_scale = np.sqrt(
    (kB * T / m) * (1 - baoab_damping_factor**2)
)

print("BAOAB timestep =", dt_baoab, "s")
print("BAOAB sampling frequency =", 1 / dt_baoab, "Hz")
print("BAOAB damping factor =", baoab_damping_factor)
print("BAOAB thermal velocity scale =", baoab_thermal_velocity_scale, "m/s")
print("Laser noise levels =", P_laser * laser_allowed_power_factors, "W")
print("Laser noise requested step duration =", laser_noise_step_duration, "s")
print("Laser noise actual step duration =", laser_step_duration_actual, "s")


def deterministic_force_no_drag(z, power_factor=1.0):
    """
    Position-dependent deterministic force excluding drag.

    BAOAB handles damping and Brownian noise in the O step.
    """
    return (
        F_scat(z, power_factor)
        + F_grad(z, power_factor)
        + F_photo(z, p, power_factor)
        - F_weight_effective
    )


def solve_baoab_with_power(power_factor_time, brownian_normals):
    """
    Run BAOAB for a specified laser-power factor time series.

    The same brownian_normals can be reused for two runs so that any
    difference between them is caused by laser-power noise rather than by a
    different random Brownian sequence.
    """
    z_out = np.zeros_like(t_baoab)
    v_out = np.zeros_like(t_baoab)

    z_out[0] = z0
    v_out[0] = v0

    for i in range(len(t_baoab) - 1):
        z_i = z_out[i]
        v_i = v_out[i]
        power_factor_i = power_factor_time[i]

        # B: half deterministic-force kick
        v_i = v_i + 0.5 * dt_baoab * deterministic_force_no_drag(z_i, power_factor_i) / m

        # A: half drift
        z_i = z_i + 0.5 * dt_baoab * v_i

        # O: exact damping + thermal Brownian velocity update
        v_i = (
            baoab_damping_factor * v_i
            + baoab_thermal_velocity_scale * brownian_normals[i]
        )

        # A: second half drift
        z_i = z_i + 0.5 * dt_baoab * v_i

        # B: second half deterministic-force kick
        v_i = v_i + 0.5 * dt_baoab * deterministic_force_no_drag(z_i, power_factor_i) / m

        z_out[i + 1] = z_i
        v_out[i + 1] = v_i

    return z_out, v_out


brownian_normals = rng.normal(size=len(t_baoab) - 1)
constant_power_factor = np.ones_like(t_baoab)

z_baoab_constant_power, v_baoab_constant_power = solve_baoab_with_power(
    constant_power_factor,
    brownian_normals,
)

z_baoab, v_baoab = solve_baoab_with_power(
    laser_power_factor,
    brownian_normals,
)

laser_noise_displacement_difference = z_baoab - z_baoab_constant_power

print(
    "BAOAB RMS displacement from equilibrium =",
    np.std(z_baoab - z_eq) * 1e6,
    "micrometres"
)
print(
    "RMS laser-noise displacement effect =",
    np.std(laser_noise_displacement_difference) * 1e9,
    "nm"
)
print(
    "Maximum absolute laser-noise displacement effect =",
    np.max(np.abs(laser_noise_displacement_difference)) * 1e9,
    "nm"
)

# ******************************************************************************************************************
# Plot laser-power noise
# ******************************************************************************************************************
plt.figure(figsize=(8, 5))
plt.step(
    t_baoab,
    laser_power_time,
    where="post",
    linewidth=1.0,
    label="stepwise laser power"
)
plt.axhline(P_laser, linestyle="--", label="nominal power")
plt.axhline(P_laser * (1 + laser_noise_fraction), linestyle=":", label="+1% level")
plt.axhline(P_laser * (1 - laser_noise_fraction), linestyle=":", label="-1% level")
plt.xlabel("Time / s")
plt.ylabel("Laser power / W")
plt.title("Stepwise laser-power noise")
plt.legend()
plt.grid()
plt.show()

# ******************************************************************************************************************
# Plot isolated effect of laser-power noise
# ******************************************************************************************************************
plt.figure(figsize=(8, 5))
plt.plot(
    t_baoab,
    z_baoab_constant_power * 1e6,
    label="BAOAB, constant laser power"
)
plt.plot(
    t_baoab,
    z_baoab * 1e6,
    linewidth=0.9,
    alpha=0.8,
    label="BAOAB, stepwise laser-power noise"
)
plt.axhline(z_eq * 1e6, linestyle=":", label="Equilibrium")
plt.xlabel("Time / s")
plt.ylabel("Height z / micrometres")
plt.title("Motion with and without laser-power noise")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    t_baoab,
    laser_noise_displacement_difference * 1e9,
    label="noisy power minus constant power"
)
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Time / s")
plt.ylabel("Displacement difference / nm")
plt.title("Isolated displacement effect of laser-power noise")
plt.legend()
plt.grid()
plt.show()

# ******************************************************************************************************************
# Extract solutions
# ******************************************************************************************************************
t = sol_nonlinear.t

z_nonlinear = sol_nonlinear.y[0]
v_nonlinear = sol_nonlinear.y[1]

x_linear = sol_linear.y[0]
v_linear = sol_linear.y[1]

# Convert linear displacement back to real position
z_linear = z_eq + x_linear

# ******************************************************************************************************************
# Plot nonlinear vs SHM comparison
# ******************************************************************************************************************
plt.figure(figsize=(8, 5))

plt.plot(t, z_nonlinear * 1e6, label="Nonlinear model")
#plt.plot(t, z_linear * 1e6, "--", label="Linear SHM approximation")
plt.plot(
    t_baoab,
    z_baoab * 1e6,
    linewidth=0.8,
    alpha=0.75,
    label="Nonlinear BAOAB Brownian model with laser noise"
)
plt.axhline(z_eq * 1e6, linestyle=":", label="Equilibrium")

plt.xlabel("Time / s")
plt.ylabel("Height z / micrometres")
plt.title("Separated optical forces vs linear SHM approximation")
plt.legend()
plt.grid()
plt.show()

# ******************************************************************************************************************
# Plot force curves
# ******************************************************************************************************************

# z_values = np.linspace(-150e-6, 500e-6, 1000)

# plt.figure(figsize=(8, 5))

# plt.plot(z_values * 1e6, F_scat(z_values), label="Scattering force")
# plt.plot(z_values * 1e6, F_grad(z_values), label="Gradient force")
# plt.plot(z_values * 1e6, F_photo(z_values, p), label="Photophoretic force")
# plt.plot(z_values * 1e6, F_total(z_values), label="Total upward force")

# plt.axhline(F_weight_effective, linestyle="--", label="Effective weight")
# plt.axhline(0, linewidth=0.8)

# plt.xlabel("z / micrometres")
# plt.ylabel("Force / N")
# plt.title("Optical and photophoretic forces")
# plt.legend()
# plt.grid()
# plt.show()
# plt.show()

# ******************************************************************************************************************
# Plot net force near equilibrium
# ******************************************************************************************************************
# plt.figure(figsize=(8, 5))

# plt.plot(z_values * 1e6, F_net(z_values), label="Nonlinear net force")

# # Linearised net force:
# # F_net ≈ -k (z - z_eq)
# F_net_linear = -k * (z_values - z_eq)

# plt.plot(z_values * 1e6, F_net_linear, "--", label="Linearised net force")
# plt.axvline(z_eq * 1e6, linestyle=":", label="Equilibrium")
# plt.axhline(0, linewidth=0.8)

# plt.xlabel("z / micrometres")
# plt.ylabel("Net force / N")
# plt.title("Nonlinear force compared with linear approximation")
# plt.legend()
# plt.grid()
# plt.show()

# ******************************************************************************************************************
#  Fourier Transform 
# ******************************************************************************************************************
# fft_mask = t > 0.2

# t_fft = t[fft_mask]
# z_signal = z_nonlinear[fft_mask] - z_eq

# z_signal = z_signal - np.mean(z_signal)

# dt_fft = t_fft[1] - t_fft[0]
# window = np.hanning(len(t_fft))

# z_fft = np.fft.rfft(z_signal * window)
# freqs = np.fft.rfftfreq(len(t_fft), dt_fft)
# z_amp = np.abs(z_fft)
# z_amp_norm = z_amp / np.max(z_amp)


z_signal = z_nonlinear - z_eq


dt_fft = t[1] - t[0]

z_signal = z_signal - np.mean(z_signal)


window = np.hanning(len(t))

z_fft = np.fft.rfft(z_signal * window)


freqs = np.fft.rfftfreq(len(t), dt_fft)

z_amp = np.abs(z_fft)
z_amp_norm = z_amp / np.max(z_amp)

sampling_frequency = 1 / dt_fft
nyquist_frequency = sampling_frequency / 2
frequency_bin_size = freqs[1] - freqs[0]
fft_plot_max = 100_000

print("FFT sampling frequency =", sampling_frequency, "Hz")
print("FFT Nyquist frequency =", nyquist_frequency, "Hz")
print("FFT frequency bin size =", frequency_bin_size, "Hz")

# The first FFT bin is 0 Hz, which is the DC/mean component rather than an
# oscillation frequency. Start the plot at the first non-zero bin instead.
positive_frequency_mask = freqs > 0
freqs_to_plot = freqs[positive_frequency_mask]
z_amp_norm_to_plot = z_amp_norm[positive_frequency_mask]
minimum_resolvable_frequency = freqs_to_plot[0]
minimum_plot_frequency = 0.95 * minimum_resolvable_frequency #np.ceil(minimum_resolvable_frequency)

print("Minimum resolvable non-zero frequency =", minimum_resolvable_frequency, "Hz")
print("Lower frequency shown on plot =", minimum_plot_frequency, "Hz")

# # Linear-amplitude Fourier spectrum
# plt.figure(figsize=(8, 5))
# plt.plot(freqs_to_plot, z_amp_norm_to_plot, label="z spectrum")
# plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
# #plt.axvline(minimum_resolvable_frequency, color="grey", linestyle=":", label="first non-zero bin")
# plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
# #plt.xscale("log")
# plt.xlim(2e0, fft_plot_max/10)
# plt.ylim(0, 1.05)
# plt.xlabel("Frequency / Hz")
# plt.ylabel("Normalized amplitude")
# plt.title("Fourier spectrum of 1D motion, linear scale")
# plt.legend()
# plt.grid(True, which="both")
# plt.show()

# Log-amplitude Fourier spectrum
# plt.figure(figsize=(8, 5))
# plt.semilogy(freqs_to_plot, z_amp_norm_to_plot, label="z spectrum")
# plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
# #plt.axvline(minimum_resolvable_frequency, color="grey", linestyle=":", label="first non-zero bin")
# plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
# plt.xscale("log")
# plt.xlim(2e0, fft_plot_max/10)
# plt.ylim(1e-12, 1.2)
# plt.xlabel("Frequency / Hz")
# plt.ylabel("Normalized amplitude")
# plt.title("Fourier spectrum of 1D motion")
# plt.legend()
# plt.grid(True, which="both")
# plt.show()

# ******************************************************************************************************************
#  BAOAB Brownian Fourier analysis
# ******************************************************************************************************************


def normalised_fft(signal, time_array):
    """
    Return positive-frequency normalized FFT amplitude.
    """
    signal = signal - np.mean(signal)
    dt_local = time_array[1] - time_array[0]
    window_local = np.hanning(len(time_array))

    fft_values = np.fft.rfft(signal * window_local)
    freqs_local = np.fft.rfftfreq(len(time_array), dt_local)
    amp = np.abs(fft_values)

    if np.max(amp) == 0:
        raise ValueError("FFT amplitude is zero everywhere.")

    amp_norm = amp / np.max(amp)
    positive_mask = (freqs_local > 0) & np.isfinite(amp_norm) & (amp_norm > 0)

    return freqs_local[positive_mask], amp_norm[positive_mask], dt_local


# Full BAOAB trajectory FFT. This includes both the initial condition and the
# later Brownian steady-state motion.
baoab_freqs, baoab_amp_norm, dt_fft_baoab = normalised_fft(
    z_baoab - z_eq,
    t_baoab,
)

baoab_nyquist = 0.5 / dt_fft_baoab
baoab_fft_max = min(fft_plot_max / 10, baoab_nyquist)

plt.figure(figsize=(8, 5))
plt.semilogy(baoab_freqs, baoab_amp_norm, label="BAOAB full trajectory")
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.xscale("log")
plt.xlim(max(baoab_freqs[0] * 0.95, 1e-12), baoab_fft_max)
plt.ylim(1e-12, 1.2)
plt.xlabel("Frequency / Hz")
plt.ylabel("Normalized amplitude")
plt.title("BAOAB full-trajectory Fourier spectrum")
plt.legend()
plt.grid(True, which="both")
plt.show()

if np.max(np.abs(laser_noise_displacement_difference)) > 0:
    laser_effect_freqs, laser_effect_amp_norm, dt_fft_laser_effect = normalised_fft(
        laser_noise_displacement_difference,
        t_baoab,
    )

    laser_effect_nyquist = 0.5 / dt_fft_laser_effect
    laser_effect_fft_max = min(fft_plot_max / 10, laser_effect_nyquist)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        laser_effect_freqs,
        laser_effect_amp_norm,
        label="laser-noise displacement effect"
    )
    plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
    plt.xscale("log")
    plt.xlim(max(laser_effect_freqs[0] * 0.95, 1e-12), laser_effect_fft_max)
    plt.ylim(1e-12, 1.2)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Normalized amplitude")
    plt.title("Fourier spectrum of isolated laser-noise effect")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    laser_effect_signal = laser_noise_displacement_difference - np.mean(
        laser_noise_displacement_difference
    )
    laser_effect_sampling_frequency = 1 / dt_fft_laser_effect
    laser_effect_target_bin_width = max(1.0, (omega / (2*np.pi)) / 20)
    laser_effect_nperseg = min(
        len(laser_effect_signal),
        int(np.ceil(laser_effect_sampling_frequency / laser_effect_target_bin_width))
    )
    laser_effect_nperseg = max(256, laser_effect_nperseg)

    laser_effect_psd_freqs, laser_effect_psd = welch(
        laser_effect_signal,
        fs=laser_effect_sampling_frequency,
        window="hann",
        nperseg=laser_effect_nperseg,
        noverlap=laser_effect_nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    laser_effect_psd_mask = (
        (laser_effect_psd_freqs > 0)
        & np.isfinite(laser_effect_psd)
        & (laser_effect_psd > 0)
    )

    print(
        "Laser-noise effect PSD frequency bin size =",
        laser_effect_sampling_frequency / laser_effect_nperseg,
        "Hz"
    )

    plt.figure(figsize=(8, 5))
    plt.loglog(
        laser_effect_psd_freqs[laser_effect_psd_mask],
        laser_effect_psd[laser_effect_psd_mask] * 1e18,
        label="laser-noise displacement effect PSD"
    )
    plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
    plt.xlim(
        max(laser_effect_psd_freqs[laser_effect_psd_mask][0] * 0.95, 1e-12),
        laser_effect_fft_max
    )
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / nm$^2$/Hz")
    plt.title("PSD of isolated laser-noise effect")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


# Early BAOAB FFT. This is the one to inspect when displacement is large and
# you expect coherent oscillation peaks/harmonics like the deterministic model.
early_fft_duration = min(0.03, t_end)
early_mask = t_baoab <= (t_start + early_fft_duration)

t_early = t_baoab[early_mask]
z_early = z_baoab[early_mask] - z_eq

early_freqs, early_amp_norm, dt_fft_early = normalised_fft(z_early, t_early)
early_nyquist = 0.5 / dt_fft_early
early_fft_max = min(fft_plot_max / 10, early_nyquist)

print("Early BAOAB FFT duration =", t_early[-1] - t_early[0], "s")
print("Early BAOAB FFT frequency bin size =", early_freqs[1] - early_freqs[0], "Hz")

plt.figure(figsize=(8, 5))
plt.semilogy(early_freqs, early_amp_norm, label="BAOAB early transient")
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third fz")
plt.xscale("log")
plt.xlim(max(early_freqs[0] * 0.95, 1e-12), early_fft_max)
plt.ylim(1e-12, 1.2)
plt.xlabel("Frequency / Hz")
plt.ylabel("Normalized amplitude")
plt.title("BAOAB early-transient Fourier spectrum")
plt.legend()
plt.grid(True, which="both")
plt.show()


# Late-time Welch PSD. This is the one to inspect when displacement = 0 and
# you expect the Brownian damped-harmonic-oscillator thermal spectrum.
settled_mask = t_baoab > (0.25 * t_end)

if np.count_nonzero(settled_mask) < 1024:
    settled_mask = np.ones_like(t_baoab, dtype=bool)

z_psd_signal = z_baoab[settled_mask] - z_eq
z_psd_signal = z_psd_signal - np.mean(z_psd_signal)
dt_psd = t_baoab[1] - t_baoab[0]
sampling_frequency_psd = 1 / dt_psd

target_psd_bin_width = max(1.0, (omega / (2*np.pi)) / 20)
nperseg = min(len(z_psd_signal), int(np.ceil(sampling_frequency_psd / target_psd_bin_width)))
nperseg = max(256, nperseg)

psd_freqs, z_psd = welch(
    z_psd_signal,
    fs=sampling_frequency_psd,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg // 2,
    detrend="constant",
    scaling="density",
)

omega_psd = 2 * np.pi * psd_freqs
z_psd_theory = (
    4 * kB * T * b
    / ((k - m * omega_psd**2)**2 + (b * omega_psd)**2)
)

positive_psd_mask = (
    (psd_freqs > 0)
    & np.isfinite(z_psd)
    & np.isfinite(z_psd_theory)
    & (z_psd > 0)
    & (z_psd_theory > 0)
)

print("Welch PSD nperseg =", nperseg)
print("Welch PSD frequency bin size =", sampling_frequency_psd / nperseg, "Hz")

plt.figure(figsize=(8, 5))
plt.semilogy(
    psd_freqs[positive_psd_mask],
    z_psd[positive_psd_mask] * 1e18,
    label="BAOAB late-time PSD"
)
plt.semilogy(
    psd_freqs[positive_psd_mask],
    z_psd_theory[positive_psd_mask] * 1e18,
    "--",
    label="linear theory PSD"
)
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.xscale("log")
plt.xlim(max(psd_freqs[positive_psd_mask][0] * 0.95, 1e-12), baoab_fft_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("BAOAB late-time Brownian PSD")
plt.legend()
plt.grid(True, which="both")
plt.show()
