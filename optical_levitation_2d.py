import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ****************************************************************************************************************************************************
# Constants
# ****************************************************************************************************************************************************
g = 9.81

# ****************************************************************************************************************************************************
# Particle properties
# ****************************************************************************************************************************************************
radius = 5e-6          # m
density = 2200         # kg/m^3

volume = (4/3) * np.pi * radius**3
m = density * volume

print("Particle mass =", m, "kg")
print("Weight mg =", m*g, "N")

# ****************************************************************************************************************************************************
# Gas properties
# ****************************************************************************************************************************************************
p = 100          # pressure, Pa
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

# Force amplitudes relative to particle weight
Fs0 = 2.5 * m * g       # upward scattering force strength
Fg0 = 1.0 * m * g       # old 1D gradient-force strength

# Calibrated so that on-axis Fz_grad(x=0, z) matches the old 1D gradient force.
alpha_grad = Fg0 * zR / 2

print("alpha_grad =", alpha_grad, "N m")


def beam_width(z):
    """
    Gaussian-beam-like width.
    """
    return w0 * np.sqrt(1 + (z / zR)**2)


def intensity(x, z):
    """
    Dimensionless 2D intensity profile.

    On-axis this reduces to:
        I(0, z) = 1 / (1 + (z/zR)^2)

    so the scattering force matches the old 1D model along x = 0.
    """
    s = 1 + (z / zR)**2
    return (1 / s) * np.exp(-2 * x**2 / (w0**2 * s))


def grad_intensity(x, z):
    """
    Analytic gradient of the dimensionless intensity.
    Returns dI/dx and dI/dz.
    """
    s = 1 + (z / zR)**2
    I = intensity(x, z)

    dIdx = I * (-4 * x / (w0**2 * s))

    dsdz = 2 * z / zR**2
    dlogI_dz = dsdz * (-1 / s + 2 * x**2 / (w0**2 * s**2))
    dIdz = I * dlogI_dz

    return dIdx, dIdz


def F_grad_2d(x, z):
    """
    Gradient force pulling the particle toward higher intensity.
    """
    dIdx, dIdz = grad_intensity(x, z)
    return alpha_grad * dIdx, alpha_grad * dIdz


def F_scat_2d(x, z):
    """
    Scattering/radiation-pressure force.
    In this simple model it points only upward, in +z.
    """
    return 0.0, Fs0 * intensity(x, z)


# ****************************************************************************************************************************************************
# Photophoretic force parameters
# ****************************************************************************************************************************************************
p_max = 3000      # pressure where photophoretic force is strongest, Pa
Fph_max = 1.0 * m * g


def pressure_factor(p):
    """
    Pressure dependence of photophoretic force.
    This equals 1 when p = p_max.
    """
    return 2 / ((p / p_max) + (p_max / p))


def F_photo_2d(x, z):
    """
    Photophoretic force.
    This simple model points only upward and is proportional to intensity.
    """
    return 0.0, Fph_max * pressure_factor(p) * intensity(x, z)


def F_optical_2d(x, z):
    """
    Total optical/photophoretic force, excluding gravity and damping.
    """
    Fx_grad, Fz_grad = F_grad_2d(x, z)
    Fx_scat, Fz_scat = F_scat_2d(x, z)
    Fx_photo, Fz_photo = F_photo_2d(x, z)

    Fx = Fx_grad + Fx_scat + Fx_photo
    Fz = Fz_grad + Fz_scat + Fz_photo

    return Fx, Fz


def Fz_net_on_axis(z):
    """
    Net vertical force along x = 0, excluding damping.
    """
    _, Fz = F_optical_2d(0.0, z)
    return Fz - m*g


# ****************************************************************************************************************************************************
# Find on-axis equilibrium
# ****************************************************************************************************************************************************
z_min = -500e-6
z_max = 500e-6

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
def cunningham_correction(Kn):
    A = 1.257
    B = 0.4
    C = 1.1
    return 1 + Kn * (A + B * np.exp(-C / Kn))

def damping_coefficient_stokes_cunningham(p):
    lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
    Kn = lambda_mfp / radius

    Cc = cunningham_correction(Kn)
    b = 6 * np.pi * eta * radius / Cc

    return b

b = damping_coefficient_stokes_cunningham(p)

print("Pressure-dependent damping b =", b, "kg/s")
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
# Initial conditions
# ****************************************************************************************************************************************************
x_displacement = 14e-6
z_displacement = 100e-6

x0 = x_eq + x_displacement
z0 = z_eq + z_displacement
vx0 = 0.0
vz0 = 0.0

Y0 = [x0, z0, vx0, vz0]

# ****************************************************************************************************************************************************
# Time range
# ****************************************************************************************************************************************************
t_start = 0
t_end = 1.0
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

t = sol_2d.t
x_det = sol_2d.y[0]
z_det = sol_2d.y[1]
vx_det = sol_2d.y[2]
vz_det = sol_2d.y[3]



# ****************************************************************************************************************************************************
# Brownian / Langevin trajectory
# ****************************************************************************************************************************************************
rng = np.random.default_rng(seed=4)

dt = 1e-6
t_brownian = np.arange(t_start, t_end + dt, dt)

x_brownian = np.zeros_like(t_brownian)
z_brownian = np.zeros_like(t_brownian)
vx_brownian = np.zeros_like(t_brownian)
vz_brownian = np.zeros_like(t_brownian)

x_brownian[0] = x0
z_brownian[0] = z0
vx_brownian[0] = vx0
vz_brownian[0] = vz0

sigma_v = np.sqrt(2 * b * kB * T / m**2)

for i in range(len(t_brownian) - 1):
    x_i = x_brownian[i]
    z_i = z_brownian[i]
    vx_i = vx_brownian[i]
    vz_i = vz_brownian[i]

    Fx, Fz = F_optical_2d(x_i, z_i)

    Fx_det = Fx - b * vx_i
    Fz_det = Fz - m*g - b * vz_i

    vx_brownian[i + 1] = (
        vx_i
        + (Fx_det / m) * dt
        + sigma_v * np.sqrt(dt) * rng.normal()
    )
    vz_brownian[i + 1] = (
        vz_i
        + (Fz_det / m) * dt
        + sigma_v * np.sqrt(dt) * rng.normal()
    )

    x_brownian[i + 1] = x_i + vx_brownian[i + 1] * dt
    z_brownian[i + 1] = z_i + vz_brownian[i + 1] * dt

settled = t_brownian > (0.5 * t_end)

print(
    "Late-time simulated x Brownian RMS =",
    np.std(x_brownian[settled] - x_eq) * 1e6,
    "micrometres"
)
print(
    "Late-time simulated z Brownian RMS =",
    np.std(z_brownian[settled] - z_eq) * 1e6,
    "micrometres"
)

# ****************************************************************************************************************************************************
# Plot x(t) and z(t)
# ****************************************************************************************************************************************************
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(t, (x_det - x_eq) * 1e6, label="deterministic")
axes[0].plot(t_brownian,(x_brownian - x_eq) * 1e6,linewidth=0.8,alpha=0.75,label="with Brownian motion")
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x displacement / micrometres")
axes[0].legend()
axes[0].grid()

axes[1].plot(t, (z_det - z_eq) * 1e6, label="deterministic")
axes[1].plot(t_brownian,(z_brownian - z_eq) * 1e6,linewidth=0.8,alpha=0.75,label="with Brownian motion")
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_xlabel("Time / s")
axes[1].set_ylabel("z displacement / micrometres")
axes[1].legend()
axes[1].grid()

fig.suptitle("2D optical levitation motion")
plt.tight_layout()
plt.show()

#****************************************************************************************************************************************************
#Fourier Tranform 
#****************************************************************************************************************************************************
x_signal = x_det - x_eq
z_signal = z_det - z_eq

dt_fft = t[1] - t[0]

x_signal = x_signal - np.mean(x_signal)
z_signal = z_signal - np.mean(z_signal)

window = np.hanning(len(t))

x_fft = np.fft.rfft(x_signal * window)
z_fft = np.fft.rfft(z_signal * window)

freqs = np.fft.rfftfreq(len(t), dt_fft)

x_amp = np.abs(x_fft)
z_amp = np.abs(z_fft)

plt.figure(figsize=(8, 5))
plt.plot(freqs, x_amp / np.max(x_amp), label="x spectrum")
plt.plot(freqs, z_amp / np.max(z_amp), label="z spectrum")
plt.axvline(omega_x / (2*np.pi), linestyle="--", label="linear fx")
plt.axvline(omega_z / (2*np.pi), linestyle="--", label="linear fz")
plt.xlim(0, 300)
plt.xlabel("Frequency / Hz")
plt.ylabel("Normalized amplitude")
plt.title("Fourier spectrum of 2D motion")
plt.legend()
plt.grid()
plt.show()


# ****************************************************************************************************************************************************
# Plot 2D trajectory
# ****************************************************************************************************************************************************
plt.figure(figsize=(7, 6))

plt.plot(x_det * 1e6,z_det * 1e6,label="deterministic trajectory")
plt.plot(x_brownian * 1e6,z_brownian * 1e6,linewidth=0.8,alpha=0.75,label="Brownian trajectory")
plt.scatter([x_eq * 1e6], [z_eq * 1e6], color="black", s=30, label="equilibrium")

plt.xlabel("x / micrometres")
plt.ylabel("z / micrometres")
plt.title("2D trajectory in the x-z plane")
plt.axis("equal")
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
