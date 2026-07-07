import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# -----------------------------
# Constants
# -----------------------------
g = 9.81

# -----------------------------
# Particle properties
# -----------------------------
radius = 5e-6          # m
density = 2200         # kg/m^3

volume = (4/3) * np.pi * radius**3
m = density * volume

print("Particle mass =", m, "kg")
print("Weight mg =", m*g, "N")

# -----------------------------
# Gas properties
# -----------------------------
p = 101325        # pressure, Pa
T = 293           # temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
R = 8.314         # gas constant, J/(mol K)
rho_g = p * M_air / (R * T)
kB = 1.38e-23
d_air = 3.7e-10

lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

# -----------------------------
# Laser / force parameters
# -----------------------------
zR = 100e-6             # axial length scale, m

# Force amplitudes relative to particle weight
Fs0 = 2.5 * m * g      # upward scattering force strength
Fg0 = 1.0 * m * g      # gradient force strength

def F_scat(z):
    """
    Upward scattering/radiation-pressure force.
    """
    return Fs0 / (1 + (z / zR)**2)

def F_grad(z):
    """
    Gradient force pulling particle back toward the focus at z = 0.

    For z > 0, this is negative.
    For z < 0, this is positive.
    """
    return -Fg0 * (z / zR) / (1 + (z / zR)**2)**2


# -----------------------------
# Photophoretic force parameters
# -----------------------------
         # gas pressure, Pa
p=3000
p_max = 3000      # pressure where photophoretic force is strongest, Pa

Fph_max = 1.0 * m * g   # maximum photophoretic force, N

z_focus = 0.0     # laser focus position, m

def pressure_factor(p):
    """
    Pressure dependence of photophoretic force.
    This equals 1 when p = p_max.
    """
    return 2 / ((p / p_max) + (p_max / p))

def intensity_shape(z):
    """
    Simple 1D laser intensity profile along z.
    Maximum at the focus.
    """
    z_rel = z - z_focus
    return 1 / (1 + (z_rel / zR)**2)

def F_photo(z, p):
    """
    Photophoretic force.

    This simple model assumes the photophoretic force is:
    - proportional to laser intensity,
    - pressure dependent,
    - upward in the +z direction.

    For a first model, the sign is chosen positive.
    """
    return Fph_max * pressure_factor(p) * intensity_shape(z)




def F_total(z):
    """
    Total vertical optical force.
    """
    return F_scat(z) + F_grad(z) + F_photo(z, p)

def F_net(z):
    """
    Net force excluding damping.
    Equilibrium occurs when F_net = 0.
    """
    return F_total(z) - m*g



#*****************************************************PRESSURE TESTING ***************************************************
# pressures = np.logspace(1, 5, 200)  # 10 Pa to 100,000 Pa


# def numerical_derivative(func, z, h=1e-9):
#     return (func(z + h) - func(z - h)) / (2*h)

# z_eq_values = []
# freq_values = []

# for p_i in pressures:
#     def F_total_p(z):
#         return F_scat(z) + F_grad(z) + F_photo(z, p_i)

#     def F_net_p(z):
#         return F_total_p(z) - m*g

#     z_scan = np.linspace(-300e-6, 300e-6, 3000)
#     F_scan = F_net_p(z_scan)

#     roots = []

#     for i in range(len(z_scan) - 1):
#         if F_scan[i] * F_scan[i + 1] < 0:
#             root = brentq(F_net_p, z_scan[i], z_scan[i + 1])

#             slope = numerical_derivative(F_total_p, root)

#             if slope < 0:  # stable equilibrium
#                 roots.append(root)

#     if len(roots) > 0:
#         z_eq = roots[0]
#         k = -numerical_derivative(F_total_p, z_eq)
#         frequency = np.sqrt(k / m) / (2*np.pi)

#         z_eq_values.append(z_eq)
#         freq_values.append(frequency)
#     else:
#         z_eq_values.append(np.nan)
#         freq_values.append(np.nan)

# plt.figure(figsize=(8, 5))
# plt.semilogx(pressures, np.array(z_eq_values) * 1e6)
# plt.xlabel("Pressure / Pa")
# plt.ylabel("Stable equilibrium z / micrometres")
# plt.title("Equilibrium position vs pressure")
# plt.grid()
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.semilogx(pressures, freq_values)
# plt.xlabel("Pressure / Pa")
# plt.ylabel("Small-oscillation frequency / Hz")
# plt.title("Trap frequency vs pressure")
# plt.grid()
# plt.show()











#******************************************************************************************************************

# -----------------------------
# Find equilibrium position
# -----------------------------
# We look for z_eq where:
#
# F_scat(z_eq) + F_grad(z_eq) +F_photo(z_eg) = mg

z_min = -500e-6
z_max = 500e-6

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

# -----------------------------
# Linear SHM approximation
# -----------------------------
# Near equilibrium:
#
# F_total(z) - mg ≈ -k (z - z_eq)
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

# -----------------------------
# Damping
# -----------------------------
#b = 6 * np.pi * eta * radius   # kg/s
#print("b is", b)
#b=1.7e-10
damping_ratio = 0.1   # < 1 gives oscillations
b = damping_ratio * 2 * np.sqrt(m * k)


# -----------------------------
# Nonlinear separated-force equation
# -----------------------------

def nonlinear_system(t, Y):
    z = Y[0]
    v = Y[1]

    dzdt = v

    Fopt = F_scat(z) + F_grad(z)
    Fph =F_photo(z, p)
    Fdrag = -b * v

    dvdt = (Fopt + Fph - m*g + Fdrag) / m

    return [dzdt, dvdt]
# -----------------------------
# Linear SHM equation
# -----------------------------
def linear_system(t, Y):
    x = Y[0]   # displacement from equilibrium
    v = Y[1]

    dxdt = v
    dvdt = -(b/m)*v - (k/m)*x

    return [dxdt, dvdt]

# -----------------------------
# Initial conditions
# -----------------------------
# Start slightly away from equilibrium
displacement = 10e-6

z0 = z_eq + displacement
v0 = 0.0

x0 = z0 - z_eq

Y0_nonlinear = [z0, v0]
Y0_linear = [x0, v0]

# -----------------------------
# Time range
# -----------------------------
t_start = 0
t_end = 0.20
t_eval = np.linspace(t_start, t_end, 4000)

# -----------------------------
# Solve nonlinear model
# -----------------------------
sol_nonlinear = solve_ivp(
    nonlinear_system,
    [t_start, t_end],
    Y0_nonlinear,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
    max_step=1e-4
)

# -----------------------------
# Solve linear SHM approximation
# -----------------------------
sol_linear = solve_ivp(
    linear_system,
    [t_start, t_end],
    Y0_linear,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
    max_step=1e-4
)

# -----------------------------
# Extract solutions
# -----------------------------
t = sol_nonlinear.t

z_nonlinear = sol_nonlinear.y[0]
v_nonlinear = sol_nonlinear.y[1]

x_linear = sol_linear.y[0]
v_linear = sol_linear.y[1]

# Convert linear displacement back to real position
z_linear = z_eq + x_linear

# -----------------------------
# Plot nonlinear vs SHM comparison
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(t, z_nonlinear * 1e6, label="Nonlinear separated-force model")
plt.plot(t, z_linear * 1e6, "--", label="Linear SHM approximation")
plt.axhline(z_eq * 1e6, linestyle=":", label="Equilibrium")

plt.xlabel("Time / s")
plt.ylabel("Height z / micrometres")
plt.title("Separated optical forces vs linear SHM approximation")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Plot force curves
# -----------------------------

z_values = np.linspace(-150e-6, 250e-6, 1000)

plt.figure(figsize=(8, 5))

plt.plot(z_values * 1e6, F_scat(z_values), label="Scattering force")
plt.plot(z_values * 1e6, F_grad(z_values), label="Gradient force")
plt.plot(z_values * 1e6, F_photo(z_values, p), label="Photophoretic force")
plt.plot(z_values * 1e6, F_total(z_values), label="Total upward force")

plt.axhline(m*g, linestyle="--", label="Gravity mg")
plt.axhline(0, linewidth=0.8)

plt.xlabel("z / micrometres")
plt.ylabel("Force / N")
plt.title("Optical and photophoretic forces")
plt.legend()
plt.grid()
plt.show()
plt.show()

# -----------------------------
# Plot net force near equilibrium
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(z_values * 1e6, F_net(z_values), label="Nonlinear net force")

# Linearised net force:
# F_net ≈ -k (z - z_eq)
F_net_linear = -k * (z_values - z_eq)

plt.plot(z_values * 1e6, F_net_linear, "--", label="Linearised net force")
plt.axvline(z_eq * 1e6, linestyle=":", label="Equilibrium")
plt.axhline(0, linewidth=0.8)

plt.xlabel("z / micrometres")
plt.ylabel("Net force / N")
plt.title("Nonlinear force compared with linear approximation")
plt.legend()
plt.grid()
plt.show()