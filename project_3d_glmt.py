import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


# =====================================================================
# Constants
# =====================================================================
g = 9.81
kB = 1.380649e-23
R = 8.314462618
c_light = 299792458


# =====================================================================
# Particle properties
# =====================================================================
radius = 5e-6          # m
density = 2200         # kg/m^3

# Needed by a GLMT force calculator.
particle_refractive_index = 1.46 + 0.0j
medium_refractive_index = 1.00027

volume = (4 / 3) * np.pi * radius**3
m = density * volume

print("Particle mass =", m, "kg")
print("Weight mg =", m * g, "N")


# =====================================================================
# Gas properties
# =====================================================================
p = 3000          # pressure, Pa
T = 293           # temperature, K
eta = 1.8e-5      # dynamic viscosity of air, Pa s
M_air = 0.029     # molar mass of air, kg/mol
d_air = 3.7e-10

rho_g = p * M_air / (R * T)
lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
Kn = lambda_mfp / radius

print("Gas density =", rho_g, "kg/m^3")
print("Mean free path =", lambda_mfp, "m")
print("Knudsen number =", Kn)


# =====================================================================
# Laser / beam properties
# =====================================================================
wavelength = 532e-9
w0 = 20e-6
zR = np.pi * w0**2 / wavelength
P_laser = 0.1
polarization = "linear-x"
numerical_aperture = None

I0 = 2 * P_laser / (np.pi * w0**2)

print("Wavelength =", wavelength, "m")
print("Beam waist w0 =", w0, "m")
print("Rayleigh range zR =", zR, "m")
print("Laser power =", P_laser, "W")
print("Peak Gaussian intensity I0 =", I0, "W/m^2")


def intensity_shape(x, y, z):
    """
    Dimensionless Gaussian beam intensity shape.

    This is still used for photophoresis. The optical gradient/scattering
    force is not calculated from this function in GLMT mode.
    """
    s = 1 + (z / zR)**2
    return (1 / s) * np.exp(-2 * (x**2 + y**2) / (w0**2 * s))


# =====================================================================
# GLMT optical force interface
# =====================================================================
# The intended workflow is:
#
# 1. Use a GLMT code/toolbox to compute the total optical force
#    F_opt = (Fx, Fy, Fz) on a regular x-y-z grid.
# 2. Save the grid as an .npz file containing:
#
#        x_grid, y_grid, z_grid, Fx, Fy, Fz
#
#    where Fx, Fy, Fz have shape:
#
#        (len(x_grid), len(y_grid), len(z_grid))
#
# 3. This simulation loads that table and interpolates F_opt during the ODE.
#
# This replaces the old split:
#
#        F_grad + F_scat
#
# with:
#
#        F_opt_GLMT(x, y, z)

glmt_force_table_path = "outputs/glmt_force_table.npz"


class GLMTForceTable:
    def __init__(self, path):
        data = np.load(path)

        self.x_grid = data["x_grid"]
        self.y_grid = data["y_grid"]
        self.z_grid = data["z_grid"]

        self.Fx_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid),
            data["Fx"],
            bounds_error=False,
            fill_value=None,
        )
        self.Fy_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid),
            data["Fy"],
            bounds_error=False,
            fill_value=None,
        )
        self.Fz_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid),
            data["Fz"],
            bounds_error=False,
            fill_value=None,
        )

    def force(self, x, y, z):
        xb, yb, zb = np.broadcast_arrays(x, y, z)
        points = np.column_stack([xb.ravel(), yb.ravel(), zb.ravel()])

        Fx = self.Fx_interp(points)
        Fy = self.Fy_interp(points)
        Fz = self.Fz_interp(points)

        shape = xb.shape
        if shape == ():
            return float(Fx[0]), float(Fy[0]), float(Fz[0])

        return Fx.reshape(shape), Fy.reshape(shape), Fz.reshape(shape)


def make_placeholder_force_table(path=glmt_force_table_path):
    """
    Creates a file with the correct grid format.

    The force values are zeros. Replace Fx, Fy, and Fz with values computed
    from a real GLMT code before running a physical simulation.
    """
    x_grid = np.linspace(-80e-6, 80e-6, 41)
    y_grid = np.linspace(-80e-6, 80e-6, 41)
    z_grid = np.linspace(-2e-3, 4e-3, 101)

    shape = (len(x_grid), len(y_grid), len(z_grid))
    Fx = np.zeros(shape)
    Fy = np.zeros(shape)
    Fz = np.zeros(shape)

    np.savez(
        path,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        Fx=Fx,
        Fy=Fy,
        Fz=Fz,
    )


if "--make-template" in sys.argv:
    make_placeholder_force_table()
    print("Wrote placeholder GLMT force table to", glmt_force_table_path)
    print("Replace Fx, Fy, and Fz with real GLMT-computed forces before use.")
    raise SystemExit


try:
    glmt_force = GLMTForceTable(glmt_force_table_path)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        "No GLMT force table was found. Create one at "
        f"{glmt_force_table_path!r} with arrays x_grid, y_grid, z_grid, "
        "Fx, Fy, and Fz. You can call make_placeholder_force_table() to "
        "create the required file format, but the placeholder contains zero "
        "optical force and is not physical."
    ) from exc


def F_optical_glmt(x, y, z):
    """
    Total optical force from GLMT.

    This replaces the previous phenomenological gradient and scattering
    forces. It should include all optical momentum-transfer effects:
    gradient-like restoring force, radiation pressure, aberration effects,
    polarization effects, and Mie resonances if present in the GLMT table.
    """
    return glmt_force.force(x, y, z)


# =====================================================================
# Photophoretic force
# =====================================================================
k_particle = 1.4
alpha_acc = 1.0
kappa_t = 1.14
absorption_fraction = 1e-4


def mean_thermal_speed():
    return np.sqrt(8 * R * T / (np.pi * M_air))


def photophoretic_D():
    c_bar = mean_thermal_speed()
    return (np.pi * c_bar * eta / (2 * T)) * np.sqrt(np.pi * kappa_t / 3)


def photophoretic_p_max():
    D = photophoretic_D()
    return (3 * T / (np.pi * radius)) * D * np.sqrt(2 / alpha_acc)


def absorbed_intensity(x, y, z):
    return absorption_fraction * I0 * intensity_shape(x, y, z)


def photophoretic_force_magnitude(x, y, z):
    D = photophoretic_D()
    p_max_ph = photophoretic_p_max()
    I_abs = absorbed_intensity(x, y, z)

    F_max = (
        0.5
        * radius**2
        * D
        * np.sqrt(alpha_acc / 2)
        * I_abs
        / k_particle
    )

    return 2 * F_max / ((p / p_max_ph) + (p_max_ph / p))


def F_photo(x, y, z):
    return 0.0, 0.0, photophoretic_force_magnitude(x, y, z)


# =====================================================================
# Total non-drag force
# =====================================================================
def F_total_non_drag(x, y, z):
    Fx_opt, Fy_opt, Fz_opt = F_optical_glmt(x, y, z)
    Fx_ph, Fy_ph, Fz_ph = F_photo(x, y, z)

    return Fx_opt + Fx_ph, Fy_opt + Fy_ph, Fz_opt + Fz_ph


def Fz_net_on_axis(z):
    _, _, Fz = F_total_non_drag(0.0, 0.0, z)
    return Fz - m * g


# =====================================================================
# Equilibrium search
# =====================================================================
z_min = -2e-3
z_max = 4e-3

z_scan = np.linspace(z_min, z_max, 3000)
F_scan = Fz_net_on_axis(z_scan)

roots = []
for i in range(len(z_scan) - 1):
    if F_scan[i] * F_scan[i + 1] < 0:
        root = brentq(
            Fz_net_on_axis,
            z_scan[i],
            z_scan[i + 1],
            xtol=1e-15,
            rtol=1e-15,
        )
        roots.append(root)

print("On-axis equilibrium positions / micrometres:")
for root in roots:
    print(root * 1e6)


def numerical_derivative_1d(func, z, h=1e-9):
    return (func(z + h) - func(z - h)) / (2 * h)


stable_roots = []
for root in roots:
    slope = numerical_derivative_1d(Fz_net_on_axis, root)
    if slope < 0:
        stable_roots.append(root)

if len(stable_roots) == 0:
    raise ValueError("No stable on-axis equilibrium found in the GLMT force table.")

x_eq = 0.0
y_eq = 0.0
z_eq = stable_roots[0]

print("Chosen equilibrium x =", x_eq * 1e6, "micrometres")
print("Chosen equilibrium y =", y_eq * 1e6, "micrometres")
print("Chosen equilibrium z =", z_eq * 1e6, "micrometres")


# =====================================================================
# Local spring constants
# =====================================================================
def Fx_at_x(x):
    Fx, _, _ = F_total_non_drag(x, y_eq, z_eq)
    return Fx


def Fy_at_y(y):
    _, Fy, _ = F_total_non_drag(x_eq, y, z_eq)
    return Fy


def Fz_net_at_z(z):
    _, _, Fz = F_total_non_drag(x_eq, y_eq, z)
    return Fz - m * g


kx = -numerical_derivative_1d(Fx_at_x, x_eq)
ky = -numerical_derivative_1d(Fy_at_y, y_eq)
kz = -numerical_derivative_1d(Fz_net_at_z, z_eq)

if kx <= 0 or ky <= 0 or kz <= 0:
    raise ValueError("The chosen equilibrium is not stable in all directions.")

omega_x = np.sqrt(kx / m)
omega_y = np.sqrt(ky / m)
omega_z = np.sqrt(kz / m)

print("kx =", kx, "N/m")
print("ky =", ky, "N/m")
print("kz =", kz, "N/m")
print("fx =", omega_x / (2 * np.pi), "Hz")
print("fy =", omega_y / (2 * np.pi), "Hz")
print("fz =", omega_z / (2 * np.pi), "Hz")


# =====================================================================
# Damping
# =====================================================================
def cunningham_correction(Kn):
    A = 1.257
    B = 0.4
    C = 1.1
    return 1 + Kn * (A + B * np.exp(-C / Kn))


def damping_coefficient_stokes_cunningham(p):
    lambda_mfp = kB * T / (np.sqrt(2) * np.pi * d_air**2 * p)
    Kn = lambda_mfp / radius
    Cc = cunningham_correction(Kn)
    return 6 * np.pi * eta * radius / Cc


b = damping_coefficient_stokes_cunningham(p)
print("Pressure-dependent damping b =", b, "kg/s")


# =====================================================================
# Deterministic equations of motion
# =====================================================================
def nonlinear_system_3d(t, Y):
    x, y, z, vx, vy, vz = Y

    Fx, Fy, Fz = F_total_non_drag(x, y, z)

    ax = (Fx - b * vx) / m
    ay = (Fy - b * vy) / m
    az = (Fz - m * g - b * vz) / m

    return [vx, vy, vz, ax, ay, az]


# =====================================================================
# Initial conditions
# =====================================================================
x_displacement = 0.3e-6
y_displacement = -0.5e-6
z_displacement = 0.7e-6

x0 = x_eq + x_displacement
y0 = y_eq + y_displacement
z0 = z_eq + z_displacement
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0

Y0 = [x0, y0, z0, vx0, vy0, vz0]


# =====================================================================
# Time integration
# =====================================================================
t_start = 0
t_end = 0.2
t_eval = np.linspace(t_start, t_end, 10000)

sol = solve_ivp(
    nonlinear_system_3d,
    [t_start, t_end],
    Y0,
    t_eval=t_eval,
    rtol=1e-10,
    atol=[1e-14, 1e-14, 1e-14, 1e-12, 1e-12, 1e-12],
    max_step=1e-4,
)

t = sol.t
x_det = sol.y[0]
y_det = sol.y[1]
z_det = sol.y[2]


# =====================================================================
# Plots
# =====================================================================
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

axes[0].plot(t, (x_det - x_eq) * 1e6)
axes[0].axhline(0, linestyle=":", color="black")
axes[0].set_ylabel("x displacement / micrometres")
axes[0].grid()

axes[1].plot(t, (y_det - y_eq) * 1e6)
axes[1].axhline(0, linestyle=":", color="black")
axes[1].set_ylabel("y displacement / micrometres")
axes[1].grid()

axes[2].plot(t, (z_det - z_eq) * 1e6)
axes[2].axhline(0, linestyle=":", color="black")
axes[2].set_xlabel("Time / s")
axes[2].set_ylabel("z displacement / micrometres")
axes[2].grid()

fig.suptitle("3D levitation with GLMT optical force table")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_det * 1e6, y_det * 1e6, z_det * 1e6)
ax.scatter([x_eq * 1e6], [y_eq * 1e6], [z_eq * 1e6], color="red")
ax.set_xlabel("x / micrometres")
ax.set_ylabel("y / micrometres")
ax.set_zlabel("z / micrometres")
ax.set_title("3D trajectory")
plt.show()
