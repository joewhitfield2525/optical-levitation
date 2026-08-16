import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import welch

# ****************************************************************************************************************************************************
# Noise switches
# ****************************************************************************************************************************************************
# Turn these on/off to choose which noise sources are included in the main
# BAOAB trajectory and which noise-specific plots are shown.

USE_BROWNIAN_NOISE = True
USE_LASER_POWER_NOISE = True 
USE_PD_FEEDBACK = True 

# ****************************************************************************************************************************************************
# Feedback settings
# ****************************************************************************************************************************************************
# The PD loop reads the measured particle position, estimates velocity, and
# changes laser power to push the particle back towards equilibrium.

USE_FEEDBACK_POSITION_NOISE = False
FEEDBACK_POSITION_NOISE_RMS = 50e-9         # m RMS, represents a <100 nm PSD readout
TOTAL_LOOP_DELAY_SECONDS = 50e-6           # s, PSD/TIA/ADC/computer/laser delay
FEEDBACK_KP_MULTIPLIER = 0.2                # Kp = this number * trap spring constant
FEEDBACK_KD_MULTIPLIER = 8.0                # Kd = this number * gas damping coefficient
FEEDBACK_POWER_MIN_FACTOR = 0.80            # minimum command = this * nominal power
FEEDBACK_POWER_MAX_FACTOR = 1.20            # maximum command = this * nominal power
VELOCITY_FILTER_ALPHA = 0.25                # lower value gives smoother velocity estimate

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
T = 300           # temperature, K
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
# F_total(z) - F_weight_effective approx -k (z - z_eq)
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
t_end = 1.0
experimental_sampling_frequency = 20_000.0
experimental_timestep = 1 / experimental_sampling_frequency
simulation_oversampling_factor = 10
dt_simulation = experimental_timestep / simulation_oversampling_factor
t_eval = np.arange(t_start, t_end + 0.5 * dt_simulation, dt_simulation)
dt_output = t_eval[1] - t_eval[0]
solve_ivp_max_step = dt_simulation

print("Experimental sampling frequency =", experimental_sampling_frequency, "Hz")
print("Experimental timestep =", experimental_timestep, "s")
print("Simulation oversampling factor =", simulation_oversampling_factor)
print("Simulation timestep =", dt_simulation, "s")
print("Number of output samples =", len(t_eval))
print("Output timestep t_eval[1] - t_eval[0] =", dt_output, "s")
print("solve_ivp maximum internal timestep =", solve_ivp_max_step, "s")

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
    max_step=solve_ivp_max_step
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
    max_step=solve_ivp_max_step
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

rng = np.random.default_rng(seed=7)

dt_baoab = dt_output
t_baoab = np.arange(t_start, t_end + 0.5 * dt_baoab, dt_baoab)
constant_power_factor = np.ones_like(t_baoab)

laser_noise_fraction = 0.01
laser_noise_step_duration = 0.01

if USE_LASER_POWER_NOISE:
    laser_step_samples = max(1, int(round(laser_noise_step_duration / dt_baoab)))
    laser_step_duration_actual = laser_step_samples * dt_baoab
    number_of_laser_steps = int(np.ceil(len(t_baoab) / laser_step_samples))
    laser_allowed_power_factors = np.array(
        [
            1 - laser_noise_fraction,
            1.0,
            1 + laser_noise_fraction,
        ]
    )

    laser_step_power_factors = np.empty(number_of_laser_steps)
    laser_step_power_factors[0] = 1.0

    for j in range(1, number_of_laser_steps):
        previous_factor = laser_step_power_factors[j - 1]

        if np.isclose(previous_factor, 1.0):
            possible_factors = laser_allowed_power_factors
        elif np.isclose(previous_factor, 1 + laser_noise_fraction):
            possible_factors = np.array([1.0, 1 + laser_noise_fraction])
        elif np.isclose(previous_factor, 1 - laser_noise_fraction):
            possible_factors = np.array([1 - laser_noise_fraction, 1.0])
        else:
            possible_factors = laser_allowed_power_factors

        laser_step_power_factors[j] = rng.choice(possible_factors)

    laser_power_factor = np.repeat(laser_step_power_factors, laser_step_samples)[:len(t_baoab)]
else:
    laser_step_duration_actual = 0.0
    laser_power_factor = constant_power_factor.copy()

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
print("Nominal laser power =", P_laser, "W")
print("Brownian noise ON =", USE_BROWNIAN_NOISE)
print("Laser-power noise ON =", USE_LASER_POWER_NOISE)

if USE_LASER_POWER_NOISE:
    print("Laser noise fraction =", laser_noise_fraction)
    print("Requested laser noise step duration =", laser_noise_step_duration, "s")
    print("Actual laser noise step duration =", laser_step_duration_actual, "s")
else:
    print("Laser power is held constant in the BAOAB trajectory.")


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


FEEDBACK_UPDATE_FREQUENCY = experimental_sampling_frequency
PD_KP = FEEDBACK_KP_MULTIPLIER * k
PD_KD = FEEDBACK_KD_MULTIPLIER * b
P_MIN = FEEDBACK_POWER_MIN_FACTOR * P_laser
P_MAX = FEEDBACK_POWER_MAX_FACTOR * P_laser


def measure_position_for_feedback(z_actual, rng):
    """
    Synthetic position measurement used by the feedback loop.

    For now this is a simple calibrated PSD-like readout: the true position
    plus optional measurement noise. The delay is handled in the controller.
    """
    z_measured = z_actual

    if USE_FEEDBACK_POSITION_NOISE and FEEDBACK_POSITION_NOISE_RMS > 0:
        z_measured = z_measured + rng.normal(0.0, FEEDBACK_POSITION_NOISE_RMS)

    return z_measured


def solve_baoab_with_pd_feedback(disturbance_power_factor_time, brownian_normals):
    """
    Run BAOAB with a delayed PD feedback loop controlling the laser power.

    disturbance_power_factor_time is the laser-power noise. The controller
    command is multiplied by this disturbance, so the feedback fights the same
    laser noise used in the no-feedback BAOAB trajectory.
    """
    feedback_rng = np.random.default_rng(seed=107)
    baoab_sampling_frequency = 1 / dt_baoab
    control_decimation = int(round(baoab_sampling_frequency / FEEDBACK_UPDATE_FREQUENCY))

    if control_decimation < 1:
        raise ValueError("FEEDBACK_UPDATE_FREQUENCY cannot exceed the BAOAB sampling frequency.")

    if not np.isclose(baoab_sampling_frequency / FEEDBACK_UPDATE_FREQUENCY, control_decimation):
        raise ValueError("Choose FEEDBACK_UPDATE_FREQUENCY so it divides the BAOAB sampling frequency.")

    dt_control = control_decimation * dt_baoab

    if TOTAL_LOOP_DELAY_SECONDS <= 0:
        loop_delay_control_steps = 0
    else:
        loop_delay_control_steps = int(np.ceil(TOTAL_LOOP_DELAY_SECONDS / dt_control))

    loop_delay_baoab_steps = loop_delay_control_steps * control_decimation
    actual_loop_delay = loop_delay_control_steps * dt_control

    force_per_watt = F_total(z_eq, power_factor=1.0) / P_laser
    if abs(force_per_watt) < 1e-30:
        raise ValueError("Force per watt is too close to zero for laser-power feedback.")

    z_out = np.zeros_like(t_baoab)
    v_out = np.zeros_like(t_baoab)
    command_power_time = np.zeros_like(t_baoab)
    actual_power_time = np.zeros_like(t_baoab)

    control_times = []
    measurement_times = []
    measured_positions = []
    filtered_positions = []
    requested_powers = []

    z_out[0] = z0
    v_out[0] = v0

    command_power = P_laser
    filtered_z_previous = None

    for i in range(len(t_baoab) - 1):
        if i % control_decimation == 0:
            delayed_i = max(0, i - loop_delay_baoab_steps)
            z_measured = measure_position_for_feedback(z_out[delayed_i], feedback_rng)

            if filtered_z_previous is None:
                z_filtered = z_measured
                measured_velocity = 0.0
            else:
                z_filtered = (
                    VELOCITY_FILTER_ALPHA * z_measured
                    + (1 - VELOCITY_FILTER_ALPHA) * filtered_z_previous
                )
                measured_velocity = (z_filtered - filtered_z_previous) / dt_control

            error = z_filtered - z_eq
            feedback_force = -PD_KP * error - PD_KD * measured_velocity
            power_change = feedback_force / force_per_watt
            command_power = np.clip(P_laser + power_change, P_MIN, P_MAX)

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
            FEEDBACK_POWER_MIN_FACTOR,
            FEEDBACK_POWER_MAX_FACTOR,
        )

        z_i = z_out[i]
        v_i = v_out[i]

        # B: half deterministic-force kick
        v_i = v_i + 0.5 * dt_baoab * deterministic_force_no_drag(z_i, actual_power_factor) / m

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
        v_i = v_i + 0.5 * dt_baoab * deterministic_force_no_drag(z_i, actual_power_factor) / m

        z_out[i + 1] = z_i
        v_out[i + 1] = v_i
        command_power_time[i] = command_power
        actual_power_time[i] = P_laser * actual_power_factor

    command_power_time[-1] = command_power
    actual_power_time[-1] = actual_power_time[-2]

    return {
        "t": t_baoab,
        "z": z_out,
        "v": v_out,
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
    }


brownian_normals = rng.normal(size=len(t_baoab) - 1)
zero_brownian_normals = np.zeros_like(brownian_normals)

main_brownian_normals = brownian_normals if USE_BROWNIAN_NOISE else zero_brownian_normals
main_power_factor = laser_power_factor if USE_LASER_POWER_NOISE else constant_power_factor

z_baoab_baseline, v_baoab_baseline = solve_baoab_with_power(
    constant_power_factor,
    zero_brownian_normals,
)

if USE_BROWNIAN_NOISE:
    z_baoab_brownian_only, v_baoab_brownian_only = solve_baoab_with_power(
        constant_power_factor,
        brownian_normals,
    )
    z_brownian_displacement_effect = z_baoab_brownian_only - z_baoab_baseline
else:
    z_baoab_brownian_only = None
    v_baoab_brownian_only = None
    z_brownian_displacement_effect = None

if USE_LASER_POWER_NOISE:
    z_baoab_laser_only, v_baoab_laser_only = solve_baoab_with_power(
        laser_power_factor,
        zero_brownian_normals,
    )
    z_laser_displacement_effect = z_baoab_laser_only - z_baoab_baseline
else:
    z_baoab_laser_only = None
    v_baoab_laser_only = None
    z_laser_displacement_effect = None

z_baoab, v_baoab = solve_baoab_with_power(
    main_power_factor,
    main_brownian_normals,
)

if USE_PD_FEEDBACK:
    feedback_result = solve_baoab_with_pd_feedback(
        main_power_factor,
        main_brownian_normals,
    )
    z_baoab_feedback = feedback_result["z"]
    v_baoab_feedback = feedback_result["v"]
else:
    feedback_result = None
    z_baoab_feedback = None
    v_baoab_feedback = None

if USE_BROWNIAN_NOISE:
    print(
        "BAOAB Brownian-only RMS displacement from equilibrium =",
        np.std(z_baoab_brownian_only - z_eq) * 1e6,
        "micrometres"
    )
    print(
        "RMS isolated Brownian displacement effect =",
        np.std(z_brownian_displacement_effect) * 1e9,
        "nm"
    )
else:
    print("Brownian noise is OFF, so Brownian-only plots and RMS values are skipped.")

if USE_LASER_POWER_NOISE:
    print(
        "RMS isolated laser-power displacement effect =",
        np.std(z_laser_displacement_effect) * 1e9,
        "nm"
    )
else:
    print("Laser-power noise is OFF, so laser-noise-only plots and RMS values are skipped.")

if USE_PD_FEEDBACK:
    no_feedback_rms_nm = np.std(z_baoab - z_eq) * 1e9
    feedback_rms_nm = np.std(z_baoab_feedback - z_eq) * 1e9

    print("PD feedback ON =", USE_PD_FEEDBACK)
    print("Feedback update frequency =", 1 / feedback_result["dt_control"], "Hz")
    print("Requested total loop delay =", TOTAL_LOOP_DELAY_SECONDS, "s")
    print("Actual total loop delay used =", feedback_result["actual_loop_delay"], "s")
    print("Loop delay in controller updates =", feedback_result["loop_delay_control_steps"])
    print("PD Kp =", PD_KP, "N/m")
    print("PD Kd =", PD_KD, "kg/s")
    print("Feedback position noise ON =", USE_FEEDBACK_POSITION_NOISE)
    print("Feedback position noise RMS =", FEEDBACK_POSITION_NOISE_RMS * 1e9, "nm")
    print("Laser command limits =", P_MIN, "to", P_MAX, "W")
    print("Force per watt at equilibrium =", feedback_result["force_per_watt"], "N/W")
    print("No-feedback RMS displacement =", no_feedback_rms_nm, "nm")
    print("With-feedback RMS displacement =", feedback_rms_nm, "nm")
    print("Feedback RMS / no-feedback RMS =", feedback_rms_nm / no_feedback_rms_nm)
    print("Minimum feedback command power =", np.min(feedback_result["command_power"]), "W")
    print("Maximum feedback command power =", np.max(feedback_result["command_power"]), "W")
    print("Minimum actual feedback laser power =", np.min(feedback_result["actual_power"]), "W")
    print("Maximum actual feedback laser power =", np.max(feedback_result["actual_power"]), "W")
else:
    print("PD feedback is OFF.")

# ******************************************************************************************************************
# Laser power as a function of time
# ******************************************************************************************************************
if USE_LASER_POWER_NOISE:
    plt.figure(figsize=(8, 4))
    plt.step(t_baoab, laser_power_time, where="post", label="laser power")
    plt.axhline(P_laser, linestyle="--", color="black", label="nominal laser power")
    plt.axhline(P_laser * (1 + laser_noise_fraction), linestyle=":", color="tab:red")
    plt.axhline(P_laser * (1 - laser_noise_fraction), linestyle=":", color="tab:red")
    plt.xlabel("Time / s")
    plt.ylabel("Laser power / W")
    plt.title("Stepwise laser-power noise")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if USE_PD_FEEDBACK:
    actual_power_label = "actual power after laser noise" if USE_LASER_POWER_NOISE else "actual laser power"

    plt.figure(figsize=(8, 4))
    plt.plot(
        feedback_result["t"],
        feedback_result["command_power"] * 1e3,
        label="PD command power"
    )
    plt.plot(
        feedback_result["t"],
        feedback_result["actual_power"] * 1e3,
        alpha=0.65,
        label=actual_power_label
    )
    plt.axhline(P_laser * 1e3, linestyle="--", color="black", label="nominal power")
    plt.axhline(P_MIN * 1e3, linestyle=":", color="tab:red", label="command limits")
    plt.axhline(P_MAX * 1e3, linestyle=":", color="tab:red")
    plt.xlabel("Time / s")
    plt.ylabel("Laser power / mW")
    plt.title("PD feedback laser power")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.plot(
    #     feedback_result["control_t"],
    #     (feedback_result["z_measured"] - z_eq) * 1e9,
    #     alpha=0.45,
    #     label="delayed noisy measured position"
    # )
    # plt.plot(
    #     feedback_result["control_t"],
    #     (feedback_result["z_filtered"] - z_eq) * 1e9,
    #     label="filtered position used by controller"
    # )
    # plt.axhline(0, color="black", linestyle=":", linewidth=0.8)
    # plt.xlabel("Time / s")
    # plt.ylabel("Measured displacement / nm")
    # plt.title("PD feedback measurement signal")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

# ******************************************************************************************************************
# Isolated displacement effects from each noise source
# ******************************************************************************************************************
if USE_LASER_POWER_NOISE:
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_baoab,
        z_laser_displacement_effect * 1e9,
        label="laser-power noise only"
    )
    plt.axhline(0, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("Time / s")
    plt.ylabel("Displacement difference / nm")
    plt.title("Isolated displacement due to laser-power noise")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if USE_BROWNIAN_NOISE:
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_baoab,
        z_brownian_displacement_effect * 1e9,
        label="Brownian noise only"
    )
    plt.axhline(0, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("Time / s")
    plt.ylabel("Displacement difference / nm")
    plt.title("Isolated displacement due to Brownian motion")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ******************************************************************************************************************
# Position histogram for the Brownian trajectory
# ******************************************************************************************************************
# A harmonic thermal distribution should be Gaussian:
#
#     p(z) proportional to exp[-k (z - z_eq)^2 / (2 kB T)]
#
# The late-time mask removes the large initial transient, so the histogram is
# mostly testing the steady Brownian motion in the trap.
histogram_mask = t_baoab > (0.25 * t_end)

if np.count_nonzero(histogram_mask) < 1024:
    histogram_mask = np.ones_like(t_baoab, dtype=bool)

z_hist_nm = (z_baoab_brownian_only[histogram_mask] - z_eq) * 1e9
z_hist_std_nm = np.std(z_hist_nm)
z_thermal_std_nm = np.sqrt(kB * T / k) * 1e9

hist_counts, hist_edges = np.histogram(z_hist_nm, bins=80, density=True)
hist_centres = 0.5 * (hist_edges[:-1] + hist_edges[1:])

z_gaussian_nm = np.linspace(hist_edges[0], hist_edges[-1], 600)
gaussian_pdf = (
    1
    / (np.sqrt(2 * np.pi) * z_thermal_std_nm)
    * np.exp(-0.5 * (z_gaussian_nm / z_thermal_std_nm)**2)
)

print("Position histogram RMS =", z_hist_std_nm, "nm")
print("Thermal Gaussian RMS =", z_thermal_std_nm, "nm")

plt.figure(figsize=(8, 5))
plt.hist(
    z_hist_nm,
    bins=80,
    density=True,
    alpha=0.65,
    label="BAOAB position histogram"
)
plt.plot(
    z_gaussian_nm,
    gaussian_pdf,
    "k--",
    linewidth=2,
    label="thermal Gaussian theory"
)
plt.xlabel("Displacement from equilibrium / nm")
plt.ylabel("Probability density / nm$^{-1}$")
plt.title("Position histogram of Brownian motion")
plt.legend()
plt.grid()
plt.show()

positive_hist_mask = hist_counts > 0

plt.figure(figsize=(8, 5))
plt.semilogy(
    hist_centres[positive_hist_mask],
    hist_counts[positive_hist_mask],
    "o",
    markersize=4,
    label="BAOAB position histogram"
)
plt.semilogy(
    z_gaussian_nm,
    gaussian_pdf,
    "k--",
    linewidth=2,
    label="thermal Gaussian theory"
)
plt.xlabel("Displacement from equilibrium / nm")
plt.ylabel("Probability density / nm$^{-1}$")
plt.title("Position histogram, semilog view")
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
noise_label_parts = []

if USE_BROWNIAN_NOISE:
    noise_label_parts.append("Brownian")

if USE_LASER_POWER_NOISE:
    noise_label_parts.append("laser-power")

if noise_label_parts:
    baoab_noise_label = "Nonlinear BAOAB with " + " and ".join(noise_label_parts) + " noise"
else:
    baoab_noise_label = "Nonlinear BAOAB without Brownian or laser-power noise"

plt.figure(figsize=(8, 5))

#plt.plot(t, z_nonlinear * 1e6, label="Nonlinear model")
#plt.plot(t, z_linear * 1e6, "--", label="Linear SHM approximation")
plt.plot(
    t_baoab,
    z_baoab * 1e6,
    linewidth=0.8,
    alpha=0.75,
    label=baoab_noise_label
)

if USE_PD_FEEDBACK:
    plt.plot(
        t_baoab,
        z_baoab_feedback * 1e6,
        linewidth=1.2,
        label="Nonlinear BAOAB with PD feedback"
    )

plt.axhline(z_eq * 1e6, linestyle=":", label="Equilibrium")

plt.xlabel("Time / s")
plt.ylabel("Height z / micrometres")
plt.title("Separated optical forces vs linear SHM approximation")
plt.legend()
plt.grid()
plt.show()

if USE_PD_FEEDBACK:
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_baoab,
        (z_baoab - z_eq) * 1e6,
        label="without feedback"
    )
    plt.plot(
        t_baoab,
        (z_baoab_feedback - z_eq) * 1e6,
        label="with PD feedback"
    )
    plt.axhline(0, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("Time / s")
    plt.ylabel("Displacement from equilibrium / micrometres")
    plt.title("Particle trajectory with and without feedback")
    plt.legend()
    plt.grid()
    plt.tight_layout()
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
# # F_net approx -k (z - z_eq)
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
#  BAOAB Brownian PSD analysis
# ******************************************************************************************************************


def positive_welch_psd(signal, time_array, target_bin_width=None):
    """
    Return positive-frequency Welch PSD in nm^2/Hz.
    """
    signal = signal - np.mean(signal)
    dt_local = time_array[1] - time_array[0]
    sampling_frequency_local = 1 / dt_local

    if target_bin_width is None:
        target_bin_width = max(1.0, (omega / (2*np.pi)) / 20)

    nperseg_local = int(np.ceil(sampling_frequency_local / target_bin_width))
    nperseg_local = min(len(signal), max(256, nperseg_local))

    freqs_local, psd_local = welch(
        signal,
        fs=sampling_frequency_local,
        window="hann",
        nperseg=nperseg_local,
        noverlap=nperseg_local // 2,
        detrend="constant",
        scaling="density",
    )

    positive_mask = (
        (freqs_local > 0)
        & np.isfinite(psd_local)
        & (psd_local > 0)
    )

    return (
        freqs_local[positive_mask],
        psd_local[positive_mask] * 1e18,
        dt_local,
        nperseg_local,
    )


def positive_raw_periodogram_psd(signal, time_array):
    """
    Return a single-segment, non-Welch-averaged PSD in nm^2/Hz.

    This uses the whole signal at once with a Hann window. It is noisier than
    Welch averaging, but it can make narrow deterministic peaks easier to see.
    """
    signal = signal - np.mean(signal)
    dt_local = time_array[1] - time_array[0]
    sampling_frequency_local = 1 / dt_local
    window_local = np.hanning(len(signal))
    window_power = np.sum(window_local**2)

    fft_values = np.fft.rfft(signal * window_local)
    freqs_local = np.fft.rfftfreq(len(signal), dt_local)

    psd_local = np.abs(fft_values)**2 / (sampling_frequency_local * window_power)

    if len(psd_local) > 2:
        psd_local[1:-1] *= 2

    positive_mask = (
        (freqs_local > 0)
        & np.isfinite(psd_local)
        & (psd_local > 0)
    )

    return (
        freqs_local[positive_mask],
        psd_local[positive_mask] * 1e18,
        dt_local,
    )


# Deterministic nonlinear PSD. This uses the solve_ivp nonlinear trajectory
# without Brownian motion or laser-power noise.
nonlinear_psd_freqs, nonlinear_psd_nm2, dt_psd_nonlinear, nonlinear_psd_nperseg = (
    positive_welch_psd(
        z_nonlinear - z_eq,
        t,
    )
)

nonlinear_nyquist = 0.5 / dt_psd_nonlinear
nonlinear_frequency_plot_max = 1.05 * nonlinear_nyquist

print("Nonlinear deterministic PSD nperseg =", nonlinear_psd_nperseg)
print("Nonlinear deterministic PSD Nyquist frequency =", nonlinear_nyquist, "Hz")

plt.figure(figsize=(8, 5))
plt.loglog(
    nonlinear_psd_freqs,
    nonlinear_psd_nm2,
    label="nonlinear deterministic PSD"
)
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(nonlinear_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(
    max(nonlinear_psd_freqs[0] * 0.95, 1e-12),
    nonlinear_frequency_plot_max
)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("PSD of nonlinear model without noise")
plt.legend()
plt.grid(True, which="both")
plt.show()

nonlinear_raw_freqs, nonlinear_raw_psd_nm2, dt_raw_nonlinear = (
    positive_raw_periodogram_psd(
        z_nonlinear - z_eq,
        t,
    )
)

nonlinear_raw_nyquist = 0.5 / dt_raw_nonlinear

plt.figure(figsize=(8, 5))
plt.loglog(
    nonlinear_raw_freqs,
    nonlinear_raw_psd_nm2,
    label="nonlinear deterministic raw PSD"
)
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(nonlinear_raw_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(
    max(nonlinear_raw_freqs[0] * 0.95, 1e-12),
    1.05 * nonlinear_raw_nyquist
)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Raw PSD of nonlinear model without noise")
plt.legend()
plt.grid(True, which="both")
plt.show()


# Full BAOAB trajectory PSD. This includes whichever noise sources are enabled.
baoab_psd_freqs, baoab_psd_nm2, dt_psd_baoab, baoab_psd_nperseg = (
    positive_welch_psd(
        z_baoab - z_eq,
        t_baoab,
    )
)

baoab_nyquist = 0.5 / dt_psd_baoab
baoab_frequency_plot_max = min(fft_plot_max / 10, baoab_nyquist)
baoab_frequency_plot_max = 1.05 * baoab_nyquist

print("BAOAB full-trajectory PSD nperseg =", baoab_psd_nperseg)
print("BAOAB PSD Nyquist frequency =", baoab_nyquist, "Hz")

plt.figure(figsize=(8, 5))
plt.loglog(baoab_psd_freqs, baoab_psd_nm2, label=baoab_noise_label + " PSD")

if USE_PD_FEEDBACK:
    feedback_psd_freqs, feedback_psd_nm2, dt_psd_feedback, feedback_psd_nperseg = (
        positive_welch_psd(
            z_baoab_feedback - z_eq,
            t_baoab,
        )
    )
    plt.loglog(
        feedback_psd_freqs,
        feedback_psd_nm2,
        label="Nonlinear BAOAB with PD feedback PSD"
    )
    print("Feedback PSD nperseg =", feedback_psd_nperseg)

plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(baoab_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(max(baoab_psd_freqs[0] * 0.95, 1e-12), baoab_frequency_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("BAOAB full-trajectory PSD: " + baoab_noise_label)
plt.legend()
plt.grid(True, which="both")
plt.show()

baoab_raw_freqs, baoab_raw_psd_nm2, dt_raw_baoab = (
    positive_raw_periodogram_psd(
        z_baoab - z_eq,
        t_baoab,
    )
)

baoab_raw_nyquist = 0.5 / dt_raw_baoab

plt.figure(figsize=(8, 5))
plt.loglog(baoab_raw_freqs, baoab_raw_psd_nm2, label=baoab_noise_label + " raw PSD")
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(baoab_raw_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(max(baoab_raw_freqs[0] * 0.95, 1e-12), 1.05 * baoab_raw_nyquist)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Raw PSD of BAOAB full trajectory: " + baoab_noise_label)
plt.legend()
plt.grid(True, which="both")
plt.show()


# Early BAOAB PSD. This is the one to inspect when displacement is large and
# you expect coherent oscillation peaks/harmonics like the deterministic model.
early_fft_duration = min(0.35 , t_end)
early_mask = t_baoab <= (t_start + early_fft_duration)

t_early = t_baoab[early_mask]
z_early = z_baoab[early_mask] - z_eq

early_psd_freqs, early_psd_nm2, dt_psd_early, early_psd_nperseg = (
    positive_welch_psd(
        z_early,
        t_early,
        target_bin_width=max(1.0, (omega / (2*np.pi)) / 10),
    )
)
early_nyquist = 0.5 / dt_psd_early
early_frequency_plot_max = min(fft_plot_max / 10, early_nyquist)
early_frequency_plot_max = 1.05 * early_nyquist

print("Early BAOAB PSD duration =", t_early[-1] - t_early[0], "s")
print("Early BAOAB PSD nperseg =", early_psd_nperseg)
print("Early BAOAB PSD Nyquist frequency =", early_nyquist, "Hz")

plt.figure(figsize=(8, 5))
plt.loglog(early_psd_freqs, early_psd_nm2, label=baoab_noise_label + " early transient PSD")
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(early_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(max(early_psd_freqs[0] * 0.95, 1e-12), early_frequency_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("BAOAB early-transient PSD: " + baoab_noise_label)
plt.legend()
plt.grid(True, which="both")
plt.show()

early_raw_freqs, early_raw_psd_nm2, dt_raw_early = (
    positive_raw_periodogram_psd(
        z_early,
        t_early,
    )
)

early_raw_nyquist = 0.5 / dt_raw_early

plt.figure(figsize=(8, 5))
plt.loglog(early_raw_freqs, early_raw_psd_nm2, label=baoab_noise_label + " early transient raw PSD")
plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
plt.axvline(early_raw_nyquist, color="red", linestyle=":", label="Nyquist limit")
plt.xlim(max(early_raw_freqs[0] * 0.95, 1e-12), 1.05 * early_raw_nyquist)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Raw PSD of BAOAB early transient: " + baoab_noise_label)
plt.legend()
plt.grid(True, which="both")
plt.show()


# Late-time Welch PSD. This is the one to inspect when displacement = 0 and
# you expect the Brownian damped-harmonic-oscillator thermal spectrum.
if USE_BROWNIAN_NOISE:
    settled_mask = t_baoab > (0.25 * t_end)

    if np.count_nonzero(settled_mask) < 1024:
        settled_mask = np.ones_like(t_baoab, dtype=bool)

    z_psd_signal = z_baoab_brownian_only[settled_mask] - z_eq
    z_psd_signal = z_psd_signal - np.mean(z_psd_signal)
    dt_psd = t_baoab[1] - t_baoab[0]
    sampling_frequency_psd = 1 / dt_psd
    nyquist_frequency_psd = sampling_frequency_psd / 2

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
    print("Welch PSD Nyquist frequency =", nyquist_frequency_psd, "Hz")

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        psd_freqs[positive_psd_mask],
        z_psd[positive_psd_mask] * 1e18,
        label="BAOAB late-time Brownian PSD"
    )
    plt.semilogy(
        psd_freqs[positive_psd_mask],
        z_psd_theory[positive_psd_mask] * 1e18,
        "--",
        label="linear theory PSD"
    )
    plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
    plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
    plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
    plt.axvline(nyquist_frequency_psd, color="red", linestyle=":", label="Nyquist limit")
    plt.xscale("log")
    plt.xlim(max(psd_freqs[positive_psd_mask][0] * 0.95, 1e-12), baoab_frequency_plot_max)
    plt.xlim(max(psd_freqs[positive_psd_mask][0] * 0.95, 1e-12), 1.05 * nyquist_frequency_psd)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / nm$^2$/Hz")
    plt.title("BAOAB late-time Brownian PSD")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    late_raw_freqs, late_raw_psd_nm2, dt_raw_late = (
        positive_raw_periodogram_psd(
            z_psd_signal,
            t_baoab[settled_mask],
        )
    )

    late_raw_nyquist = 0.5 / dt_raw_late

    plt.figure(figsize=(8, 5))
    plt.loglog(late_raw_freqs, late_raw_psd_nm2, label="BAOAB late-time Brownian raw PSD")
    plt.axvline(omega / (2*np.pi), linestyle="--", label="linear fz")
    plt.axvline(2*omega / (2*np.pi), linestyle="--", label="second harmonic fz")
    plt.axvline(3*omega / (2*np.pi), linestyle="--", label="third harmonic fz")
    plt.axvline(late_raw_nyquist, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(max(late_raw_freqs[0] * 0.95, 1e-12), 1.05 * late_raw_nyquist)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Displacement PSD / nm$^2$/Hz")
    plt.title("Raw PSD of BAOAB late-time Brownian motion")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
else:
    print("Late-time Brownian PSD plots skipped because USE_BROWNIAN_NOISE is False.")
