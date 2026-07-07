import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Brownian white noise driving a mass on a spring
# -------------------------------------------------------------------------
# This script is more representative of the noise you would get for a trapped
# particle / mass-on-a-spring system.
#
# The model is the Langevin oscillator:
#
#     m x'' + b x' + k x = F_th(t)
#
# where F_th(t) is a white Brownian thermal force with
#
#     <F_th(t) F_th(t')> = 2 b k_B T delta(t - t')
#
# The force noise is white, but the displacement x(t) is not white. The spring
# and damping filter the force noise, producing a displacement spectrum with a
# resonance peak near the natural frequency.


# -------------------------------------------------------------------------
# Sampling parameters
# -------------------------------------------------------------------------
fs = 35_000.0       # sampling frequency / Hz
t_end = 10.0        # total signal duration / s
dt = 1.0 / fs

# Frequency range shown in the FFT/PSD plots.
# With fs = 35 kHz, the Nyquist frequency is 17.5 kHz.
f_plot_min = 1.0
nyquist_frequency = fs / 2
f_plot_max = nyquist_frequency

t = np.arange(0.0, t_end, dt)
N = len(t)

df = fs / N
print("Number of samples N =", N)
print("Sampling interval dt =", dt, "s")
print("FFT bin size df =", df, "Hz")
print("Nyquist frequency =", nyquist_frequency, "Hz")


# -------------------------------------------------------------------------
# Physical parameters
# -------------------------------------------------------------------------
kB = 1.380649e-23
T = 293.0

# Example particle properties, close to your levitation model.
radius = 5e-6       # m
density = 2200.0    # kg/m^3
volume = (4.0 / 3.0) * np.pi * radius**3
m = density * volume

# Choose a simple oscillator frequency and damping ratio.
# Change these to match your trap frequency and damping.
f0 = 50.0
omega0 = 2 * np.pi * f0
k = m * omega0**2

damping_ratio = 0.05
b = 2 * damping_ratio * m * omega0
gamma = b / m

if gamma < 2 * omega0:
    omega_d = np.sqrt(omega0**2 - (gamma / 2)**2)
    f_oscillation = omega_d / (2 * np.pi)
else:
    omega_d = np.nan
    f_oscillation = np.nan

f_damping = gamma / (2 * np.pi)

print("Mass m =", m, "kg")
print("Spring constant k =", k, "N/m")
print("Damping coefficient b =", b, "kg/s")
print("Natural frequency f0 =", f0, "Hz")
print("Damped oscillation frequency fd =", f_oscillation, "Hz")
print("Damping frequency f_gamma =", f_damping, "Hz")
print("Damping ratio =", damping_ratio)
print("Thermal RMS displacement sqrt(kBT/k) =", np.sqrt(kB * T / k), "m")


# -------------------------------------------------------------------------
# Brownian thermal force
# -------------------------------------------------------------------------
rng = np.random.default_rng(seed=1)

# Continuous-time Brownian force:
#     <F(t)F(t')> = 2 b kBT delta(t-t')
#
# In a discrete timestep dt, the timestep-averaged force has standard
# deviation:
#     sigma_F = sqrt(2 b kBT / dt)
sigma_F = np.sqrt(2 * b * kB * T / dt)
F_th = sigma_F * rng.normal(size=N)


# -------------------------------------------------------------------------
# Integrate the stochastic oscillator
# -------------------------------------------------------------------------
x = np.zeros(N)
v = np.zeros(N)

for i in range(N - 1):
    # Semi-implicit Euler update. For more serious stochastic simulations,
    # a Langevin-specific integrator such as BAOAB is usually better.
    a = (F_th[i] - b * v[i] - k * x[i]) / m
    v[i + 1] = v[i] + a * dt
    x[i + 1] = x[i] + v[i + 1] * dt


# -------------------------------------------------------------------------
# Fourier transform helpers
# -------------------------------------------------------------------------
def fft_amplitude_and_psd(signal):
    """
    Return one-sided FFT amplitude and PSD estimates using a Hann window.
    """
    signal = signal - np.mean(signal)

    window = np.hanning(N)
    coherent_gain = np.mean(window)
    window_power = np.mean(window**2)

    fft_values = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(N, dt)

    amplitude = 2.0 * np.abs(fft_values) / (N * coherent_gain)
    amplitude[0] *= 0.5

    psd = (2.0 * dt / (N * window_power)) * np.abs(fft_values)**2
    psd[0] *= 0.5

    if N % 2 == 0:
        amplitude[-1] *= 0.5
        psd[-1] *= 0.5

    return freqs, amplitude, psd


freqs, force_amp, force_psd = fft_amplitude_and_psd(F_th)
_, x_amp, x_psd = fft_amplitude_and_psd(x)


# -------------------------------------------------------------------------
# Theoretical PSDs
# -------------------------------------------------------------------------
omega = 2 * np.pi * freqs

# One-sided white force PSD.
S_F_theory = 4 * kB * T * b * np.ones_like(freqs)

# One-sided displacement PSD of a thermally driven damped harmonic oscillator:
#
#     S_x(f) = S_F / [(k - m omega^2)^2 + (b omega)^2]
#
x_psd_theory = S_F_theory / ((k - m * omega**2) ** 2 + (b * omega) ** 2)


def add_frequency_reference_lines():
    """
    Add the calculated oscillator and damping-frequency reference lines.
    """
    if np.isfinite(f_oscillation):
        plt.axvline(
            f_oscillation,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="damped oscillation frequency",
        )

    plt.axvline(
        f_damping,
        color="tab:red",
        linestyle=":",
        linewidth=1.6,
        label="damping frequency",
    )


# -------------------------------------------------------------------------
# Plot time-domain signals
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(t, F_th, linewidth=0.7)
plt.xlim(0, 0.05)
plt.xlabel("Time / s")
plt.ylabel("Force / N")
plt.title("Brownian thermal force, short time segment")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(9, 4))
plt.plot(t, x * 1e9, linewidth=0.8)
plt.xlim(0, 1.0)
plt.xlabel("Time / s")
plt.ylabel("Displacement / nm")
plt.title("Mass-spring displacement driven by Brownian force")
plt.grid()
plt.tight_layout()


# -------------------------------------------------------------------------
# Linear FFT amplitude of displacement
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(freqs[1:], x_amp[1:] * 1e9, linewidth=0.9)
add_frequency_reference_lines()
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Amplitude / nm")
plt.title("Displacement FFT amplitude, linear y-scale")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Log FFT amplitude of displacement
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], x_amp[1:] * 1e9, linewidth=0.9)
add_frequency_reference_lines()
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Amplitude / nm")
plt.title("Displacement FFT amplitude, log y-scale")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Force PSD: should be approximately white
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], force_psd[1:], linewidth=0.7, label="simulated force PSD")
plt.semilogy(freqs[1:], S_F_theory[1:], "k--", linewidth=1.5, label="theory: 4 kBT b")
add_frequency_reference_lines()
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Force PSD / N$^2$/Hz")
plt.title("Brownian force PSD, approximately white")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Displacement PSD: filtered by the oscillator response
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], x_psd[1:] * 1e18, linewidth=0.7, label="simulated displacement PSD")
plt.semilogy(freqs[1:], x_psd_theory[1:] * 1e18, "k--", linewidth=1.5, label="theory")
add_frequency_reference_lines()
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Mass-spring Brownian displacement PSD")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


plt.show()
