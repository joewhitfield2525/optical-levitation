import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Brownian white noise driving a free damped particle
# -------------------------------------------------------------------------
# This is the free-particle version of the Langevin model.
#
# Instead of a trapped mass on a spring,
#
#     m x'' + b x' + k x = F_th(t),
#
# this code uses
#
#     m x'' + b x' = F_th(t).
#
# There is no restoring spring force, so the particle is not confined around
# x = 0. Its position diffuses over time.


# -------------------------------------------------------------------------
# Sampling parameters
# -------------------------------------------------------------------------
fs = 120_000.0     # sampling frequency / Hz
t_end = 10.0       # total signal duration / s
dt = 1.0 / fs

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

radius = 5e-6       # m
density = 2200.0    # kg/m^3
eta = 1.8e-5        # air viscosity / Pa s

volume = (4.0 / 3.0) * np.pi * radius**3
m = density * volume

# Free-particle damping. This is ordinary Stokes drag.
# For low-pressure levitation you may want Cunningham or Epstein drag instead.
b = 6.0 * np.pi * eta * radius
gamma = b / m
velocity_relaxation_time = 1.0 / gamma

print("Mass m =", m, "kg")
print("Damping coefficient b =", b, "kg/s")
print("Velocity damping rate gamma = b/m =", gamma, "1/s")
print("Velocity relaxation time m/b =", velocity_relaxation_time, "s")


# -------------------------------------------------------------------------
# Brownian thermal force
# -------------------------------------------------------------------------
rng = np.random.default_rng(seed=1)

# Continuous-time Brownian force:
#
#     <F_th(t) F_th(t')> = 2 b k_B T delta(t - t')
#
# In a discrete timestep dt, the timestep-averaged force has standard
# deviation:
#
#     sigma_F = sqrt(2 b k_B T / dt)
sigma_F = np.sqrt(2.0 * b * kB * T / dt)
F_th = sigma_F * rng.normal(size=N)

print("Brownian force standard deviation per timestep =", sigma_F, "N")


# -------------------------------------------------------------------------
# Integrate the free Langevin particle
# -------------------------------------------------------------------------
x = np.zeros(N)
v = np.zeros(N)

for i in range(N - 1):
    # Free damped particle:
    #     a = (F_th - b v) / m
    #
    # There is deliberately no -k*x term here.
    a = (F_th[i] - b * v[i]) / m
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


def full_fft_wrapped_amplitude(signal):
    """
    Return a temporary full-FFT amplitude view from 0 to fs.

    The part between fs/2 and fs is not new physical information. It is the
    negative-frequency half of the FFT wrapped into positive-frequency labels.
    This is useful only for visualising why the one-sided FFT stops at Nyquist.
    """
    signal = signal - np.mean(signal)

    window = np.hanning(N)
    coherent_gain = np.mean(window)

    fft_values = np.fft.fft(signal * window)
    fft_freqs = np.fft.fftfreq(N, dt)

    wrapped_freqs = np.where(fft_freqs < 0, fft_freqs + fs, fft_freqs)
    sort_order = np.argsort(wrapped_freqs)

    wrapped_freqs = wrapped_freqs[sort_order]
    wrapped_amplitude = np.abs(fft_values[sort_order]) / (N * coherent_gain)

    return wrapped_freqs, wrapped_amplitude


freqs, force_amp, force_psd = fft_amplitude_and_psd(F_th)
_, v_amp, v_psd = fft_amplitude_and_psd(v)
_, x_amp, x_psd = fft_amplitude_and_psd(x)

full_freqs, full_x_amp = full_fft_wrapped_amplitude(x)
_, full_force_amp = full_fft_wrapped_amplitude(F_th)


# -------------------------------------------------------------------------
# Theoretical PSDs
# -------------------------------------------------------------------------
omega = 2.0 * np.pi * freqs

# One-sided Brownian force PSD.
S_F_theory = 4.0 * kB * T * b * np.ones_like(freqs)

# Free-particle response:
#
#     x(omega) / F(omega) = 1 / (-m omega^2 + i b omega)
#
# so:
#
#     S_x = S_F / [(m omega^2)^2 + (b omega)^2].
#
# The zero-frequency value diverges for a free particle, so set it to NaN for
# plotting.
denominator_x = (m * omega**2) ** 2 + (b * omega) ** 2
x_psd_theory = S_F_theory / denominator_x
x_psd_theory[0] = np.nan

# Velocity response:
#
#     v(omega) / F(omega) = i omega / (-m omega^2 + i b omega)
#
# which simplifies to:
#
#     S_v = S_F / [(m omega)^2 + b^2].
v_psd_theory = S_F_theory / ((m * omega) ** 2 + b**2)


# -------------------------------------------------------------------------
# Plot time-domain signals
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(t, F_th, linewidth=0.7)
plt.xlim(0, 0.01)
plt.xlabel("Time / s")
plt.ylabel("Force / N")
plt.title("Brownian thermal force, short time segment")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(9, 4))
plt.plot(t, x * 1e9, linewidth=0.8)
plt.xlim(0, 1.0)
plt.xlabel("Time / s")
plt.ylabel("Position / nm")
plt.title("Free Brownian particle position")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(9, 4))
plt.plot(t, v * 1e6, linewidth=0.8)
plt.xlim(0, 0.1)
plt.xlabel("Time / s")
plt.ylabel("Velocity / micrometres s$^{-1}$")
plt.title("Free Brownian particle velocity")
plt.grid()
plt.tight_layout()


# -------------------------------------------------------------------------
# FFT amplitude of displacement
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(freqs[1:], x_amp[1:] * 1e9, linewidth=0.9)
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Amplitude / nm")
plt.title("Free-particle displacement FFT amplitude, linear y-scale")
plt.grid(which="both")
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], x_amp[1:] * 1e9, linewidth=0.9)
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Amplitude / nm")
plt.title("Free-particle displacement FFT amplitude, log y-scale")
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Temporary full FFT views up to Nyquist
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(full_freqs[1:], full_x_amp[1:] * 1e9, linewidth=0.8, label="full FFT amplitude")
plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Wrapped FFT frequency / Hz")
plt.ylabel("Amplitude / nm")
plt.title("Temporary full FFT displacement view, up to Nyquist")
plt.legend()
plt.grid(which="both")
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.semilogy(full_freqs[1:], full_force_amp[1:], linewidth=0.8, label="full FFT amplitude")
plt.axvline(nyquist_frequency, color="red", linestyle=":", label="Nyquist limit")
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Wrapped FFT frequency / Hz")
plt.ylabel("Force amplitude / N")
plt.title("Temporary full FFT force-noise view, up to Nyquist")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Force PSD: should be approximately white
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], force_psd[1:], linewidth=0.7, label="simulated force PSD")
plt.semilogy(freqs[1:], S_F_theory[1:], "k--", linewidth=1.5, label="theory: 4 kBT b")
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Force PSD / N$^2$/Hz")
plt.title("Brownian force PSD, approximately white")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Velocity PSD: Lorentzian roll-off
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], v_psd[1:] * 1e12, linewidth=0.7, label="simulated velocity PSD")
plt.semilogy(freqs[1:], v_psd_theory[1:] * 1e12, "k--", linewidth=1.5, label="theory")
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Velocity PSD / ($\\mu$m/s)$^2$/Hz")
plt.title("Free-particle velocity PSD")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Displacement PSD: free-particle Brownian spectrum
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.semilogy(freqs[1:], x_psd[1:] * 1e18, linewidth=0.7, label="simulated displacement PSD")
plt.semilogy(freqs[1:], x_psd_theory[1:] * 1e18, "k--", linewidth=1.5, label="theory")
plt.xscale("log")
plt.xlim(f_plot_min, f_plot_max)
plt.xlabel("Frequency / Hz")
plt.ylabel("Displacement PSD / nm$^2$/Hz")
plt.title("Free-particle Brownian displacement PSD")
plt.legend()
plt.grid(which="both")
plt.tight_layout()


plt.show()
