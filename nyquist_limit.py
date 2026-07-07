import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Nyquist aliasing demo
# -------------------------------------------------------------------------
# This script deliberately samples a sine wave whose true frequency is above
# the Nyquist limit. The FFT cannot show the true above-Nyquist frequency.
# Instead, the signal appears at a lower "aliased" frequency.


# Sampling setup
fs = 1000.0          # sampling frequency / Hz
dt = 1.0 / fs
t_end = 1.0
t = np.arange(0.0, t_end, dt)
N = len(t)

nyquist = fs / 2
df = fs / N


# Deliberately choose a signal above Nyquist.
f_true = 700.0       # Hz, greater than Nyquist = 500 Hz


def alias_frequency(f, fs):
    """
    Fold frequency f into the observable range 0 to fs/2.
    """
    f_mod = f % fs
    if f_mod > fs / 2:
        return fs - f_mod
    return f_mod


f_alias = alias_frequency(f_true, fs)

print("Sampling frequency fs =", fs, "Hz")
print("Nyquist frequency =", nyquist, "Hz")
print("FFT bin size df =", df, "Hz")
print("True signal frequency =", f_true, "Hz")
print("Expected aliased frequency =", f_alias, "Hz")


# Signal sampled at fs.
sampled_signal = np.sin(2 * np.pi * f_true * t)


# A finely sampled version just for visualising the true continuous signal.
t_fine = np.linspace(0.0, 0.02, 10000)
true_continuous_signal = np.sin(2 * np.pi * f_true * t_fine)
aliased_continuous_signal = np.sin(2 * np.pi * f_alias * t_fine)


# FFT of sampled signal.
window = np.hanning(N)
fft_values = np.fft.rfft((sampled_signal - np.mean(sampled_signal)) * window)
freqs = np.fft.rfftfreq(N, dt)
amplitude = np.abs(fft_values)
amplitude = amplitude / np.max(amplitude)

peak_frequency = freqs[np.argmax(amplitude)]
print("FFT peak frequency =", peak_frequency, "Hz")


# -------------------------------------------------------------------------
# Plot sampled time signal
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(t_fine * 1e3, true_continuous_signal, label=f"true {f_true:.0f} Hz signal")
plt.plot(
    t_fine * 1e3,
    aliased_continuous_signal,
    "--",
    label=f"aliased {f_alias:.0f} Hz signal",
)
plt.scatter(
    t[t <= 0.02] * 1e3,
    sampled_signal[t <= 0.02],
    color="black",
    s=18,
    label="sampled points",
)
plt.xlabel("Time / ms")
plt.ylabel("Signal")
plt.title("Above-Nyquist sine wave looks like a lower-frequency sine after sampling")
plt.legend()
plt.grid()
plt.tight_layout()


# -------------------------------------------------------------------------
# Plot FFT
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(freqs, amplitude, label="FFT of sampled signal")
plt.axvline(f_true, color="red", linestyle="--", label="true frequency")
plt.axvline(f_alias, color="black", linestyle=":", label="aliased frequency")
plt.axvline(nyquist, color="grey", linestyle="-.", label="Nyquist limit")
plt.xlim(0, fs)
plt.xlabel("Frequency / Hz")
plt.ylabel("Normalized amplitude")
plt.title("FFT cannot show frequencies above Nyquist; they alias below it")
plt.legend()
plt.grid()
plt.tight_layout()


plt.show()
