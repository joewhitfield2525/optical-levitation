import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


# -------------------------------------------------------------------------
# FFT window comparison
# -------------------------------------------------------------------------
# This script compares several common FFT windows using the same basic
# workflow as the optical levitation code:
#
#     signal = signal - mean(signal)
#     fft = np.fft.rfft(signal * window)
#
# It tests two pure sine waves:
#   1. A bin-centred sine wave, whose frequency lies exactly on an FFT bin.
#   2. A non-bin-centred sine wave, whose frequency lies between FFT bins.
#
# The second case is usually the more realistic/useful one because your
# oscillation frequency will almost never land exactly on an FFT bin.


# -------------------------------------------------------------------------
# Sampling parameters
# -------------------------------------------------------------------------
fs = 2000.0          # sampling frequency / Hz
t_end = 5.0          # signal length / s
dt = 1.0 / fs
t = np.arange(0.0, t_end, dt)
N = len(t)

df = fs / N
print("Number of samples N =", N)
print("Frequency resolution df =", df, "Hz")


# -------------------------------------------------------------------------
# Test frequencies
# -------------------------------------------------------------------------
# With t_end = 5 s, df = 0.2 Hz. 50.0 Hz is exactly bin-centred.
# 50.37 Hz is deliberately not bin-centred, so it causes spectral leakage.
f_bin_centred = 50.0
f_not_bin_centred = 50.37


# -------------------------------------------------------------------------
# Window definitions
# -------------------------------------------------------------------------
windows = {
    "rectangular": np.ones(N),
    "hann": np.hanning(N),
    "hamming": np.hamming(N),
    "blackman": np.blackman(N),
    "kaiser beta=8.6": np.kaiser(N, beta=8.6),
}


def single_sine(f0, amplitude=1.0):
    """A clean single-frequency test signal."""
    return amplitude * np.sin(2 * np.pi * f0 * t)


def fft_amplitude(signal, window):
    """
    Return the one-sided amplitude spectrum.

    The division by coherent_gain corrects for the fact that different
    windows reduce the signal amplitude by different amounts.
    """
    signal = signal - np.mean(signal)

    coherent_gain = np.mean(window)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(N, dt)

    amplitude = 2.0 * np.abs(spectrum) / (N * coherent_gain)
    amplitude[0] *= 0.5

    if N % 2 == 0:
        amplitude[-1] *= 0.5

    return freqs, amplitude


def analyse_window(signal, f0, window_name, window):
    freqs, amp = fft_amplitude(signal, window)

    peak_index = np.argmax(amp)
    peak_freq = freqs[peak_index]
    peak_amp = amp[peak_index]

    # A simple leakage metric: compare all power outside +/- 3 bins around the
    # strongest peak with the total spectral power. This is not a perfect
    # universal metric because windows have different main-lobe widths, but it
    # is useful for a quick comparison.
    main_lobe_half_width_bins = 3
    keep = np.zeros_like(amp, dtype=bool)
    lo = max(0, peak_index - main_lobe_half_width_bins)
    hi = min(len(amp), peak_index + main_lobe_half_width_bins + 1)
    keep[lo:hi] = True

    total_power = np.sum(amp**2)
    leakage_power = np.sum(amp[~keep] ** 2)
    leakage_fraction = leakage_power / total_power
    leakage_db = 10 * np.log10(leakage_fraction + 1e-300)

    return {
        "window": window_name,
        "true_frequency": f0,
        "peak_frequency": peak_freq,
        "peak_amplitude": peak_amp,
        "leakage_db": leakage_db,
    }


def print_summary(signal, f0, title):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'window':<16} {'peak freq / Hz':>14} "
        f"{'peak amp':>12} {'leakage / dB':>14}"
    )

    for name, window in windows.items():
        result = analyse_window(signal, f0, name, window)
        print(
            f"{result['window']:<16} "
            f"{result['peak_frequency']:>14.4f} "
            f"{result['peak_amplitude']:>12.5f} "
            f"{result['leakage_db']:>14.2f}"
        )


def plot_window_shapes():
    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    for name, window in windows.items():
        axes[0].plot(t, window, label=name)
        axes[1].plot(t, window, label=name)

    axes[0].set_xlim(0, t_end)
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Window value")
    axes[0].set_title("Window functions over the full time record")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xlim(0, min(0.15, t_end))
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Window value")
    axes[1].set_title("Zoom near the start of the time record")
    axes[1].grid()

    plt.tight_layout()


def plot_spectra(signal, f0, title):
    if plt is None:
        return

    plt.figure(figsize=(9, 5))

    for name, window in windows.items():
        freqs, amp = fft_amplitude(signal, window)

        # Normalize each spectrum to its own peak so the plot compares leakage
        # shape rather than absolute amplitude.
        amp_db = 20 * np.log10(amp / np.max(amp) + 1e-15)
        plt.plot(freqs, amp_db, label=name)

    plt.axvline(f0, color="black", linestyle="--", linewidth=1.0, label="true frequency")
    plt.xlim(f0 - 20, f0 + 20)
    plt.ylim(-120, 5)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Amplitude / dB, normalized to peak")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_spectra_linear(signal, f0, title):
    if plt is None:
        return

    plt.figure(figsize=(9, 5))

    for name, window in windows.items():
        freqs, amp = fft_amplitude(signal, window)
        plt.plot(freqs, amp / np.max(amp), label=name)

    plt.axvline(f0, color="black", linestyle="--", linewidth=1.0, label="true frequency")
    plt.xlim(f0 - 5, f0 + 5)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Normalized amplitude")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()


def main():
    signal_bin = single_sine(f_bin_centred)
    signal_off_bin = single_sine(f_not_bin_centred)

    print_summary(signal_bin, f_bin_centred, "Bin-centred sine wave")
    print_summary(signal_off_bin, f_not_bin_centred, "Non-bin-centred sine wave")

    plot_window_shapes()

    plot_spectra_linear(
        signal_bin,
        f_bin_centred,
        "Linear FFT spectrum: bin-centred sine wave",
    )
    plot_spectra(
        signal_bin,
        f_bin_centred,
        "Log FFT spectrum: bin-centred sine wave",
    )

    plot_spectra_linear(
        signal_off_bin,
        f_not_bin_centred,
        "Linear FFT spectrum: non-bin-centred sine wave",
    )
    plot_spectra(
        signal_off_bin,
        f_not_bin_centred,
        "Log FFT spectrum: non-bin-centred sine wave",
    )

    if plt is None:
        print()
        print("matplotlib is not installed in this Python environment, so plots were skipped.")
        print("Install/use matplotlib to see the comparison figures.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
