import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

try:
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
except OSError:
    pass

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl = None
    plt = None


# -------------------------------------------------------------------------
# House plotting style
# -------------------------------------------------------------------------
save_path = SCRIPT_DIR / "Figures"
save_figures = True
display_plots = False

if plt is not None:
    os.makedirs(save_path, exist_ok=True)

    fontsize = 7
    mpl.rcParams.update({
        "figure.figsize": (3.4, 2.6),
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
    })


def finish_plot(name=None, fig=None):
    if plt is None:
        return

    target_fig = fig if fig is not None else plt.gcf()
    target_fig.tight_layout()

    if name is not None and save_figures:
        target_fig.savefig(save_path / f"{name}.png", bbox_inches="tight")
        target_fig.savefig(save_path / f"{name}.pdf", bbox_inches="tight")
        print("Wrote", save_path / f"{name}.png")

    if display_plots:
        plt.show()
    else:
        plt.close(target_fig)


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
# User-adjustable parameters
# -------------------------------------------------------------------------
fs = 2000.0
t_end = 5.0
f_bin_centred = 50.0
f_not_bin_centred = 50.37

plot_window_shape_comparison = False
plot_bin_centred_spectra = False
plot_non_bin_centred_spectra = True
plot_linear_spectra = False

fft_plot_half_width_hz = 20.0
fft_plot_ylim_db = (-120.0, 5.0)


# -------------------------------------------------------------------------
# Sampling setup
# -------------------------------------------------------------------------
dt = 1.0 / fs
t = np.arange(0.0, t_end, dt)
N = len(t)

df = fs / N
print("Number of samples N =", N)
print("Frequency resolution df =", df, "Hz")


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
    # strongest peak with the total spectral power.
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

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=False)

    for name, window in windows.items():
        axes[0].plot(t, window, label=name)
        axes[1].plot(t, window, label=name)

    axes[0].set_xlim(0, t_end)
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Window value")
    axes[0].set_title("FFT window functions")
    axes[0].legend(frameon=False)
    axes[0].grid()

    axes[1].set_xlim(0, min(0.15, t_end))
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Window value")
    axes[1].set_title("Window start")
    axes[1].grid()

    finish_plot("fft_window_shapes", fig)


def plot_spectra(signal, f0, title, output_name):
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for name, window in windows.items():
        freqs, amp = fft_amplitude(signal, window)
        amp_db = 20 * np.log10(amp / np.max(amp) + 1e-15)
        ax.plot(freqs, amp_db, label=name)

    ax.axvline(
        f0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="true frequency",
    )
    ax.set_xlim(f0 - fft_plot_half_width_hz, f0 + fft_plot_half_width_hz)
    ax.set_ylim(*fft_plot_ylim_db)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Amplitude / dB, normalized to peak")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.grid()
    finish_plot(output_name, fig)


def plot_spectra_linear(signal, f0, title, output_name):
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for name, window in windows.items():
        freqs, amp = fft_amplitude(signal, window)
        ax.plot(freqs, amp / np.max(amp), label=name)

    ax.axvline(
        f0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="true frequency",
    )
    ax.set_xlim(f0 - 5, f0 + 5)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid()
    finish_plot(output_name, fig)


def main():
    signal_bin = single_sine(f_bin_centred)
    signal_off_bin = single_sine(f_not_bin_centred)

    print_summary(signal_bin, f_bin_centred, "Bin-centred sine wave")
    print_summary(signal_off_bin, f_not_bin_centred, "Non-bin-centred sine wave")

    if plot_window_shape_comparison:
        plot_window_shapes()

    if plot_bin_centred_spectra:
        if plot_linear_spectra:
            plot_spectra_linear(
                signal_bin,
                f_bin_centred,
                "Linear FFT spectrum: bin-centred sine wave",
                "fft_spectrum_bin_centred_linear",
            )
        plot_spectra(
            signal_bin,
            f_bin_centred,
            "Log FFT spectrum: bin-centred sine wave",
            "fft_spectrum_bin_centred_log",
        )

    if plot_non_bin_centred_spectra:
        if plot_linear_spectra:
            plot_spectra_linear(
                signal_off_bin,
                f_not_bin_centred,
                "Linear FFT spectrum: non-bin-centred sine wave",
                "fft_spectrum_non_bin_centred_linear",
            )
        plot_spectra(
            signal_off_bin,
            f_not_bin_centred,
            "Log FFT spectrum: non-bin-centred sine wave",
            "fft_spectrum_non_bin_centred_log",
        )

    if plt is None:
        print()
        print("matplotlib is not installed in this Python environment, so plots were skipped.")
        print("Install/use matplotlib to see the comparison figures.")


if __name__ == "__main__":
    main()
