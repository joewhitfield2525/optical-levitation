from pathlib import Path
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "Power_30min.csv"
OUTPUT_DIR = SCRIPT_DIR / "power_noise_output"


# -------------------------------------------------------------------------
# Publication style
# -------------------------------------------------------------------------
fontsize = 9
mpl.rcParams.update({
    # Figure
    "figure.figsize": (3.4, 2.6),  # single column
    "figure.dpi": 300,

    # Font
    "font.family": "sans-serif",
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize - 1,
    "ytick.labelsize": fontsize - 1,
    "legend.fontsize": fontsize - 1,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # Axes
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    # Grid
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
})


PSD_MATCHED_NOISE_SEED = 12345
LONG_TIME_TRACE_DURATION_S = 10.0


def load_power_data(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    try:
        graph_data_index = next(
            index for index, row in enumerate(rows) if row and row[0] == "--Graph Data--"
        )
    except StopIteration as exc:
        raise ValueError("Could not find '--Graph Data--' section in the CSV file.") from exc

    header_index = graph_data_index + 1
    data_rows = rows[header_index + 1 :]

    time_ms = []
    power_w = []

    for row in data_rows:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue

        try:
            time_ms.append(float(row[0]))
            power_w.append(float(row[1]))
        except ValueError:
            continue

    if not time_ms:
        raise ValueError("No numeric power data was found in the CSV file.")

    return np.array(time_ms), np.array(power_w)


def resample_to_uniform_grid(time_s, signal):
    order = np.argsort(time_s)
    time_s = time_s[order]
    signal = signal[order]

    finite = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[finite]
    signal = signal[finite]

    time_s = time_s - time_s[0]
    dt = np.median(np.diff(time_s))
    uniform_time_s = np.arange(0.0, time_s[-1], dt)
    uniform_signal = np.interp(uniform_time_s, time_s, signal)
    fs = 1.0 / dt

    return uniform_time_s, uniform_signal, fs


def generate_psd_matched_noise(reference_relative_noise):
    reference_relative_noise = (
        reference_relative_noise - np.mean(reference_relative_noise)
    )
    reference_spectrum = np.fft.rfft(reference_relative_noise)
    reference_magnitudes = np.abs(reference_spectrum)

    rng = np.random.default_rng(seed=PSD_MATCHED_NOISE_SEED)
    random_phases = rng.uniform(0.0, 2.0 * np.pi, len(reference_spectrum))
    randomised_spectrum = reference_magnitudes * np.exp(1j * random_phases)

    randomised_spectrum[0] = 0.0
    if len(reference_relative_noise) % 2 == 0:
        randomised_spectrum[-1] = reference_magnitudes[-1] * rng.choice([-1.0, 1.0])

    psd_matched_noise = np.fft.irfft(
        randomised_spectrum,
        n=len(reference_relative_noise),
    )

    target_std = np.std(reference_relative_noise, ddof=1)
    synthetic_std = np.std(psd_matched_noise, ddof=1)
    if synthetic_std > 0:
        psd_matched_noise *= target_std / synthetic_std

    return psd_matched_noise


def relative_power_fluctuation(power_w):
    return (power_w - np.mean(power_w)) / np.mean(power_w)


def power_trace_with_common_mean(relative_power_noise, common_mean_power_w):
    relative_power_noise = relative_power_noise - np.mean(relative_power_noise)
    return common_mean_power_w * (1.0 + relative_power_noise)


def welch_relative_psd(relative_power_noise, fs, segment_duration_s=2.0):
    # The experimental power-meter trace is sampled at about 375 Hz, so its
    # measured PSD is only defined up to the Nyquist frequency of about 190 Hz.
    # Any laser-power noise above this frequency is not constrained by this
    # dataset.
    nperseg = min(
        len(relative_power_noise),
        max(8, int(round(segment_duration_s * fs))),
    )
    noverlap = nperseg // 2

    frequencies, psd = welch(
        relative_power_noise,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    positive = frequencies > 0
    return frequencies[positive], psd[positive], nperseg


def main():
    time_ms, measured_power_w = load_power_data(CSV_PATH)
    measured_time_s, measured_power_uniform_w, measured_fs = resample_to_uniform_grid(
        time_ms / 1000.0,
        measured_power_w,
    )

    measured_relative = relative_power_fluctuation(measured_power_uniform_w)
    psd_matched_relative = generate_psd_matched_noise(measured_relative)
    measured_relative_rms = np.std(measured_relative, ddof=1)

    common_mean_power_w = np.mean(measured_power_uniform_w)
    measured_common_mean_power_w = power_trace_with_common_mean(
        measured_relative,
        common_mean_power_w,
    )
    psd_matched_common_mean_power_w = power_trace_with_common_mean(
        psd_matched_relative,
        common_mean_power_w,
    )

    measured_freqs, measured_psd, measured_nperseg = welch_relative_psd(
        measured_relative,
        measured_fs,
    )
    psd_matched_freqs, psd_matched_psd, psd_matched_nperseg = welch_relative_psd(
        psd_matched_relative,
        measured_fs,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    def plot_psd_comparison(ax, x_max):
        ax.loglog(
            measured_freqs,
            measured_psd,
            color="tab:blue",
            linewidth=0.9,
            label="experimental laser noise",
        )
        ax.loglog(
            psd_matched_freqs,
            psd_matched_psd,
            color="tab:green",
            linewidth=0.9,
            label="PSD-matched synthetic noise",
        )

        ax.set_xlim(1.0, x_max)
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel(r"Relative power PSD / Hz$^{-1}$")
        ax.grid(True, which="both")
        ax.legend(loc="lower left", frameon=False)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    plot_psd_comparison(ax, measured_fs / 2.0)
    ax.set_title("Experimental and synthetic laser-noise PSD")
    fig.tight_layout()

    full_output_png = OUTPUT_DIR / "laser_noise_relative_psd_comparison.png"
    full_output_pdf = OUTPUT_DIR / "laser_noise_relative_psd_comparison.pdf"
    fig.savefig(full_output_png, dpi=300, bbox_inches="tight")
    fig.savefig(full_output_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    common_band_max_frequency = measured_fs / 2.0
    plot_psd_comparison(ax, common_band_max_frequency)
    ax.set_title("Laser-noise PSD, common bandwidth")
    fig.tight_layout()

    common_output_png = OUTPUT_DIR / "laser_noise_relative_psd_common_band.png"
    common_output_pdf = OUTPUT_DIR / "laser_noise_relative_psd_common_band.pdf"
    fig.savefig(common_output_png, dpi=300, bbox_inches="tight")
    fig.savefig(common_output_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    time_trace_max = min(0.1, measured_time_s[-1])
    measured_time_mask = measured_time_s <= time_trace_max
    ax.plot(
        measured_time_s[measured_time_mask],
        measured_relative[measured_time_mask],
        color="tab:blue",
        linewidth=0.7,
        label="experimental laser noise",
    )
    ax.plot(
        measured_time_s[measured_time_mask],
        psd_matched_relative[measured_time_mask],
        color="tab:green",
        linewidth=0.7,
        label="PSD-matched synthetic noise",
    )
    ax.set_xlim(0.0, time_trace_max)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"Relative power fluctuation")
    ax.set_title("Laser-noise time trace")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()

    time_output_png = OUTPUT_DIR / "laser_noise_relative_time_trace.png"
    time_output_pdf = OUTPUT_DIR / "laser_noise_relative_time_trace.pdf"
    fig.savefig(time_output_png, dpi=300, bbox_inches="tight")
    fig.savefig(time_output_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    long_time_trace_max = min(LONG_TIME_TRACE_DURATION_S, measured_time_s[-1])
    measured_long_time_mask = measured_time_s <= long_time_trace_max

    ax.plot(
        measured_time_s[measured_long_time_mask],
        measured_common_mean_power_w[measured_long_time_mask],
        color="tab:blue",
        linewidth=0.7,
        label="experimental laser noise",
    )
    ax.plot(
        measured_time_s[measured_long_time_mask],
        psd_matched_common_mean_power_w[measured_long_time_mask],
        color="tab:green",
        linewidth=0.7,
        label="PSD-matched synthetic noise",
    )
    ax.axhline(
        common_mean_power_w,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="common mean power",
    )
    ax.set_xlim(0.0, long_time_trace_max)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Power / W")
    ax.set_title("Laser-noise time trace with common mean power")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    long_time_output_png = OUTPUT_DIR / "laser_noise_common_mean_time_trace_long.png"
    long_time_output_pdf = OUTPUT_DIR / "laser_noise_common_mean_time_trace_long.pdf"
    fig.savefig(long_time_output_png, dpi=300, bbox_inches="tight")
    fig.savefig(long_time_output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded: {CSV_PATH}")
    print(f"Measured samples: {len(measured_power_w)}")
    print(f"Measured mean power: {np.mean(measured_power_w):.6g} W")
    print(f"Measured standard deviation: {np.std(measured_power_w, ddof=1):.6g} W")
    print(f"Measured relative RMS: {measured_relative_rms:.6g}")
    print(f"Measured resampled frequency: {measured_fs:.3f} Hz")
    print(
        "Experimental PSD is limited to "
        f"{measured_fs / 2.0:.3f} Hz by the measured sampling rate."
    )
    print(f"Measured Welch nperseg: {measured_nperseg}")
    print(
        "PSD-matched relative standard deviation: "
        f"{np.std(psd_matched_relative, ddof=1):.6g}"
    )
    print(f"PSD-matched Welch nperseg: {psd_matched_nperseg}")
    print(f"Common mean power used for time traces: {common_mean_power_w:.6g} W")
    print(
        "Long time-trace means: "
        f"experimental={np.mean(measured_common_mean_power_w):.6g} W, "
        f"PSD-matched={np.mean(psd_matched_common_mean_power_w):.6g} W"
    )
    print("Saved PSD comparisons:")
    print(f"  {full_output_png}")
    print(f"  {full_output_pdf}")
    print(f"  {common_output_png}")
    print(f"  {common_output_pdf}")
    print("Saved time-trace comparison:")
    print(f"  {time_output_png}")
    print(f"  {time_output_pdf}")
    print("Saved long common-mean time-trace comparison:")
    print(f"  {long_time_output_png}")
    print(f"  {long_time_output_pdf}")


if __name__ == "__main__":
    main()
