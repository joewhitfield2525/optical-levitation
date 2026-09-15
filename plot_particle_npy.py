from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/josephwhitfield/Masters/Summer Project/11_06_2026 PSD")
OUTPUT_DIR = SCRIPT_DIR / "experimental data"

ADC_MIN = 32500
ADC_MAX = 31
POSITION_SCALE_UM = 1750/6
MAX_PLOT_POINTS = 200_000
psd_plot_max_frequency_override = None
psd_min_frequency_factor = 0.95
psd_min_value = 1e-30
psd_max_value = None
welch_average_segment_duration_seconds = 2.0
psd_overlap_fraction = 0.5
SPECTRUM_FIGSIZE = (8, 5)
TEMPERATURE_K = 300.0
KB = 1.380649e-23
UM_TO_M = 1e-6
N_PER_M_TO_PN_PER_UM = 1e6


def save_raw_npy_data_to_csv(data, npy_path):
    csv_path = npy_path.with_name(f"{npy_path.stem}_raw_npy_data.csv")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D NPY array, got shape {data.shape}.")

    rows_for_csv = data.T
    headers = ["channel_1_raw", "channel_2_raw", "channel_3_raw", "channel_4_raw", "time_ns"]

    if data.shape[0] > len(headers):
        headers.extend(
            f"extra_row_{row_number}"
            for row_number in range(len(headers) + 1, data.shape[0] + 1)
        )

    fmt = "%d" if np.issubdtype(data.dtype, np.integer) else "%.18e"
    np.savetxt(
        csv_path,
        rows_for_csv,
        delimiter=",",
        header=",".join(headers[: data.shape[0]]),
        comments="",
        fmt=fmt,
    )

    return csv_path


def downsample_for_plot(x_values, y_values, max_points=MAX_PLOT_POINTS):
    step = max(1, len(x_values) // max_points)
    return x_values[::step], y_values[::step]


def show_position_plot(time_s, position_m, title, output_path):
    displacement_m = position_m - np.mean(position_m)
    plot_time_s, plot_displacement_m = downsample_for_plot(time_s, displacement_m)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_time_s, plot_displacement_m * 1e6, linewidth=0.8)
    plt.xlabel("Time / s")
    plt.ylabel("Displacement from mean / micrometres")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def median_positive_timestep(time_values):
    time_differences = np.diff(time_values)
    finite_positive_dt = time_differences[
        np.isfinite(time_differences) & (time_differences > 0)
    ]

    if len(finite_positive_dt) == 0:
        raise ValueError("Need increasing finite time values.")

    return np.median(finite_positive_dt)


def positive_welch_averaged_psd(signal, time_values):
    dt = median_positive_timestep(time_values)
    fs = 1 / dt
    signal = signal - np.mean(signal)

    nperseg = min(
        max(2, int(round(welch_average_segment_duration_seconds * fs))),
        len(signal)
    )

    if nperseg < 2:
        raise ValueError("Welch PSD settings must leave at least two samples.")

    noverlap = min(int(round(psd_overlap_fraction * nperseg)), nperseg - 1)

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        return_onesided=True
    )

    positive_mask = (frequencies > 0) & (psd > 0)
    frequencies = frequencies[positive_mask]
    psd = psd[positive_mask]

    return frequencies, psd, nperseg, fs / nperseg


def show_psd_plot(time_s, position_m, title):
    frequencies_hz, psd_m2_per_hz, nperseg, bin_size_hz = (
        positive_welch_averaged_psd(position_m, time_s)
    )
    dt_s = median_positive_timestep(time_s)
    sampling_frequency_hz = 1.0 / dt_s
    nyquist_frequency_hz = sampling_frequency_hz / 2
    psd_plot_max_frequency = (
        nyquist_frequency_hz
        if psd_plot_max_frequency_override is None
        else psd_plot_max_frequency_override
    )
    minimum_plot_frequency = psd_min_frequency_factor * frequencies_hz[0]

    plt.figure(figsize=SPECTRUM_FIGSIZE)
    plt.loglog(frequencies_hz, psd_m2_per_hz, linewidth=0.8)
    plt.axvline(nyquist_frequency_hz, color="red", linestyle=":", label="Nyquist limit")
    plt.xlim(minimum_plot_frequency, psd_plot_max_frequency)

    if psd_max_value is None:
        plt.ylim(bottom=psd_min_value)
    else:
        plt.ylim(psd_min_value, psd_max_value)

    plt.xlabel("Frequency / Hz")
    plt.ylabel("PSD / m^2 Hz^-1")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return nperseg, bin_size_hz


def calculate_effective_spring_constant(position_m, temperature_k=TEMPERATURE_K):
    finite_position_m = np.asarray(position_m)[np.isfinite(position_m)]

    if len(finite_position_m) < 2:
        raise ValueError("Need at least two finite position samples.")

    displacement_m = finite_position_m - np.mean(finite_position_m)
    variance_m2 = np.var(displacement_m, ddof=1)

    if variance_m2 <= 0.0:
        raise ValueError("Position variance must be positive.")

    k_eff_n_per_m = KB * temperature_k / variance_m2

    return {
        "mean_m": np.mean(finite_position_m),
        "rms_m": np.sqrt(variance_m2),
        "variance_m2": variance_m2,
        "k_eff_n_per_m": k_eff_n_per_m,
        "k_eff_pn_per_um": k_eff_n_per_m * N_PER_M_TO_PN_PER_UM,
    }


def print_spring_constant(label, spring_constant):
    print(f"{label} effective spring constant from equipartition:")
    print(f"  mean position = {spring_constant['mean_m'] * 1e6:.6g} micrometres")
    print(f"  RMS displacement = {spring_constant['rms_m'] * 1e6:.6g} micrometres")
    print(f"  variance = {spring_constant['variance_m2']:.6e} m^2")
    print(
        f"  k_eff = {spring_constant['k_eff_n_per_m']:.6e} N/m "
        f"= {spring_constant['k_eff_pn_per_um']:.6g} pN/um"
    )


def process_npy_file(npy_path):
    data = np.load(npy_path)

    print(f"\nLoaded: {npy_path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    if data.shape[0] < 5:
        raise ValueError(
            "Expected at least 5 rows: four photodiode channels plus time data."
        )

    channel_1 = data[0].astype(float)
    channel_2 = data[1].astype(float)
    channel_3 = data[2].astype(float)
    channel_4 = data[3].astype(float)
    recorded_time_s = data[4].astype(float) / 1e9

    adc_delta = ADC_MIN - ADC_MAX
    channel_1 = channel_1 - adc_delta
    channel_2 = channel_2 - adc_delta
    channel_3 = channel_3 - adc_delta
    channel_4 = channel_4 - adc_delta

    with np.errstate(divide="ignore", invalid="ignore"):
        vertical_um = ((channel_2 - channel_1) / (channel_1 + channel_2)) * POSITION_SCALE_UM
        horizontal_um = ((channel_4 - channel_3) / (channel_3 + channel_4)) * POSITION_SCALE_UM
        vertical_m = vertical_um * UM_TO_M
        horizontal_m = horizontal_um * UM_TO_M

    finite_samples = (
        np.isfinite(recorded_time_s)
        & np.isfinite(vertical_m)
        & np.isfinite(horizontal_m)
    )
    recorded_time_s = recorded_time_s[finite_samples]
    vertical_m = vertical_m[finite_samples]
    horizontal_m = horizontal_m[finite_samples]

    if len(recorded_time_s) < 2:
        raise ValueError("Need at least two finite samples after position conversion.")

    time_s = recorded_time_s - recorded_time_s[0]
    recorded_dt_s = median_positive_timestep(time_s)
    sampling_frequency_hz = 1 / recorded_dt_s

    vertical_spring = calculate_effective_spring_constant(vertical_m)
    horizontal_spring = calculate_effective_spring_constant(horizontal_m)

    print(f"Finite samples used: {len(time_s)}")
    print("Recorded duration =", time_s[-1] - time_s[0], "s")
    print("Median sampling frequency =", sampling_frequency_hz, "Hz")
    print_spring_constant("Vertical", vertical_spring)
    print_spring_constant("Horizontal", horizontal_spring)

    safe_stem = npy_path.stem.replace("/", "_")
    vertical_output_path = OUTPUT_DIR / f"{safe_stem}_vertical_trajectory.png"
    horizontal_output_path = OUTPUT_DIR / f"{safe_stem}_horizontal_trajectory.png"

    show_position_plot(
        time_s,
        vertical_m,
        f"{npy_path.stem} - Vertical Position vs Time",
        vertical_output_path,
    )
    print(f"Saved vertical trajectory plot to: {vertical_output_path}")

    show_position_plot(
        time_s,
        horizontal_m,
        f"{npy_path.stem} - Horizontal Position vs Time",
        horizontal_output_path,
    )
    print(f"Saved horizontal trajectory plot to: {horizontal_output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npy_paths = sorted(DATA_DIR.glob("*.npy"))

    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {DATA_DIR}")

    print(f"Found {len(npy_paths)} NPY files in: {DATA_DIR}")

    for npy_path in npy_paths:
        process_npy_file(npy_path)

    print(f"\nSaved trajectory plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
