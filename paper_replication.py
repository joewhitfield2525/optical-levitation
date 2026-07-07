import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Berg-Sorensen and Flyvbjerg 2004: simple Lorentzian PSD replication
# -------------------------------------------------------------------------
# Paper:
#   K. Berg-Sorensen and H. Flyvbjerg,
#   "Power spectrum analysis for optical tweezers",
#   Review of Scientific Instruments 75, 594-612 (2004).
#
# This script reproduces the first/simple calibration model:
#
#   dx/dt = -2 pi f_c x + sqrt(2D) eta(t)
#
# where x is one coordinate of a bead in a harmonic optical trap.
# The expected one-sided displacement PSD is
#
#   P(f) = D / [2 pi^2 (f_c^2 + f^2)].
#
# The more advanced paper corrections, hydrodynamic friction, detector
# filtering, and aliasing are not included here. This is the clean first
# comparison target.


# -------------------------------------------------------------------------
# Parameters close to the paper's Fig. 2 / Fig. 5 data set
# -------------------------------------------------------------------------
fs = 16_000.0
dt = 1.0 / fs
samples_per_window = 2**18
n_windows = 5
t_window = samples_per_window * dt
t_total = n_windows * t_window

fc_true = 374.0
D_um2_per_s = 0.41
D_nm2_per_s = D_um2_per_s * 1e6

f_fit_min = 110.0
f_fit_max_lorentzian = 1_000.0
f_plot_min = 10.0
f_plot_max = fs / 2

lambda_trap = 2 * np.pi * fc_true
x_rms_theory_nm = np.sqrt(D_nm2_per_s / lambda_trap)

print("Sampling frequency fs =", fs, "Hz")
print("Samples per window =", samples_per_window)
print("Window duration =", t_window, "s")
print("Number of windows =", n_windows)
print("Total simulated time =", t_total, "s")
print("True corner frequency fc =", fc_true, "Hz")
print("Diffusion coefficient D =", D_um2_per_s, "um^2/s")
print("Thermal RMS displacement =", x_rms_theory_nm, "nm")


# -------------------------------------------------------------------------
# Exact discrete Ornstein-Uhlenbeck simulation
# -------------------------------------------------------------------------
rng = np.random.default_rng(seed=7)
n_total = samples_per_window * n_windows
t = np.arange(n_total) * dt

ou_decay = np.exp(-lambda_trap * dt)
ou_noise_scale = np.sqrt((D_nm2_per_s / lambda_trap) * (1 - ou_decay**2))

x = np.zeros(n_total)
x[0] = rng.normal(scale=x_rms_theory_nm)

for i in range(n_total - 1):
    x[i + 1] = ou_decay * x[i] + ou_noise_scale * rng.normal()

print("Simulated RMS displacement =", np.std(x), "nm")


# -------------------------------------------------------------------------
# PSD helpers
# -------------------------------------------------------------------------
def one_sided_periodogram(signal_nm, sample_rate):
    """
    One-sided raw periodogram in nm^2/Hz.

    No window is used here because the paper notes that leakage is negligible
    for a smooth Lorentzian-like spectrum.
    """
    signal_nm = signal_nm - np.mean(signal_nm)
    n = len(signal_nm)
    dt_local = 1.0 / sample_rate

    fft_values = np.fft.rfft(signal_nm)
    freqs = np.fft.rfftfreq(n, dt_local)

    psd = (dt_local / n) * np.abs(fft_values)**2
    psd[1:-1] *= 2.0

    return freqs, psd


def lorentzian_psd(freqs, fc, D_nm2_s):
    return D_nm2_s / (2 * np.pi**2 * (fc**2 + freqs**2))


def log_block_spectrum(freqs, psd, n_blocks=150, f_min=1.0, f_max=None):
    """
    Block a noisy spectrum into logarithmically spaced frequency bins.
    """
    if f_max is None:
        f_max = freqs[-1]

    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_blocks + 1)
    blocked_freqs = []
    blocked_psd = []
    blocked_std = []
    blocked_counts = []

    for left, right in zip(edges[:-1], edges[1:]):
        mask = (freqs >= left) & (freqs < right)
        if np.count_nonzero(mask) == 0:
            continue

        blocked_freqs.append(np.mean(freqs[mask]))
        blocked_psd.append(np.mean(psd[mask]))
        blocked_std.append(np.std(psd[mask]) / np.sqrt(np.count_nonzero(mask)))
        blocked_counts.append(np.count_nonzero(mask))

    return (
        np.array(blocked_freqs),
        np.array(blocked_psd),
        np.array(blocked_std),
        np.array(blocked_counts),
    )


def fit_lorentzian_by_grid(freqs, psd, f_min, f_max):
    """
    Fit P(f) = D / [2 pi^2 (fc^2 + f^2)] by a simple grid search over fc.

    For each trial fc, the best D is found analytically by least squares.
    This avoids requiring scipy for the first replication script.
    """
    mask = (freqs >= f_min) & (freqs <= f_max) & np.isfinite(psd) & (psd > 0)
    f_fit = freqs[mask]
    p_fit = psd[mask]

    fc_grid = np.linspace(100.0, 1_000.0, 5_000)
    best_error = np.inf
    best_fc = np.nan
    best_D = np.nan

    for fc_trial in fc_grid:
        basis = 1.0 / (2 * np.pi**2 * (fc_trial**2 + f_fit**2))
        D_trial = np.sum(basis * p_fit) / np.sum(basis**2)
        residual = p_fit - D_trial * basis
        error = np.mean(residual**2)

        if error < best_error:
            best_error = error
            best_fc = fc_trial
            best_D = D_trial

    return best_fc, best_D


# -------------------------------------------------------------------------
# Calculate spectra from five windows, as in the paper's first data set
# -------------------------------------------------------------------------
window_psds = []

for window_index in range(n_windows):
    start = window_index * samples_per_window
    stop = start + samples_per_window
    freqs, psd_window = one_sided_periodogram(x[start:stop], fs)
    window_psds.append(psd_window)

window_psds = np.array(window_psds)
psd_average = np.mean(window_psds, axis=0)
psd_raw_first_window = window_psds[0]

positive_mask = freqs > 0
freqs_positive = freqs[positive_mask]
psd_average_positive = psd_average[positive_mask]
psd_raw_positive = psd_raw_first_window[positive_mask]

theory_psd = lorentzian_psd(freqs_positive, fc_true, D_nm2_per_s)

blocked_freqs, blocked_psd, blocked_std, blocked_counts = log_block_spectrum(
    freqs_positive,
    psd_average_positive,
    n_blocks=150,
    f_min=f_plot_min,
    f_max=f_plot_max,
)

fc_fit, D_fit_nm2_per_s = fit_lorentzian_by_grid(
    blocked_freqs,
    blocked_psd,
    f_fit_min,
    f_fit_max_lorentzian,
)

print("Lorentzian fit range =", f_fit_min, "to", f_fit_max_lorentzian, "Hz")
print("Fitted fc =", fc_fit, "Hz")
print("Fitted D =", D_fit_nm2_per_s / 1e6, "um^2/s")


# -------------------------------------------------------------------------
# Plot trajectory segment
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(t[: int(0.2 * fs)], x[: int(0.2 * fs)], linewidth=0.8)
plt.xlabel("Time / s")
plt.ylabel("x / nm")
plt.title("Simulated overdamped Brownian motion in harmonic trap")
plt.grid()
plt.tight_layout()


# -------------------------------------------------------------------------
# Plot raw, averaged, blocked, and theoretical PSD
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.loglog(
    freqs_positive,
    psd_raw_positive,
    linewidth=0.4,
    alpha=0.35,
    label="raw PSD, one window",
)
plt.loglog(
    freqs_positive,
    psd_average_positive,
    linewidth=0.55,
    alpha=0.65,
    label="average of 5 raw PSDs",
)
plt.errorbar(
    blocked_freqs,
    blocked_psd,
    yerr=blocked_std,
    fmt="o",
    markersize=3.0,
    linewidth=0.8,
    capsize=2,
    label="log-blocked average PSD",
)
plt.loglog(
    freqs_positive,
    theory_psd,
    "k--",
    linewidth=2.0,
    label="Lorentzian theory",
)
plt.axvspan(
    f_fit_min,
    f_fit_max_lorentzian,
    color="tab:green",
    alpha=0.12,
    label="fit range",
)
plt.xlabel("Frequency / Hz")
plt.ylabel("PSD / nm$^2$/Hz")
plt.title("Lorentzian PSD replication")
plt.xlim(f_plot_min, f_plot_max)
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Plot inverse PSD, similar in spirit to paper Fig. 1/Fig. 2 diagnostics
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(
    blocked_freqs,
    1.0 / blocked_psd,
    "o",
    markersize=3.0,
    label="blocked simulated data",
)
plt.plot(
    freqs_positive,
    1.0 / theory_psd,
    "k--",
    linewidth=2.0,
    label="inverse Lorentzian theory",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency / Hz")
plt.ylabel("1 / PSD / Hz nm$^{-2}$")
plt.title("Inverse PSD diagnostic")
plt.xlim(f_plot_min, f_plot_max)
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()


# -------------------------------------------------------------------------
# Check the exponential scatter of raw periodogram values
# -------------------------------------------------------------------------
fit_mask_raw = (freqs_positive >= f_fit_min) & (freqs_positive <= f_fit_max_lorentzian)
raw_ratio = psd_raw_positive[fit_mask_raw] / lorentzian_psd(
    freqs_positive[fit_mask_raw],
    fc_true,
    D_nm2_per_s,
)

hist_values, hist_edges = np.histogram(raw_ratio, bins=80, range=(0, 6), density=True)
hist_centres = 0.5 * (hist_edges[:-1] + hist_edges[1:])

plt.figure(figsize=(9, 5))
plt.semilogy(hist_centres, hist_values, drawstyle="steps-mid", label="raw PSD / theory")
plt.semilogy(hist_centres, np.exp(-hist_centres), "k--", label="exp(-x)")
plt.xlabel("Raw PSD / expected PSD")
plt.ylabel("Probability density")
plt.title("Raw periodogram scatter is approximately exponential")
plt.legend()
plt.grid()
plt.tight_layout()


plt.show()
