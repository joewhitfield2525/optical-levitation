"""
Plot Rohatschek photophoretic force vs pressure for absorption fractions.

The listed absorption values are treated as dimensionless absorbed fractions

    f_abs = absorbed power / incident geometrical power.

The compact Rohatschek equation needs a photophoretic asymmetry factor J1, not
only a total absorbed fraction. By default this script assumes front-loaded
absorption, so

    J1 = -0.5 f_abs

in the sign convention used by compare_photophoresis_models.py.

Run:
    python3 plot_rohatschek_absorption_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).with_name(".matplotlib-cache")
    mpl_cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(__file__).with_name(".cache")
    xdg_cache.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib.pyplot as plt

try:
    from photophoresis_testing import (
        Beam,
        Gas,
        Particle,
        rohatschek_characteristic_values,
        rohatschek_force,
    )
except ImportError:
    from compare_photophoresis_models import (
        Beam,
        Gas,
        Particle,
        rohatschek_characteristic_values,
        rohatschek_force,
    )


DEFAULT_ABSORPTION_VALUES = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="rohatschek_absorption_sweep")
    parser.add_argument(
        "--absorption-values",
        type=float,
        nargs="+",
        default=DEFAULT_ABSORPTION_VALUES,
        help="absorbed fractions f_abs to compare",
    )
    parser.add_argument(
        "--asymmetry-factor",
        type=float,
        default=0.5,
        help=(
            "sets J1 = -asymmetry_factor * f_abs; 0.5 is the front-loaded "
            "opaque-absorber convention"
        ),
    )
    parser.add_argument("--radius-um", type=float, default=6.3)
    parser.add_argument("--particle-k", type=float, default=0.135)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument("--n-real", type=float, default=1.555)
    parser.add_argument("--medium-n", type=float, default=1.00027)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--waist-um", type=float, default=20.0)
    parser.add_argument("--m2", type=float, default=1.2)
    parser.add_argument("--manual-zr-um", type=float, default=100.0)
    parser.add_argument(
        "--use-m2-rayleigh-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match project_3d_cython_new.py by default",
    )
    parser.add_argument("--power", type=float, default=0.225)
    parser.add_argument("--gas-temperature", type=float, default=300.0)
    parser.add_argument("--gas-molar-mass", type=float, default=0.029)
    parser.add_argument("--gas-viscosity", type=float, default=1.8e-5)
    parser.add_argument("--gas-k", type=float, default=0.0262)
    parser.add_argument("--pressure-min", type=float, default=0.03)
    parser.add_argument("--pressure-max", type=float, default=2.0e5)
    parser.add_argument("--pressure-points", type=int, default=260)
    parser.add_argument("--ray-grid-points", type=int, default=200)
    parser.add_argument("--equilibrium-z-min-mm", type=float, default=-10.0)
    parser.add_argument("--equilibrium-z-max-mm", type=float, default=10.0)
    parser.add_argument("--equilibrium-scan-points", type=int, default=500)
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="also save the plotted force data as a CSV file",
    )
    return parser.parse_args()


def rayleigh_range(args: argparse.Namespace, beam: Beam) -> float:
    if args.use_m2_rayleigh_range:
        return np.pi * beam.waist**2 / (args.m2 * beam.wavelength)
    return args.manual_zr_um * 1.0e-6


def on_axis_intensity(z: float, beam: Beam, z_r: float) -> float:
    return beam.peak_intensity / (1.0 + (z / z_r) ** 2)


def build_on_axis_ray_force(args: argparse.Namespace, beam: Beam, z_r: float):
    """
    Return the same on-axis Ashkin-style ray-optics force used by the 3D code.
    """
    radius = args.radius_um * 1.0e-6
    n_medium = args.medium_n
    n_particle = args.n_real
    c_light = 299792458.0

    u_values = np.linspace(-radius, radius, args.ray_grid_points)
    v_values = np.linspace(-radius, radius, args.ray_grid_points)
    du = u_values[1] - u_values[0]
    dv = v_values[1] - v_values[0]
    dA = du * dv

    U, V = np.meshgrid(u_values, v_values, indexing="ij")
    rho = np.sqrt(U**2 + V**2)
    ray_mask = rho < radius

    U_hit = U[ray_mask]
    V_hit = V[ray_mask]
    rho_hit = rho[ray_mask]

    sin_theta_i = rho_hit / radius
    theta_i = np.arcsin(np.clip(sin_theta_i, 0.0, 1.0))
    sin_theta_r = (n_medium / n_particle) * sin_theta_i
    theta_r = np.arcsin(np.clip(sin_theta_r, 0.0, 1.0))

    cos_i = np.cos(theta_i)
    cos_r = np.cos(theta_r)

    Rs = ((n_medium * cos_i - n_particle * cos_r) / (n_medium * cos_i + n_particle * cos_r))**2
    Rp = ((n_medium * cos_r - n_particle * cos_i) / (n_medium * cos_r + n_particle * cos_i))**2

    def ashkin_efficiencies_from_reflectance(fresnel_R):
        fresnel_T = 1 - fresnel_R
        denom = 1 + fresnel_R**2 + 2 * fresnel_R * np.cos(2 * theta_r)

        Q_scat = (
            1
            + fresnel_R * np.cos(2 * theta_i)
            - (
                fresnel_T**2
                * (np.cos(2 * theta_i - 2 * theta_r) + fresnel_R * np.cos(2 * theta_i))
                / denom
            )
        )
        Q_grad = (
            fresnel_R * np.sin(2 * theta_i)
            - (
                fresnel_T**2
                * (np.sin(2 * theta_i - 2 * theta_r) + fresnel_R * np.sin(2 * theta_i))
                / denom
            )
        )
        return Q_scat, Q_grad

    Q_s_s_pol, Q_g_s_pol = ashkin_efficiencies_from_reflectance(Rs)
    Q_s_p_pol, Q_g_p_pol = ashkin_efficiencies_from_reflectance(Rp)
    Q_s = 0.5 * (Q_s_s_pol + Q_s_p_pol)
    Q_g = 0.5 * (Q_g_s_pol + Q_g_p_pol)

    rho_safe = np.where(rho_hit > 0, rho_hit, 1.0)
    u_hat = U_hit / rho_safe

    def fz_ray_on_axis(z: float) -> float:
        s = 1.0 + (z / z_r) ** 2
        dP = beam.peak_intensity * (1.0 / s) * np.exp(
            -2.0 * (U_hit**2 + V_hit**2) / (beam.waist**2 * s)
        ) * dA

        if abs(z) < 1.0e-30:
            s_x = np.zeros_like(U_hit)
        else:
            wavefront_radius = z * (1.0 + (z_r / z) ** 2)
            s_x = U_hit / wavefront_radius

        s_z = np.ones_like(U_hit)
        s_norm = np.sqrt(s_x**2 + s_z**2)
        s_x = s_x / s_norm
        s_z = s_z / s_norm

        e_perp_z = -s_x
        prefactor = n_medium / c_light
        fz_scat = prefactor * np.sum(dP * Q_s * s_z)
        fz_grad = prefactor * np.sum(dP * (-Q_g * u_hat * e_perp_z))
        return float(fz_scat + fz_grad)

    return fz_ray_on_axis


def find_stable_equilibrium(
    pressure: float,
    j1: float,
    gas: Gas,
    particle: Particle,
    beam: Beam,
    z_r: float,
    fz_ray_on_axis,
    particle_weight: float,
    args: argparse.Namespace,
    previous_root: float | None = None,
) -> float:
    def fz_net(z: float) -> float:
        local_intensity = on_axis_intensity(z, beam, z_r)
        f_photo = rohatschek_force(pressure, local_intensity, gas, particle, j1)
        return float(fz_ray_on_axis(z) + f_photo - particle_weight)

    def slope_at(z: float) -> float:
        h = 1.0e-9
        return (fz_net(z + h) - fz_net(z - h)) / (2.0 * h)

    z_min = args.equilibrium_z_min_mm * 1.0e-3
    z_max = args.equilibrium_z_max_mm * 1.0e-3

    if previous_root is not None and np.isfinite(previous_root):
        for half_width in (5e-6, 20e-6, 50e-6, 100e-6, 250e-6, 1e-3, 5e-3):
            left = max(z_min, previous_root - half_width)
            right = min(z_max, previous_root + half_width)
            if fz_net(left) * fz_net(right) <= 0.0:
                root = brentq(fz_net, left, right, xtol=1e-14, rtol=1e-14)
                if slope_at(root) < 0.0:
                    return root

    z_scan = np.linspace(z_min, z_max, args.equilibrium_scan_points)
    f_scan = np.array([fz_net(z_value) for z_value in z_scan])
    stable_roots = []

    for i in range(len(z_scan) - 1):
        if f_scan[i] * f_scan[i + 1] <= 0.0:
            root = brentq(fz_net, z_scan[i], z_scan[i + 1], xtol=1e-14, rtol=1e-14)
            if slope_at(root) < 0.0:
                stable_roots.append(root)

    if not stable_roots:
        return np.nan

    return stable_roots[0]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gas = Gas(
        temperature=args.gas_temperature,
        molar_mass=args.gas_molar_mass,
        viscosity=args.gas_viscosity,
        thermal_conductivity=args.gas_k,
    )
    beam = Beam(
        wavelength=args.wavelength_nm * 1.0e-9,
        waist=args.waist_um * 1.0e-6,
        power=args.power,
    )
    pressures = np.geomspace(args.pressure_min, args.pressure_max, args.pressure_points)

    curves: dict[str, np.ndarray] = {}
    raw_curves: list[tuple[str, np.ndarray]] = []
    summary_rows: list[tuple[float, float, float, float]] = []

    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    for absorption_fraction in args.absorption_values:
        if absorption_fraction < 0.0 or absorption_fraction > 1.0:
            raise ValueError("absorption fractions must lie between 0 and 1")
        particle = Particle(
            radius=args.radius_um * 1.0e-6,
            thermal_conductivity=args.particle_k,
            density=args.density,
            refractive_index=complex(args.n_real, 0.0),
            thermal_accommodation=args.alpha,
        )
        j1 = -args.asymmetry_factor * absorption_fraction
        force = np.asarray(
            rohatschek_force(
                pressures,
                beam.peak_intensity,
                gas,
                particle,
                j1,
            )
        )
        p_max, f_max = rohatschek_characteristic_values(
            beam.peak_intensity,
            gas,
            particle,
            j1,
        )
        label = rf"$f_{{abs}}={absorption_fraction:.0e}$"
        raw_curves.append((label, force))
        summary_rows.append((absorption_fraction, j1, p_max, f_max))

    normalisation_force = max(
        np.max(np.abs(force))
        for _, force in raw_curves
    )
    if normalisation_force <= 0.0:
        raise ValueError("cannot normalise zero photophoretic force curves")

    particle_radius = args.radius_um * 1.0e-6
    particle_mass = args.density * (4.0 / 3.0) * np.pi * particle_radius**3
    particle_weight = particle_mass * 9.81
    normalised_particle_weight = particle_weight / normalisation_force

    for label, force in raw_curves:
        normalised_force = np.abs(force) / normalisation_force
        curves[label] = normalised_force
        ax.plot(pressures, normalised_force, linewidth=2.0, label=label)

    ax.axhline(
        normalised_particle_weight,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=rf"particle weight $mg$ = {normalised_particle_weight:.1e}"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pressure / Pa")
    ax.set_ylabel(r"normalised force, $|F| / \max(|F|)$")
    ax.set_title("Rohatschek force normalised to the largest curve peak")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="absorbed fraction")

    plot_path = output_dir / "rohatschek_force_vs_pressure_absorption.png"
    fig.savefig(plot_path, dpi=220)

    particle_focus = Particle(
        radius=args.radius_um * 1.0e-6,
        thermal_conductivity=args.particle_k,
        density=args.density,
        refractive_index=complex(args.n_real, 0.0),
        thermal_accommodation=args.alpha,
    )
    z_r = rayleigh_range(args, beam)
    focus_z = 0.0
    focus_intensity = on_axis_intensity(focus_z, beam, z_r)

    focus_raw_curves: list[tuple[str, np.ndarray]] = []

    for absorption_fraction in args.absorption_values:
        j1 = -args.asymmetry_factor * absorption_fraction
        label = rf"$f_{{abs}}={absorption_fraction:.0e}$"
        force_at_focus = np.asarray(
            rohatschek_force(
                pressures,
                focus_intensity,
                gas,
                particle_focus,
                j1,
            )
        )

        focus_raw_curves.append((label, force_at_focus))

    focus_normalisation_force = max(
        np.max(np.abs(force))
        for _, force in focus_raw_curves
    )
    if focus_normalisation_force <= 0.0:
        raise ValueError("cannot normalise zero focus photophoretic force curves")

    normalised_focus_weight = particle_weight / focus_normalisation_force

    fig_focus, ax_focus = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    focus_curves: dict[str, np.ndarray] = {}
    for label, force in focus_raw_curves:
        normalised_force = np.abs(force) / focus_normalisation_force
        focus_curves[label] = normalised_force
        ax_focus.plot(pressures, normalised_force, linewidth=2.0, label=label)

    ax_focus.axhline(
        normalised_focus_weight,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=rf"particle weight $mg$ = {normalised_focus_weight:.1e}"
    )
    ax_focus.set_xscale("log")
    ax_focus.set_yscale("log")
    ax_focus.set_xlabel("pressure / Pa")
    ax_focus.set_ylabel(r"normalised force, $|F(z=0)| / \max(|F(z=0)|)$")
    ax_focus.set_title("Rohatschek force evaluated at the beam focus")
    ax_focus.grid(True, which="both", alpha=0.3)
    ax_focus.legend(title="absorbed fraction")

    focus_plot_path = output_dir / "rohatschek_force_vs_pressure_absorption_at_focus.png"
    fig_focus.savefig(focus_plot_path, dpi=220)

    with (output_dir / "summary.txt").open("w") as f:
        f.write("Rohatschek absorption sweep\n")
        f.write("===========================\n\n")
        f.write("Absorption values are dimensionless absorbed fractions.\n")
        f.write(f"J1 = -{args.asymmetry_factor:.9e} * f_abs\n\n")
        f.write(f"particle_weight_N,{particle_weight:.9e}\n")
        f.write(f"normalisation_force_N,{normalisation_force:.9e}\n")
        f.write(f"normalised_particle_weight,{normalised_particle_weight:.9e}\n\n")
        f.write(f"focus_normalisation_force_N,{focus_normalisation_force:.9e}\n")
        f.write(f"normalised_focus_particle_weight,{normalised_focus_weight:.9e}\n")
        f.write(f"focus_z_m,{focus_z:.9e}\n")
        f.write(f"focus_intensity_W_m2,{focus_intensity:.9e}\n")
        f.write(f"rayleigh_range_used_m,{z_r:.9e}\n\n")
        f.write("f_abs,J1,p_max_Pa,F_max_N\n")
        for row in summary_rows:
            f.write(",".join(f"{value:.9e}" for value in row) + "\n")

    if args.save_csv:
        csv_path = output_dir / "rohatschek_force_vs_pressure_absorption.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pressure_Pa", *curves.keys()])
            for index, pressure in enumerate(pressures):
                writer.writerow([pressure, *[force[index] for force in curves.values()]])

        focus_csv_path = output_dir / "rohatschek_force_vs_pressure_absorption_at_focus.csv"
        with focus_csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pressure_Pa", *focus_curves.keys()])
            for index, pressure in enumerate(pressures):
                writer.writerow([
                    pressure,
                    *[force[index] for force in focus_curves.values()],
                ])

    print(f"Wrote plot to {plot_path.resolve()}")
    print(f"Wrote focus plot to {focus_plot_path.resolve()}")
    print("Rohatschek absorption sweep")
    print(f"Using J1 = -{args.asymmetry_factor:g} * f_abs")
    print("Particle weight / shared force normalisation =", normalised_particle_weight)
    print(
        "Particle weight / focus force normalisation =",
        normalised_focus_weight,
    )
    print("f_abs       J1           p_max / Pa    F_max / N")
    for absorption_fraction, j1, p_max, f_max in summary_rows:
        print(
            f"{absorption_fraction:10.1e}  {j1:11.3e}  "
            f"{p_max:11.3e}  {f_max:11.3e}"
        )
    plt.show()


if __name__ == "__main__":
    main()
