"""
Compare Hettner, Rohatschek, Loesche-Husmann, and Beresnev
photophoretic force models for an illuminated spherical particle.

The script keeps the optical/thermal part explicit:

* absorption is represented by a Beer-Lambert volumetric heat source inside
  the sphere, optionally normalized to the Lorenz-Mie absorption efficiency;
* J0 and J1 are computed from the heat-source moments;
* the steady heat equation is solved in Legendre modes to get the particle
  surface coefficient A1 and adjacent-gas coefficient B1;
* all force models use the calculated optical/thermal asymmetry by default;
* Rohatschek can optionally use a fixed asymmetry factor with --rohatschek-j1.

Run:
    python3 photophoresis_testing.py

Use --save-csv if you also want CSV data tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_legendre, roots_legendre, spherical_jn, spherical_yn

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).with_name(".matplotlib-cache")
    mpl_cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(__file__).with_name(".cache")
    xdg_cache.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib.pyplot as plt


R_GAS = 8.31446261815324
K_BOLTZMANN = 1.380649e-23
SIGMA_SB = 5.670374419e-8
C_LIGHT = 299_792_458.0
OPAQUE_ROHATSCHEK_J1 = -0.5


@dataclass(frozen=True)
class Gas:
    temperature: float = 300.0
    molar_mass: float = 0.0289652
    viscosity: float = 1.846e-5
    thermal_conductivity: float = 0.0262
    rohatschek_creep: float = 1.14

    def density(self, pressure: np.ndarray | float) -> np.ndarray | float:
        return pressure * self.molar_mass / (R_GAS * self.temperature)

    @property
    def mean_thermal_speed(self) -> float:
        return np.sqrt(8.0 * R_GAS * self.temperature / (np.pi * self.molar_mass))

    @property
    def molecule_mass(self) -> float:
        return self.molar_mass / 6.02214076e23

    def mean_free_path(self, pressure: np.ndarray | float) -> np.ndarray | float:
        return (
            self.viscosity
            / pressure
            * np.sqrt(np.pi * R_GAS * self.temperature / (2.0 * self.molar_mass))
        )


@dataclass(frozen=True)
class LoescheCoefficients:
    thermal_creep: float = 2.179
    temperature_jump: float = 2.179
    frictional_slip: float = 1.131
    thermal_stress_slip: float = 1.0
    momentum_accommodation: float = 1.0
    energy_accommodation: float = 1.0
    heat_transfer_factor: float = 0.75


@dataclass(frozen=True)
class Particle:
    radius: float = 5.0e-6
    thermal_conductivity: float = 1.4
    density: float = 2200.0
    refractive_index: complex = 1.57 + 0.02j
    thermal_accommodation: float = 1.0
    emissivity: float = 0.9


@dataclass(frozen=True)
class Beam:
    wavelength: float = 532.0e-9
    waist: float = 50.0e-6
    power: float = 0.1
    m2: float = 1.0

    @property
    def peak_intensity(self) -> float:
        return 2.0 * self.power / (np.pi * self.waist**2)

    @property
    def rayleigh_range(self) -> float:
        return np.pi * self.waist**2 / (self.m2 * self.wavelength)

    def axial_intensity(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.peak_intensity / (1.0 + (np.asarray(z) / self.rayleigh_range) ** 2)

    def transverse_intensity(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.peak_intensity * np.exp(-2.0 * (np.asarray(x) / self.waist) ** 2)


@dataclass(frozen=True)
class OpticalMoments:
    radial_grid: np.ndarray
    source_coefficients: dict[int, np.ndarray]
    j0: float
    j1: float
    q_abs_mie: float
    q_abs_beer: float
    beta_absorption: float
    source_scale_to_mie: float


@dataclass(frozen=True)
class ThermalState:
    j0: float
    j1: float
    particle_A0: float
    particle_A1: float
    gas_B0: float
    gas_B1: float
    delta_t_front_minus_back: float
    axial_gradient: float


def mie_coefficients(size_parameter: float, refractive_index: complex) -> tuple[np.ndarray, np.ndarray]:
    if size_parameter <= 0.0:
        raise ValueError("size_parameter must be positive")

    n_stop = int(np.ceil(size_parameter + 4.0 * size_parameter ** (1.0 / 3.0) + 2.0))
    n = np.arange(1, n_stop + 1)
    mx = refractive_index * size_parameter

    j_x = spherical_jn(n, size_parameter)
    j_x_p = spherical_jn(n, size_parameter, derivative=True)
    y_x = spherical_yn(n, size_parameter)
    y_x_p = spherical_yn(n, size_parameter, derivative=True)
    h_x = j_x + 1j * y_x
    h_x_p = j_x_p + 1j * y_x_p

    j_mx = spherical_jn(n, mx)
    j_mx_p = spherical_jn(n, mx, derivative=True)

    psi_x = size_parameter * j_x
    psi_x_p = j_x + size_parameter * j_x_p
    xi_x = size_parameter * h_x
    xi_x_p = h_x + size_parameter * h_x_p
    psi_mx = mx * j_mx
    psi_mx_p = j_mx + mx * j_mx_p

    a_n = (
        refractive_index * psi_mx * psi_x_p - psi_x * psi_mx_p
    ) / (
        refractive_index * psi_mx * xi_x_p - xi_x * psi_mx_p
    )
    b_n = (
        psi_mx * psi_x_p - refractive_index * psi_x * psi_mx_p
    ) / (
        psi_mx * xi_x_p - refractive_index * xi_x * psi_mx_p
    )
    return a_n, b_n


def mie_absorption_efficiency(radius: float, beam: Beam, particle: Particle) -> float:
    x = 2.0 * np.pi * radius / beam.wavelength
    a_n, b_n = mie_coefficients(x, particle.refractive_index)
    n = np.arange(1, len(a_n) + 1)
    q_ext = (2.0 / x**2) * np.sum((2 * n + 1) * np.real(a_n + b_n))
    q_sca = (2.0 / x**2) * np.sum(
        (2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2)
    )
    return float(max(0.0, np.real(q_ext - q_sca)))


def cumulative_from_right(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return -cumulative_trapezoid(y[::-1], x[::-1], initial=0.0)[::-1]


def build_optical_moments(
    particle: Particle,
    beam: Beam,
    l_max: int = 1,
    nr: int = 320,
    nmu: int = 320,
    normalize_to_mie: bool = True,
) -> OpticalMoments:
    radius = particle.radius
    beta = 4.0 * np.pi * max(0.0, particle.refractive_index.imag) / beam.wavelength
    radial_grid = np.linspace(0.0, radius, nr)
    mu, w_mu = roots_legendre(nmu)

    r = radial_grid[:, None]
    mu_grid = mu[None, :]
    rho_squared = r**2 * (1.0 - mu_grid**2)
    z = r * mu_grid
    path_from_illuminated_surface = z + np.sqrt(np.maximum(radius**2 - rho_squared, 0.0))

    if beta == 0.0:
        source_per_intensity = np.zeros_like(path_from_illuminated_surface)
    else:
        source_per_intensity = beta * np.exp(-beta * path_from_illuminated_surface)

    angular_integral_q = source_per_intensity @ w_mu
    cross_section_beer = 2.0 * np.pi * np.trapz(radial_grid**2 * angular_integral_q, radial_grid)
    q_abs_beer = cross_section_beer / (np.pi * radius**2) if radius > 0 else 0.0

    q_abs_mie = mie_absorption_efficiency(radius, beam, particle)
    source_scale = 1.0
    if normalize_to_mie and cross_section_beer > 0.0:
        source_scale = (np.pi * radius**2 * q_abs_mie) / cross_section_beer
        source_per_intensity = source_per_intensity * source_scale

    source_coefficients: dict[int, np.ndarray] = {}
    for ell in range(l_max + 1):
        p_ell = eval_legendre(ell, mu)
        source_coefficients[ell] = (2 * ell + 1) / 2.0 * (source_per_intensity * p_ell) @ w_mu

    angular_integral_q = source_per_intensity @ w_mu
    angular_integral_q_mu = (source_per_intensity * mu_grid) @ w_mu
    volume = 4.0 / 3.0 * np.pi * radius**3
    cross_section = 2.0 * np.pi * np.trapz(radial_grid**2 * angular_integral_q, radial_grid)
    weighted_moment = 2.0 * np.pi * np.trapz(
        radial_grid**3 * angular_integral_q_mu,
        radial_grid,
    )

    return OpticalMoments(
        radial_grid=radial_grid,
        source_coefficients=source_coefficients,
        j0=float(cross_section / (4.0 * np.pi * radius**2)),
        j1=float(weighted_moment / volume),
        q_abs_mie=q_abs_mie,
        q_abs_beer=float(q_abs_beer),
        beta_absorption=beta,
        source_scale_to_mie=float(source_scale),
    )


def surface_mode_from_heat_equation(
    ell: int,
    radius: float,
    particle_k: float,
    gas_k: float,
    temperature_jump: float,
    kn: float,
    radiation_linear_coeff: float,
    radial_grid: np.ndarray,
    source_l_actual: np.ndarray,
) -> float:
    r_safe = radial_grid.copy()
    if r_safe[0] == 0.0:
        r_safe[0] = radial_grid[1] * 1.0e-6

    int_left = cumulative_trapezoid(source_l_actual * radial_grid ** (ell + 2), radial_grid, initial=0.0)
    int_right = cumulative_from_right(source_l_actual * r_safe ** (1 - ell), radial_grid)

    particular_surface = int_left[-1] / (particle_k * (2 * ell + 1) * radius ** (ell + 1))
    particular_derivative_surface = (
        -(ell + 1) * int_left[-1] / (particle_k * (2 * ell + 1) * radius ** (ell + 2))
    )

    gas_conductance = gas_k * (ell + 1) / (radius * (1.0 + temperature_jump * kn * (ell + 1)))
    boundary_conductance = gas_conductance + radiation_linear_coeff

    if ell == 0:
        homogeneous_denominator = boundary_conductance
    else:
        homogeneous_denominator = (
            particle_k * ell * radius ** (ell - 1) + boundary_conductance * radius**ell
        )

    homogeneous_coefficient = -(
        particle_k * particular_derivative_surface + boundary_conductance * particular_surface
    ) / homogeneous_denominator

    return float(homogeneous_coefficient * radius**ell + particular_surface)


def solve_thermal_state(
    optics: OpticalMoments,
    gas: Gas,
    particle: Particle,
    pressure: float,
    intensity: float,
    loesche: LoescheCoefficients,
    radiation_temperature: float,
) -> ThermalState:
    radius = particle.radius
    kn = gas.mean_free_path(pressure) / radius
    # Linearize radiation about a one-node estimate of the mean temperature.
    absorbed_power = max(0.0, intensity * np.pi * radius**2 * 4.0 * optics.j0)
    t_rad_ref = (radiation_temperature**4 + absorbed_power / (4.0 * np.pi * radius**2 * SIGMA_SB * max(particle.emissivity, 1e-12))) ** 0.25
    h_rad = 4.0 * SIGMA_SB * particle.emissivity * t_rad_ref**3

    surface_coefficients: dict[int, float] = {}
    for ell in (0, 1):
        source_l_actual = intensity * optics.source_coefficients[ell]
        surface_coefficients[ell] = surface_mode_from_heat_equation(
            ell=ell,
            radius=radius,
            particle_k=particle.thermal_conductivity,
            gas_k=gas.thermal_conductivity,
            temperature_jump=loesche.temperature_jump,
            kn=float(kn),
            radiation_linear_coeff=h_rad,
            radial_grid=optics.radial_grid,
            source_l_actual=source_l_actual,
        )

    particle_A0 = gas.temperature + surface_coefficients[0]
    particle_A1 = surface_coefficients[1]
    gas_B0 = gas.temperature + surface_coefficients[0] / (1.0 + loesche.temperature_jump * kn)
    gas_B1 = surface_coefficients[1] / (1.0 + 2.0 * loesche.temperature_jump * kn)

    return ThermalState(
        j0=optics.j0,
        j1=optics.j1,
        particle_A0=float(particle_A0),
        particle_A1=float(particle_A1),
        gas_B0=float(gas_B0),
        gas_B1=float(gas_B1),
        delta_t_front_minus_back=float(-2.0 * particle_A1),
        axial_gradient=float(particle_A1 / radius),
    )


def signed_harmonic(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray | float:
    first_arr = np.asarray(first, dtype=float)
    second_arr = np.asarray(second, dtype=float)
    same_sign = np.sign(first_arr) == np.sign(second_arr)
    result = np.full(np.broadcast(first_arr, second_arr).shape, np.nan, dtype=float)
    first_b, second_b = np.broadcast_arrays(first_arr, second_arr)
    mask = same_sign & (np.abs(first_b) > 0.0) & (np.abs(second_b) > 0.0)
    result[mask] = np.sign(first_b[mask]) / (
        1.0 / np.abs(first_b[mask]) + 1.0 / np.abs(second_b[mask])
    )
    if np.isscalar(first) and np.isscalar(second):
        return float(result)
    return result


def beresnev_c_star(gas: Gas) -> float:
    return np.sqrt(2.0 * R_GAS * gas.temperature / (np.pi * gas.molar_mass))


def beresnev_gas_thermal_conductivity(gas: Gas) -> float:
    """Gas conductivity implied by the S-model kinetic closure."""

    return 15.0 / 4.0 * gas.viscosity * R_GAS / gas.molar_mass


def beresnev_mean_free_path(pressure: np.ndarray | float, gas: Gas) -> np.ndarray | float:
    """Mean free path consistent with the Beresnev kinetic conductivity closure."""

    return (
        4.0
        * beresnev_gas_thermal_conductivity(gas)
        * gas.temperature
        / (15.0 * np.asarray(pressure, dtype=float) * beresnev_c_star(gas))
    )


def beresnev_knudsen_number(
    pressure: np.ndarray | float,
    gas: Gas,
    particle: Particle,
) -> np.ndarray | float:
    return beresnev_mean_free_path(pressure, gas) / particle.radius


def hettner_force(
    pressure: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    thermal: ThermalState,
) -> np.ndarray | float:
    diameter = 2.0 * particle.radius
    grad_t = thermal.axial_gradient
    f_cont = -(
        3.0
        * np.pi**2
        * gas.viscosity**2
        * diameter
        * R_GAS
        / (np.asarray(pressure) * gas.molar_mass)
        * grad_t
    )
    f_free = -(
        np.pi
        / 24.0
        * particle.thermal_accommodation
        * diameter**3
        * np.asarray(pressure)
        * grad_t
        / gas.temperature
    )
    return signed_harmonic(f_free, f_cont)


def rohatschek_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> np.ndarray | float:
    """Standard Rohatschek all-pressure Delta-T photophoresis equation."""

    p_max, f_max = rohatschek_characteristic_values(intensity, gas, particle, j1)
    pressure_arr = np.asarray(pressure)
    return 2.0 * f_max / (pressure_arr / p_max + p_max / pressure_arr)


def rohatschek_characteristic_values(
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    d_factor = (
        np.pi
        * gas.mean_thermal_speed
        * gas.viscosity
        / (2.0 * gas.temperature)
        * np.sqrt(np.pi * gas.rohatschek_creep / 3.0)
    )
    p_max = d_factor * 3.0 * gas.temperature / (np.pi * particle.radius) * np.sqrt(
        2.0 / particle.thermal_accommodation
    )
    f_max = -(
        d_factor
        * np.sqrt(particle.thermal_accommodation / 2.0)
        * particle.radius**2
        * np.asarray(intensity)
        * j1
        / particle.thermal_conductivity
    )
    return p_max, f_max


def rohatschek_low_pressure_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> np.ndarray | float:
    p_max, f_max = rohatschek_characteristic_values(intensity, gas, particle, j1)
    return 2.0 * f_max * np.asarray(pressure, dtype=float) / p_max


def rohatschek_high_pressure_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> np.ndarray | float:
    p_max, f_max = rohatschek_characteristic_values(intensity, gas, particle, j1)
    return 2.0 * f_max * p_max / np.asarray(pressure, dtype=float)


def loesche_mean_temperature_fm(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    j0: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressure_arr, intensity_arr = np.broadcast_arrays(
        np.asarray(pressure, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    if particle_temperature is not None:
        a0_fm = np.full_like(pressure_arr, particle_temperature, dtype=float)
        t_tilde = np.full_like(pressure_arr, particle_temperature, dtype=float)
        t_g_scattered = gas.temperature + particle.thermal_accommodation * (
            a0_fm - gas.temperature
        )
        return a0_fm, t_tilde, t_g_scattered

    h = (
        loesche.heat_transfer_factor
        * loesche.momentum_accommodation
        * particle.thermal_accommodation
        * pressure_arr
        / gas.temperature
        * gas.mean_thermal_speed
    )
    t_tilde = (intensity_arr / (4.0 * SIGMA_SB) + radiation_temperature**4) ** 0.25
    for _ in range(iterations):
        numerator = (
            intensity_arr * j0
            + h * gas.temperature
            + SIGMA_SB
            * particle.emissivity
            * (3.0 * t_tilde**4 + radiation_temperature**4)
        )
        denominator = h + 4.0 * SIGMA_SB * particle.emissivity * t_tilde**3
        next_t = numerator / denominator
        t_tilde = 0.5 * t_tilde + 0.5 * next_t
    a0_fm = (
        intensity_arr * j0
        + h * gas.temperature
        + SIGMA_SB * particle.emissivity * (3.0 * t_tilde**4 + radiation_temperature**4)
    ) / (h + 4.0 * SIGMA_SB * particle.emissivity * t_tilde**3)
    t_g_scattered = gas.temperature + particle.thermal_accommodation * (a0_fm - gas.temperature)
    return a0_fm, t_tilde, t_g_scattered


def loesche_husmann_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    j0: float,
    j1: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
) -> np.ndarray | float:
    f_fm, f_co = loesche_husmann_limit_forces(
        pressure,
        intensity,
        gas,
        particle,
        loesche,
        j0,
        j1,
        radiation_temperature,
        particle_temperature,
    )
    return signed_harmonic(f_fm, f_co)


def loesche_husmann_limit_forces(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    j0: float,
    j1: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    pressure_arr, intensity_arr = np.broadcast_arrays(
        np.asarray(pressure, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    radius = particle.radius

    a0_fm, t_tilde_fm, t_g_scattered = loesche_mean_temperature_fm(
        pressure_arr,
        intensity_arr,
        gas,
        particle,
        loesche,
        j0,
        radiation_temperature,
        particle_temperature,
    )
    h = (
        loesche.heat_transfer_factor
        * loesche.momentum_accommodation
        * particle.thermal_accommodation
        * pressure_arr
        / gas.temperature
        * gas.mean_thermal_speed
    )
    denom_fm = (
        particle.thermal_conductivity / radius
        + h
        + 4.0 * SIGMA_SB * particle.emissivity * t_tilde_fm**3
    )
    f_fm = -(
        np.pi
        / 3.0
        * particle.thermal_accommodation
        * loesche.momentum_accommodation
        * pressure_arr
        * radius**2
        * intensity_arr
        * j1
        / (np.sqrt(t_g_scattered * gas.temperature) * denom_fm)
    )

    if particle_temperature is None:
        t_bb = (intensity_arr / (4.0 * SIGMA_SB) + radiation_temperature**4) ** 0.25
        a0_co = (
            intensity_arr * j0
            + gas.thermal_conductivity / radius * gas.temperature
            + SIGMA_SB * particle.emissivity * (3.0 * t_bb**4 + radiation_temperature**4)
        ) / (
            gas.thermal_conductivity / radius
            + 4.0 * SIGMA_SB * particle.emissivity * t_bb**3
        )
    else:
        t_bb = np.full_like(pressure_arr, particle_temperature, dtype=float)
        a0_co = np.full_like(pressure_arr, particle_temperature, dtype=float)

    density = gas.density(pressure_arr)
    denom_co = (
        particle.thermal_conductivity / radius
        + 2.0 * gas.thermal_conductivity / radius
        + 4.0 * SIGMA_SB * particle.emissivity * t_bb**3
    )
    f_co = -(
        4.0
        * np.pi
        * loesche.thermal_creep
        * gas.viscosity**2
        / (density * a0_co)
        * intensity_arr
        * j1
        / denom_co
    )

    return f_fm, f_co


def loesche_husmann_slip_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    j0: float,
    j1: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
) -> np.ndarray | float:
    pressure_arr, intensity_arr = np.broadcast_arrays(
        np.asarray(pressure, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    radius = particle.radius
    kn = gas.mean_free_path(pressure_arr) / radius
    if particle_temperature is None:
        t_bb = (intensity_arr / (4.0 * SIGMA_SB) + radiation_temperature**4) ** 0.25
        gas_surface_temperature = gas.temperature + (
            (
                intensity_arr * j0
                - 4.0 * SIGMA_SB * particle.emissivity * t_bb**3 * gas.temperature
                + SIGMA_SB * particle.emissivity * (3.0 * t_bb**4 + radiation_temperature**4)
            )
            / (1.0 + loesche.temperature_jump * kn)
            / (
                gas.thermal_conductivity / radius / (1.0 + loesche.temperature_jump * kn)
                + 4.0 * SIGMA_SB * particle.emissivity * t_bb**3
            )
        )
    else:
        t_bb = np.full_like(pressure_arr, particle_temperature, dtype=float)
        gas_surface_temperature = gas.temperature + particle.thermal_accommodation * (
            particle_temperature - gas.temperature
        ) / (1.0 + loesche.temperature_jump * kn)

    density = gas.density(pressure_arr)
    slip_factor = (
        (loesche.thermal_creep + 3.0 * loesche.thermal_stress_slip * loesche.frictional_slip * kn)
        / (1.0 + 3.0 * loesche.frictional_slip * kn)
    )
    denom = (
        particle.thermal_conductivity / radius
        + gas.thermal_conductivity / radius * 2.0 / (1.0 + 2.0 * loesche.temperature_jump * kn)
        + 4.0 * SIGMA_SB * particle.emissivity * t_bb**3
    )
    return -(
        4.0
        * np.pi
        * gas.viscosity**2
        / (density * gas_surface_temperature)
        * slip_factor
        / (1.0 + 2.0 * loesche.temperature_jump * kn)
        * intensity_arr
        * j1
        / denom
    )


def beresnev_low_kn_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> np.ndarray | float:
    pressure_arr = np.asarray(pressure, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    kn = beresnev_knudsen_number(pressure_arr, gas, particle)
    beresnev_k_g = beresnev_gas_thermal_conductivity(gas)
    conductivity_ratio = (
        particle.thermal_conductivity
        + 4.0 * particle.emissivity * SIGMA_SB * gas.temperature**3 * particle.radius
    ) / beresnev_k_g
    bracket = (
        1.125
        - (2.136 + 11.09 / (conductivity_ratio + 2.0)) * kn
        - (20.81 + 57.62 / (conductivity_ratio + 2.0)) * kn**2
    )
    density = gas.density(pressure_arr)
    return -(
        4.0
        * np.pi
        * gas.viscosity**2
        * particle.radius
        * intensity_arr
        * j1
        / (density * gas.temperature * beresnev_k_g * (conductivity_ratio + 2.0))
        * bracket
    )


def beresnev_all_kn_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    j1: float,
) -> np.ndarray | float:
    """Beresnev-Chernyak-Fomyagin all-Kn kinetic approximation.

    This is the practical closed expression given after the integral-moment
    solution in Beresnev et al. It represents the S-model/BGK kinetic result
    over the full Knudsen range for normal and tangential accommodation equal
    to one, with arbitrary energy accommodation. It is not the unpublished
    raw Galerkin matrix solve.
    """

    pressure_arr = np.asarray(pressure, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    kn = beresnev_knudsen_number(pressure_arr, gas, particle)
    alpha_energy = particle.thermal_accommodation
    lambda_eff = (
        particle.thermal_conductivity
        + 4.0 * particle.emissivity * SIGMA_SB * gas.temperature**3 * particle.radius
    )
    conductivity_ratio = lambda_eff / beresnev_gas_thermal_conductivity(gas)
    c_star = beresnev_c_star(gas)
    sqrt_pi = np.sqrt(np.pi)

    phi_1 = (
        1.0
        + 2.0
        * sqrt_pi
        * kn
        / (5.0 * kn**2 + sqrt_pi * kn + np.pi / 4.0)
    ) * kn / (kn + 5.0 * np.pi / 18.0)
    phi_2 = (
        1.0
        - 1.21
        * sqrt_pi
        * kn
        / (100.0 * kn**2 + np.pi / 4.0)
    ) * (0.5 + 15.0 * kn / 4.0)
    denominator = (
        alpha_energy
        + 15.0 / 4.0 * conductivity_ratio * kn * (1.0 - alpha_energy)
        + alpha_energy * conductivity_ratio * phi_2
    )

    return -(
        np.pi
        / 3.0
        * particle.radius**2
        * intensity_arr
        * j1
        / c_star
        * alpha_energy
        * phi_1
        / denominator
    )


def beresnev_fm_limit_force(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    j1: float,
) -> np.ndarray | float:
    pressure_arr = np.asarray(pressure, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    alpha_energy = particle.thermal_accommodation
    alpha_normal = loesche.momentum_accommodation
    denominator_common = 32.0 - np.pi * (9.0 - alpha_energy) * (1.0 - alpha_normal)
    f_1 = 32.0 * alpha_energy * alpha_normal / denominator_common
    f_2 = (
        alpha_energy
        * (32.0 - 9.0 * np.pi * (1.0 - alpha_normal))
        / denominator_common
    )
    c_star = beresnev_c_star(gas)
    denominator = (
        particle.thermal_conductivity * gas.temperature / particle.radius
        + 4.0 * SIGMA_SB * particle.emissivity * gas.temperature**4
        + f_2 * pressure_arr * c_star
    )
    return -(
        np.pi
        / 3.0
        * pressure_arr
        * particle.radius**2
        * intensity_arr
        * j1
        * f_1
        / denominator
    )


def hettner_force_from_conditions(
    pressure: np.ndarray | float,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    optics: OpticalMoments,
    radiation_temperature: float,
) -> np.ndarray:
    pressure_arr, intensity_arr = np.broadcast_arrays(
        np.asarray(pressure, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    values = np.empty_like(pressure_arr, dtype=float)
    for index in np.ndindex(pressure_arr.shape):
        local_thermal = solve_thermal_state(
            optics,
            gas,
            particle,
            pressure=float(pressure_arr[index]),
            intensity=float(intensity_arr[index]),
            loesche=loesche,
            radiation_temperature=radiation_temperature,
        )
        values[index] = hettner_force(
            float(pressure_arr[index]),
            gas,
            particle,
            local_thermal,
        )
    return values


def timed_result(
    timings: dict[str, float] | None,
    label: str,
    calculation,
):
    start = time.perf_counter()
    result = calculation()
    elapsed = time.perf_counter() - start
    if timings is not None:
        timings[label] = timings.get(label, 0.0) + elapsed
    return result


def format_runtime(seconds: float) -> str:
    if seconds < 1.0e-6:
        return f"{seconds * 1.0e9:.3g} ns"
    if seconds < 1.0e-3:
        return f"{seconds * 1.0e6:.3g} us"
    if seconds < 1.0:
        return f"{seconds * 1.0e3:.3g} ms"
    return f"{seconds:.3g} s"


def print_runtime_table(title: str, timings: dict[str, float]) -> None:
    if not timings:
        return
    print(title)
    for name, seconds in timings.items():
        print(f"  {name}: {format_runtime(seconds)}")


def write_runtime_table(file_obj, title: str, timings: dict[str, float]) -> None:
    if not timings:
        return
    file_obj.write(f"\n{title}\n")
    file_obj.write(f"{'-' * len(title)}\n")
    for name, seconds in timings.items():
        file_obj.write(f"{name}: {format_runtime(seconds)} ({seconds:.6e} s)\n")


def evaluate_models(
    pressure: np.ndarray,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    optics: OpticalMoments,
    thermal: ThermalState,
    rohatschek_j1: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    kn = gas.mean_free_path(np.asarray(pressure, dtype=float)) / particle.radius
    beresnev_kn = beresnev_knudsen_number(pressure, gas, particle)
    slip_force = timed_result(
        timings,
        "Loesche-Husmann slip",
        lambda: np.asarray(
            loesche_husmann_slip_force(
                pressure,
                intensity,
                gas,
                particle,
                loesche,
                thermal.j0,
                thermal.j1,
                radiation_temperature,
                particle_temperature,
            )
        )
    )
    slip_force = np.where(kn <= 0.3, slip_force, np.nan)

    beresnev_low = timed_result(
        timings,
        "Beresnev low-Kn",
        lambda: np.asarray(
            beresnev_low_kn_force(pressure, intensity, gas, particle, thermal.j1)
        ),
    )
    beresnev_low = np.where(beresnev_kn <= 0.1, beresnev_low, np.nan)

    beresnev_fm = timed_result(
        timings,
        "Beresnev fm limit",
        lambda: np.asarray(
            beresnev_fm_limit_force(pressure, intensity, gas, particle, loesche, thermal.j1)
        ),
    )
    beresnev_fm = np.where(beresnev_kn >= 10.0, beresnev_fm, np.nan)

    return {
        "Hettner": timed_result(
            timings,
            "Hettner",
            lambda: hettner_force_from_conditions(
                pressure,
                intensity,
                gas,
                particle,
                loesche,
                optics,
                radiation_temperature,
            ),
        ),
        "Rohatschek": timed_result(
            timings,
            "Rohatschek",
            lambda: np.asarray(
                rohatschek_force(pressure, intensity, gas, particle, rohatschek_j1)
            ),
        ),
        "Loesche-Husmann": timed_result(
            timings,
            "Loesche-Husmann",
            lambda: np.asarray(
                loesche_husmann_force(
                    pressure,
                    intensity,
                    gas,
                    particle,
                    loesche,
                    thermal.j0,
                    thermal.j1,
                    radiation_temperature,
                    particle_temperature,
                )
            ),
        ),
        "Loesche-Husmann slip": slip_force,
        "Beresnev all-Kn kinetic": timed_result(
            timings,
            "Beresnev all-Kn kinetic",
            lambda: np.asarray(
                beresnev_all_kn_force(pressure, intensity, gas, particle, thermal.j1)
            ),
        ),
        "Beresnev fm limit": beresnev_fm,
        "Beresnev low-Kn": beresnev_low,
    }


def evaluate_pressure_branch_models(
    pressure: np.ndarray,
    intensity: np.ndarray | float,
    gas: Gas,
    particle: Particle,
    loesche: LoescheCoefficients,
    optics: OpticalMoments,
    thermal: ThermalState,
    rohatschek_j1: float,
    radiation_temperature: float,
    particle_temperature: float | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    pressure_arr = np.asarray(pressure, dtype=float)
    hettner = timed_result(
        timings,
        "Hettner",
        lambda: hettner_force_from_conditions(
            pressure_arr,
            intensity,
            gas,
            particle,
            loesche,
            optics,
            radiation_temperature,
        ),
    )

    kn = gas.mean_free_path(pressure_arr) / particle.radius
    roh_low = timed_result(
        timings,
        "Rohatschek low pressure",
        lambda: np.asarray(
            rohatschek_low_pressure_force(
                pressure_arr, intensity, gas, particle, rohatschek_j1
            )
        ),
    )
    roh_high = timed_result(
        timings,
        "Rohatschek high pressure",
        lambda: np.asarray(
            rohatschek_high_pressure_force(
                pressure_arr, intensity, gas, particle, rohatschek_j1
            )
        ),
    )
    roh_general = timed_result(
        timings,
        "Rohatschek",
        lambda: np.asarray(
            rohatschek_force(pressure_arr, intensity, gas, particle, rohatschek_j1)
        ),
    )
    roh_low = np.where(kn >= 10.0, roh_low, np.nan)
    roh_high = np.where(kn <= 0.3, roh_high, np.nan)

    lh_low = timed_result(
        timings,
        "Loesche-Husmann low pressure",
        lambda: np.asarray(
            loesche_husmann_limit_forces(
                pressure_arr,
                intensity,
                gas,
                particle,
                loesche,
                thermal.j0,
                thermal.j1,
                radiation_temperature,
                particle_temperature,
            )[0]
        ),
    )
    lh_high = timed_result(
        timings,
        "Loesche-Husmann high pressure",
        lambda: np.asarray(
            loesche_husmann_limit_forces(
                pressure_arr,
                intensity,
                gas,
                particle,
                loesche,
                thermal.j0,
                thermal.j1,
                radiation_temperature,
                particle_temperature,
            )[1]
        ),
    )
    lh_general = timed_result(
        timings,
        "Loesche-Husmann",
        lambda: np.asarray(
            loesche_husmann_force(
                pressure_arr,
                intensity,
                gas,
                particle,
                loesche,
                thermal.j0,
                thermal.j1,
                radiation_temperature,
                particle_temperature,
            )
        ),
    )
    lh_low = np.where(kn >= 10.0, lh_low, np.nan)
    lh_high = np.where(kn <= 0.3, lh_high, np.nan)

    return {
        "Hettner": np.asarray(hettner),
        "Rohatschek low pressure": roh_low,
        "Rohatschek": roh_general,
        "Rohatschek high pressure": roh_high,
        "Loesche-Husmann low pressure": lh_low,
        "Loesche-Husmann": lh_general,
        "Loesche-Husmann high pressure": lh_high,
        "Beresnev all-Kn kinetic": timed_result(
            timings,
            "Beresnev all-Kn kinetic",
            lambda: np.asarray(
                beresnev_all_kn_force(pressure_arr, intensity, gas, particle, thermal.j1)
            ),
        ),
    }


def write_csv(path: Path, x_name: str, x_values: np.ndarray, data: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, *data.keys()])
        for i, x in enumerate(x_values):
            writer.writerow([x, *[values[i] for values in data.values()]])


def plot_force_comparison(
    path: Path,
    x_values: np.ndarray,
    data: dict[str, np.ndarray],
    xlabel: str,
    title: str,
    log_x: bool = False,
    style_map: dict[str, dict[str, object]] | None = None,
    normalisation_force: float | None = None,
    normalisation_label: str = "N",
    weight_reference: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    plot_data = data
    if normalisation_force is not None:
        if normalisation_force <= 0.0:
            raise ValueError("normalisation_force must be positive")
        plot_data = {
            name: values / normalisation_force
            for name, values in data.items()
        }

    for name, values in plot_data.items():
        if not np.any(np.isfinite(values)):
            continue
        style = {"linewidth": 2.0, "label": name}
        if name.endswith("slip"):
            style.update({"linestyle": "--", "linewidth": 1.5})
        if style_map and name in style_map:
            style.update(style_map[name])
        ax.plot(x_values, values, **style)
    if log_x:
        ax.set_xscale("log")
    if weight_reference:
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="particle weight $mg$",
        )

    finite_values = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in plot_data.values() if np.any(np.isfinite(v))])
    if finite_values.size > 0 and np.all(finite_values > 0.0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-18)
        ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"photophoretic force along beam / {normalisation_label}")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)


def plot_surface_temperature(path: Path, thermal: ThermalState) -> None:
    mu = np.linspace(-1.0, 1.0, 400)
    theta_deg = np.degrees(np.arccos(mu))
    particle_surface = thermal.particle_A0 + thermal.particle_A1 * mu
    gas_surface = thermal.gas_B0 + thermal.gas_B1 * mu
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(theta_deg, particle_surface, label="particle surface")
    ax.plot(theta_deg, gas_surface, label="adjacent gas", linestyle="--")
    ax.set_xlabel("polar angle from +z / degrees")
    ax.set_ylabel("temperature / K")
    ax.set_title("First two Legendre modes from heat equation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="photophoresis_output")
    parser.add_argument("--radius-um", type=float, default=6.3)
    parser.add_argument("--particle-k", type=float, default=0.135)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument("--n-real", type=float, default=1.555)
    parser.add_argument("--n-imag", type=float, default=0.02)
    parser.add_argument("--emissivity", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--waist-um", type=float, default=20.0)
    parser.add_argument("--m2", type=float, default=1.2)
    parser.add_argument("--power", type=float, default=0.225)
    parser.add_argument("--gas-temperature", type=float, default=300.0)
    parser.add_argument("--radiation-temperature", type=float, default=300.0)
    parser.add_argument(
        "--particle-temperature",
        type=float,
        default=500.0,
        help=(
            "fixed particle surface/mean temperature used by the "
            "Loesche-Husmann model; set <= 0 to use the original internally "
            "solved temperature estimate"
        ),
    )
    parser.add_argument("--gas-molar-mass", type=float, default=0.029)
    parser.add_argument("--gas-viscosity", type=float, default=1.8e-5)
    parser.add_argument("--gas-k", type=float, default=0.0262)
    parser.add_argument("--pressure-min", type=float, default=0.03)
    parser.add_argument("--pressure-max", type=float, default=2.0e5)
    parser.add_argument("--pressure-points", type=int, default=260)
    parser.add_argument("--fixed-pressure", type=float, default=100000.0)
    parser.add_argument("--displacement-mm", type=float, default=30.0)
    parser.add_argument("--displacement-points", type=int, default=260)
    parser.add_argument("--displacement-axis", choices=("axial", "transverse"), default="axial")
    parser.add_argument(
        "--rohatschek-j1",
        type=float,
        default=None,
        help=(
            "optional fixed Rohatschek photophoretic asymmetry factor; by "
            "default the code uses the calculated heat-source J1. In this "
            f"sign convention {OPAQUE_ROHATSCHEK_J1:g} is the opaque "
            "black-sphere value"
        ),
    )
    parser.add_argument("--no-mie-normalization", action="store_true")
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="also save CSV data tables for the pressure and displacement plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    gas = Gas(
        temperature=args.gas_temperature,
        molar_mass=args.gas_molar_mass,
        viscosity=args.gas_viscosity,
        thermal_conductivity=args.gas_k,
    )
    particle = Particle(
        radius=args.radius_um * 1e-6,
        thermal_conductivity=args.particle_k,
        density=args.density,
        refractive_index=complex(args.n_real, args.n_imag),
        thermal_accommodation=args.alpha,
        emissivity=args.emissivity,
    )
    particle_mass = particle.density * (4.0 / 3.0) * np.pi * particle.radius**3
    particle_weight = particle_mass * 9.81
    beam = Beam(
        wavelength=args.wavelength_nm * 1e-9,
        waist=args.waist_um * 1e-6,
        power=args.power,
        m2=args.m2,
    )
    loesche = LoescheCoefficients()
    loesche_particle_temperature = (
        args.particle_temperature if args.particle_temperature > 0.0 else None
    )

    optics = build_optical_moments(
        particle,
        beam,
        normalize_to_mie=not args.no_mie_normalization,
    )
    thermal = solve_thermal_state(
        optics,
        gas,
        particle,
        pressure=args.fixed_pressure,
        intensity=beam.peak_intensity,
        loesche=loesche,
        radiation_temperature=args.radiation_temperature,
    )
    rohatschek_j1 = thermal.j1 if args.rohatschek_j1 is None else args.rohatschek_j1
    rohatschek_j1_source = (
        "calculated heat-source J1" if args.rohatschek_j1 is None else "command-line override"
    )
    roh_p_max, roh_f_max = rohatschek_characteristic_values(
        beam.peak_intensity,
        gas,
        particle,
        rohatschek_j1,
    )

    pressures = np.geomspace(args.pressure_min, args.pressure_max, args.pressure_points)
    pressure_png = output_dir / "force_vs_pressure.png"
    pressure_csv = output_dir / "force_vs_pressure.csv"
    displacement_png = output_dir / "force_vs_displacement.png"
    displacement_csv = output_dir / "force_vs_displacement.csv"
    surface_png = output_dir / "surface_temperature_modes.png"
    pressure_timings: dict[str, float] = {}
    displacement_timings: dict[str, float] = {}
    if not args.save_csv:
        pressure_csv.unlink(missing_ok=True)
        displacement_csv.unlink(missing_ok=True)
    pressure_data = evaluate_pressure_branch_models(
        pressures,
        beam.peak_intensity,
        gas,
        particle,
        loesche,
        optics,
        thermal,
        rohatschek_j1,
        args.radiation_temperature,
        loesche_particle_temperature,
        timings=pressure_timings,
    )
    pressure_styles = {
        "Hettner": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.0},
        "Rohatschek low pressure": {
            "color": "tab:orange",
            "linestyle": "--",
            "linewidth": 1.6,
        },
        "Rohatschek": {"color": "tab:orange", "linestyle": "-", "linewidth": 2.2},
        "Rohatschek high pressure": {
            "color": "tab:orange",
            "linestyle": "--",
            "linewidth": 1.6,
        },
        "Loesche-Husmann low pressure": {
            "color": "tab:green",
            "linestyle": "--",
            "linewidth": 1.6,
        },
        "Loesche-Husmann": {"color": "tab:green", "linestyle": "-", "linewidth": 2.2},
        "Loesche-Husmann high pressure": {
            "color": "tab:green",
            "linestyle": "--",
            "linewidth": 1.6,
        },
        "Beresnev all-Kn kinetic": {
            "color": "tab:purple",
            "linestyle": "-",
            "linewidth": 2.0,
        },
    }
    if args.save_csv:
        write_csv(pressure_csv, "pressure_Pa", pressures, pressure_data)
    plot_force_comparison(
        pressure_png,
        pressures,
        pressure_data,
        "pressure / Pa",
        "Photophoretic force vs pressure",
        log_x=True,
        style_map=pressure_styles,
        normalisation_force=particle_weight,
        normalisation_label="$mg$",
        weight_reference=True,
    )

    displacements = np.linspace(
        -args.displacement_mm * 1e-3,
        args.displacement_mm * 1e-3,
        args.displacement_points,
    )
    if args.displacement_axis == "axial":
        intensities = beam.axial_intensity(displacements)
        displacement_label = "axial displacement from focus / m"
    else:
        intensities = beam.transverse_intensity(displacements)
        displacement_label = "transverse displacement from beam axis / m"

    fixed_pressure = np.full_like(displacements, args.fixed_pressure)
    displacement_data = evaluate_models(
        fixed_pressure,
        intensities,
        gas,
        particle,
        loesche,
        optics,
        thermal,
        rohatschek_j1,
        args.radiation_temperature,
        loesche_particle_temperature,
        timings=displacement_timings,
    )
    if args.save_csv:
        write_csv(
            displacement_csv,
            "displacement_m",
            displacements,
            displacement_data,
        )
    plot_force_comparison(
        displacement_png,
        displacements,
        displacement_data,
        displacement_label,
        f"Photophoretic force vs {args.displacement_axis} displacement at {args.fixed_pressure:g} Pa",
        normalisation_force=particle_weight,
        normalisation_label="$mg$",
        weight_reference=True,
    )
    plot_surface_temperature(surface_png, thermal)

    with (output_dir / "summary.txt").open("w") as f:
        f.write("Photophoresis model comparison summary\n")
        f.write("======================================\n\n")
        f.write(f"radius = {particle.radius:.6e} m\n")
        f.write(f"particle mass = {particle_mass:.6e} kg\n")
        f.write(f"particle weight mg = {particle_weight:.6e} N\n")
        f.write(f"particle k = {particle.thermal_conductivity:.6g} W/(m K)\n")
        f.write(f"refractive index = {particle.refractive_index}\n")
        f.write(f"beam peak intensity = {beam.peak_intensity:.6e} W/m^2\n")
        f.write(f"beam waist = {beam.waist:.6e} m\n")
        f.write(f"beam M2 = {beam.m2:.6g}\n")
        f.write(f"beam Rayleigh range = {beam.rayleigh_range:.6e} m\n")
        f.write(f"fixed pressure = {args.fixed_pressure:.6g} Pa\n")
        if loesche_particle_temperature is None:
            f.write("Loesche-Husmann particle temperature = internally solved\n")
        else:
            f.write(
                "Loesche-Husmann particle temperature override = "
                f"{loesche_particle_temperature:.6g} K\n"
            )
        f.write(f"Kn(fixed pressure) = {gas.mean_free_path(args.fixed_pressure) / particle.radius:.6g}\n")
        f.write(
            "Beresnev S-model gas k = "
            f"{beresnev_gas_thermal_conductivity(gas):.6g} W/(m K)\n"
        )
        f.write(
            "Beresnev kinetic Kn(fixed pressure) = "
            f"{beresnev_knudsen_number(args.fixed_pressure, gas, particle):.6g}\n"
        )
        f.write(f"Mie Q_abs = {optics.q_abs_mie:.6g}\n")
        f.write(f"Beer-only Q_abs = {optics.q_abs_beer:.6g}\n")
        f.write(f"source scale to Mie = {optics.source_scale_to_mie:.6g}\n")
        f.write(f"J0 = {thermal.j0:.6g}\n")
        f.write(f"heat-equation J1 = {thermal.j1:.6g}\n")
        f.write(f"Rohatschek J1 used = {rohatschek_j1:.6g}\n")
        f.write(f"Rohatschek J1 source = {rohatschek_j1_source}\n")
        f.write(f"Rohatschek p_max = {roh_p_max:.6g} Pa\n")
        f.write(f"Rohatschek F_max = {roh_f_max:.6g} N\n")
        f.write(f"A0 particle surface = {thermal.particle_A0:.6g} K\n")
        f.write(f"A1 particle surface = {thermal.particle_A1:.6g} K\n")
        f.write(f"B0 adjacent gas = {thermal.gas_B0:.6g} K\n")
        f.write(f"B1 adjacent gas = {thermal.gas_B1:.6g} K\n")
        f.write(f"front-minus-back surface temperature = {thermal.delta_t_front_minus_back:.6g} K\n")
        f.write(f"axial gradient = {thermal.axial_gradient:.6g} K/m\n")
        write_runtime_table(f, "Pressure model runtimes", pressure_timings)
        write_runtime_table(f, "Displacement model runtimes", displacement_timings)

    if args.save_csv:
        print(f"Wrote plots and CSV files to {output_dir.resolve()}")
    else:
        print(f"Wrote plots to {output_dir.resolve()}")
    print(
        f"J0={thermal.j0:.6g}, heat-equation J1={thermal.j1:.6g}, "
        f"Rohatschek J1 used={rohatschek_j1:.6g} "
        f"({rohatschek_j1_source}), "
        f"A1={thermal.particle_A1:.6g} K"
    )
    print(f"Rohatschek p_max={roh_p_max:.6g} Pa, F_max={roh_f_max:.6g} N")
    print_runtime_table("Pressure model runtimes", pressure_timings)
    print_runtime_table("Displacement model runtimes", displacement_timings)
    plt.show()


if __name__ == "__main__":
    main()
