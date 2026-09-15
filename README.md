# 3D Optical Levitation Simulation

This folder contains a numerical model for simulating a micron-scale particle held in an optical levitation trap. The main script is:

```text
project_3d_cython.py
```

This is the main 3D simulation script. Most settings are changed near the top of the file under `User-adjustable parameters`, then the script is run directly.

## What The Code Does

The simulation calculates the forces on a trapped particle, finds the trap equilibrium, and then evolves the particle motion in time using a BAOAB Langevin integrator. It includes optical forces, gravity, buoyancy, gas damping, Brownian motion, photophoretic force, laser-power noise, and an optional delayed PD feedback loop.

The code then saves diagnostic plots and data products such as trajectories, power spectral densities, force checks, trap-frequency comparisons, and 3D trajectory projections.

## Main Files

```text
project_3d_cython.py                 main 3D simulation script
baoab_3d_cython.pyx                  Cython source for the fast solver
baoab_3d_cython.cpython-312-darwin.so compiled Cython solver used when available
setup_baoab_3d.py                    build script for the Cython extension

```

## Script Guide

The files in this folder are related variants of the same core 3D optical levitation model. In most cases, `project_3d_cython.py` is the best starting point. The other scripts keep the same force model and BAOAB simulation structure, but focus on a specific experiment, diagnostic or parameter sweep.

| Script | Purpose |
| --- | --- |
| `project_3d_cython.py` | Main 3D BAOAB simulation. Use this for ordinary trajectory, PSD, force, and trap-frequency calculations. |
| `project_3d_cython_experimental_comparison.py` | Compares simulation results with experimental data. |
| `project_3d_cython_driven_dynamics.py` | Drives the trap by deliberately modulating laser power. Used for step response, sine/square modulation, frequency response, and resonance tests. |
| `particle_loading_batch.py` | Batch version of the particle-loading study. Produces large phase-space maps and optional bottle-loading Monte Carlo results. |
| `particle_moving_trap_velocity_sweep.py` | Tests how fast the trap can be moved before the particle is lost, including pressure-dependent maximum velocity scans. |
| `project_3d_linewidth_sweep_backup3.py` | Uses simulated PSD linewidths to recover Brownian/surface temperature and test pressure/temperature fitting accuracy. |
| `project_3d_cython_power_ramp.py` | Simulates a laser-power ramp and tracks how the equilibrium position, stability, trajectory, and PSD change during the ramp. |
| `project_3d_cython_pressure_ramp.py` | Simulates a pressure ramp, including pressure-dependent damping, Brownian forcing, and photophoretic force changes. |

## Choosing A Script

Use this rough guide:

```text
ordinary 3D trajectory or PSD      -> project_3d_cython.py
laser modulation / driven response -> project_3d_cython_driven_dynamics.py
feedback control                   -> project_3d_cython_laser_feedback.py
runtime profiling                  -> project_3d_cython_benchmarking.py
capture / loading maps             -> project_3d_cython_particle_loading.py or particle_loading_batch.py
moving trap speed limit            -> particle_moving_trap_velocity_sweep.py
surface-temperature recovery       -> project_3d_linewidth_sweep_backup3.py
power ramp                         -> project_3d_cython_power_ramp.py
pressure ramp                      -> project_3d_cython_pressure_ramp.py
experimental comparison            -> project_3d_cython_experimental_comparison.py
```


## Requirements

The script uses Python 3 with:

```text
numpy
scipy
matplotlib
```

The Cython extension is optional but strongly recommended for long runs. If it is available, the script prints:

```text
Cython BAOAB solver available = True
```

## Quick Start

From this folder, run:

```bash
python3 project_3d_cython.py
```

By default, plots are saved to:

```text
cython_outputs/
```

To change the simulation, edit the values near the top of `project_3d_cython.py` under `User-adjustable parameters`. For example, change `t_end` for simulation length, `p` for pressure, `P_laser` for laser power, and the `use_...` switches for noise, Brownian motion, feedback and scan modes.

To save outputs somewhere else, set the `PROJECT_3D_SAVE_PATH` environment variable before running the script. For example:

```bash
PROJECT_3D_SAVE_PATH=cython_outputs/test_run python3 project_3d_cython.py
```

## Main User Settings Inside The Script

Most important settings are near the top of the script under `User-adjustable parameters`. These include:

```text
radius                         particle radius
density                        particle density
p                              gas pressure
T                              gas temperature
w0                             laser beam waist
wavelength                     laser wavelength
P_laser                        laser power
use_laser_power_noise          laser noise on/off
laser_noise_model              square or psd_matched
use_brownian_motion            Brownian motion on/off
t_end                          simulation duration
dt_baoab                       integration time step
use_pd_feedback                feedback on/off
run_power_pressure_scan        power/pressure scan mode
run_diameter_scan              diameter scan mode
```

## What Happens In Order

1. Imports packages, loads the Cython solver, and applies plot styling.
2. Defines particle, gas, laser, damping, feedback, and plotting parameters.
3. Uses the chosen settings from the user-adjustable parameter block.
4. Runs optional scan modes if selected.
5. Sets up plot-saving helpers.
6. Calculates derived quantities such as mass, gas density, mean free path, Knudsen number, Rayleigh range, and peak intensity.
7. Builds the Gaussian beam model.
8. Calculates optical forces using an Ashkin-style ray-optics model.
9. Adds the pressure-dependent photophoretic force.
10. Builds a 3D force lookup table for faster time stepping.
11. Finds the stable axial equilibrium position.
12. Calculates local trap stiffness and expected resonant frequencies.
13. Saves force-linearity and stiffness diagnostic plots.
14. Calculates the gas damping coefficient.
15. Sets the initial particle position and velocity.
16. Generates laser-power noise if enabled.
17. Runs the BAOAB trajectory simulation.
18. Optionally runs the delayed PD feedback simulation.
19. Compares constant-power, noisy-power, and no-Brownian trajectories.
20. Saves time-domain trajectory plots.
21. Saves position histograms if enabled.
22. Calculates raw and Welch-averaged power spectral densities.
23. Compares simulated trap frequencies with the linear prediction.
24. Saves PSD plots for motion, laser power, and laser-noise effects.
25. Saves 3D trajectory plots, projection plots, and force-field diagnostics.

## Main Outputs

The exact outputs depend on which switches are enabled, but common files include:

```text
xyz_time_traces.png
laser_power_noise.png
xyz_laser_noise_comparison.png
laser_noise_effect.png
brownian_effect.png
psd_x.png
psd_y.png
psd_z.png
psd_xz_welch.png
psd_laser_noise_effect_welch.png
trap_frequency_comparison.csv
trap_frequency_comparison_table.png
trajectory_3d.png
trajectory_projections.png
fx_magnitude_heatmap.png
```

Plot data are also saved as CSV files for many figures.

## Notes

- Long simulations can take a while, especially without the Cython solver.
- The default time step is small, so set `t_end = 1` or `t_end = 5` for quick test runs.
- The script is currently set up for the local project folder structure used during development. If moving the code to another machine, check paths such as `Power_30min.csv` and the Cython extension location.
- Generated output folders and large data files should usually be excluded from Git unless they are needed for reproducing a specific result.

## Additional Validation And Testing Scripts

The main simulation workflow is described above. The folder also contains several supporting scripts that were used while developing, checking and comparing the model. These are included for transparency and reproducibility, but they are not all needed for a standard run of the final 3D simulation.

| File | Purpose |
| --- | --- |
| `project_1dnoise.py` | Early 1D Brownian levitation model used to test the basic stochastic dynamics and PSD analysis. |
| `project_1d_lasernoise.py` | 1D model with laser-power noise included. |
| `project_1d_noise_feedback.py` | 1D test case for simple feedback cooling ideas. |
| `project_2d.py` | Early 2D x-z model using a simplified force description. |
| `project_2d_improved.py` | Improved 2D model using the ray-optics force calculation. |
| `project_2d_noise.py` | 2D model including Brownian motion, laser-power noise, lookup tables and PSD analysis. |
| `project_2d_cython.py` | Cython-accelerated version of the 2D model. |
| `project_3d.py` | Earlier pure-Python 3D model, kept as a reference point for the later Cython version. |
| `project_3d_cython_backup.py` | Backup copy of an earlier 3D Cython version. |
| `project_3d_cython_heating.py` | 3D variant used for heating or particle surface-temperature checks. |
| `project_3d_cython_real_laser_noise.py` | 3D variant using measured laser-power fluctuations rather than generated noise. |
| `project_3d_cython_driven_dynamics.py` | 3D driven-dynamics version used to study the response to modulated laser power. |
| `project_3d_linewidth_sweep_backup3.py` | Pressure/linewidth sweep script used for comparison with recovered-temperature or PSD-based measurements. |
| `particle_loading_batch.py` | Batch script for running particle-loading simulations over multiple conditions. |
| `particle_moving_trap_velocity_sweep.py` | Sweep script for testing particle capture/loss while moving the trap at different velocities. |
| `brownian_solver_testing.py` | Standalone validation of the Brownian/Langevin solver against expected harmonic-trap behaviour. |
| `solver_testing.py` | Deterministic solver test using a damped oscillator, mainly for checking accuracy and runtime. |
| `fourier_testing.py` | Tests of Fourier transforms, windowing and PSD estimation choices. |
| `photophoresis_testing.py` | Checks of photophoretic force models and heat-source assumptions. |
| `plot_rohatschek_absorption_sweep.py` | Plotting utility for Rohatschek/absorption parameter sweeps. |
| `plot_power_csv.py` | Utility for plotting measured laser-power CSV files. |
| `plot_particle_npy.py` | Utility for plotting experimental particle trajectory `.npy` files. |

The `.npy` files are experimental data files used by the plotting and comparison scripts. The `__pycache__/` folder is created automatically by Python and is not part of the research code.
| `particle_moving_trap_velocity_sweep.py` | Tests how fast the trap can be moved before the particle is lost, including pressure-dependent maximum velocity scans. | `Moving_trap_velocity/` |
| `project_3d_linewidth_sweep_backup3.py` | Uses simulated PSD linewidths to recover Brownian/surface temperature and test pressure/temperature fitting accuracy. | linewidth, pressure, and surface-temperature recovery plots |
| `project_3d_cython_power_ramp.py` | Simulates a laser-power ramp and tracks how the equilibrium position, stability, trajectory, and PSD change during the ramp. | `power_ramp/` |
| `project_3d_cython_pressure_ramp.py` | Simulates a pressure ramp, including pressure-dependent damping, Brownian forcing, and photophoretic force changes. | `pressure_ramp/` |

## Choosing A Script

Use this rough guide:

```text
ordinary 3D trajectory or PSD      -> project_3d_cython.py
laser modulation / driven response -> project_3d_cython_driven_dynamics.py
feedback control                   -> project_3d_cython_laser_feedback.py
runtime profiling                  -> project_3d_cython_benchmarking.py
capture / loading maps             -> project_3d_cython_particle_loading.py or particle_loading_batch.py
moving trap speed limit            -> particle_moving_trap_velocity_sweep.py
surface-temperature recovery       -> project_3d_linewidth_sweep_backup3.py
power ramp                         -> project_3d_cython_power_ramp.py
pressure ramp                      -> project_3d_cython_pressure_ramp.py
improved force/noise validation    -> project_3d_cython_new.py
```


## Requirements

The script uses Python 3 with:

```text
numpy
scipy
matplotlib
```

The Cython extension is optional but strongly recommended for long runs. If it is available, the script prints:

```text
Cython BAOAB solver available = True
```

## Quick Start

From this folder, run:

```bash
python3 project_3d_cython.py
```

By default, plots are saved to:

```text
cython_outputs/
```

To change the simulation, edit the values near the top of `project_3d_cython.py` under `User-adjustable parameters`. For example, change `t_end` for simulation length, `p` for pressure, `P_laser` for laser power, and the `use_...` switches for noise, Brownian motion, feedback and scan modes.

To save outputs somewhere else, set the `PROJECT_3D_SAVE_PATH` environment variable before running the script. For example:

```bash
PROJECT_3D_SAVE_PATH=cython_outputs/test_run python3 project_3d_cython.py
```

## Main User Settings Inside The Script

Most important settings are near the top of the script under `User-adjustable parameters`. These include:

```text
radius                         particle radius
density                        particle density
p                              gas pressure
T                              gas temperature
w0                             laser beam waist
wavelength                     laser wavelength
P_laser                        laser power
use_laser_power_noise          laser noise on/off
laser_noise_model              square or psd_matched
use_brownian_motion            Brownian motion on/off
t_end                          simulation duration
dt_baoab                       integration time step
use_pd_feedback                feedback on/off
run_power_pressure_scan        power/pressure scan mode
run_diameter_scan              diameter scan mode
```

## What Happens In Order

1. Imports packages, loads the Cython solver, and applies plot styling.
2. Defines particle, gas, laser, damping, feedback, and plotting parameters.
3. Uses the chosen settings from the user-adjustable parameter block.
4. Runs optional scan modes if selected.
5. Sets up plot-saving helpers.
6. Calculates derived quantities such as mass, gas density, mean free path, Knudsen number, Rayleigh range, and peak intensity.
7. Builds the Gaussian beam model.
8. Calculates optical forces using an Ashkin-style ray-optics model.
9. Adds the pressure-dependent photophoretic force.
10. Builds a 3D force lookup table for faster time stepping.
11. Finds the stable axial equilibrium position.
12. Calculates local trap stiffness and expected resonant frequencies.
13. Saves force-linearity and stiffness diagnostic plots.
14. Calculates the gas damping coefficient.
15. Sets the initial particle position and velocity.
16. Generates laser-power noise if enabled.
17. Runs the BAOAB trajectory simulation.
18. Optionally runs the delayed PD feedback simulation.
19. Compares constant-power, noisy-power, and no-Brownian trajectories.
20. Saves time-domain trajectory plots.
21. Saves position histograms if enabled.
22. Calculates raw and Welch-averaged power spectral densities.
23. Compares simulated trap frequencies with the linear prediction.
24. Saves PSD plots for motion, laser power, and laser-noise effects.
25. Saves 3D trajectory plots, projection plots, and force-field diagnostics.

## Main Outputs

The exact outputs depend on which switches are enabled, but common files include:

```text
xyz_time_traces.png
laser_power_noise.png
xyz_laser_noise_comparison.png
laser_noise_effect.png
brownian_effect.png
psd_x.png
psd_y.png
psd_z.png
psd_xz_welch.png
psd_laser_noise_effect_welch.png
trap_frequency_comparison.csv
trap_frequency_comparison_table.png
trajectory_3d.png
trajectory_projections.png
fx_magnitude_heatmap.png
```

Plot data are also saved as CSV files for many figures.
=======
>>>>>>> 376b194 (Update cleaned simulation code)


## Script Guide

<<<<<<< HEAD
=======
The files in this folder are related variants of the same core 3D optical levitation model. In most cases, `project_3d_cython.py` is the best starting point. The other scripts keep the same force model and BAOAB simulation structure, but focus on a specific experiment, diagnostic or parameter sweep.

| Script | Purpose | Typical outputs |
| --- | --- | --- |
| `project_3d_cython.py` | Main 3D BAOAB simulation. Use this for ordinary trajectory, PSD, force, and trap-frequency calculations. | `cython_outputs/` |
| `project_3d_cython_new.py` | Improved general model with extra optical-force options, measured/PSD-matched laser noise, and validation modes. | `Figures/` |
| `project_3d_cython_driven_dynamics.py` | Drives the trap by deliberately modulating laser power. Used for step response, sine/square modulation, frequency response, and resonance tests. | `driven dynamics/` |
| `project_3d_cython_laser_feedback.py` | Tests delayed PD feedback where measured axial position is used to vary laser power. Includes feedback prediction and history-length sweeps. | `Figures/` |
| `project_3d_cython_benchmarking.py` | Times the main parts of the simulation to identify slow sections and compare solver performance. | `plots/` plus terminal timing table |
| `project_3d_cython_particle_loading.py` | Studies particle capture/loading, including phase-space capture maps and trap-loss boundaries. | phase-space plots/CSVs, depending on enabled switches |
| `particle_loading_batch.py` | Batch version of the particle-loading study. Produces large phase-space maps and optional bottle-loading Monte Carlo results. | `Phase_space/` |
| `particle_moving_trap_velocity_sweep.py` | Tests how fast the trap can be moved before the particle is lost, including pressure-dependent maximum velocity scans. | `Moving_trap_velocity/` |
| `project_3d_linewidth_sweep_backup3.py` | Uses simulated PSD linewidths to recover Brownian/surface temperature and test pressure/temperature fitting accuracy. | linewidth, pressure, and surface-temperature recovery plots |
| `project_3d_cython_power_ramp.py` | Simulates a laser-power ramp and tracks how the equilibrium position, stability, trajectory, and PSD change during the ramp. | `power_ramp/` |
| `project_3d_cython_pressure_ramp.py` | Simulates a pressure ramp, including pressure-dependent damping, Brownian forcing, and photophoretic force changes. | `pressure_ramp/` |

## Choosing A Script

Use this rough guide:

```text
ordinary 3D trajectory or PSD      -> project_3d_cython.py
laser modulation / driven response -> project_3d_cython_driven_dynamics.py
feedback control                   -> project_3d_cython_laser_feedback.py
runtime profiling                  -> project_3d_cython_benchmarking.py
capture / loading maps             -> project_3d_cython_particle_loading.py or particle_loading_batch.py
moving trap speed limit            -> particle_moving_trap_velocity_sweep.py
surface-temperature recovery       -> project_3d_linewidth_sweep_backup3.py
power ramp                         -> project_3d_cython_power_ramp.py
pressure ramp                      -> project_3d_cython_pressure_ramp.py
improved force/noise validation    -> project_3d_cython_new.py
```


## Requirements

The script uses Python 3 with:

```text
numpy
scipy
matplotlib
```

The Cython extension is optional but strongly recommended for long runs. If it is available, the script prints:

```text
Cython BAOAB solver available = True
```

## Quick Start

From this folder, run:

```bash
python3 project_3d_cython.py
```

By default, plots are saved to:

```text
cython_outputs/
```

To change the simulation, edit the values near the top of `project_3d_cython.py` under `User-adjustable parameters`. For example, change `t_end` for simulation length, `p` for pressure, `P_laser` for laser power, and the `use_...` switches for noise, Brownian motion, feedback and scan modes.

To save outputs somewhere else, set the `PROJECT_3D_SAVE_PATH` environment variable before running the script. For example:

```bash
PROJECT_3D_SAVE_PATH=cython_outputs/test_run python3 project_3d_cython.py
```

## Main User Settings Inside The Script

Most important settings are near the top of the script under `User-adjustable parameters`. These include:

```text
radius                         particle radius
density                        particle density
p                              gas pressure
T                              gas temperature
w0                             laser beam waist
wavelength                     laser wavelength
P_laser                        laser power
use_laser_power_noise          laser noise on/off
laser_noise_model              square or psd_matched
use_brownian_motion            Brownian motion on/off
t_end                          simulation duration
dt_baoab                       integration time step
use_pd_feedback                feedback on/off
run_power_pressure_scan        power/pressure scan mode
run_diameter_scan              diameter scan mode
```

## What Happens In Order

1. Imports packages, loads the Cython solver, and applies plot styling.
2. Defines particle, gas, laser, damping, feedback, and plotting parameters.
3. Uses the chosen settings from the user-adjustable parameter block.
4. Runs optional scan modes if selected.
5. Sets up plot-saving helpers.
6. Calculates derived quantities such as mass, gas density, mean free path, Knudsen number, Rayleigh range, and peak intensity.
7. Builds the Gaussian beam model.
8. Calculates optical forces using an Ashkin-style ray-optics model.
9. Adds the pressure-dependent photophoretic force.
10. Builds a 3D force lookup table for faster time stepping.
11. Finds the stable axial equilibrium position.
12. Calculates local trap stiffness and expected resonant frequencies.
13. Saves force-linearity and stiffness diagnostic plots.
14. Calculates the gas damping coefficient.
15. Sets the initial particle position and velocity.
16. Generates laser-power noise if enabled.
17. Runs the BAOAB trajectory simulation.
18. Optionally runs the delayed PD feedback simulation.
19. Compares constant-power, noisy-power, and no-Brownian trajectories.
20. Saves time-domain trajectory plots.
21. Saves position histograms if enabled.
22. Calculates raw and Welch-averaged power spectral densities.
23. Compares simulated trap frequencies with the linear prediction.
24. Saves PSD plots for motion, laser power, and laser-noise effects.
25. Saves 3D trajectory plots, projection plots, and force-field diagnostics.

## Main Outputs

The exact outputs depend on which switches are enabled, but common files include:

```text
xyz_time_traces.png
laser_power_noise.png
xyz_laser_noise_comparison.png
laser_noise_effect.png
brownian_effect.png
psd_x.png
psd_y.png
psd_z.png
psd_xz_welch.png
psd_laser_noise_effect_welch.png
trap_frequency_comparison.csv
trap_frequency_comparison_table.png
trajectory_3d.png
trajectory_projections.png
fx_magnitude_heatmap.png
```

Plot data are also saved as CSV files for many figures.

## Notes

- Long simulations can take a while, especially without the Cython solver.
- The default time step is small, so set `t_end = 1` or `t_end = 5` for quick test runs.
- The script is currently set up for the local project folder structure used during development. If moving the code to another machine, check paths such as `Power_30min.csv` and the Cython extension location.
- Generated output folders and large data files should usually be excluded from Git unless they are needed for reproducing a specific result.

>>>>>>> 376b194 (Update cleaned simulation code)
## Additional Validation And Testing Scripts

The main simulation workflow is described above. The folder also contains several supporting scripts that were used while developing, checking and comparing the model. These are included for transparency and reproducibility, but they are not all needed for a standard run of the final 3D simulation.

| File | Purpose |
| --- | --- |
| `project_1dnoise.py` | Early 1D Brownian levitation model used to test the basic stochastic dynamics and PSD analysis. |
| `project_1d_lasernoise.py` | 1D model with laser-power noise included. |
| `project_1d_noise_feedback.py` | 1D test case for simple feedback cooling ideas. |
| `project_2d.py` | Early 2D x-z model using a simplified force description. |
| `project_2d_improved.py` | Improved 2D model using the ray-optics force calculation. |
| `project_2d_noise.py` | 2D model including Brownian motion, laser-power noise, lookup tables and PSD analysis. |
| `project_2d_cython.py` | Cython-accelerated version of the 2D model. |
| `project_3d.py` | Earlier pure-Python 3D model, kept as a reference point for the later Cython version. |
| `project_3d_cython_backup.py` | Backup copy of an earlier 3D Cython version. |
| `project_3d_cython_heating.py` | 3D variant used for heating or particle surface-temperature checks. |
| `project_3d_cython_real_laser_noise.py` | 3D variant using measured laser-power fluctuations rather than generated noise. |
| `project_3d_cython_driven_dynamics.py` | 3D driven-dynamics version used to study the response to modulated laser power. |
| `project_3d_linewidth_sweep_backup3.py` | Pressure/linewidth sweep script used for comparison with recovered-temperature or PSD-based measurements. |
<<<<<<< HEAD
=======
| `particle_loading_batch.py` | Batch script for running particle-loading simulations over multiple conditions. |
| `particle_moving_trap_velocity_sweep.py` | Sweep script for testing particle capture/loss while moving the trap at different velocities. |
>>>>>>> 376b194 (Update cleaned simulation code)
| `brownian_solver_testing.py` | Standalone validation of the Brownian/Langevin solver against expected harmonic-trap behaviour. |
| `solver_testing.py` | Deterministic solver test using a damped oscillator, mainly for checking accuracy and runtime. |
| `fourier_testing.py` | Tests of Fourier transforms, windowing and PSD estimation choices. |
| `photophoresis_testing.py` | Checks of photophoretic force models and heat-source assumptions. |
| `plot_rohatschek_absorption_sweep.py` | Plotting utility for Rohatschek/absorption parameter sweeps. |
| `plot_power_csv.py` | Utility for plotting measured laser-power CSV files. |
| `plot_particle_npy.py` | Utility for plotting experimental particle trajectory `.npy` files. |

The `.npy` files are experimental data files used by the plotting and comparison scripts. The `__pycache__/` folder is created automatically by Python and is not part of the research code.
