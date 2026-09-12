# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport ceil, isfinite, sqrt


cdef inline double clamp_double(double value, double lower, double upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef inline int trap_lost_3d(
    double x,
    double y,
    double z,
    double vx,
    double vz,
    double axial_net_force,
    double axial_potential,
    double mass,
    bint use_axial_energy_loss,
    double axial_lower_escape_z,
    double axial_lower_escape_energy,
    double x_eq,
    double y_eq,
    double z_eq,
    double x_outward_limit,
    double below_equilibrium_limit
):
    cdef double dx
    cdef double below_equilibrium_displacement
    cdef double axial_energy

    if not isfinite(x) or not isfinite(y) or not isfinite(z):
        return 1

    if not isfinite(vx) or not isfinite(vz):
        return 1

    if not isfinite(axial_net_force):
        return 1

    dx = x - x_eq
    below_equilibrium_displacement = z_eq - z

    if (dx > x_outward_limit and vx > 0.0) or (
        dx < -x_outward_limit and vx < 0.0
    ):
        return 1

    if use_axial_energy_loss:
        if (
            not isfinite(axial_potential)
            or not isfinite(axial_lower_escape_z)
            or not isfinite(axial_lower_escape_energy)
        ):
            return 1

        axial_energy = 0.5 * mass * vz * vz + axial_potential
        return (
            (
                axial_energy >= axial_lower_escape_energy
                and vz < 0.0
            )
            or (
                z <= axial_lower_escape_z
                and vz <= 0.0
            )
        )

    return (
        below_equilibrium_displacement > below_equilibrium_limit
        and axial_net_force < 0.0
    )


cdef inline double lookup_linear_1d(
    double[::1] x_values,
    double[::1] y_values,
    double x
):
    cdef Py_ssize_t n = x_values.shape[0]
    cdef double x_min
    cdef double x_max
    cdef double dx
    cdef double grid_position
    cdef Py_ssize_t index
    cdef double weight

    if n <= 1:
        return y_values[0]

    x_min = x_values[0]
    x_max = x_values[n - 1]

    if x <= x_min:
        return y_values[0]
    if x >= x_max:
        return y_values[n - 1]

    dx = (x_max - x_min) / (n - 1)
    grid_position = (x - x_min) / dx
    index = <Py_ssize_t>grid_position

    if index >= n - 1:
        index = n - 2
        weight = 1.0
    else:
        weight = grid_position - index

    return (1.0 - weight) * y_values[index] + weight * y_values[index + 1]


cdef inline void lookup_force_3d(
    double[::1] r_values,
    double[::1] z_values,
    double[:, ::1] Fr_table,
    double[:, ::1] Fz_table,
    double x,
    double y,
    double z,
    double power_factor,
    double radial_zero_tolerance,
    double* Fx,
    double* Fy,
    double* Fz,
    int* was_clamped
):
    cdef Py_ssize_t n_r = r_values.shape[0]
    cdef Py_ssize_t n_z = z_values.shape[0]
    cdef double r_min = r_values[0]
    cdef double r_max = r_values[n_r - 1]
    cdef double z_min = z_values[0]
    cdef double z_max = z_values[n_z - 1]
    cdef double r = sqrt(x * x + y * y)
    cdef double r_lookup = r
    cdef double z_lookup = z
    cdef double dr = (r_max - r_min) / (n_r - 1)
    cdef double dz = (z_max - z_min) / (n_z - 1)
    cdef double r_grid_position
    cdef double z_grid_position
    cdef Py_ssize_t ir
    cdef Py_ssize_t iz
    cdef double r_weight
    cdef double z_weight
    cdef double Fr00
    cdef double Fr10
    cdef double Fr01
    cdef double Fr11
    cdef double Fz00
    cdef double Fz10
    cdef double Fz01
    cdef double Fz11
    cdef double Fr_nominal
    cdef double Fz_nominal
    cdef double Fr

    was_clamped[0] = 0

    if r_lookup < r_min:
        r_lookup = r_min
        was_clamped[0] = 1
    elif r_lookup > r_max:
        r_lookup = r_max
        was_clamped[0] = 1

    if z_lookup < z_min:
        z_lookup = z_min
        was_clamped[0] = 1
    elif z_lookup > z_max:
        z_lookup = z_max
        was_clamped[0] = 1

    r_grid_position = (r_lookup - r_min) / dr
    z_grid_position = (z_lookup - z_min) / dz

    ir = <Py_ssize_t>r_grid_position
    iz = <Py_ssize_t>z_grid_position

    if ir >= n_r - 1:
        ir = n_r - 2
        r_weight = 1.0
    else:
        r_weight = r_grid_position - ir

    if iz >= n_z - 1:
        iz = n_z - 2
        z_weight = 1.0
    else:
        z_weight = z_grid_position - iz

    Fr00 = Fr_table[ir, iz]
    Fr10 = Fr_table[ir + 1, iz]
    Fr01 = Fr_table[ir, iz + 1]
    Fr11 = Fr_table[ir + 1, iz + 1]

    Fz00 = Fz_table[ir, iz]
    Fz10 = Fz_table[ir + 1, iz]
    Fz01 = Fz_table[ir, iz + 1]
    Fz11 = Fz_table[ir + 1, iz + 1]

    Fr_nominal = (
        (1.0 - r_weight) * (1.0 - z_weight) * Fr00
        + r_weight * (1.0 - z_weight) * Fr10
        + (1.0 - r_weight) * z_weight * Fr01
        + r_weight * z_weight * Fr11
    )
    Fz_nominal = (
        (1.0 - r_weight) * (1.0 - z_weight) * Fz00
        + r_weight * (1.0 - z_weight) * Fz10
        + (1.0 - r_weight) * z_weight * Fz01
        + r_weight * z_weight * Fz11
    )

    Fr = power_factor * Fr_nominal
    Fz[0] = power_factor * Fz_nominal

    if r > radial_zero_tolerance:
        Fx[0] = Fr * x / r
        Fy[0] = Fr * y / r
    else:
        Fx[0] = 0.0
        Fy[0] = 0.0


def solve_baoab_3d_lookup_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] r_values,
    cnp.ndarray[cnp.float64_t, ndim=1] z_values,
    cnp.ndarray[cnp.float64_t, ndim=2] Fr_table,
    cnp.ndarray[cnp.float64_t, ndim=2] Fz_table,
    cnp.ndarray[cnp.float64_t, ndim=1] power_factor_time,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_x,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_y,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_z,
    double dt,
    double mass,
    double weight,
    double damping_factor,
    double thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double radial_zero_tolerance,
    bint use_axial_energy_loss,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_energy_z_values,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_potential_values,
    double axial_lower_escape_z,
    double axial_lower_escape_energy,
    bint terminate_on_trap_loss=False,
    long trap_loss_check_interval_steps=1,
    double x_eq=0.0,
    double y_eq=0.0,
    double z_eq=0.0,
    double trap_loss_x_outward_limit=1.0,
    double trap_loss_below_equilibrium_limit=1.0,
):
    cdef Py_ssize_t n = power_factor_time.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t sample_index
    cdef Py_ssize_t stop_index = n - 1
    cdef int lost_flag = 0
    cdef double x_i
    cdef double y_i
    cdef double z_i
    cdef double vx_i
    cdef double vy_i
    cdef double vz_i
    cdef double power_factor_i
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double axial_potential_i
    cdef int clamped
    cdef long out_of_bounds_count = 0

    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vx_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vy_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vz_out = np.zeros(n, dtype=np.float64)

    cdef double[::1] r_values_mv = r_values
    cdef double[::1] z_values_mv = z_values
    cdef double[:, ::1] Fr_table_mv = Fr_table
    cdef double[:, ::1] Fz_table_mv = Fz_table
    cdef double[::1] power_mv = power_factor_time
    cdef double[::1] brownian_x_mv = brownian_normals_x
    cdef double[::1] brownian_y_mv = brownian_normals_y
    cdef double[::1] brownian_z_mv = brownian_normals_z
    cdef double[::1] axial_energy_z_mv = axial_energy_z_values
    cdef double[::1] axial_potential_mv = axial_potential_values
    cdef double[::1] x_out_mv = x_out
    cdef double[::1] y_out_mv = y_out
    cdef double[::1] z_out_mv = z_out
    cdef double[::1] vx_out_mv = vx_out
    cdef double[::1] vy_out_mv = vy_out
    cdef double[::1] vz_out_mv = vz_out

    if trap_loss_check_interval_steps < 1:
        trap_loss_check_interval_steps = 1

    x_out_mv[0] = x0
    y_out_mv[0] = y0
    z_out_mv[0] = z0
    vx_out_mv[0] = vx0
    vy_out_mv[0] = vy0
    vz_out_mv[0] = vz0

    if terminate_on_trap_loss:
        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x0,
            y0,
            z0,
            power_mv[0],
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped
        axial_potential_i = lookup_linear_1d(
            axial_energy_z_mv,
            axial_potential_mv,
            z0
        )

        if trap_lost_3d(
            x0,
            y0,
            z0,
            vx0,
            vz0,
            Fz_i - weight,
            axial_potential_i,
            mass,
            use_axial_energy_loss,
            axial_lower_escape_z,
            axial_lower_escape_energy,
            x_eq,
            y_eq,
            z_eq,
            trap_loss_x_outward_limit,
            trap_loss_below_equilibrium_limit
        ):
            return x_out, y_out, z_out, vx_out, vy_out, vz_out, out_of_bounds_count, 0, 1

    for i in range(n - 1):
        x_i = x_out_mv[i]
        y_i = y_out_mv[i]
        z_i = z_out_mv[i]
        vx_i = vx_out_mv[i]
        vy_i = vy_out_mv[i]
        vz_i = vz_out_mv[i]
        power_factor_i = power_mv[i]

        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            power_factor_i,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        vx_i = damping_factor * vx_i + thermal_velocity_scale * brownian_x_mv[i]
        vy_i = damping_factor * vy_i + thermal_velocity_scale * brownian_y_mv[i]
        vz_i = damping_factor * vz_i + thermal_velocity_scale * brownian_z_mv[i]

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            power_factor_i,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        sample_index = i + 1
        x_out_mv[sample_index] = x_i
        y_out_mv[sample_index] = y_i
        z_out_mv[sample_index] = z_i
        vx_out_mv[sample_index] = vx_i
        vy_out_mv[sample_index] = vy_i
        vz_out_mv[sample_index] = vz_i
        axial_potential_i = lookup_linear_1d(
            axial_energy_z_mv,
            axial_potential_mv,
            z_i
        )

        if (
            terminate_on_trap_loss
            and (
                sample_index % trap_loss_check_interval_steps == 0
                or sample_index == n - 1
            )
            and trap_lost_3d(
                x_i,
                y_i,
                z_i,
                vx_i,
                vz_i,
                Fz_i - weight,
                axial_potential_i,
                mass,
                use_axial_energy_loss,
                axial_lower_escape_z,
                axial_lower_escape_energy,
                x_eq,
                y_eq,
                z_eq,
                trap_loss_x_outward_limit,
                trap_loss_below_equilibrium_limit
            )
        ):
            stop_index = sample_index
            lost_flag = 1
            break

    return (
        x_out,
        y_out,
        z_out,
        vx_out,
        vy_out,
        vz_out,
        out_of_bounds_count,
        stop_index,
        lost_flag
    )


def solve_baoab_3d_feedback_lookup_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] r_values,
    cnp.ndarray[cnp.float64_t, ndim=1] z_values,
    cnp.ndarray[cnp.float64_t, ndim=2] Fr_table,
    cnp.ndarray[cnp.float64_t, ndim=2] Fz_table,
    cnp.ndarray[cnp.float64_t, ndim=1] disturbance_power_factor_time,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_x,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_y,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_z,
    cnp.ndarray[cnp.float64_t, ndim=1] feedback_position_noise,
    double dt,
    double mass,
    double weight,
    double damping_factor,
    double thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double z_eq,
    double P_laser,
    double feedback_kp,
    double feedback_kd,
    double force_per_watt,
    double feedback_power_min_factor,
    double feedback_power_max_factor,
    double feedback_velocity_filter_alpha,
    long control_decimation,
    long loop_delay_baoab_steps,
    double dt_control,
    double radial_zero_tolerance,
    bint use_axial_energy_loss,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_energy_z_values,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_potential_values,
    double axial_lower_escape_z,
    double axial_lower_escape_energy,
    bint terminate_on_trap_loss=False,
    long trap_loss_check_interval_steps=1,
    double x_eq=0.0,
    double y_eq=0.0,
    double trap_loss_x_outward_limit=1.0,
    double trap_loss_below_equilibrium_limit=1.0,
):
    cdef Py_ssize_t n = disturbance_power_factor_time.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t delayed_i
    cdef Py_ssize_t sample_index
    cdef Py_ssize_t stop_index = n - 1
    cdef int lost_flag = 0
    cdef long out_of_bounds_count = 0
    cdef int clamped
    cdef int have_filtered_z_previous = 0
    cdef Py_ssize_t control_count = 0
    cdef Py_ssize_t n_control_updates = 0
    cdef double feedback_power_min = feedback_power_min_factor * P_laser
    cdef double feedback_power_max = feedback_power_max_factor * P_laser
    cdef double command_power = P_laser
    cdef double filtered_z_previous = 0.0
    cdef double z_measured
    cdef double z_filtered
    cdef double measured_vz
    cdef double z_error
    cdef double feedback_force
    cdef double power_change
    cdef double command_factor
    cdef double actual_power_factor
    cdef double x_i
    cdef double y_i
    cdef double z_i
    cdef double vx_i
    cdef double vy_i
    cdef double vz_i
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double axial_potential_i

    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vx_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vy_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vz_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] command_power_time
    cdef cnp.ndarray[cnp.float64_t, ndim=1] actual_power_time
    cdef cnp.ndarray[cnp.float64_t, ndim=1] control_times
    cdef cnp.ndarray[cnp.float64_t, ndim=1] measurement_times
    cdef cnp.ndarray[cnp.float64_t, ndim=1] measured_positions
    cdef cnp.ndarray[cnp.float64_t, ndim=1] filtered_positions
    cdef cnp.ndarray[cnp.float64_t, ndim=1] requested_powers

    cdef double[::1] r_values_mv
    cdef double[::1] z_values_mv
    cdef double[:, ::1] Fr_table_mv
    cdef double[:, ::1] Fz_table_mv
    cdef double[::1] disturbance_power_mv
    cdef double[::1] brownian_x_mv
    cdef double[::1] brownian_y_mv
    cdef double[::1] brownian_z_mv
    cdef double[::1] feedback_noise_mv
    cdef double[::1] axial_energy_z_mv
    cdef double[::1] axial_potential_mv
    cdef double[::1] x_out_mv
    cdef double[::1] y_out_mv
    cdef double[::1] z_out_mv
    cdef double[::1] vx_out_mv
    cdef double[::1] vy_out_mv
    cdef double[::1] vz_out_mv
    cdef double[::1] command_power_mv
    cdef double[::1] actual_power_mv
    cdef double[::1] control_times_mv
    cdef double[::1] measurement_times_mv
    cdef double[::1] measured_positions_mv
    cdef double[::1] filtered_positions_mv
    cdef double[::1] requested_powers_mv

    if n >= 2:
        n_control_updates = ((n - 2) // control_decimation) + 1

    if trap_loss_check_interval_steps < 1:
        trap_loss_check_interval_steps = 1

    x_out = np.zeros(n, dtype=np.float64)
    y_out = np.zeros(n, dtype=np.float64)
    z_out = np.zeros(n, dtype=np.float64)
    vx_out = np.zeros(n, dtype=np.float64)
    vy_out = np.zeros(n, dtype=np.float64)
    vz_out = np.zeros(n, dtype=np.float64)
    command_power_time = np.zeros(n, dtype=np.float64)
    actual_power_time = np.zeros(n, dtype=np.float64)
    control_times = np.zeros(n_control_updates, dtype=np.float64)
    measurement_times = np.zeros(n_control_updates, dtype=np.float64)
    measured_positions = np.zeros(n_control_updates, dtype=np.float64)
    filtered_positions = np.zeros(n_control_updates, dtype=np.float64)
    requested_powers = np.zeros(n_control_updates, dtype=np.float64)

    r_values_mv = r_values
    z_values_mv = z_values
    Fr_table_mv = Fr_table
    Fz_table_mv = Fz_table
    disturbance_power_mv = disturbance_power_factor_time
    brownian_x_mv = brownian_normals_x
    brownian_y_mv = brownian_normals_y
    brownian_z_mv = brownian_normals_z
    feedback_noise_mv = feedback_position_noise
    axial_energy_z_mv = axial_energy_z_values
    axial_potential_mv = axial_potential_values
    x_out_mv = x_out
    y_out_mv = y_out
    z_out_mv = z_out
    vx_out_mv = vx_out
    vy_out_mv = vy_out
    vz_out_mv = vz_out
    command_power_mv = command_power_time
    actual_power_mv = actual_power_time
    control_times_mv = control_times
    measurement_times_mv = measurement_times
    measured_positions_mv = measured_positions
    filtered_positions_mv = filtered_positions
    requested_powers_mv = requested_powers

    x_out_mv[0] = x0
    y_out_mv[0] = y0
    z_out_mv[0] = z0
    vx_out_mv[0] = vx0
    vy_out_mv[0] = vy0
    vz_out_mv[0] = vz0
    command_power_mv[0] = command_power
    actual_power_mv[0] = P_laser * clamp_double(
        disturbance_power_mv[0],
        feedback_power_min_factor,
        feedback_power_max_factor
    )

    if terminate_on_trap_loss:
        actual_power_factor = actual_power_mv[0] / P_laser
        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x0,
            y0,
            z0,
            actual_power_factor,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped
        axial_potential_i = lookup_linear_1d(
            axial_energy_z_mv,
            axial_potential_mv,
            z0
        )

        if trap_lost_3d(
            x0,
            y0,
            z0,
            vx0,
            vz0,
            Fz_i - weight,
            axial_potential_i,
            mass,
            use_axial_energy_loss,
            axial_lower_escape_z,
            axial_lower_escape_energy,
            x_eq,
            y_eq,
            z_eq,
            trap_loss_x_outward_limit,
            trap_loss_below_equilibrium_limit
        ):
            return (
                x_out,
                y_out,
                z_out,
                vx_out,
                vy_out,
                vz_out,
                command_power_time,
                actual_power_time,
                control_times[:0].copy(),
                measurement_times[:0].copy(),
                measured_positions[:0].copy(),
                filtered_positions[:0].copy(),
                requested_powers[:0].copy(),
                out_of_bounds_count,
                0,
                1
            )

    for i in range(n - 1):
        if i % control_decimation == 0:
            delayed_i = i - loop_delay_baoab_steps
            if delayed_i < 0:
                delayed_i = 0

            z_measured = z_out_mv[delayed_i] + feedback_noise_mv[control_count]

            if not have_filtered_z_previous:
                z_filtered = z_measured
                measured_vz = 0.0
                have_filtered_z_previous = 1
            else:
                z_filtered = (
                    feedback_velocity_filter_alpha * z_measured
                    + (1.0 - feedback_velocity_filter_alpha) * filtered_z_previous
                )
                measured_vz = (z_filtered - filtered_z_previous) / dt_control

            z_error = z_filtered - z_eq
            feedback_force = -feedback_kp * z_error - feedback_kd * measured_vz
            power_change = feedback_force / force_per_watt
            command_power = clamp_double(
                P_laser + power_change,
                feedback_power_min,
                feedback_power_max
            )

            control_times_mv[control_count] = i * dt
            measurement_times_mv[control_count] = delayed_i * dt
            measured_positions_mv[control_count] = z_measured
            filtered_positions_mv[control_count] = z_filtered
            requested_powers_mv[control_count] = command_power
            control_count += 1

            filtered_z_previous = z_filtered

        command_factor = command_power / P_laser
        actual_power_factor = clamp_double(
            command_factor * disturbance_power_mv[i],
            feedback_power_min_factor,
            feedback_power_max_factor
        )

        x_i = x_out_mv[i]
        y_i = y_out_mv[i]
        z_i = z_out_mv[i]
        vx_i = vx_out_mv[i]
        vy_i = vy_out_mv[i]
        vz_i = vz_out_mv[i]

        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            actual_power_factor,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        vx_i = damping_factor * vx_i + thermal_velocity_scale * brownian_x_mv[i]
        vy_i = damping_factor * vy_i + thermal_velocity_scale * brownian_y_mv[i]
        vz_i = damping_factor * vz_i + thermal_velocity_scale * brownian_z_mv[i]

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            actual_power_factor,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        sample_index = i + 1
        x_out_mv[sample_index] = x_i
        y_out_mv[sample_index] = y_i
        z_out_mv[sample_index] = z_i
        vx_out_mv[sample_index] = vx_i
        vy_out_mv[sample_index] = vy_i
        vz_out_mv[sample_index] = vz_i
        command_power_mv[i] = command_power
        actual_power_mv[i] = P_laser * actual_power_factor
        command_power_mv[sample_index] = command_power
        actual_power_mv[sample_index] = P_laser * actual_power_factor
        axial_potential_i = lookup_linear_1d(
            axial_energy_z_mv,
            axial_potential_mv,
            z_i
        )

        if (
            terminate_on_trap_loss
            and (
                sample_index % trap_loss_check_interval_steps == 0
                or sample_index == n - 1
            )
            and trap_lost_3d(
                x_i,
                y_i,
                z_i,
                vx_i,
                vz_i,
                Fz_i - weight,
                axial_potential_i,
                mass,
                use_axial_energy_loss,
                axial_lower_escape_z,
                axial_lower_escape_energy,
                x_eq,
                y_eq,
                z_eq,
                trap_loss_x_outward_limit,
                trap_loss_below_equilibrium_limit
            )
        ):
            stop_index = sample_index
            lost_flag = 1
            break

    if stop_index == n - 1 and n >= 2:
        command_power_mv[n - 1] = command_power
        actual_power_mv[n - 1] = actual_power_mv[n - 2]

    return (
        x_out,
        y_out,
        z_out,
        vx_out,
        vy_out,
        vz_out,
        command_power_time,
        actual_power_time,
        control_times[:control_count].copy(),
        measurement_times[:control_count].copy(),
        measured_positions[:control_count].copy(),
        filtered_positions[:control_count].copy(),
        requested_powers[:control_count].copy(),
        out_of_bounds_count,
        stop_index,
        lost_flag
    )


def classify_baoab_3d_lookup_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] r_values,
    cnp.ndarray[cnp.float64_t, ndim=1] z_values,
    cnp.ndarray[cnp.float64_t, ndim=2] Fr_table,
    cnp.ndarray[cnp.float64_t, ndim=2] Fz_table,
    long n_steps,
    double dt,
    double mass,
    double weight,
    double damping_factor,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double power_factor,
    double radial_zero_tolerance,
    bint use_axial_energy_loss,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_energy_z_values,
    cnp.ndarray[cnp.float64_t, ndim=1] axial_potential_values,
    double axial_lower_escape_z,
    double axial_lower_escape_energy,
    long trap_loss_check_interval_steps,
    double x_eq,
    double y_eq,
    double z_eq,
    double trap_loss_x_outward_limit,
    double trap_loss_below_equilibrium_limit,
):
    """
    Fast deterministic capture/loss classifier for parameter sweeps.

    This uses the same BAOAB drift/kick structure and force lookup as the full
    trajectory solver, but it stores only the current state. Brownian kicks and
    laser-power noise are intentionally omitted so the returned basin boundary is
    deterministic.
    """
    cdef long i
    cdef long sample_index
    cdef long stop_index = n_steps
    cdef int lost_flag = 0
    cdef int clamped
    cdef long out_of_bounds_count = 0
    cdef double x_i = x0
    cdef double y_i = y0
    cdef double z_i = z0
    cdef double vx_i = vx0
    cdef double vy_i = vy0
    cdef double vz_i = vz0
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double axial_potential_i

    cdef double[::1] r_values_mv = r_values
    cdef double[::1] z_values_mv = z_values
    cdef double[:, ::1] Fr_table_mv = Fr_table
    cdef double[:, ::1] Fz_table_mv = Fz_table
    cdef double[::1] axial_energy_z_mv = axial_energy_z_values
    cdef double[::1] axial_potential_mv = axial_potential_values

    if trap_loss_check_interval_steps < 1:
        trap_loss_check_interval_steps = 1

    lookup_force_3d(
        r_values_mv,
        z_values_mv,
        Fr_table_mv,
        Fz_table_mv,
        x_i,
        y_i,
        z_i,
        power_factor,
        radial_zero_tolerance,
        &Fx_i,
        &Fy_i,
        &Fz_i,
        &clamped
    )
    out_of_bounds_count += clamped
    axial_potential_i = lookup_linear_1d(
        axial_energy_z_mv,
        axial_potential_mv,
        z_i
    )

    if trap_lost_3d(
        x_i,
        y_i,
        z_i,
        vx_i,
        vz_i,
        Fz_i - weight,
        axial_potential_i,
        mass,
        use_axial_energy_loss,
        axial_lower_escape_z,
        axial_lower_escape_energy,
        x_eq,
        y_eq,
        z_eq,
        trap_loss_x_outward_limit,
        trap_loss_below_equilibrium_limit
    ):
        return (
            1,
            0,
            x_i,
            y_i,
            z_i,
            vx_i,
            vy_i,
            vz_i,
            out_of_bounds_count
        )

    for i in range(n_steps):
        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            power_factor,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        vx_i = damping_factor * vx_i
        vy_i = damping_factor * vy_i
        vz_i = damping_factor * vz_i

        x_i += 0.5 * dt * vx_i
        y_i += 0.5 * dt * vy_i
        z_i += 0.5 * dt * vz_i

        lookup_force_3d(
            r_values_mv,
            z_values_mv,
            Fr_table_mv,
            Fz_table_mv,
            x_i,
            y_i,
            z_i,
            power_factor,
            radial_zero_tolerance,
            &Fx_i,
            &Fy_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vy_i += 0.5 * dt * Fy_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        sample_index = i + 1
        axial_potential_i = lookup_linear_1d(
            axial_energy_z_mv,
            axial_potential_mv,
            z_i
        )

        if (
            sample_index % trap_loss_check_interval_steps == 0
            or sample_index == n_steps
        ) and trap_lost_3d(
            x_i,
            y_i,
            z_i,
            vx_i,
            vz_i,
            Fz_i - weight,
            axial_potential_i,
            mass,
            use_axial_energy_loss,
            axial_lower_escape_z,
            axial_lower_escape_energy,
            x_eq,
            y_eq,
            z_eq,
            trap_loss_x_outward_limit,
            trap_loss_below_equilibrium_limit
        ):
            stop_index = sample_index
            lost_flag = 1
            break

    return (
        lost_flag,
        stop_index,
        x_i,
        y_i,
        z_i,
        vx_i,
        vy_i,
        vz_i,
        out_of_bounds_count
    )
