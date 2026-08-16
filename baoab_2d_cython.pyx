# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as cnp


cdef inline void lookup_force_2d(
    double[::1] x_values,
    double[::1] z_values,
    double[:, ::1] Fx_table,
    double[:, ::1] Fz_table,
    double x,
    double z,
    double power_factor,
    double* Fx,
    double* Fz,
    int* was_clamped
):
    cdef Py_ssize_t n_x = x_values.shape[0]
    cdef Py_ssize_t n_z = z_values.shape[0]
    cdef double x_min = x_values[0]
    cdef double x_max = x_values[n_x - 1]
    cdef double z_min = z_values[0]
    cdef double z_max = z_values[n_z - 1]
    cdef double x_lookup = x
    cdef double z_lookup = z
    cdef double dx = (x_max - x_min) / (n_x - 1)
    cdef double dz = (z_max - z_min) / (n_z - 1)
    cdef double x_grid_position
    cdef double z_grid_position
    cdef Py_ssize_t ix
    cdef Py_ssize_t iz
    cdef double x_weight
    cdef double z_weight
    cdef double Fx00
    cdef double Fx10
    cdef double Fx01
    cdef double Fx11
    cdef double Fz00
    cdef double Fz10
    cdef double Fz01
    cdef double Fz11
    cdef double Fx_nominal
    cdef double Fz_nominal

    was_clamped[0] = 0

    if x_lookup < x_min:
        x_lookup = x_min
        was_clamped[0] = 1
    elif x_lookup > x_max:
        x_lookup = x_max
        was_clamped[0] = 1

    if z_lookup < z_min:
        z_lookup = z_min
        was_clamped[0] = 1
    elif z_lookup > z_max:
        z_lookup = z_max
        was_clamped[0] = 1

    x_grid_position = (x_lookup - x_min) / dx
    z_grid_position = (z_lookup - z_min) / dz

    ix = <Py_ssize_t>x_grid_position
    iz = <Py_ssize_t>z_grid_position

    if ix >= n_x - 1:
        ix = n_x - 2
        x_weight = 1.0
    else:
        x_weight = x_grid_position - ix

    if iz >= n_z - 1:
        iz = n_z - 2
        z_weight = 1.0
    else:
        z_weight = z_grid_position - iz

    Fx00 = Fx_table[ix, iz]
    Fx10 = Fx_table[ix + 1, iz]
    Fx01 = Fx_table[ix, iz + 1]
    Fx11 = Fx_table[ix + 1, iz + 1]

    Fz00 = Fz_table[ix, iz]
    Fz10 = Fz_table[ix + 1, iz]
    Fz01 = Fz_table[ix, iz + 1]
    Fz11 = Fz_table[ix + 1, iz + 1]

    Fx_nominal = (
        (1.0 - x_weight) * (1.0 - z_weight) * Fx00
        + x_weight * (1.0 - z_weight) * Fx10
        + (1.0 - x_weight) * z_weight * Fx01
        + x_weight * z_weight * Fx11
    )
    Fz_nominal = (
        (1.0 - x_weight) * (1.0 - z_weight) * Fz00
        + x_weight * (1.0 - z_weight) * Fz10
        + (1.0 - x_weight) * z_weight * Fz01
        + x_weight * z_weight * Fz11
    )

    Fx[0] = power_factor * Fx_nominal
    Fz[0] = power_factor * Fz_nominal


def solve_baoab_2d_lookup_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] x_values,
    cnp.ndarray[cnp.float64_t, ndim=1] z_values,
    cnp.ndarray[cnp.float64_t, ndim=2] Fx_table,
    cnp.ndarray[cnp.float64_t, ndim=2] Fz_table,
    cnp.ndarray[cnp.float64_t, ndim=1] power_factor_time,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_x,
    cnp.ndarray[cnp.float64_t, ndim=1] brownian_normals_z,
    double dt,
    double mass,
    double weight,
    double damping_factor,
    double thermal_velocity_scale,
    double x0,
    double z0,
    double vx0,
    double vz0,
):
    """
    Run the 2D BAOAB loop using a bilinear force lookup table.

    Forces in Fx_table/Fz_table are nominal optical + photophoretic forces at
    power_factor=1.0. The instantaneous force is scaled by power_factor_time.
    Positions outside the lookup table are clamped to the table edge.
    """
    cdef Py_ssize_t n = power_factor_time.shape[0]
    cdef Py_ssize_t i
    cdef double x_i
    cdef double z_i
    cdef double vx_i
    cdef double vz_i
    cdef double power_factor_i
    cdef double Fx_i
    cdef double Fz_i
    cdef int clamped
    cdef long out_of_bounds_count = 0

    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vx_out = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vz_out = np.zeros(n, dtype=np.float64)

    cdef double[::1] x_values_mv = x_values
    cdef double[::1] z_values_mv = z_values
    cdef double[:, ::1] Fx_table_mv = Fx_table
    cdef double[:, ::1] Fz_table_mv = Fz_table
    cdef double[::1] power_mv = power_factor_time
    cdef double[::1] brownian_x_mv = brownian_normals_x
    cdef double[::1] brownian_z_mv = brownian_normals_z
    cdef double[::1] x_out_mv = x_out
    cdef double[::1] z_out_mv = z_out
    cdef double[::1] vx_out_mv = vx_out
    cdef double[::1] vz_out_mv = vz_out

    x_out_mv[0] = x0
    z_out_mv[0] = z0
    vx_out_mv[0] = vx0
    vz_out_mv[0] = vz0

    for i in range(n - 1):
        x_i = x_out_mv[i]
        z_i = z_out_mv[i]
        vx_i = vx_out_mv[i]
        vz_i = vz_out_mv[i]
        power_factor_i = power_mv[i]

        lookup_force_2d(
            x_values_mv,
            z_values_mv,
            Fx_table_mv,
            Fz_table_mv,
            x_i,
            z_i,
            power_factor_i,
            &Fx_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        x_i += 0.5 * dt * vx_i
        z_i += 0.5 * dt * vz_i

        vx_i = damping_factor * vx_i + thermal_velocity_scale * brownian_x_mv[i]
        vz_i = damping_factor * vz_i + thermal_velocity_scale * brownian_z_mv[i]

        x_i += 0.5 * dt * vx_i
        z_i += 0.5 * dt * vz_i

        lookup_force_2d(
            x_values_mv,
            z_values_mv,
            Fx_table_mv,
            Fz_table_mv,
            x_i,
            z_i,
            power_factor_i,
            &Fx_i,
            &Fz_i,
            &clamped
        )
        out_of_bounds_count += clamped

        vx_i += 0.5 * dt * Fx_i / mass
        vz_i += 0.5 * dt * (Fz_i - weight) / mass

        x_out_mv[i + 1] = x_i
        z_out_mv[i + 1] = z_i
        vx_out_mv[i + 1] = vx_i
        vz_out_mv[i + 1] = vz_i

    return x_out, z_out, vx_out, vz_out, out_of_bounds_count
