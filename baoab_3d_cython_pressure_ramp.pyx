# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

ctypedef cnp.float64_t DTYPE_t

cnp.import_array()


cdef inline int lookup_force_clamped(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_table,
    double x,
    double y,
    double z,
    double power_factor,
    double radial_zero_tolerance,
    double* Fx,
    double* Fy,
    double* Fz
) noexcept nogil:
    cdef Py_ssize_t n_r = r_values.shape[0]
    cdef Py_ssize_t n_z = z_values.shape[0]
    cdef double r = sqrt(x * x + y * y)
    cdef double r_min = r_values[0]
    cdef double r_max = r_values[n_r - 1]
    cdef double z_min = z_values[0]
    cdef double z_max = z_values[n_z - 1]
    cdef double r_lookup = r
    cdef double z_lookup = z
    cdef int out_of_bounds = 0
    cdef double dr
    cdef double dz
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

    if r_lookup < r_min:
        r_lookup = r_min
        out_of_bounds = 1
    elif r_lookup > r_max:
        r_lookup = r_max
        out_of_bounds = 1

    if z_lookup < z_min:
        z_lookup = z_min
        out_of_bounds = 1
    elif z_lookup > z_max:
        z_lookup = z_max
        out_of_bounds = 1

    dr = (r_max - r_min) / (n_r - 1)
    dz = (z_max - z_min) / (n_z - 1)

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

    return out_of_bounds


cdef inline int lookup_force_clamped_pressure(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_ray_table,
    DTYPE_t[:, ::1] Fz_photo_base_table,
    double x,
    double y,
    double z,
    double power_factor,
    double photo_pressure_factor,
    double radial_zero_tolerance,
    double* Fx,
    double* Fy,
    double* Fz
) noexcept nogil:
    cdef Py_ssize_t n_r = r_values.shape[0]
    cdef Py_ssize_t n_z = z_values.shape[0]
    cdef double r = sqrt(x * x + y * y)
    cdef double r_min = r_values[0]
    cdef double r_max = r_values[n_r - 1]
    cdef double z_min = z_values[0]
    cdef double z_max = z_values[n_z - 1]
    cdef double r_lookup = r
    cdef double z_lookup = z
    cdef int out_of_bounds = 0
    cdef double dr
    cdef double dz
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
    cdef double Fz_ray00
    cdef double Fz_ray10
    cdef double Fz_ray01
    cdef double Fz_ray11
    cdef double Fz_photo00
    cdef double Fz_photo10
    cdef double Fz_photo01
    cdef double Fz_photo11
    cdef double Fr_nominal
    cdef double Fz_ray_nominal
    cdef double Fz_photo_base
    cdef double Fz_nominal
    cdef double Fr

    if r_lookup < r_min:
        r_lookup = r_min
        out_of_bounds = 1
    elif r_lookup > r_max:
        r_lookup = r_max
        out_of_bounds = 1

    if z_lookup < z_min:
        z_lookup = z_min
        out_of_bounds = 1
    elif z_lookup > z_max:
        z_lookup = z_max
        out_of_bounds = 1

    dr = (r_max - r_min) / (n_r - 1)
    dz = (z_max - z_min) / (n_z - 1)

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

    Fz_ray00 = Fz_ray_table[ir, iz]
    Fz_ray10 = Fz_ray_table[ir + 1, iz]
    Fz_ray01 = Fz_ray_table[ir, iz + 1]
    Fz_ray11 = Fz_ray_table[ir + 1, iz + 1]

    Fz_photo00 = Fz_photo_base_table[ir, iz]
    Fz_photo10 = Fz_photo_base_table[ir + 1, iz]
    Fz_photo01 = Fz_photo_base_table[ir, iz + 1]
    Fz_photo11 = Fz_photo_base_table[ir + 1, iz + 1]

    Fr_nominal = (
        (1.0 - r_weight) * (1.0 - z_weight) * Fr00
        + r_weight * (1.0 - z_weight) * Fr10
        + (1.0 - r_weight) * z_weight * Fr01
        + r_weight * z_weight * Fr11
    )
    Fz_ray_nominal = (
        (1.0 - r_weight) * (1.0 - z_weight) * Fz_ray00
        + r_weight * (1.0 - z_weight) * Fz_ray10
        + (1.0 - r_weight) * z_weight * Fz_ray01
        + r_weight * z_weight * Fz_ray11
    )
    Fz_photo_base = (
        (1.0 - r_weight) * (1.0 - z_weight) * Fz_photo00
        + r_weight * (1.0 - z_weight) * Fz_photo10
        + (1.0 - r_weight) * z_weight * Fz_photo01
        + r_weight * z_weight * Fz_photo11
    )
    Fz_nominal = Fz_ray_nominal + photo_pressure_factor * Fz_photo_base

    Fr = power_factor * Fr_nominal
    Fz[0] = power_factor * Fz_nominal

    if r > radial_zero_tolerance:
        Fx[0] = Fr * x / r
        Fy[0] = Fr * y / r
    else:
        Fx[0] = 0.0
        Fy[0] = 0.0

    return out_of_bounds


def solve_baoab_3d_lookup_cython(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_table,
    DTYPE_t[::1] power_factor_time,
    DTYPE_t[::1] brownian_normals_x,
    DTYPE_t[::1] brownian_normals_y,
    DTYPE_t[::1] brownian_normals_z,
    double dt,
    double mass,
    double weight,
    damping_factor,
    thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double radial_zero_tolerance=1.0e-30,
):
    cdef Py_ssize_t n = power_factor_time.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] y_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] z_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vx_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vy_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vz_out_arr = np.zeros(n, dtype=np.float64)
    cdef DTYPE_t[::1] x_out = x_out_arr
    cdef DTYPE_t[::1] y_out = y_out_arr
    cdef DTYPE_t[::1] z_out = z_out_arr
    cdef DTYPE_t[::1] vx_out = vx_out_arr
    cdef DTYPE_t[::1] vy_out = vy_out_arr
    cdef DTYPE_t[::1] vz_out = vz_out_arr
    cdef Py_ssize_t i
    cdef long out_of_bounds_count = 0
    cdef double x_i
    cdef double y_i
    cdef double z_i
    cdef double vx_i
    cdef double vy_i
    cdef double vz_i
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double power_factor_i
    cdef double damping_factor_i
    cdef double thermal_velocity_scale_i
    cdef double damping_factor_constant = 0.0
    cdef double thermal_velocity_scale_constant = 0.0
    cdef bint use_damping_factor_time = False
    cdef bint use_thermal_velocity_scale_time = False
    cdef cnp.ndarray[DTYPE_t, ndim=1] damping_factor_time_arr = np.empty(0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] thermal_velocity_scale_time_arr = np.empty(0, dtype=np.float64)
    cdef DTYPE_t[::1] damping_factor_time = damping_factor_time_arr
    cdef DTYPE_t[::1] thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    if n < 1:
        raise ValueError("power_factor_time must contain at least one sample.")
    if r_values.shape[0] < 2 or z_values.shape[0] < 2:
        raise ValueError("Force lookup table needs at least two grid points per axis.")
    if brownian_normals_x.shape[0] < n - 1 or brownian_normals_y.shape[0] < n - 1 or brownian_normals_z.shape[0] < n - 1:
        raise ValueError("Brownian normal arrays must have len(power_factor_time) - 1 samples.")

    if np.ndim(damping_factor) == 0:
        damping_factor_constant = float(damping_factor)
    else:
        use_damping_factor_time = True
        damping_factor_time_arr = np.ascontiguousarray(damping_factor, dtype=np.float64)
        if damping_factor_time_arr.shape[0] < n - 1:
            raise ValueError("damping_factor must be scalar or have len(power_factor_time) - 1 samples.")
        damping_factor_time = damping_factor_time_arr

    if np.ndim(thermal_velocity_scale) == 0:
        thermal_velocity_scale_constant = float(thermal_velocity_scale)
    else:
        use_thermal_velocity_scale_time = True
        thermal_velocity_scale_time_arr = np.ascontiguousarray(thermal_velocity_scale, dtype=np.float64)
        if thermal_velocity_scale_time_arr.shape[0] < n - 1:
            raise ValueError("thermal_velocity_scale must be scalar or have len(power_factor_time) - 1 samples.")
        thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    x_out[0] = x0
    y_out[0] = y0
    z_out[0] = z0
    vx_out[0] = vx0
    vy_out[0] = vy0
    vz_out[0] = vz0

    with nogil:
        for i in range(n - 1):
            x_i = x_out[i]
            y_i = y_out[i]
            z_i = z_out[i]
            vx_i = vx_out[i]
            vy_i = vy_out[i]
            vz_i = vz_out[i]
            power_factor_i = power_factor_time[i]
            if use_damping_factor_time:
                damping_factor_i = damping_factor_time[i]
            else:
                damping_factor_i = damping_factor_constant
            if use_thermal_velocity_scale_time:
                thermal_velocity_scale_i = thermal_velocity_scale_time[i]
            else:
                thermal_velocity_scale_i = thermal_velocity_scale_constant

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            vx_i = damping_factor_i * vx_i + thermal_velocity_scale_i * brownian_normals_x[i]
            vy_i = damping_factor_i * vy_i + thermal_velocity_scale_i * brownian_normals_y[i]
            vz_i = damping_factor_i * vz_i + thermal_velocity_scale_i * brownian_normals_z[i]

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_out[i + 1] = x_i
            y_out[i + 1] = y_i
            z_out[i + 1] = z_i
            vx_out[i + 1] = vx_i
            vy_out[i + 1] = vy_i
            vz_out[i + 1] = vz_i

    return (
        x_out_arr,
        y_out_arr,
        z_out_arr,
        vx_out_arr,
        vy_out_arr,
        vz_out_arr,
        out_of_bounds_count,
    )


def solve_baoab_3d_pressure_lookup_cython(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_ray_table,
    DTYPE_t[:, ::1] Fz_photo_base_table,
    DTYPE_t[::1] power_factor_time,
    DTYPE_t[::1] photo_pressure_factor_time,
    DTYPE_t[::1] brownian_normals_x,
    DTYPE_t[::1] brownian_normals_y,
    DTYPE_t[::1] brownian_normals_z,
    double dt,
    double mass,
    double weight,
    damping_factor,
    thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double radial_zero_tolerance=1.0e-30,
):
    cdef Py_ssize_t n = power_factor_time.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] y_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] z_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vx_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vy_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vz_out_arr = np.zeros(n, dtype=np.float64)
    cdef DTYPE_t[::1] x_out = x_out_arr
    cdef DTYPE_t[::1] y_out = y_out_arr
    cdef DTYPE_t[::1] z_out = z_out_arr
    cdef DTYPE_t[::1] vx_out = vx_out_arr
    cdef DTYPE_t[::1] vy_out = vy_out_arr
    cdef DTYPE_t[::1] vz_out = vz_out_arr
    cdef Py_ssize_t i
    cdef long out_of_bounds_count = 0
    cdef double x_i
    cdef double y_i
    cdef double z_i
    cdef double vx_i
    cdef double vy_i
    cdef double vz_i
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double power_factor_i
    cdef double photo_pressure_factor_i
    cdef double damping_factor_i
    cdef double thermal_velocity_scale_i
    cdef double damping_factor_constant = 0.0
    cdef double thermal_velocity_scale_constant = 0.0
    cdef bint use_damping_factor_time = False
    cdef bint use_thermal_velocity_scale_time = False
    cdef cnp.ndarray[DTYPE_t, ndim=1] damping_factor_time_arr = np.empty(0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] thermal_velocity_scale_time_arr = np.empty(0, dtype=np.float64)
    cdef DTYPE_t[::1] damping_factor_time = damping_factor_time_arr
    cdef DTYPE_t[::1] thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    if n < 1:
        raise ValueError("power_factor_time must contain at least one sample.")
    if r_values.shape[0] < 2 or z_values.shape[0] < 2:
        raise ValueError("Force lookup table needs at least two grid points per axis.")
    if photo_pressure_factor_time.shape[0] < n - 1:
        raise ValueError("photo_pressure_factor_time must have len(power_factor_time) - 1 samples.")
    if brownian_normals_x.shape[0] < n - 1 or brownian_normals_y.shape[0] < n - 1 or brownian_normals_z.shape[0] < n - 1:
        raise ValueError("Brownian normal arrays must have len(power_factor_time) - 1 samples.")

    if np.ndim(damping_factor) == 0:
        damping_factor_constant = float(damping_factor)
    else:
        use_damping_factor_time = True
        damping_factor_time_arr = np.ascontiguousarray(damping_factor, dtype=np.float64)
        if damping_factor_time_arr.shape[0] < n - 1:
            raise ValueError("damping_factor must be scalar or have len(power_factor_time) - 1 samples.")
        damping_factor_time = damping_factor_time_arr

    if np.ndim(thermal_velocity_scale) == 0:
        thermal_velocity_scale_constant = float(thermal_velocity_scale)
    else:
        use_thermal_velocity_scale_time = True
        thermal_velocity_scale_time_arr = np.ascontiguousarray(thermal_velocity_scale, dtype=np.float64)
        if thermal_velocity_scale_time_arr.shape[0] < n - 1:
            raise ValueError("thermal_velocity_scale must be scalar or have len(power_factor_time) - 1 samples.")
        thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    x_out[0] = x0
    y_out[0] = y0
    z_out[0] = z0
    vx_out[0] = vx0
    vy_out[0] = vy0
    vz_out[0] = vz0

    with nogil:
        for i in range(n - 1):
            x_i = x_out[i]
            y_i = y_out[i]
            z_i = z_out[i]
            vx_i = vx_out[i]
            vy_i = vy_out[i]
            vz_i = vz_out[i]
            power_factor_i = power_factor_time[i]
            photo_pressure_factor_i = photo_pressure_factor_time[i]
            if use_damping_factor_time:
                damping_factor_i = damping_factor_time[i]
            else:
                damping_factor_i = damping_factor_constant
            if use_thermal_velocity_scale_time:
                thermal_velocity_scale_i = thermal_velocity_scale_time[i]
            else:
                thermal_velocity_scale_i = thermal_velocity_scale_constant

            out_of_bounds_count += lookup_force_clamped_pressure(
                r_values,
                z_values,
                Fr_table,
                Fz_ray_table,
                Fz_photo_base_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                photo_pressure_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            vx_i = damping_factor_i * vx_i + thermal_velocity_scale_i * brownian_normals_x[i]
            vy_i = damping_factor_i * vy_i + thermal_velocity_scale_i * brownian_normals_y[i]
            vz_i = damping_factor_i * vz_i + thermal_velocity_scale_i * brownian_normals_z[i]

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            out_of_bounds_count += lookup_force_clamped_pressure(
                r_values,
                z_values,
                Fr_table,
                Fz_ray_table,
                Fz_photo_base_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                photo_pressure_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_out[i + 1] = x_i
            y_out[i + 1] = y_i
            z_out[i + 1] = z_i
            vx_out[i + 1] = vx_i
            vy_out[i + 1] = vy_i
            vz_out[i + 1] = vz_i

    return (
        x_out_arr,
        y_out_arr,
        z_out_arr,
        vx_out_arr,
        vy_out_arr,
        vz_out_arr,
        out_of_bounds_count,
    )


def solve_baoab_3d_lookup_cython_positions_into(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_table,
    DTYPE_t[::1] power_factor_time,
    DTYPE_t[::1] brownian_normals_x,
    DTYPE_t[::1] brownian_normals_y,
    DTYPE_t[::1] brownian_normals_z,
    double dt,
    double mass,
    double weight,
    damping_factor,
    thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    DTYPE_t[::1] x_out,
    DTYPE_t[::1] y_out,
    DTYPE_t[::1] z_out,
    double radial_zero_tolerance=1.0e-30,
):
    cdef Py_ssize_t n = power_factor_time.shape[0]
    cdef Py_ssize_t i
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
    cdef double power_factor_i
    cdef double damping_factor_i
    cdef double thermal_velocity_scale_i
    cdef double damping_factor_constant = 0.0
    cdef double thermal_velocity_scale_constant = 0.0
    cdef bint use_damping_factor_time = False
    cdef bint use_thermal_velocity_scale_time = False
    cdef cnp.ndarray[DTYPE_t, ndim=1] damping_factor_time_arr = np.empty(0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] thermal_velocity_scale_time_arr = np.empty(0, dtype=np.float64)
    cdef DTYPE_t[::1] damping_factor_time = damping_factor_time_arr
    cdef DTYPE_t[::1] thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    if n < 1:
        raise ValueError("power_factor_time must contain at least one sample.")
    if r_values.shape[0] < 2 or z_values.shape[0] < 2:
        raise ValueError("Force lookup table needs at least two grid points per axis.")
    if brownian_normals_x.shape[0] < n - 1 or brownian_normals_y.shape[0] < n - 1 or brownian_normals_z.shape[0] < n - 1:
        raise ValueError("Brownian normal arrays must have len(power_factor_time) - 1 samples.")
    if x_out.shape[0] < n or y_out.shape[0] < n or z_out.shape[0] < n:
        raise ValueError("Output arrays must have len(power_factor_time) samples.")

    if np.ndim(damping_factor) == 0:
        damping_factor_constant = float(damping_factor)
    else:
        use_damping_factor_time = True
        damping_factor_time_arr = np.ascontiguousarray(damping_factor, dtype=np.float64)
        if damping_factor_time_arr.shape[0] < n - 1:
            raise ValueError("damping_factor must be scalar or have len(power_factor_time) - 1 samples.")
        damping_factor_time = damping_factor_time_arr

    if np.ndim(thermal_velocity_scale) == 0:
        thermal_velocity_scale_constant = float(thermal_velocity_scale)
    else:
        use_thermal_velocity_scale_time = True
        thermal_velocity_scale_time_arr = np.ascontiguousarray(thermal_velocity_scale, dtype=np.float64)
        if thermal_velocity_scale_time_arr.shape[0] < n - 1:
            raise ValueError("thermal_velocity_scale must be scalar or have len(power_factor_time) - 1 samples.")
        thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    x_out[0] = x_i
    y_out[0] = y_i
    z_out[0] = z_i

    with nogil:
        for i in range(n - 1):
            power_factor_i = power_factor_time[i]
            if use_damping_factor_time:
                damping_factor_i = damping_factor_time[i]
            else:
                damping_factor_i = damping_factor_constant
            if use_thermal_velocity_scale_time:
                thermal_velocity_scale_i = thermal_velocity_scale_time[i]
            else:
                thermal_velocity_scale_i = thermal_velocity_scale_constant

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            vx_i = damping_factor_i * vx_i + thermal_velocity_scale_i * brownian_normals_x[i]
            vy_i = damping_factor_i * vy_i + thermal_velocity_scale_i * brownian_normals_y[i]
            vz_i = damping_factor_i * vz_i + thermal_velocity_scale_i * brownian_normals_z[i]

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                power_factor_i,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_out[i + 1] = x_i
            y_out[i + 1] = y_i
            z_out[i + 1] = z_i

    return out_of_bounds_count


def solve_baoab_3d_feedback_lookup_cython(
    DTYPE_t[::1] r_values,
    DTYPE_t[::1] z_values,
    DTYPE_t[:, ::1] Fr_table,
    DTYPE_t[:, ::1] Fz_table,
    DTYPE_t[::1] disturbance_power_factor_time,
    DTYPE_t[::1] brownian_normals_x,
    DTYPE_t[::1] brownian_normals_y,
    DTYPE_t[::1] brownian_normals_z,
    DTYPE_t[::1] feedback_position_noise,
    double dt,
    double mass,
    double weight,
    damping_factor,
    thermal_velocity_scale,
    double x0,
    double y0,
    double z0,
    double vx0,
    double vy0,
    double vz0,
    double z_eq,
    double laser_power,
    double feedback_kp,
    double feedback_kd,
    double force_per_watt,
    double feedback_power_min_factor,
    double feedback_power_max_factor,
    double feedback_velocity_filter_alpha,
    Py_ssize_t control_decimation,
    Py_ssize_t loop_delay_baoab_steps,
    double dt_control,
    double radial_zero_tolerance=1.0e-30,
):
    cdef Py_ssize_t n = disturbance_power_factor_time.shape[0]
    cdef Py_ssize_t n_control = 0
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] y_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] z_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vx_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vy_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vz_out_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] command_power_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] actual_power_arr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] control_t_arr
    cdef cnp.ndarray[DTYPE_t, ndim=1] measurement_t_arr
    cdef cnp.ndarray[DTYPE_t, ndim=1] z_measured_arr
    cdef cnp.ndarray[DTYPE_t, ndim=1] z_filtered_arr
    cdef cnp.ndarray[DTYPE_t, ndim=1] requested_power_arr
    cdef DTYPE_t[::1] x_out = x_out_arr
    cdef DTYPE_t[::1] y_out = y_out_arr
    cdef DTYPE_t[::1] z_out = z_out_arr
    cdef DTYPE_t[::1] vx_out = vx_out_arr
    cdef DTYPE_t[::1] vy_out = vy_out_arr
    cdef DTYPE_t[::1] vz_out = vz_out_arr
    cdef DTYPE_t[::1] command_power = command_power_arr
    cdef DTYPE_t[::1] actual_power = actual_power_arr
    cdef DTYPE_t[::1] control_t
    cdef DTYPE_t[::1] measurement_t
    cdef DTYPE_t[::1] z_measured_values
    cdef DTYPE_t[::1] z_filtered_values
    cdef DTYPE_t[::1] requested_power
    cdef Py_ssize_t i
    cdef Py_ssize_t delayed_i
    cdef Py_ssize_t control_count = 0
    cdef long out_of_bounds_count = 0
    cdef int have_filtered = 0
    cdef double x_i
    cdef double y_i
    cdef double z_i
    cdef double vx_i
    cdef double vy_i
    cdef double vz_i
    cdef double Fx_i
    cdef double Fy_i
    cdef double Fz_i
    cdef double command_power_i = laser_power
    cdef double command_factor
    cdef double actual_power_factor
    cdef double z_measured
    cdef double z_filtered = 0.0
    cdef double filtered_z_previous = 0.0
    cdef double measured_vz
    cdef double z_error
    cdef double feedback_force
    cdef double power_change
    cdef double feedback_power_min = feedback_power_min_factor * laser_power
    cdef double feedback_power_max = feedback_power_max_factor * laser_power
    cdef double damping_factor_i
    cdef double thermal_velocity_scale_i
    cdef double damping_factor_constant = 0.0
    cdef double thermal_velocity_scale_constant = 0.0
    cdef bint use_damping_factor_time = False
    cdef bint use_thermal_velocity_scale_time = False
    cdef cnp.ndarray[DTYPE_t, ndim=1] damping_factor_time_arr = np.empty(0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] thermal_velocity_scale_time_arr = np.empty(0, dtype=np.float64)
    cdef DTYPE_t[::1] damping_factor_time = damping_factor_time_arr
    cdef DTYPE_t[::1] thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    if n < 1:
        raise ValueError("disturbance_power_factor_time must contain at least one sample.")
    if r_values.shape[0] < 2 or z_values.shape[0] < 2:
        raise ValueError("Force lookup table needs at least two grid points per axis.")
    if control_decimation < 1:
        raise ValueError("control_decimation must be at least 1.")
    if brownian_normals_x.shape[0] < n - 1 or brownian_normals_y.shape[0] < n - 1 or brownian_normals_z.shape[0] < n - 1:
        raise ValueError("Brownian normal arrays must have len(disturbance_power_factor_time) - 1 samples.")

    if n >= 2:
        n_control = ((n - 2) // control_decimation) + 1
    else:
        n_control = 0

    if feedback_position_noise.shape[0] < n_control:
        raise ValueError("feedback_position_noise must contain one sample per control update.")

    if np.ndim(damping_factor) == 0:
        damping_factor_constant = float(damping_factor)
    else:
        use_damping_factor_time = True
        damping_factor_time_arr = np.ascontiguousarray(damping_factor, dtype=np.float64)
        if damping_factor_time_arr.shape[0] < n - 1:
            raise ValueError("damping_factor must be scalar or have len(disturbance_power_factor_time) - 1 samples.")
        damping_factor_time = damping_factor_time_arr

    if np.ndim(thermal_velocity_scale) == 0:
        thermal_velocity_scale_constant = float(thermal_velocity_scale)
    else:
        use_thermal_velocity_scale_time = True
        thermal_velocity_scale_time_arr = np.ascontiguousarray(thermal_velocity_scale, dtype=np.float64)
        if thermal_velocity_scale_time_arr.shape[0] < n - 1:
            raise ValueError("thermal_velocity_scale must be scalar or have len(disturbance_power_factor_time) - 1 samples.")
        thermal_velocity_scale_time = thermal_velocity_scale_time_arr

    control_t_arr = np.zeros(n_control, dtype=np.float64)
    measurement_t_arr = np.zeros(n_control, dtype=np.float64)
    z_measured_arr = np.zeros(n_control, dtype=np.float64)
    z_filtered_arr = np.zeros(n_control, dtype=np.float64)
    requested_power_arr = np.zeros(n_control, dtype=np.float64)
    control_t = control_t_arr
    measurement_t = measurement_t_arr
    z_measured_values = z_measured_arr
    z_filtered_values = z_filtered_arr
    requested_power = requested_power_arr

    x_out[0] = x0
    y_out[0] = y0
    z_out[0] = z0
    vx_out[0] = vx0
    vy_out[0] = vy0
    vz_out[0] = vz0

    with nogil:
        for i in range(n - 1):
            if i % control_decimation == 0:
                delayed_i = i - loop_delay_baoab_steps
                if delayed_i < 0:
                    delayed_i = 0

                z_measured = z_out[delayed_i] + feedback_position_noise[control_count]

                if have_filtered == 0:
                    z_filtered = z_measured
                    measured_vz = 0.0
                    have_filtered = 1
                else:
                    z_filtered = (
                        feedback_velocity_filter_alpha * z_measured
                        + (1.0 - feedback_velocity_filter_alpha) * filtered_z_previous
                    )
                    measured_vz = (z_filtered - filtered_z_previous) / dt_control

                z_error = z_filtered - z_eq
                feedback_force = -feedback_kp * z_error - feedback_kd * measured_vz
                power_change = feedback_force / force_per_watt
                command_power_i = laser_power + power_change

                if command_power_i < feedback_power_min:
                    command_power_i = feedback_power_min
                elif command_power_i > feedback_power_max:
                    command_power_i = feedback_power_max

                control_t[control_count] = i * dt
                measurement_t[control_count] = delayed_i * dt
                z_measured_values[control_count] = z_measured
                z_filtered_values[control_count] = z_filtered
                requested_power[control_count] = command_power_i
                control_count += 1
                filtered_z_previous = z_filtered

            command_factor = command_power_i / laser_power
            actual_power_factor = command_factor * disturbance_power_factor_time[i]
            if use_damping_factor_time:
                damping_factor_i = damping_factor_time[i]
            else:
                damping_factor_i = damping_factor_constant
            if use_thermal_velocity_scale_time:
                thermal_velocity_scale_i = thermal_velocity_scale_time[i]
            else:
                thermal_velocity_scale_i = thermal_velocity_scale_constant

            if actual_power_factor < feedback_power_min_factor:
                actual_power_factor = feedback_power_min_factor
            elif actual_power_factor > feedback_power_max_factor:
                actual_power_factor = feedback_power_max_factor

            x_i = x_out[i]
            y_i = y_out[i]
            z_i = z_out[i]
            vx_i = vx_out[i]
            vy_i = vy_out[i]
            vz_i = vz_out[i]

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                actual_power_factor,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            vx_i = damping_factor_i * vx_i + thermal_velocity_scale_i * brownian_normals_x[i]
            vy_i = damping_factor_i * vy_i + thermal_velocity_scale_i * brownian_normals_y[i]
            vz_i = damping_factor_i * vz_i + thermal_velocity_scale_i * brownian_normals_z[i]

            x_i += 0.5 * dt * vx_i
            y_i += 0.5 * dt * vy_i
            z_i += 0.5 * dt * vz_i

            out_of_bounds_count += lookup_force_clamped(
                r_values,
                z_values,
                Fr_table,
                Fz_table,
                x_i,
                y_i,
                z_i,
                actual_power_factor,
                radial_zero_tolerance,
                &Fx_i,
                &Fy_i,
                &Fz_i,
            )
            Fz_i -= weight
            vx_i += 0.5 * dt * Fx_i / mass
            vy_i += 0.5 * dt * Fy_i / mass
            vz_i += 0.5 * dt * Fz_i / mass

            x_out[i + 1] = x_i
            y_out[i + 1] = y_i
            z_out[i + 1] = z_i
            vx_out[i + 1] = vx_i
            vy_out[i + 1] = vy_i
            vz_out[i + 1] = vz_i
            command_power[i] = command_power_i
            actual_power[i] = laser_power * actual_power_factor

    if n >= 2:
        command_power[n - 1] = command_power_i
        actual_power[n - 1] = actual_power[n - 2]
    elif n == 1:
        command_power[0] = command_power_i
        actual_power[0] = laser_power

    return (
        x_out_arr,
        y_out_arr,
        z_out_arr,
        vx_out_arr,
        vy_out_arr,
        vz_out_arr,
        command_power_arr,
        actual_power_arr,
        control_t_arr,
        measurement_t_arr,
        z_measured_arr,
        z_filtered_arr,
        requested_power_arr,
        out_of_bounds_count,
    )
