import numpy as np


output_path = "Summmer\Project/glmt_force_table.npz"

x_grid = np.linspace(-80e-6, 80e-6, 41)
y_grid = np.linspace(-80e-6, 80e-6, 41)
z_grid = np.linspace(-2e-3, 4e-3, 101)

shape = (len(x_grid), len(y_grid), len(z_grid))

# Replace these arrays with GLMT-computed total optical forces.
# Units must be newtons.
Fx = np.zeros(shape)
Fy = np.zeros(shape)
Fz = np.zeros(shape)

np.savez(
    output_path,
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    Fx=Fx,
    Fy=Fy,
    Fz=Fz,
)

print("Wrote", output_path)
print("Fx, Fy, Fz currently contain zeros. Replace them with GLMT forces.")
