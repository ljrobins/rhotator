import sys
import numpy as np
import taichi as ti
import tater
import matplotlib.pyplot as plt
import polars as pl

t = np.linspace(0, 10, 30)
lc_true = np.sin(t) / 2 + 1


svi = np.tile(np.array([1.0, 0.0, 0.0]), (len(t), 1))
ovi = np.tile(np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]), (len(t), 1))
obj_path = "/Users/liamrobinson/Downloads/irregular.obj"

res = tater.direct_invert_attitude(
    t,
    svi,
    ovi,
    lc_true,
    n_particles=int(1e5),
    max_angular_velocity_magnitude=0.3,
    obj_file_path=obj_path,
    remove_nans=True,
)

print(res)

fvals = res["fun"].to_numpy()

res.write_parquet("saved.parquet")

ti.profiler.print_scoped_profiler_info()

print(res['xk'].to_numpy()[:10,:])

plt.plot(res["lcs"].to_numpy()[:10, :].T)
plt.plot(lc_true, linewidth=3)
plt.show()
