import sys
import numpy as np

sys.path.append("../tibfgs")

import taichi as ti
import tater
import tibfgs
import time
import matplotlib.pyplot as plt
import polars as pl

t = np.linspace(0, 10, 30)
lc_true = np.sin(t) / 2 + 1

svi = np.array([1.0, 0.0, 0.0])
ovi = np.array([1.0, 1.0, 0.0])
obj_path = "/Users/liamrobinson/Documents/mirage/mirage/resources/models/cube.obj"

x0 = tater.initialize(
    obj_path,
    svi,
    ovi,
    lc_true,
    max_angular_velocity_magnitude=3.0,
    n_particles=int(1e5),
)

t1 = time.time()
res = tibfgs.minimize(tater.core.propagate_one, x0)
print(time.time() - t1)

res = res.sort("fun").filter(pl.col("fun").is_not_nan())

print(res)

fvals = res["fun"].to_numpy()
print(np.sum(np.isnan(fvals)))
print(np.min(fvals[~np.isnan(fvals)]))

res.write_parquet("saved.parquet")

ti.profiler.print_scoped_profiler_info()

x0s = res["xk"].to_numpy()
lcs = tater.compute_light_curves(x0s)

plt.plot(lcs[:3, :].T)
plt.plot(lc_true, linewidth=3)
plt.show()
