import sys
import numpy as np

sys.path.append("../tibfgs")

import taichi as ti
import tater
import tibfgs

t = np.linspace(0, 10, 30)
lc_true = np.sin(t) + 1
x0 = np.random.rand(1, 6)

svi = np.array([1.0, 0.0, 0.0])
ovi = np.array([1.0, 1.0, 0.0])
obj_path = "/Users/liamrobinson/Documents/mirage/mirage/resources/models/cube.obj"

tater.initialize(obj_path, svi, ovi, lc_true, x0)
x0 = tater.generate_initial_states(int(1e3))

res = tibfgs.minimize(tater.core.propagate_one, x0)

print(res.sort("fun"))


fvals = res["fun"].to_numpy()
print(np.sum(np.isnan(fvals)))
print(np.min(fvals[~np.isnan(fvals)]))

res.write_parquet("saved.parquet")

ti.profiler.print_scoped_profiler_info()
