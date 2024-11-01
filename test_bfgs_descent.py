import sys
import numpy as np
import taichi as ti

sys.path.append("../../maintained-wip/taichi-polyclip")
import ticlip
import tater
import matplotlib.pyplot as plt
import os

ti.init(
    arch=ti.cpu,
    cfg_optimization=False,
    opt_level=1,
    fast_math=False,
    advanced_optimization=False,
)

t = np.linspace(0, 1, 30)
lc_true = np.sin(10 * t) / 2 + 1

svi = np.tile(np.array([1.0, 0.0, 0.0]), (len(t), 1))
ovi = np.tile(np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]), (len(t), 1))
# ovi = np.array([np.cos(t), np.sin(t), 0.0*t]).T

obj_path = "/Users/liamrobinson/Documents/maintained-research/mirage-models/Non-Convex/irregular.obj"

ticlip.load_finfo(obj_path)
imat = ticlip.gen_cache(500, 500)

print(imat.to_numpy().sum())
# endd

x0s = tater.initialize(
    obj_path,
    svi,
    ovi,
    lc_true,
    max_angular_velocity_magnitude=0.2,
    n_particles=1,
    dimx=6,
)


from tater.core import (
    load_unshadowed_areas,
    compute_lc,
)
from ticlip.handler import unshadowed_areas_handle

load_unshadowed_areas(unshadowed_areas_handle())

x0 = ti.Vector([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dt=ti.float32)


@ti.kernel
def test():
    print(compute_lc(x0))


test()

endd


# res = tater.direct_invert_attitude(
#     svi,
#     ovi,
#     lc_true,
#     n_particles=int(1e4),
#     max_angular_velocity_magnitude=0.3,
#     obj_file_path=obj_path,
#     remove_nans=True,
# )

print(res)

fvals = res["fun"].to_numpy()

res.write_parquet("saved.parquet")

ti.profiler.print_scoped_profiler_info()

print(res["xk"].to_numpy()[:10, :])

plt.plot(res["lcs"].to_numpy()[:10, :].T)
plt.plot(lc_true, linewidth=3)
plt.show()

os.system("cp saved.parquet ../../maintained-research/mirage")
