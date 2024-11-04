import sys
import numpy as np
import taichi as ti

sys.path.append("../../maintained-wip/taichi-polyclip")
import ticlip
import tater
import matplotlib.pyplot as plt
import os

np.random.seed(2)

t = np.linspace(0, 1, 30)
lc_mean = np.sin(10 * t) / 2 + 1
sigma_obs = 0.1 * lc_mean
lc_obs = lc_mean + np.random.normal(loc=np.zeros_like(t), scale=sigma_obs)

svi = np.tile(np.array([1.0, 0.0, 0.0]), (len(t), 1))
# ovi = np.tile(np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]), (len(t), 1))
ovi = np.array([np.cos(t), np.sin(t), 0.0 * t]).T

# plt.plot(t, lc_mean)
# plt.scatter(t, lc_obs)
# plt.plot(t, sigma_obs)
# plt.fill_between(t, lc_mean - 2 * sigma_obs, lc_mean + 2 * sigma_obs, alpha=0.2, label='True uncertainty $\pm 2 \sigma$')
# plt.show()
# endd

obj_name = "irregular"
obj_path = f"/Users/liamrobinson/Documents/maintained-research/mirage-models/Non-Convex/{obj_name}.obj"

ti.init(
    arch=ti.cpu,
    cfg_optimization=False,
    opt_level=1,
    fast_math=False,
    advanced_optimization=False,
)

for loss_function in ["l2"]:
    res = tater.direct_invert_attitude(
        svi,
        ovi,
        lc_obs,
        n_particles=int(1e4),
        max_angular_velocity_magnitude=4.0,
        obj_file_path=obj_path,
        self_shadowing=True,
        loss_function=loss_function,
        sigma_obs=sigma_obs,
    )

    save_name = f"{obj_name}-{loss_function}.parquet"
    res.write_parquet(save_name)
    ti.profiler.print_scoped_profiler_info()

    print(res)

    os.system("cp saved.parquet ../../maintained-research/mirage")
