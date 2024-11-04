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

np.random.seed(2)

t = np.linspace(0, 1, 30)
lc_true = np.sin(10 * t) / 2 + 1 + np.random.normal(loc=np.zeros_like(t), scale=0.3)

svi = np.tile(np.array([1.0, 0.0, 0.0]), (len(t), 1))
ovi = np.tile(np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]), (len(t), 1))
# ovi = np.array([np.cos(t), np.sin(t), 0.0*t]).T

obj_path = "/Users/liamrobinson/Documents/maintained-research/mirage-models/Non-Convex/irregular.obj"

x0s = tater.initialize(
    obj_path,
    svi,
    ovi,
    lc_true,
    max_angular_velocity_magnitude=0.2,
    n_particles=1,
    dimx=6,
    self_shadowing=True,
    loss_function="log-likelihood",
    sigma_obs=0.5,
)

from tater.core import compute_lc, compute_loss, sigma_obs

x0 = ti.Vector([0.0, 0.0, 0.0, -3.0, 3.0, 1.0], dt=ti.float32)


@ti.kernel
def test():
    lc = compute_lc(x0)
    print(lc)
    print(compute_loss(lc))


test()
