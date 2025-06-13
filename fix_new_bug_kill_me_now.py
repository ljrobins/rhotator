import numpy as np
import tater
import sys
import matplotlib.pyplot as plt

sys.path.append("/Users/liamrobinson/Documents/maintained-wip/taichi-polyclip")

import taichi as ti
from alive_progress import alive_bar

ti.init(
    arch=ti.cpu,
    cfg_optimization=False,
    opt_level=1,
    fast_math=False,
    advanced_optimization=False,
    default_fp=ti.f64,
)

n_times = 50

np.random.seed(1)
svi = np.array([[-0.96, -0.27, -0.12]])
svi /= np.linalg.norm(svi)
svi = np.tile(svi, (n_times, 1))
ovi = np.array([[0.71, -0.71, 0.03]])
ovi /= np.linalg.norm(ovi)
ovi = np.tile(ovi, (n_times, 1))
x0 = np.array([[-0.33, -0.33, -0.33, 0.1, 0.1, 0.1]])
x0 = np.tile(x0, (10, 1))
x0[:,0] += np.random.rand(x0.shape[0])/100
invert_obj_path = "/Users/liamrobinson/Documents/maintained-research/mirage-models/accurate_sats/box_wing_antenna_quad.obj"
iratios = np.array([1.0, 0.34, 1.03])
pixel_shadow_val = 7.053493
convex_val = 7.883861923637019
self_shadowing = True

epsecs = np.linspace(0, 1, n_times)

tater.initialize(
    obj_path=invert_obj_path,
    svi=svi,
    ovi=ovi,
    epsecs=epsecs,
    lc_true=epsecs,
    x0=x0,
    loss_function="log-likelihood",
    sigma_obs=0.5,
    cache_size=200,
    self_shadowing=self_shadowing,
    iratios=iratios,
)

eps = 1e-4

res = tater.compute_light_curves(
    x0s=x0,
    vary_itensor=False,
    vary_global_mats=False,
    substeps=1,
    cs_scale=0.0,
    n_scale=0.0,
    i_scale=0.0,
    return_g=True,
    compute_material_sensitivity=True,
    material_eps=eps,
)
crap = res['lcs'].flatten()[0]

# print(crap)

# plt.figure()
# plt.imshow(H_x_theta[0])
# plt.figure()
# plt.imshow(H_x_x[0])
# plt.figure()
plt.imshow(res['dxhat_dtheta'][0])
plt.show()
