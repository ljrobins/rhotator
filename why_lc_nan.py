import os
import sys
import numpy as np

import taichi as ti

ti.init(
    arch=ti.cpu,
    default_fp=ti.float32,
    cfg_optimization=False,
    offline_cache=True,
    opt_level=2,
    check_out_of_bound=False,
    fast_math=True,
    simplify_before_lower_access=False,
    lower_access=False,
    simplify_after_lower_access=False,
    advanced_optimization=False,
    num_compile_threads=32,
    make_block_local=True,
)


NTIMES = 20
NSTATES = 6


sys.path.append("..")
NPART = 1
NTIMES = 25
NSTATES = 6
sys.path.append("../tibfgs")
os.environ["TI_DIM_X"] = str(NSTATES)
os.environ["TI_NUM_PARTICLES"] = str(NPART)
os.environ["TI_NUM_TIMES"] = str(NTIMES)


import tater

tspace = np.linspace(0, 6.0, NTIMES, dtype=np.float32)
lc_clean = np.sin(tspace * 2) + 1.0
lc_true = np.clip(lc_clean + np.random.randn(NTIMES) / 5, 0, np.inf)
tater.load_lc_true(lc_true)


@ti.kernel
def run() -> tater.VLTYPE:
    x0 = ti.Vector(
        [-0.42594984, -0.5526155, -0.070913285, 0.3925352, 4.002883, -1.5416336]
    )
    lc = tater.compute_lc(x0)
    loss = tater.compute_loss(lc)
    return lc


lc = run().to_numpy()

import matplotlib.pyplot as plt

plt.plot(tspace, lc)
plt.plot(tspace, lc_true)
plt.plot(tspace, lc_clean)
plt.show()
