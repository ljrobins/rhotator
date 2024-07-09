
import time
import os
import sys
import numpy as np

import taichi as ti
ti.init(arch=ti.cpu,
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

import sys
sys.path.append('..')
NPART = 1
NTIMES = 20
NSTATES = 6
sys.path.append('../tibfgs')
os.environ['TI_DIM_X'] = str(NSTATES)
os.environ['TI_NUM_PARTICLES'] = str(NPART)
os.environ['TI_NUM_TIMES'] = str(NTIMES)

import tibfgs
import tater

tspace = np.linspace(0, 1.0, NTIMES, dtype=np.float32)
lc_true = np.sin(tspace * 2) + 0.3
tater.load_lc_true(lc_true)

@ti.kernel
def run() -> tater.VLTYPE:
    lc = tater.VLTYPE(0.0)
    for i in range(1):
        x0 = ti.Vector([-0.8835819, -0.42102355, -0.064743236, -1.5945487, 0.07931882, 0.78442633])
        lc = tater.compute_lc(x0)
        loss = tater.compute_loss(lc)
    return lc
lc = run().to_numpy()

import matplotlib.pyplot as plt

plt.plot(lc / np.linalg.norm(lc))
plt.plot(lc_true / np.linalg.norm(lc_true))
plt.show()