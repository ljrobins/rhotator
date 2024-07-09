import sys, os

import taichi as ti
ti.init(arch=ti.metal,
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

NPART = int(500_000)
NTIMES = 20
NSTATES = 6
sys.path.append('../tibfgs')
os.environ['TI_DIM_X'] = str(NSTATES)
os.environ['TI_NUM_PARTICLES'] = str(NPART)
os.environ['TI_NUM_TIMES'] = str(NTIMES)

import tater

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def poly():
    v = x[None]
    y[None] = v**2 - 2*v + 3

with ti.ad.Tape(y):
    poly()

print(x.grad)
