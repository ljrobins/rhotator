import time
import os
import sys
import numpy as np

np.finfo(np.float32)
import matplotlib.pyplot as plt

import taichi as ti

ti.init(
    arch=ti.metal,
    default_fp=ti.float32,
    num_compile_threads=4,
)


NPART = int(1e5)
NTIMES = 32
NSTATES = 8
sys.path.append("../tibfgs")
os.environ["TI_DIM_X"] = str(NSTATES)
os.environ["TI_NUM_PARTICLES"] = str(NPART)
os.environ["TI_NUM_TIMES"] = str(NTIMES)
import tibfgs

import tater

MAX_ANG_VEL_MAG = 3.0

# tspace = np.linspace(0, 6.0, NTIMES, dtype=np.float32)
# noise = np.random.randn(NTIMES) / 5
# lct = np.clip(np.sin(tspace * 2) + 1.0, 0, np.inf).astype(np.float32)


@ti.kernel
def get_lc_true() -> tater.VLTYPE:
    x0 = ti.Vector([1.0, 2.0, 3.0, 1.0, 4.0, 0.0, 1.0, 2.0])
    lc = tater.compute_lc(x0)
    return lc


lct = get_lc_true().to_numpy()
tater.load_lc_true(lct)
plt.plot(lct)
plt.show()

# %%
# Initializing the quaternion and angular velocity guesses


tibfgs.set_f(tater.propagate_one)

res = ti.types.struct(
    x0=tibfgs.VTYPE,
    fun=ti.f32,
    jac=tibfgs.VTYPE,
    hess_inv=tibfgs.MTYPE,
    status=ti.u8,
    xk=tibfgs.VTYPE,
    k=ti.u32,
)

res_field = ti.Struct.field(
    dict(
        x0=tibfgs.VTYPE,
        fun=ti.f32,
        jac=tibfgs.VTYPE,
        hess_inv=tibfgs.MTYPE,
        status=ti.u8,
        xk=tibfgs.VTYPE,
        k=ti.u32,
    ),
    shape=(tibfgs.NPART,),
)


@ti.kernel
def run() -> int:
    for i in range(NPART):
        x0 = tater.gen_x0(MAX_ANG_VEL_MAG)
        fval, gfk, Hk, warnflag, xk, k = tibfgs.minimize_bfgs(
            i=i, x0=x0, gtol=1e-3, eps=1e-6, maxiter=100
        )
        res_field[i] = res(
            x0=x0, fun=fval, jac=gfk, hess_inv=Hk, status=warnflag, xk=xk, k=k
        )
    return 0


t1 = time.time()
run()
print(time.time() - t1)

r = res_field.to_numpy()

print(np.sum(np.isnan(r["fun"])))
print(np.min(r["fun"][~np.isnan(r["fun"])]))

np.savez("saved.npz", **r)

ti.profiler.print_scoped_profiler_info()
