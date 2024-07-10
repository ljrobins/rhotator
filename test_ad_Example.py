import taichi as ti

ti.init(arch=ti.gpu)

N = 8
dt = 1e-5

x0 = ti.Vector.field(6, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy

import sys, os
NPART = int(1e5)
NTIMES = 25
NSTATES = 6
sys.path.append("../tibfgs")
os.environ["TI_DIM_X"] = str(NSTATES)
os.environ["TI_NUM_PARTICLES"] = str(NPART)
os.environ["TI_NUM_TIMES"] = str(NTIMES)

import tater

itensor = ti.math.vec3([1.0, 2.0, 3.0])
h = 0.01

lc = ti.field(dtype=ti.f32, shape=NTIMES)

@ti.kernel
def iter():
    ti.loop_config(serialize=True)
    for j in range(lc.shape[0]):
        dcm = tater.mrp_to_dcm(x0[0][:3])
        lc[j] = tater.normalized_convex_light_curve(dcm @ tater.L, dcm @ tater.O)
        x0[0] = tater.rk4_step(x0[0], h, itensor)

@ti.kernel
def compute_lc():
    for i in lc:
        U[None] += lc[i]

def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        iter()
        ti.sync()
        compute_lc()  # The tape will automatically compute dU/dx and save the results in x.grad
    ti.sync()

@ti.kernel
def init():
    x0[0] = [ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random()]


init()
print(x0[0])

substep()
print(lc)

print(x0[0])
print(x0.grad[0])