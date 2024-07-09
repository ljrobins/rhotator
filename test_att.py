import taichi as ti
import os, sys

ti.init(arch=ti.cpu)

NPART = int(1e5)
NTIMES = 25
NSTATES = 6
sys.path.append('../tibfgs')
os.environ['TI_DIM_X'] = str(NSTATES)
os.environ['TI_NUM_PARTICLES'] = str(NPART)
os.environ['TI_NUM_TIMES'] = str(NTIMES)
import tibfgs

import tater
import numpy as np

NSTEP = 20

@ti.kernel
def test():
    s = ti.math.vec3([1.0, 2.0, 3.0])
    print(tater.mrp_to_dcm(s))

itensor = ti.Vector([1.0, 2.0, 3.0])
h = 0.1

@ti.kernel
def prop_att(x0: ti.types.vector(n=6, dtype=ti.f32)):
    current_state = x0
    for j in range(NSTEP):
        # print(current_state)
        dcm = tater.mrp_to_dcm(current_state[:3])
        dcm2 = tater.quat_to_dcm(tater.mrp_to_quat(current_state[:3]))
        print(dcm)
        print(dcm2)
        ell = dcm @ ti.math.vec3([1.0, 0.0, 0.0])
        oh = dcm @ ti.math.vec3([1.0, 1.0, 0.0]).normalized()
        v = tater.normalized_convex_light_curve(ell, oh)
        print(ell, oh, v)
        current_state = tater.rk4_step(current_state, h, itensor)
prop_att(ti.Vector([1.0, 2.0, 3.0, 1.0, 1.0, 1.0]))