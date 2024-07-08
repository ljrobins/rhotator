import taichi as ti
import tibfgs
import os
import numpy as np


NSTATES = int(os.environ['TI_DIM_X'])
NTIMES = int(os.environ['TI_NUM_TIMES'])

VSTYPE = ti.types.vector(n=NSTATES, dtype=ti.f32)
VLTYPE = ti.types.vector(n=NTIMES, dtype=ti.f32)


L = ti.Vector([1.0, 0.0, 0.0]).normalized()
O = ti.Vector([1.0, 1.0, 0.0]).normalized()
itensor = ti.Vector([1.0, 2.0, 3.0])

h = 1 / NTIMES

lc_true = None
def load_lc_true(lc_arr: np.ndarray):
    global lc_true
    lc_true = VLTYPE(lc_arr)

def load_obj(obj_path: str):
    import trimesh
    obj = trimesh.load(obj_path)
    v1 = obj.vertices[obj.faces[:,0]]
    v2 = obj.vertices[obj.faces[:,1]]
    v3 = obj.vertices[obj.faces[:,2]]
    fnn = np.cross(v2 - v1, v3 - v1)
    fan = np.linalg.norm(fnn, axis=1, keepdims=True) / 2
    fnn = fnn / fan / 2

    fa = ti.field(dtype=ti.f32, shape=fan.shape)
    fa.from_numpy(fan.astype(np.float32))

    fn = ti.Vector.field(n=3, dtype=ti.f32, shape=fnn.shape[0])
    fn.from_numpy(fnn.astype(np.float32))

    fd = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fd.fill(0.5)
    fs = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fs.fill(0.5)
    fp = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fp.fill(3)

    return fa, fn, fd, fs, fp


@ti.func
def rand_b2(r: float) -> ti.math.vec3:
    return r * rand_s2() * ti.random() ** (1 / 3)

@ti.func
def rand_s2() -> ti.math.vec3:
    return ti.math.vec3(ti.randn(), ti.randn(), ti.randn()).normalized()


@ti.func
def rand_s3() -> ti.math.vec4:
    q = ti.math.vec4(ti.randn(), ti.randn(), ti.randn(), ti.randn()).normalized()
    if q[3] < 0:
        q = -q
    return q

@ti.func
def mrp_to_quat(s: ti.math.vec3) -> ti.math.vec4:
    s2 = s.norm_sqr()
    v = 2 * s / (1 + s2)
    return ti.math.vec4([v.x, v.y, v.z, (1 - s2) / (1 + s2)])

@ti.func
def quat_to_mrp(q: ti.math.vec4) -> ti.math.vec3:
    return q[:3] / (q[3] + 1)

@ti.func
def quat_to_dcm(q: ti.math.vec4) -> ti.math.mat3:
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    C = ti.math.mat3(
        1 - 2 * q1**2 - 2 * q2**2, 2 * (q0 * q1 + q2 * q3), 2 * (q0 * q2 - q1 * q3), 
        2 * (q0 * q1 - q2 * q3), 1 - 2 * q0**2 - 2 * q2**2, 2 * (q1 * q2 + q0 * q3), 
        2 * (q0 * q2 + q1 * q3), 2 * (q1 * q2 - q0 * q3), 1 - 2 * q0**2 - 2 * q1**2
    ).transpose()
    return C

@ti.func
def normalized_convex_light_curve(L: ti.math.vec3, O: ti.math.vec3) -> float:
    b = 0.0
    for i in range(fn.shape[0]):
        fr = brdf_phong(L, O, fn[i], fd[i], fs[i], fp[i])
        ti.atomic_add(b, fr * rdot(fn[i], O) * rdot(fn[i], L))
    return b

@ti.func
def rdot(v1, v2) -> float:
    return ti.max(ti.math.dot(v1,v2), 0.0)

@ti.func
def brdf_phong(
    L: ti.math.vec3,
    O: ti.math.vec3,
    N: ti.math.vec3,
    cd: float,
    cs: float,
    n: float,
) -> float:
    NdL = ti.math.dot(N, L)
    if NdL <= 1e-5:
        NdL = np.inf
    R = (2 * ti.math.dot(N, L) * N - L).normalized()
    fd = cd / np.pi
    fs = cs * (n + 2) / (2 * np.pi) * rdot(R, O) ** n / NdL
    return fd + fs

@ti.func
def compute_lc(x0: VSTYPE) -> VLTYPE:
    current_state = x0
    lc = VLTYPE(0.0)
    for j in range(lc.n):
        q = mrp_to_quat(current_state[:3])
        dcm = quat_to_dcm(q)
        lc[j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
        
        q, current_state[3:] = rk4_step(q, current_state[3:], h, itensor)
        q = q.normalized()
        current_state[:3] = quat_to_mrp(q)

    return lc

@ti.func
def rk4_step(q0, w0, h, itensor):
    k1q, k1w = deriv(q0, w0, itensor)
    k2q, k2w = deriv(q0 + h * k1q / 2, w0 + h * k1w / 2, itensor)
    k3q, k3w = deriv(q0 + h * k2q / 2, w0 + h * k2w / 2, itensor)
    k4q, k4w = deriv(q0 + h * k3q, w0 + h * k3w, itensor)
    return (
        q0 + h * (k1q + 2 * k2q + 2 * k3q + k4q) / 6, 
        w0 + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6)

@ti.func
def deriv(q, w, itensor):
    qmat = 0.5 * ti.math.mat4(
            q[3], -q[2], q[1], q[0],
            q[2], q[3], -q[0], q[1],
            -q[1], q[0], q[3], q[2],
            -q[0], -q[1], -q[2], q[3],
    )
    qdot = qmat @ ti.math.vec4(w.xyz, 0.0)

    wdot = ti.math.vec3(0.0, 0.0, 0.0)
    wdot.x = -1 / itensor[0] * (itensor[2] - itensor[1]) * w.y * w.z
    wdot.y = -1 / itensor[1] * (itensor[0] - itensor[2]) * w.z * w.x
    wdot.z = -1 / itensor[2] * (itensor[1] - itensor[0]) * w.x * w.y

    return qdot, wdot

@ti.func
def compute_loss(lc: VLTYPE) -> ti.f32:
    # dp = ti.math.dot(lc.normalized(), lc_true.normalized())
    # dpc = tibfgs.clip(dp, -1.0, 1.0)
    # return ti.acos(dp)
    return (lc-lc_true).norm()

@ti.func
def propagate_one(x0: VSTYPE) -> ti.f32:
    lc = compute_lc(x0)
    loss = compute_loss(lc)
    return loss

@ti.func
def gen_x0(w_max) -> VSTYPE:
    x0 = VSTYPE(0.0)
    x0[:3] = quat_to_mrp(rand_s3())
    x0[3:] = rand_b2(w_max)
    return x0

fa, fn, fd, fs, fp = load_obj('/Users/liamrobinson/Documents/mirage/mirage/resources/models/cube.obj')
