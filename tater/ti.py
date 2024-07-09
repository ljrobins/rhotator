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

def unique_areas_and_normals(fn: np.ndarray, fa: np.ndarray):
    """Finds groups of unique normals and areas to save rows elsewhere"""
    (
        unique_normals,
        all_to_unique,
        unique_to_all,
    ) = np.unique(np.round(fn, decimals=6), axis=0, return_index=True, return_inverse=True)
    unique_areas = np.zeros((unique_normals.shape[0]))
    np.add.at(unique_areas, unique_to_all, fa)
    return unique_normals, unique_areas

def load_obj(obj_path: str):
    import trimesh
    obj = trimesh.load(obj_path)
    v1 = obj.vertices[obj.faces[:,0]]
    v2 = obj.vertices[obj.faces[:,1]]
    v3 = obj.vertices[obj.faces[:,2]]
    fnn = np.cross(v2 - v1, v3 - v1)
    fan = np.linalg.norm(fnn, axis=1, keepdims=True) / 2
    fnn = fnn / fan / 2

    fnn, fan = unique_areas_and_normals(fnn, fan.flatten())

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
        dcm = mrp_to_dcm(current_state[:3])
        lc[j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
        current_state = rk4_step(current_state, h, itensor)
    return lc

@ti.func
def rk4_step(x0, h, itensor):
    k1 = state_derivative_mrp(x0, itensor)
    k2 = state_derivative_mrp(x0 + h * k1 / 2, itensor)
    k3 = state_derivative_mrp(x0 + h * k2 / 2, itensor)
    k4 = state_derivative_mrp(x0 + h * k3, itensor)
    return (
        x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

@ti.func
def mrp_kde(s: ti.math.vec3, w: ti.math.vec3) -> ti.math.vec3:
    s1, s2, s3 = s
    s12 = s1**2
    s22 = s2**2
    s32 = s3**2

    m = 1/4 * ti.math.mat3(
        [1+s12-s22-s32, 2*(s1*s2-s3),  2*(s1*s3+s2)],
        [2*(s1*s2+s3),  1-s12+s22-s32, 2*(s2*s3-s1)],
        [2*(s1*s3-s2), 2*(s2*s3+s1),  1-s12-s22+s32]
    )
    return m @ w

@ti.func
def tilde(v: ti.math.vec3) -> ti.math.mat3:
    v_tilde = ti.math.mat3(
        [0.0, -v.z, v.y],
        [v.z, 0.0, -v.x],
        [-v.y, v.x, 0.0]
    )
    return v_tilde

@ti.func
def mrp_to_dcm(s: ti.math.vec3) -> ti.math.mat3:
    s1, s2, s3 = s
    s12 = s1**2
    s22 = s2**2
    s32 = s3**2
    s2 = s12 + s22 + s32
    
    s_tilde = tilde(s)
    eye = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c = eye + (8 * s_tilde @ s_tilde - 4 * (1 - s2)) / (1 + s2) ** 2
    return c

@ti.func
def quat_kde(q: ti.math.vec4, w: ti.math.vec3) -> ti.math.vec4:
    qmat = 0.5 * ti.math.mat4(
            q[3], -q[2], q[1], q[0],
            q[2], q[3], -q[0], q[1],
            -q[1], q[0], q[3], q[2],
            -q[0], -q[1], -q[2], q[3],
    )
    qdot = qmat @ ti.math.vec4(w.xyz, 0.0)
    return qdot

@ti.func
def omega_dot(w: ti.math.vec3, itensor: ti.math.vec3) -> ti.math.vec3:
    wdot = ti.math.vec3(0.0)
    wdot.x = -1 / itensor[0] * (itensor[2] - itensor[1]) * w.y * w.z
    wdot.y = -1 / itensor[1] * (itensor[0] - itensor[2]) * w.z * w.x
    wdot.z = -1 / itensor[2] * (itensor[1] - itensor[0]) * w.x * w.y
    return wdot

@ti.func
def state_derivative_mrp(x, itensor):
    xdot = 0 * x
    s = x[:3]
    w = x[3:6]
    xdot[:3] = mrp_kde(s, w)
    xdot[3:6] = omega_dot(w, itensor)
    return xdot

@ti.func
def state_derivative_quat(x, itensor):
    q = x[:4]
    w = x[4:7]
    qdot = quat_kde(q, w)
    wdot = omega_dot(w, itensor)
    return qdot, wdot

@ti.func
def compute_loss(lc: VLTYPE) -> ti.f32:
    # lcn = lc.normalized()
    # dp = ti.math.dot(lcn, lc_true.normalized())
    # dpc = tibfgs.clip(dp, -1.0, 1.0)
    # loss = ti.acos(dp)
    # return loss
    return (ti.math.log(lc+1.0)-ti.math.log(lc_true+1)).norm_sqr()

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
