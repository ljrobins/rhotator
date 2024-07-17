import taichi as ti
import os
import numpy as np
import trimesh

NSTATES = int(os.environ["TI_DIM_X"])
NTIMES = int(os.environ["TI_NUM_TIMES"])

h: ti.f32 = 1.0 / NTIMES
itensor = ti.Vector([1.0, 2.0, 3.0])


@ti.func
def rand_b2(r: float) -> ti.math.vec3:
    return r * rand_s2() * ti.random() ** (1 / 3)


@ti.func
def rand_s2() -> ti.math.vec3:
    return ti.math.vec3(ti.randn(), ti.randn(), ti.randn()).normalized()


@ti.func
def rand_s3() -> ti.math.vec4:
    q = ti.math.vec4(ti.randn(), ti.randn(), ti.randn(), ti.randn()).normalized()
    return q


@ti.func
def quat_to_mrp(q: ti.math.vec4) -> ti.math.vec3:
    return q[:3] / (q[3] + 1)


@ti.func
def normalized_convex_light_curve(L: ti.math.vec3, O: ti.math.vec3) -> float:
    b = 0.0
    for i in range(fn.shape[0]):
        fr = brdf_phong(L, O, fn[i], fd[i], fs[i], fp[i])
        ti.atomic_add(b, fa[i] * fr * rdot(fn[i], O) * rdot(fn[i], L))
    return b


@ti.func
def rdot(v1, v2) -> float:
    return ti.max(ti.math.dot(v1, v2), 0.0)


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
    fr: ti.f32 = ti.math.nan
    if NdL <= 1e-5:
        fr = 0.0
    if ti.math.isnan(fr):
        R = (2 * NdL * N - L).normalized()
        fd = cd / ti.math.pi
        fs = cs * (n + 2) / (2 * ti.math.pi) * rdot(R, O) ** n / NdL
        fr = fd + fs
    return fr


@ti.func
def compute_lc(x0: ti.template()) -> ti.template():
    current_state = x0
    itens = itensor
    if ti.static(x0.n > 6):
        itens.y = ti.abs(x0[6])
    if ti.static(x0.n > 7):
        itens.z = ti.abs(x0[7])

    lc = ti.types.vector(n=NTIMES, dtype=ti.f32)(0.0)
    ti.loop_config(serialize=True)
    for j in range(lc.n):
        dcm = mrp_to_dcm(current_state[:3])
        lc[j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
        current_state = rk4_step(current_state, h, itens)
    return lc


@ti.func
def rk4_step(x0, h, itensor):
    k1 = h * state_derivative_mrp(x0, itensor)
    k2 = h * state_derivative_mrp(x0 + k1 / 2, itensor)
    k3 = h * state_derivative_mrp(x0 + k2 / 2, itensor)
    k4 = h * state_derivative_mrp(x0 + k3, itensor)
    return x0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


@ti.func
def mrp_kde(s: ti.math.vec3, w: ti.math.vec3) -> ti.math.vec3:
    s1, s2, s3 = s.x, s.y, s.z
    s12 = s1**2
    s22 = s2**2
    s32 = s3**2

    m = (
        1
        / 4
        * ti.math.mat3(
            [1 + s12 - s22 - s32, 2 * (s1 * s2 - s3), 2 * (s1 * s3 + s2)],
            [2 * (s1 * s2 + s3), 1 - s12 + s22 - s32, 2 * (s2 * s3 - s1)],
            [2 * (s1 * s3 - s2), 2 * (s2 * s3 + s1), 1 - s12 - s22 + s32],
        )
    )
    return m @ w


@ti.func
def tilde(v: ti.math.vec3) -> ti.math.mat3:
    v_tilde = ti.math.mat3([[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]])
    return v_tilde


@ti.func
def mrp_to_dcm(s: ti.math.vec3) -> ti.math.mat3:
    s1, s2, s3 = s.x, s.y, s.z
    s12 = s1**2
    s22 = s2**2
    s32 = s3**2
    s2 = s12 + s22 + s32

    s_tilde = tilde(s)
    eye = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c = eye + (8 * s_tilde @ s_tilde - 4 * (1 - s2) * s_tilde) / (1 + s2) ** 2
    return c


@ti.func
def quat_kde(q: ti.math.vec4, w: ti.math.vec3) -> ti.math.vec4:
    qmat = 0.5 * ti.math.mat4(
        q[3],
        -q[2],
        q[1],
        q[0],
        q[2],
        q[3],
        -q[0],
        q[1],
        -q[1],
        q[0],
        q[3],
        q[2],
        -q[0],
        -q[1],
        -q[2],
        q[3],
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
def compute_loss(lc: ti.template()) -> ti.f32:
    # lcn = lc.normalized()
    # dp = ti.math.dot(lcn, lc_true.normalized())
    # dpc = tibfgs.clip(dp, -1.0, 1.0)
    # loss = ti.acos(dp)
    # return loss
    # return (ti.math.log(lc + 1.0) - ti.math.log(lc_true + 1)).norm_sqr()
    return (lc - lc_true).norm_sqr()


@ti.func
def propagate_one(x0: ti.template()) -> ti.f32:
    lc = compute_lc(x0)
    loss = compute_loss(lc)
    return loss

@ti.kernel
def _compute_light_curves(x0s: ti.template(), lcs: ti.template()) -> int:
    for i in x0s:
        lcs[i] = compute_lc(x0s[i])
    return 0

@ti.func
def gen_x0(w_max: ti.f32) -> ti.template():
    x0 = ti.types.vector(n=NSTATES, dtype=ti.f32)(0.0)
    x0[:3] = quat_to_mrp(rand_s3())
    x0[3:6] = rand_b2(w_max)
    if ti.static(x0.n > 6):  # iy
        x0[6] = ti.random() * 3
    if ti.static(x0.n > 7):  # iz
        x0[7] = ti.random() * 3
    return x0


lc_true = None


def load_lc_true(lc_arr: np.ndarray):
    global lc_true
    lc_true = ti.types.vector(n=NTIMES, dtype=ti.f32)(lc_arr)


L = None
O = None


def load_observation_geometry(svi: np.ndarray, ovi: np.ndarray) -> None:
    global L, O
    L = ti.math.vec3(svi).normalized()
    O = ti.math.vec3(ovi).normalized()


def unique_areas_and_normals(fn: np.ndarray, fa: np.ndarray):
    """Finds groups of unique normals and areas to save rows elsewhere"""
    (
        unique_normals,
        all_to_unique,
        unique_to_all,
    ) = np.unique(
        np.round(fn, decimals=6), axis=0, return_index=True, return_inverse=True
    )
    unique_areas = np.zeros((unique_normals.shape[0]), dtype=np.float32)
    np.add.at(unique_areas, unique_to_all, fa)
    return unique_normals, unique_areas


def load_obj(obj_path: str) -> None:
    global fa, fn, fd, fs, fp
    obj = trimesh.load(obj_path)
    v1 = obj.vertices[obj.faces[:, 0]].astype(np.float32)
    v2 = obj.vertices[obj.faces[:, 1]].astype(np.float32)
    v3 = obj.vertices[obj.faces[:, 2]].astype(np.float32)
    fnn = np.cross(v2 - v1, v3 - v1)
    fan = np.linalg.norm(fnn, axis=1, keepdims=True).astype(np.float32) / 2
    fnn = fnn / fan / 2

    fnn, fan = unique_areas_and_normals(fnn, fan.flatten())

    fa = ti.field(dtype=ti.f32, shape=fan.shape)
    fa.from_numpy(fan)

    fn = ti.Vector.field(n=3, dtype=ti.f32, shape=fnn.shape[0])
    fn.from_numpy(fnn)

    fd = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fd.fill(0.5)
    fs = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fs.fill(0.5)
    fp = ti.field(dtype=ti.f32, shape=fnn.shape[0])
    fp.fill(3)
