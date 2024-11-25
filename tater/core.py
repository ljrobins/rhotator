import taichi as ti
import os
import numpy as np

NSTATES = int(os.environ["TI_DIM_X"])
NTIMES = int(os.environ["TI_NUM_TIMES"])
LOSS_TYPE = int(os.environ["TATER_LOSS_TYPE"])
SELF_SHADOWING = bool(os.environ["TATER_SELF_SHADOW"])
h: ti.f32 = float(os.environ["TATER_DT"])

SVI = ti.field(dtype=ti.math.vec3, shape=NTIMES)
OVI = ti.field(dtype=ti.math.vec3, shape=NTIMES)

itensor = ti.Vector([1.0, 2.0, 3.0])


@ti.func
def clip(x: ti.f32, min_v: ti.f32, max_v: ti.f32) -> ti.f32:
    v = x
    if x < min_v:
        v = min_v
    if x > max_v:
        v = max_v
    return v


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
def spiral_s3(n, i) -> ti.math.vec4:
    """Samples the hypersphere with a fibonacci spiral

    :param n: Number of quaternions being sampled overall
    :type n: int
    :param i: Index of the current saple
    :type i: int
    :return: Sampled unit quaternions
    :rtype: ti.math.vec4
    """
    phi = ti.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = i + 0.5
    r = ti.sqrt(s / n)
    R = ti.sqrt(1.0 - s / n)
    alpha = 2.0 * ti.math.pi * s / phi
    beta = 2.0 * ti.math.pi * s / psi
    return ti.math.vec4(
        [r * ti.sin(alpha), r * ti.cos(alpha), R * ti.sin(beta), R * ti.cos(beta)]
    )


@ti.func
def quat_to_mrp(q: ti.math.vec4) -> ti.math.vec3:
    return q[:3] / (q[3] + 1)


@ti.func
def normalized_convex_light_curve(L: ti.math.vec3, O: ti.math.vec3) -> float:
    b = 0.0

    uas = ti.types.vector(n=fn.shape[0], dtype=ti.f32)(0.0)
    ti.loop_config(serialize=True)
    for i in range(fn.shape[0]):
        uas[i] = fa[i]

    if SELF_SHADOWING:
        saz, sel = cart_to_sph(L)
        oaz, oel = cart_to_sph(O)
        uas = unshadowed_areas(oaz, oel, saz, sel)

    ti.loop_config(serialize=True)
    for i in range(fn.shape[0]):
        ndo = rdot(fn[i], O)
        ndl = rdot(fn[i], L)

        if ndo > 0 and ndl > 0:
            fr = brdf_blinn_phong(L, O, fn[i], fd[i], fs[i], fp[i])
            ti.atomic_add(b, uas[i] * fr * ndo * ndl)
    return b


@ti.func
def rdot(v1, v2) -> float:
    return ti.max(ti.math.dot(v1, v2), 0.0)


@ti.func
def brdf_blinn_phong(
    L: ti.math.vec3,
    O: ti.math.vec3,
    N: ti.math.vec3,
    cd: float,
    cs: float,
    n: float,
) -> float:
    H = (L + O).normalized()
    D = (n + 2) / (2 * np.pi) * rdot(N, H) ** n
    denom = 4 * rdot(N, L) * rdot(N, O)
    fr: ti.f32 = ti.math.nan
    if denom <= 1e-5:
        fr = 0.0
    else:
        fd = cd / ti.math.pi
        fr = fd + cs * D / denom
    return fr


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
def cart_to_sph(v: ti.math.vec3) -> ti.math.vec2:
    x, y, z = v
    az = ti.atan2(y, x)
    el = ti.atan2(z, ti.sqrt(x**2 + y**2))
    if az < 0:
        az = az + 2 * ti.math.pi
    return ti.math.vec2(az, el)


@ti.func
def compute_lc(x0: ti.template()) -> ti.template():
    current_state = x0
    itens = itensor
    if ti.static(x0.n > 6):
        itens.y = ti.abs(x0[6])
    if ti.static(x0.n > 7):
        itens.z = ti.abs(x0[7])

    # print('INERTIA TENSOR: ', itens)
    # print('x0: ', x0)

    lc = ti.types.vector(n=NTIMES, dtype=ti.f32)(0.0)
    ti.loop_config(serialize=True)
    for j in range(lc.n):
        # print("TIMESTEP", j)
        dcm = mrp_to_dcm(current_state[:3])
        svb = dcm @ SVI[j]
        ovb = dcm @ OVI[j]
        # print(j, 'h:', h, 'SVB:', svb)
        lc[j] = normalized_convex_light_curve(svb, ovb)
        if j < lc.n - 1:  # if this isnt the last step
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
    loss = 0.0

    if LOSS_TYPE == 2:
        # Cosine loss
        dp = ti.math.dot(lc.normalized(), lc_obs.normalized())
        loss = ti.acos(clip(dp, -1.0, 1.0))
    elif LOSS_TYPE == 1:
        # Negative log-likelihood
        loss = (
            ti.log(sigma_obs) + 1 / 2 * ((lc - lc_obs) / sigma_obs) ** 2
        ).sum()  # Negative Log Likelihood
    elif LOSS_TYPE == 0:
        # L2 norm error
        loss = (lc - lc_obs).norm_sqr()  # L2 error

    return loss


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
def gen_x0(n: ti.i32, i: ti.i32, w_max: ti.f32) -> ti.template():
    x0 = ti.types.vector(n=NSTATES, dtype=ti.f32)(0.0)
    q0 = spiral_s3(n, i)
    if q0.w < 0:
        q0 *= -1
    x0[:3] = quat_to_mrp(q0)
    x0[3:6] = rand_b2(w_max)
    if ti.static(x0.n > 6):  # iy
        x0[6] = ti.random() * 3
    if ti.static(x0.n > 7):  # iz
        x0[7] = ti.random() * 3
    return x0


lc_obs = None
sigma_obs = None


def load_lc_obs(lc_arr: np.ndarray, sigma: np.ndarray):
    global lc_obs, sigma_obs
    lc_obs = ti.types.vector(n=NTIMES, dtype=ti.f32)(lc_arr)
    if sigma is not None:
        sigma_obs = ti.types.vector(n=NTIMES, dtype=ti.f32)(sigma)
    else:
        sigma_obs = 0 * lc_obs


def load_observation_geometry(svi: np.ndarray, ovi: np.ndarray) -> None:
    global SVI, OVI
    if svi.size == 3 and ovi.size == 3:
        svi = np.tile(svi, (NTIMES, 1))
        ovi = np.tile(ovi, (NTIMES, 1))
    elif svi.shape == (NTIMES, 3) and ovi.shape == (NTIMES, 3):
        pass
    else:
        raise ValueError(
            f"The shapes of either svi or ovi are not (3,) or (NTIMES,3), got {svi.shape=}, {ovi.shape=}"
        )

    assert np.allclose(
        np.linalg.norm(svi, axis=1), 1.0
    ), "Make sure svi rows are normalized!"
    assert np.allclose(
        np.linalg.norm(ovi, axis=1), 1.0
    ), "Make sure ovi rows are normalized!"

    SVI.from_numpy(svi.astype(np.float32))
    OVI.from_numpy(ovi.astype(np.float32))


def load_mtllib(mtllib_path: str) -> dict:
    """Loads a .mtl file

    :param mtllib_path: Path to the .mtl file
    :type mtllib_path: str
    :return: Dictionary of materials arranged by material name
    :rtype: dict
    """
    print(f"Reading", mtllib_path)
    with open(mtllib_path, "r") as f:
        materials = {}
        current_mat_name = None
        for line in f.readlines():
            if "newmtl" in line:
                current_mat_name = line.split()[1]
                materials[current_mat_name] = {}
            if current_mat_name is not None:
                if "Kd" in line:
                    vals = list(map(float, line.split()[1:]))
                    materials[current_mat_name]["cd"] = vals[0]
                    materials[current_mat_name]["cs"] = vals[1]
                    materials[current_mat_name]["n"] = vals[2] * 1000.0
                    if (vals[0] + vals[1]) > 1.0:
                        raise ValueError(
                            "Red plus Green channel of each material must add to less than 1 for energy conservation, aborting."
                        )
    return materials


def load_obj(obj_path: str, itensor_ratios: np.ndarray) -> None:
    global fa, fn, fd, fs, fp, itensor

    if itensor_ratios is not None:
        itensor[1:] = itensor_ratios

    obj_dir = os.path.split(obj_path)[0]
    with open(obj_path, "r") as f:
        lines = f.readlines()
        mtllib_lines = [line for line in lines if "mtllib" in line]
        if not len(mtllib_lines):
            raise ValueError("No mtllib found!")
        elif len(mtllib_lines) > 1:
            raise ValueError("Multiple mtllibs found, not supported!")
        mtllib_name = mtllib_lines[0].split()[1]
        materials = load_mtllib(os.path.join(obj_dir, mtllib_name))

        v = []
        f = []
        f_cd_cs_n = []
        current_mat = None
        for i, line in enumerate(lines):
            if line.startswith("v "):
                v.append(list(map(float, line[2:].split())))
            if (
                "usemtl" in line
            ):  # then the next faces should all use the specified material
                if materials is None:
                    raise ValueError(
                        "no mtllib declaration found in .obj file, but at least one usemtl line exists"
                    )
                current_mat_name = line.split()[1]
                if current_mat_name not in materials:
                    raise ValueError(
                        f"Material {current_mat_name} not found in the material library!"
                    )
                current_mat = materials[current_mat_name]
            elif line.startswith("f "):
                if current_mat is None:
                    raise ValueError(
                        f"Encountered a face definition before the current material was set! ({obj_path}, line {i})"
                    )
                lf = (
                    np.array(
                        [
                            list(map(int, [y for y in x.split("/") if len(y)]))
                            for x in line[2:].strip().split(" ")
                        ]
                    )
                    - 1
                )
                f.append(lf[:, 0].tolist())
                f_cd_cs_n.append(
                    [current_mat["cd"], current_mat["cs"], current_mat["n"]]
                )
    f_cd_cs_n = np.array(f_cd_cs_n, dtype=np.float32)

    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.uint16)

    fd = ti.field(dtype=ti.f32, shape=f.shape[0])
    fd.from_numpy(f_cd_cs_n[:, 0])
    fs = ti.field(dtype=ti.f32, shape=f.shape[0])
    fs.from_numpy(f_cd_cs_n[:, 1])
    fp = ti.field(dtype=ti.f32, shape=f.shape[0])
    fp.from_numpy(f_cd_cs_n[:, 2])

    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fnn = np.cross(v2 - v1, v3 - v1)
    fan = np.linalg.norm(fnn, axis=1) / 2
    fnn = fnn / fan.reshape(-1, 1) / 2  # Normalizing

    fa = ti.field(dtype=ti.f32, shape=fan.shape)
    fa.from_numpy(fan)

    fn = ti.Vector.field(n=3, dtype=ti.f32, shape=fnn.shape[0])
    fn.from_numpy(fnn)

    print(fa, fp, fs, fd)


unshadowed_areas = None


def load_unshadowed_areas(fcn) -> None:
    global unshadowed_areas
    unshadowed_areas = fcn
