import taichi as ti
import os
import numpy as np

NSTATES = int(os.environ["TI_DIM_X"])
NTIMES = int(os.environ["TI_NUM_TIMES"])
LOSS_TYPE = int(os.environ["TATER_LOSS_TYPE"])
SELF_SHADOWING = bool(os.environ["TATER_SELF_SHADOW"])

SVI = ti.field(dtype=ti.math.vec3, shape=NTIMES)
OVI = ti.field(dtype=ti.math.vec3, shape=NTIMES)
HS = ti.field(dtype=ti.f64, shape=NTIMES - 1)

itensor = ti.Vector([1.0, 2.0, 3.0])
naz = None
nel = None


@ti.func
def clip(x: ti.f64, min_v: ti.f64, max_v: ti.f64) -> ti.f64:
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
def normalized_convex_light_curve(
    L: ti.math.vec3,
    O: ti.math.vec3,
    cs_factor: ti.f64,
    n_factor: ti.f64,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    cds: ti.template(),
    css: ti.template(),
    ns: ti.template(),
    save_g: bool,
    x_ind: int,
    t_ind: int,
) -> float:
    b = 0.0

    uas = ti.types.vector(n=fn.shape[0], dtype=ti.f64)(0.0)
    ti.loop_config(serialize=True)
    for i in range(fn.shape[0]):
        uas[i] = fa[i]

    # print(uas)
    if SELF_SHADOWING:
        saz, sel = cart_to_sph(L)
        oaz, oel = cart_to_sph(O)
        uas = unshadowed_areas(oaz, oel, saz, sel, naz, nel)

    # print(uas)
    ti.loop_config(serialize=True)
    for i in range(fn.shape[0]):
        ndo = rdot(fn[i], O)
        ndl = rdot(fn[i], L)

        if ndo > 0 and ndl > 0:
            cs = 0.0
            n = 0.0
            if cs_scale > 1:
                cs = css[i] * cs_scale ** ti.math.sin(ti.math.pi / 2 * cs_factor)
            else:
                cs = css[i] * (1 + cs_scale * ti.math.sin(ti.math.pi / 2 * cs_factor))
            if n_scale > 1:
                n = ns[i] * n_scale ** ti.math.sin(ti.math.pi / 2 * n_factor)
            else:
                n = ns[i] * (1 + n_scale * ti.math.sin(ti.math.pi / 2 * n_factor))

            fr = brdf_blinn_phong(L, O, fn[i], cds[i], cs, n)
            contrib = uas[i] * fr * ndo * ndl
            if save_g:
                g[x_ind, i, t_ind] = contrib
            ti.atomic_add(b, contrib)
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
    fr: ti.f64 = ti.math.nan
    if denom <= 1e-5:
        fr = 0.0
    else:
        fr = cd / ti.math.pi + cs * D / denom
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
    fr: ti.f64 = ti.math.nan
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
def compute_lc(
    x0: ti.template(),
    vary_itensor: ti.u1,
    vary_global_mats: ti.u1,
    substeps: int,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    i_scale: ti.f64,
    cds: ti.template(),
    css: ti.template(),
    ns: ti.template(),
    save_g: bool,
    x_ind: int,
) -> ti.template():
    current_state = x0
    itens = itensor

    itens_y_factor = 0.0
    itens_z_factor = 0.0
    cs_factor = 0.0
    n_factor = 0.0

    if vary_itensor and not vary_global_mats:
        if ti.static(x0.n > 6):
            itens_y_factor = x0[6]
        if ti.static(x0.n > 7):
            itens_z_factor = x0[7]
    elif not vary_itensor and vary_global_mats:
        if ti.static(x0.n > 6):
            cs_factor = x0[6]
        if ti.static(x0.n > 7):
            n_factor = x0[7]
    elif vary_itensor and vary_global_mats:
        if ti.static(x0.n > 6):
            itens_y_factor = x0[6]
        if ti.static(x0.n > 7):
            itens_z_factor = x0[7]
        if ti.static(x0.n > 8):
            cs_factor = x0[8]
        if ti.static(x0.n > 9):
            n_factor = x0[9]

    if vary_itensor:
        if i_scale > 1:
            itens.y = itens.y * i_scale ** ti.math.sin(ti.math.pi / 2 * itens_y_factor)
            itens.z = itens.z * i_scale ** ti.math.sin(ti.math.pi / 2 * itens_z_factor)
        else:
            itens.y = itens.y * (
                1 + i_scale * ti.math.sin(ti.math.pi / 2 * itens_y_factor)
            )
            itens.z = itens.z * (
                1 + i_scale * ti.math.sin(ti.math.pi / 2 * itens_z_factor)
            )

    # print('INERTIA TENSOR: ', itens)
    # print('x0: ', x0)

    lc = ti.types.vector(n=NTIMES, dtype=ti.f64)(0.0)
    ti.loop_config(serialize=True)
    for j in range(lc.n):
        # print("TIMESTEP", j)
        ti.loop_config(serialize=True)
        for k in range(substeps):
            dcm = mrp_to_dcm(current_state[:3])
            svb = dcm @ SVI[j]
            ovb = dcm @ OVI[j]
            # print(j, 'h:', h, 'SVB:', svb)
            contrib = (
                normalized_convex_light_curve(
                    svb,
                    ovb,
                    cs_factor,
                    n_factor,
                    cs_scale,
                    n_scale,
                    cds,
                    css,
                    ns,
                    save_g,
                    x_ind,
                    j,
                )
                / substeps
            )

            ti.atomic_add(lc[j], contrib)
            prev_current_state = current_state
            if j < lc.n - 1:  # if this isnt the last step
                current_state = rk4_step(current_state, HS[j] / substeps, itens)
            if ti.math.isnan(current_state).sum() > 0:
                print(f"MAYDAY: NAN STATE AT {(j+k/substeps)/lc.n}: {current_state}")
                print(f"    prev current: {prev_current_state}")
    return lc


@ti.func
def rk4_step(x0, h, itensor):
    # Compute |H| initial (correct with squares)
    H_norm = ti.math.sqrt(
        (itensor[0] * x0[3]) ** 2
        + (itensor[1] * x0[4]) ** 2
        + (itensor[2] * x0[5]) ** 2
    )

    # RK4 integration
    k1 = h * state_derivative_mrp(x0, itensor)
    k2 = h * state_derivative_mrp(x0 + k1 / 2, itensor)
    k3 = h * state_derivative_mrp(x0 + k2 / 2, itensor)
    k4 = h * state_derivative_mrp(x0 + k3, itensor)
    new_state = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Compute |H| new
    H_norm_new = ti.math.sqrt(
        (itensor[0] * new_state[3]) ** 2
        + (itensor[1] * new_state[4]) ** 2
        + (itensor[2] * new_state[5]) ** 2
    )

    # Rescale angular velocity
    new_h = itensor * new_state[3:6]
    new_h = new_h * (H_norm / H_norm_new)
    new_state[3:6] = new_h / itensor

    new_s2 = new_state[0] ** 2 + new_state[1] ** 2 + new_state[2] ** 2
    # shadow set switching, if we need to
    if new_s2 > 1.0:
        # Doing switching if we've passed outside of the unit sphere!
        new_state[:3] = -new_state[:3] / new_s2

    return new_state


@ti.func
def rk5_nystrom_step(x0, h, itensor):
    # Stage 1
    ks1 = h * state_derivative_mrp(x0, itensor)

    # Stage 2
    ks2 = h * state_derivative_mrp(x0 + (1.0 / 3.0) * ks1, itensor)

    # Stage 3
    ks3 = h * state_derivative_mrp(
        x0 + (4.0 / 25.0) * ks1 + (6.0 / 25.0) * ks2, itensor
    )

    # Stage 4
    ks4 = h * state_derivative_mrp(
        x0 + (1.0 / 4.0) * ks1 - 3.0 * ks2 + (15.0 / 4.0) * ks3, itensor
    )

    # Stage 5
    ks5 = h * state_derivative_mrp(
        x0
        + (2.0 / 27.0) * ks1
        + (10.0 / 9.0) * ks2
        - (50.0 / 81.0) * ks3
        + (8.0 / 81.0) * ks4,
        itensor,
    )

    # Stage 6
    ks6 = h * state_derivative_mrp(
        x0
        + (2.0 / 25.0) * ks1
        + (12.0 / 25.0) * ks2
        + (2.0 / 15.0) * ks3
        + (8.0 / 75.0) * ks4,
        itensor,
    )

    # Final combination (weighted sum)
    new_state = x0 + (
        (23.0 / 192.0) * ks1
        + (125.0 / 192.0) * ks3
        + (-27.0 / 64.0) * ks5
        + (125.0 / 192.0) * ks6
    )

    # Angular momentum rescaling
    H_norm = ti.math.sqrt(
        (itensor[0] * x0[3]) ** 2
        + (itensor[1] * x0[4]) ** 2
        + (itensor[2] * x0[5]) ** 2
    )
    H_norm_new = ti.math.sqrt(
        (itensor[0] * new_state[3]) ** 2
        + (itensor[1] * new_state[4]) ** 2
        + (itensor[2] * new_state[5]) ** 2
    )

    new_h = itensor * new_state[3:6]
    new_h = new_h * (H_norm / H_norm_new)
    new_state[3:6] = new_h / itensor

    # Shadow set switching
    new_s2 = new_state[0] ** 2 + new_state[1] ** 2 + new_state[2] ** 2
    if new_s2 > 1.0:
        new_state[:3] = -new_state[:3] / new_s2

    return new_state


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
def compute_loss(lc: ti.template()) -> ti.f64:
    loss = 0.0

    if LOSS_TYPE == 2:
        # Cosine loss
        dp = ti.math.dot(lc.normalized(), lc_obs.normalized())
        loss = ti.acos(clip(dp, -1.0, 1.0))
    elif LOSS_TYPE == 1:
        # Negative log-likelihood
        lc = lc / lc.norm() * lc_obs_norm  # Force est to have same magnitude as
        ti.loop_config(serialize=True)
        for i in range(lc.n):
            if ~ti.math.isnan(sigma_obs[i]) and ~ti.math.isnan(lc_obs[i]):
                loss = (
                    loss
                    + 1 / 2 * ti.log(2 * ti.math.pi)
                    + ti.log(sigma_obs[i])
                    + 1 / 2 * ((lc[i] - lc_obs[i]) / sigma_obs[i]) ** 2
                )
        loss = loss / lc.n

    elif LOSS_TYPE == 0:
        # L2 norm error
        lc = lc / lc.norm() * lc_obs_norm  # Force est to have same magnitude as
        loss = (lc - lc_obs).norm_sqr()  # L2 error

    return loss


@ti.func
def propagate_one(
    x0: ti.template(),
    vary_itensor: ti.u1,
    vary_global_mats: ti.u1,
    substeps: int,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    i_scale: ti.f64,
) -> ti.f64:
    lc = compute_lc(
        x0,
        vary_itensor,
        vary_global_mats,
        substeps,
        cs_scale,
        n_scale,
        i_scale,
        fd,
        fs,
        fp,
        False,
        0,
    )
    loss = compute_loss(lc)
    return loss


@ti.kernel
def _compute_light_curves(
    x0s: ti.template(),
    lcs: ti.template(),
    vary_itensor: ti.u1,
    vary_global_mats: ti.u1,
    substeps: int,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    i_scale: ti.f64,
) -> int:
    for i in x0s:
        lcs[i] = compute_lc(
            x0s[i],
            vary_itensor,
            vary_global_mats,
            substeps,
            cs_scale,
            n_scale,
            i_scale,
            fd,
            fs,
            fp,
            False,
            0,
        )
    return 0


@ti.kernel
def _compute_material_sensitivity(
    x0s: ti.template(),
    vary_itensor: ti.u1,
    vary_global_mats: ti.u1,
    substeps: int,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    i_scale: ti.f64,
    eps: ti.f64,
) -> ti.int32:
    print('Material sens starting...')
    ti.loop_config(serialize=True)
    for i in range(x0s.shape[0]):
        print(f'MS: {i}/{x0s.shape[0]}')
        for s1 in range(2):
            x1sign: ti.f64 = -1 + 2 * float(s1)
            for s2 in range(2):
                x2sign: ti.f64 = -1 + 2 * float(s2)
                for l in range(len(x0s[i])):
                    for j in range(l, len(x0s[i])):
                        x0s[i][l] += x1sign * eps
                        x0s[i][j] += x2sign * eps

                        v = compute_loss(
                            compute_lc(
                                x0s[i],
                                vary_itensor,
                                vary_global_mats,
                                substeps,
                                cs_scale,
                                n_scale,
                                i_scale,
                                fd,
                                fs,
                                fp,
                                False,
                                0,
                            )
                        )

                        h[i, s1, s2, l, j] = v
                        h[i, s1, s2, j, l] = v

                        x0s[i][l] -= x1sign * eps
                        x0s[i][j] -= x2sign * eps

    ti.loop_config(serialize=True)
    for i in range(x0s.shape[0]):
        # print(f"{i=}")
        for xs in range(2):
            # positive or negative perturbations
            xsign: ti.f64 = -1 + 2 * float(xs)
            for ts in range(2):
                tsign: ti.f64 = -1 + 2 * float(ts)
                for l in range(len(x0s[i])):
                    # print(f"{l=}")
                    for j in range(fd.shape[0]):
                        for k in range(3):
                            # once for each of the variables we have to change
                            if k == 0:
                                fd[j] += tsign * eps
                            elif k == 1:
                                fs[j] += tsign * eps
                            elif k == 2:
                                fp[j] += tsign * eps
                            x0s[i][l] += xsign * eps

                            m[i, xs, ts, l, j, k] = compute_loss(
                                compute_lc(
                                    x0s[i],
                                    vary_itensor,
                                    vary_global_mats,
                                    substeps,
                                    cs_scale,
                                    n_scale,
                                    i_scale,
                                    fd,
                                    fs,
                                    fp,
                                    False,
                                    0,
                                )
                            )

                            if k == 0:
                                fd[j] -= tsign * eps
                            elif k == 1:
                                fs[j] -= tsign * eps
                            elif k == 2:
                                fp[j] -= tsign * eps
                            x0s[i][l] -= xsign * eps
    return 0


@ti.kernel
def _compute_g_matrix(
    x0s: ti.template(),
    vary_itensor: ti.u1,
    vary_global_mats: ti.u1,
    substeps: int,
    cs_scale: ti.f64,
    n_scale: ti.f64,
    i_scale: ti.f64,
) -> int:
    ti.loop_config(serialize=True)
    for i in range(x0s.shape[0]):
        # print(f"{i=}")
        compute_lc(
            x0s[i],
            vary_itensor,
            vary_global_mats,
            substeps,
            cs_scale,
            n_scale,
            i_scale,
            fd,
            fs,
            fp,
            True,
            i,
        )
    return 0


lc_obs = None
sigma_obs = None
lc_obs_norm = None


def load_lc_obs(lc_arr: np.ndarray, sigma: np.ndarray):
    global lc_obs, lc_obs_norm, sigma_obs
    lc_obs = ti.types.vector(n=NTIMES, dtype=ti.f64)(lc_arr)
    lc_obs_norm = np.linalg.norm(lc_obs[~np.isnan(lc_obs)])
    if sigma is not None:
        sigma_obs = ti.types.vector(n=NTIMES, dtype=ti.f64)(sigma)
    else:
        sigma_obs = 0 * lc_obs


def load_observation_geometry(
    svi: np.ndarray, ovi: np.ndarray, cache_size: int, hs: np.ndarray
) -> None:
    global SVI, OVI, naz, nel
    naz = cache_size
    nel = cache_size
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

    SVI.from_numpy(svi.astype(np.float64))
    OVI.from_numpy(ovi.astype(np.float64))
    HS.from_numpy(hs)


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


def load_itensor(iratios: list[float]) -> None:
    global itensor
    assert len(iratios) == 3, "required to provide all xyz entries"
    for i, v in enumerate(iratios):
        itensor[i] = v


def set_fd(fd_np: np.ndarray) -> None:
    global fd
    fd.from_numpy(fd_np)


def set_fs(fs_np: np.ndarray) -> None:
    global fs
    fs.from_numpy(fs_np)


def set_fp(fp_np: np.ndarray) -> None:
    global fp
    fp.from_numpy(fp_np)

mats = []

def load_obj(obj_path: str) -> None:
    global fa, fn, fd, fs, fp, mats

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
                mats.append(current_mat_name)
    f_cd_cs_n = np.array(f_cd_cs_n, dtype=np.float64)

    v = np.array(v, dtype=np.float64)

    fd = ti.field(dtype=ti.f64, shape=len(f))
    fd.from_numpy(f_cd_cs_n[:, 0])
    fs = ti.field(dtype=ti.f64, shape=len(f))
    fs.from_numpy(f_cd_cs_n[:, 1])
    fp = ti.field(dtype=ti.f64, shape=len(f))
    fp.from_numpy(f_cd_cs_n[:, 2])

    fnn = np.zeros((len(f), 3))

    fan = np.zeros(len(f))

    for i in range(len(f)):
        for j in range(len(f[i]) - 2):
            v1, v2, v3 = v[f[i][0 + j]], v[f[i][1 + j]], v[f[i][2 + j]]
            fnc = np.cross(v2 - v1, v3 - v1)
            if j == 0:
                fnn[i, :] = fnc
                fnn[i, :] /= np.linalg.norm(fnn[i, :])
            fan[i] += np.linalg.norm(fnc) / 2

    fa = ti.field(dtype=ti.f64, shape=fan.shape)
    fa.from_numpy(fan)

    fn = ti.Vector.field(n=3, dtype=ti.f64, shape=fnn.shape[0])
    fn.from_numpy(fnn)

    # print(fa, fp, fs, fd)


unshadowed_areas = None


def load_unshadowed_areas(fcn) -> None:
    global unshadowed_areas
    unshadowed_areas = fcn


def load_g(_g: ti.field) -> None:
    global g
    g = _g


def get_g() -> np.ndarray:
    global g
    return g.to_numpy()


def load_m(_m: ti.field) -> None:
    global m
    m = _m


def get_m() -> np.ndarray:
    global m
    return m.to_numpy()


def load_h(_h: ti.field) -> None:
    global h
    h = _h


def get_h() -> np.ndarray:
    global h
    return h.to_numpy()

def get_fd() -> np.ndarray:
    global fd
    return fd.to_numpy()


def get_fs() -> np.ndarray:
    global fs
    return fs.to_numpy()


def get_fp() -> np.ndarray:
    global fp
    return fp.to_numpy()