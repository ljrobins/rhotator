import numpy as np
import taichi as ti
import os
import polars as pl
import time
import tibfgs
import taichi as ti

_LOSS_TYPES = [
    "l2",
    "log-likelihood",
    "cosine",
]


def direct_invert_attitude(
    svi: np.ndarray,
    ovi: np.ndarray,
    epsecs: np.ndarray,
    lc_true: np.ndarray,
    x0: np.ndarray,
    obj_file_path: str,
    verbose: bool = False,
    remove_nans: bool = True,
    self_shadowing: bool = False,
    loss_function: str = "l2",
    sigma_obs: np.ndarray = None,
    cache_size: int = 500,
    finite_difference_step_size: float = 1e-3,
    iratios: np.ndarray = None,
    vary_itensor: bool = False,
    vary_global_mats: bool = False,
    substeps: int = 1,
    cs_scale: float = 0,
    n_scale: float = 0,
    i_scale: float = 0,
) -> pl.DataFrame:
    """Performs direct BFGS nonlinear optimization on initial conditions ``x0``

    :param svi: Inertial unit vector from the space object to the Sun
    :type svi: np.ndarray [3,]
    :param ovi: Inertial unit vector from the space object to the observer
    :type ovi: np.ndarray [3,]
    :param lc_true: True light curve
    :type lc_true: np.ndarray [n,]
    :param obj_file_path: Path to the .obj file to use
    :type obj_file_path: str
    :param verbose: Whether to print diagnostic information about the optimization process, defaults to False
    :type verbose: bool, optional
    :param remove_nans: Whether to remove solutions that converged to NaN values for the objective function, defaults to False
    :type remove_nans: bool, optional
    :return: Dataframe of results
    :rtype: pl.DataFrame
    """

    required_state_variables = 6 + 2 * vary_itensor + 2 * vary_global_mats

    assert x0.shape[1] == required_state_variables, ""
    assert cs_scale >= 0.0
    assert n_scale >= 0.0
    assert i_scale >= 0.0

    if required_state_variables == 6:
        assert (
            iratios is not None
        ), "If only six state variables are provided, iratios must be specified as [3,]"

    epmax = epsecs.max()
    x0[:, 3:6] *= epmax
    epsecs_old = epsecs.copy()
    epsecs = epsecs / epmax

    initialize(
        obj_file_path,
        svi,
        ovi,
        epsecs,
        lc_true,
        x0=x0,
        self_shadowing=self_shadowing,
        loss_function=loss_function,
        sigma_obs=sigma_obs,
        cache_size=cache_size,
        iratios=iratios,
    )

    from .core import propagate_one, load_g, mats

    # this might be extra?
    g = ti.field(
        dtype=ti.f64,
        shape=(x0.shape[0], len(mats), len(epsecs)),
    )
    load_g(g)

    @ti.func
    def f(x: ti.template()) -> ti.f64:
        return propagate_one(
            x, vary_itensor, vary_global_mats, substeps, cs_scale, n_scale, i_scale
        )

    t1 = time.time()
    res = tibfgs.minimize(f, x0, eps=finite_difference_step_size)
    t2 = time.time() - t1
    if verbose:
        print(
            f"Optimizing {n_particles} over {len(lc_true)} data points took {t2-t1:.2f} seconds"
        )
    lcs = compute_light_curves(
        res["xk"].to_numpy(),
        vary_itensor,
        vary_global_mats,
        substeps,
        cs_scale,
        n_scale,
        i_scale,
        compute_material_sensitivity=False,
    )['lcs'].astype(np.float32)
    lcs = (
        lcs
        / np.linalg.norm(lcs, axis=1, keepdims=True)
        * np.linalg.norm(lc_true[~np.isnan(lc_true)])
    )

    res = res.with_columns(lcs=lcs)
    
    if remove_nans:
        res = res.sort("fun").filter(pl.col("fun").is_not_nan())

    for c in ["x0", "xk"]:
        x = res[c].to_numpy().copy()
        x[:, 3:6] /= epmax
        res = res.with_columns(pl.Series(name=c, values=x))
    
    from .core import get_fd, get_fs, get_fp

    res = res.with_columns(
        pl.Series("epsecs", res.height * [epsecs_old.astype(np.float32)]),
        pl.Series('obj_file_path', res.height * [obj_file_path]),
        pl.Series('substeps', res.height * [substeps]),
        pl.Series('self_shadowing', res.height * [self_shadowing]),
        pl.Series('vary_global_mats', res.height * [vary_global_mats]),
        pl.Series('vary_itensor', res.height * [vary_itensor]),
        pl.Series('cache_size', res.height * [cache_size]),
        pl.Series('cs_scale', res.height * [cs_scale]),
        pl.Series('n_scale', res.height * [n_scale]),
        pl.Series('i_scale', res.height * [i_scale]),
        pl.Series('iratios', res.height * [np.array(iratios, dtype=np.float32)]),
        pl.Series('finite_difference_step_size', res.height * [finite_difference_step_size]),
        pl.Series('mats', res.height * [mats]),
        pl.Series('cds', res.height * [get_fd()]),
        pl.Series('css', res.height * [get_fs()]),
        pl.Series('ns', res.height * [get_fp()]),
    )

    return res


def compute_light_curves(
    x0s: np.ndarray,
    vary_itensor: bool,
    vary_global_mats: bool,
    substeps: int,
    cs_scale: float,
    n_scale: float,
    i_scale: float,
    compute_material_sensitivity: bool,
    return_g: bool = False,
    material_eps: float = 1e-4,
) -> np.ndarray:
    from .core import (
        _compute_light_curves,
        _compute_g_matrix,
        _compute_material_sensitivity,
        NTIMES,
        fd,
    )

    x0s = np.atleast_2d(x0s)

    x0s_ti = ti.field(
        dtype=ti.types.vector(n=x0s.shape[1], dtype=ti.f64), shape=x0s.shape[0]
    )
    x0s_ti.from_numpy(x0s)
    lcs = ti.field(dtype=ti.types.vector(n=NTIMES, dtype=ti.f64), shape=x0s.shape[0])

    from .core import load_g, get_g, load_m, get_m, load_h, get_h

    g = ti.field(
        dtype=ti.f64,
        shape=(x0s.shape[0], fd.shape[0], NTIMES),
    )
    load_g(g)

    m = ti.field(
        dtype=ti.f64,
        shape=(x0s.shape[0], 2, 2, x0s.shape[1], fd.shape[0], 3),
    )
    load_m(m)

    h = ti.field(
        dtype=ti.f64,
        shape=(x0s.shape[0], 2, 2, x0s.shape[1], x0s.shape[1]),
    )
    load_h(h)

    _compute_light_curves(
        x0s_ti,
        lcs,
        vary_itensor,
        vary_global_mats,
        substeps,
        cs_scale,
        n_scale,
        i_scale,
    )

    ret_dict = dict(lcs=lcs.to_numpy())

    if return_g:
        _compute_g_matrix(
            x0s_ti,
            vary_itensor,
            vary_global_mats,
            substeps,
            cs_scale,
            n_scale,
            i_scale,
        )
        ret_dict['g'] = get_g()

    if compute_material_sensitivity:
        _compute_material_sensitivity(
            x0s_ti,
            vary_itensor,
            vary_global_mats,
            substeps,
            cs_scale,
            n_scale,
            i_scale,
            material_eps,
        )
        m, h = get_m(), get_h()

        # m: num ics, \pmx, \pmtheta, num x, num faces, num coefs
        # h: num ics, \pmx1, \pmx2, num x, num x

        H_x_x = np.zeros(
            (
                h.shape[0],  # number of ics
                h.shape[-2],  # number of x states
                h.shape[-1],  # number of x states
            )
        )

        for i in range(H_x_x.shape[0]):
            for j in range(H_x_x.shape[1]):
                for k in range(H_x_x.shape[2]):
                    # +pp -mp -pm +mm
                    pp = h[i, 1, 1, j, k]
                    mp = h[i, 0, 1, j, k]
                    pm = h[i, 1, 0, j, k]
                    mm = h[i, 0, 0, j, k]
                    H_x_x[i, j, k] = 1 / (4 * material_eps**2) * (pp - pm - mp + mm)

        H_x_theta = np.zeros(
            (
                m.shape[0],  # number of ics
                m.shape[-3],  # number of x states
                m.shape[-2] * m.shape[-1],  # number of parameters (3 * number of faces)
            )
        )

        for i in range(H_x_theta.shape[0]):
            for j in range(H_x_theta.shape[1]):
                for k in range(H_x_theta.shape[2]):
                    # +pp -mp -pm +mm
                    face_ind = k // 3  # which face?
                    coef_ind = k % 3  # which coeffient?
                    pp = m[i, 1, 1, j, face_ind, coef_ind]
                    mp = m[i, 0, 1, j, face_ind, coef_ind]
                    pm = m[i, 1, 0, j, face_ind, coef_ind]
                    mm = m[i, 0, 0, j, face_ind, coef_ind]
                    H_x_theta[i, j, k] = 1 / (4 * material_eps**2) * (pp - pm - mp + mm)

        dxhat_dtheta = np.zeros_like(H_x_theta)

        for i in range(H_x_theta.shape[0]):
            dxhat_dtheta[i] = -np.linalg.inv(H_x_x[i]) @ H_x_theta[i]

        ret_dict['dxhat_dtheta'] = dxhat_dtheta
        ret_dict['H_x_x'] = H_x_x
        ret_dict['H_x_theta'] = H_x_theta

    return ret_dict


def initialize(
    obj_path: str,
    svi: np.ndarray,
    ovi: np.ndarray,
    epsecs: np.ndarray,
    lc_true: np.ndarray,
    x0: np.ndarray,
    self_shadowing: bool = False,
    loss_function: str = "log-likelihood",
    sigma_obs: np.ndarray = None,
    cache_size: int = 500,
    iratios: list[float] | None = None,
):
    from ticlip.handler import load_finfo

    FINFOFIELD = load_finfo(obj_path)

    os.environ["TATER_SELF_SHADOW"] = str(self_shadowing)

    if not loss_function in _LOSS_TYPES:
        raise ValueError(
            f"Loss type {loss_function} is not one of the valid loss types: {_LOSS_TYPES}"
        )
    else:
        os.environ["TATER_LOSS_TYPE"] = str(_LOSS_TYPES.index(loss_function))

    os.environ["TI_NUM_TIMES"] = str(lc_true.size)
    os.environ["TI_DIM_X"] = str(x0.shape[1])

    from .core import load_lc_obs, load_observation_geometry, load_obj, load_itensor

    if loss_function == "log-likelihood":
        if sigma_obs is None:
            raise ValueError(
                f"sigma_obs must be an ndarray with size equal to lc_obs.size"
            )
        elif np.sum(sigma_obs == 0):  # If any are zero
            raise ValueError(
                "one or more sigma_obs[i] == 0, this will crash the solver"
            )
    load_lc_obs(lc_true, sigma_obs)
    load_observation_geometry(svi, ovi, cache_size, hs=np.diff(epsecs))
    load_obj(obj_path)

    if iratios is not None:
        load_itensor(iratios)

    from .core import load_unshadowed_areas
    from ticlip.handler import unshadowed_areas_handle

    load_unshadowed_areas(unshadowed_areas_handle())

    from ticlip.handler import gen_cache

    cache = gen_cache(
        cache_size, cache_size, empty=not self_shadowing
    )  # Only fill the cache if we're using self-shadowing

    return FINFOFIELD


def get_material_types() -> list[str]:
    from .core import mats

    return mats
