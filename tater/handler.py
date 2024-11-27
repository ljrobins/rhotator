import numpy as np
import taichi as ti
import os
import polars as pl
import time
import tibfgs

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
    )

    from .core import propagate_one

    t1 = time.time()
    res = tibfgs.minimize(propagate_one, x0, eps=1e-3)
    t2 = time.time() - t1
    if verbose:
        print(
            f"Optimizing {n_particles} over {len(lc_true)} data points took {t2-t1:.2f} seconds"
        )
    lcs = compute_light_curves(res["xk"].to_numpy())
    res = res.with_columns(lcs=lcs)
    if remove_nans:
        res = res.sort("fun").filter(pl.col("fun").is_not_nan())
    return res


def compute_light_curves(x0s: np.ndarray) -> np.ndarray:
    from .core import _compute_light_curves, NTIMES

    x0s = np.atleast_2d(x0s)

    x0s_ti = ti.field(
        dtype=ti.types.vector(n=x0s.shape[1], dtype=ti.f32), shape=x0s.shape[0]
    )
    x0s_ti.from_numpy(x0s)
    lcs = ti.field(dtype=ti.types.vector(n=NTIMES, dtype=ti.f32), shape=x0s.shape[0])
    _compute_light_curves(x0s_ti, lcs)
    return lcs.to_numpy()


def generate_initial_states(
    n_particles: int, dimx: int = 6, max_angular_velocity_mag: float = 0.3
) -> np.ndarray:
    try:
        x0s = ti.field(dtype=ti.types.vector(n=dimx, dtype=ti.f32), shape=n_particles)
    except ti.lang.exception.TaichiRuntimeError:
        raise ValueError(
            "tater.generate_initial_states() must be called after tater.initialize()"
        )

    from .core import gen_x0

    @ti.kernel
    def gen_x0s() -> int:
        for i in x0s:
            x0s[i] = gen_x0(x0s.shape[0], i, max_angular_velocity_mag)
        return 0

    gen_x0s()

    return x0s.to_numpy()


def initialize(
    obj_path: str,
    svi: np.ndarray,
    ovi: np.ndarray,
    epsecs: np.ndarray,
    lc_true: np.ndarray,
    x0: np.ndarray,
    self_shadowing: bool = False,
    loss_function: str = "l2",
    sigma_obs: np.ndarray = None,
    cache_size: int = 500,
):
    from ticlip.handler import load_finfo

    load_finfo(obj_path)

    os.environ["TATER_SELF_SHADOW"] = str(self_shadowing)
    os.environ["TATER_DT"] = str(np.diff(epsecs[:2])[0])

    if not loss_function in _LOSS_TYPES:
        raise ValueError(
            f"Loss type {loss_function} is not one of the valid loss types: {_LOSS_TYPES}"
        )
    else:
        os.environ["TATER_LOSS_TYPE"] = str(_LOSS_TYPES.index(loss_function))

    os.environ["TI_NUM_TIMES"] = str(lc_true.size)
    os.environ["TI_DIM_X"] = str(x0.shape[1])

    from .core import load_lc_obs, load_observation_geometry, load_obj

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
    load_observation_geometry(svi, ovi)
    load_obj(obj_path)

    from .core import load_unshadowed_areas
    from ticlip.handler import unshadowed_areas_handle

    load_unshadowed_areas(unshadowed_areas_handle())

    from ticlip.handler import gen_cache

    cache = gen_cache(
        cache_size, cache_size, empty=not self_shadowing
    )  # Only fill the cache if we're using self-shadowing
