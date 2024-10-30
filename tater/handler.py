import numpy as np
import taichi as ti
import os
from tibfgs import init_ti
import polars as pl
import time
import tibfgs
import ticlip


def direct_invert_attitude(
    t: np.ndarray,
    svi: np.ndarray,
    ovi: np.ndarray,
    lc_true: np.ndarray,
    n_particles: int,
    max_angular_velocity_magnitude: float,
    obj_file_path: str,
    dimx: int = 6,
    verbose: bool = False,
    remove_nans: bool = False,
) -> pl.DataFrame:
    """Performs direct BFGS nonlinear optimization on ``n_particles`` number of initial conditions

    :param t: Times [seconds]
    :type t: np.ndarray [n,]
    :param svi: Inertial unit vector from the space object to the Sun
    :type svi: np.ndarray [3,]
    :param ovi: Inertial unit vector from the space object to the observer
    :type ovi: np.ndarray [3,]
    :param lc_true: True light curve
    :type lc_true: np.ndarray [n,]
    :param n_particles: Number of particles (initial conditions) to optimize
    :type n_particles: int
    :param max_angular_velocity_magnitude: Maximum magnitude of the angular velocity vector [rad/s]
    :type max_angular_velocity_magnitude: float
    :param obj_file_path: Path to the .obj file to use
    :type obj_file_path: str
    :param dimx: Dimension of the state (6 is just orientation and angular velocity), defaults to 6
    :type dimx: int, optional
    :param verbose: Whether to print diagnostic information about the optimization process, defaults to False
    :type verbose: bool, optional
    :param remove_nans: Whether to remove solutions that converged to NaN values for the objective function, defaults to False
    :type remove_nans: bool, optional
    :return: Dataframe of results
    :rtype: pl.DataFrame
    """
    x0 = initialize(
        obj_file_path,
        svi,
        ovi,
        lc_true,
        max_angular_velocity_magnitude=max_angular_velocity_magnitude,
        n_particles=n_particles,
        dimx=dimx,
    )

    from .core import propagate_one, propagate_one_self_shadow, load_unshadowed_areas
    from ticlip.handler import unshadowed_areas_handle

    load_unshadowed_areas(unshadowed_areas_handle())

    t1 = time.time()
    res = tibfgs.minimize(propagate_one_self_shadow, x0)
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
    lc_true: np.ndarray,
    x0: np.ndarray | None = None,
    dimx: int = 6,
    n_particles: int = 10_000,
    max_angular_velocity_magnitude: float = 0.3,
    **taichi_kwargs,
) -> np.ndarray:
    # init_ti(**taichi_kwargs)

    os.environ["TI_NUM_TIMES"] = str(lc_true.size)

    if x0 is not None:
        os.environ["TI_DIM_X"] = str(x0.shape[1])
    else:
        os.environ["TI_DIM_X"] = str(dimx)
        x0 = generate_initial_states(n_particles, dimx, max_angular_velocity_magnitude)

    from .core import load_lc_true, load_observation_geometry, load_obj

    load_lc_true(lc_true)
    load_observation_geometry(svi, ovi)
    load_obj(obj_path)

    return x0
