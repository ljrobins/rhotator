import numpy as np
import taichi as ti
import os
from tibfgs import init_ti


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
    init_ti(**taichi_kwargs)

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
