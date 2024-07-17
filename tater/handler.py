import numpy as np
import taichi as ti
import os
from tibfgs import init_ti


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
            x0s[i] = gen_x0(max_angular_velocity_mag)
        return 0

    gen_x0s()

    return x0s.to_numpy()


def initialize(
    obj_path: str,
    svi: np.ndarray,
    ovi: np.ndarray,
    lc_true: np.ndarray,
    x0: np.ndarray,
    dimx: int = 6,
    max_angular_velocity_magnitude: float = 0.3,
    **taichi_kwargs,
) -> None:
    init_ti(**taichi_kwargs)

    if x0 is not None:
        os.environ["TI_DIM_X"] = str(x0.shape[1])
    else:
        os.environ["TI_DIM_X"] = str(dimx)

    os.environ["TI_NUM_TIMES"] = str(lc_true.size)

    from .core import load_lc_true, load_observation_geometry, load_obj

    load_lc_true(lc_true)
    load_observation_geometry(svi, ovi)
    load_obj(obj_path)
