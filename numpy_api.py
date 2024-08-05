import numpy as np


def get_numpy_api() -> np:
    try:
        import cupy as cp

        return cp, cp.ndarray  # type: cp
    except ImportError:
        try:
            import mlx.core as mx

            return mx, mx.ndarray  # type: mx
        except:
            return np, np.ndarray  # type: np


xp, ndarray = get_numpy_api()
