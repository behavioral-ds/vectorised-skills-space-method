def get_numpy_api():
    try:
        import cupy as cp

        return cp  # type: cp
    except ImportError:
        import mlx.core as mx

        return mx  # type: mx
    finally:
        import numpy as np

        return np  # type: np


xp = get_numpy_api()
