from typing import Tuple
from dataclasses import dataclass


@dataclass
class Parameters:
    """
    # TODO: docstring
    """
    p_re: int
    p_im: int
    I: int
    backend: str  # naive, numba, or dask
    T: int = 2
    real_val_lims: Tuple[int] = (-2.0, 1.0)
    imag_val_lims: Tuple[int] = (1.5, -1.5)
    
    # relevant for multiprocessing with dask
    n_workers: int = 4
    threads_per_worker: int = 2
