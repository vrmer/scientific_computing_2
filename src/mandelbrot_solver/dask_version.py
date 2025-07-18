import numpy as np
from dask import array as da
from .parameters import Parameters
from dask.distributed import Client
from .numba_version import compute_mandelbrot_set_vectorized_wrapper


def compute_mandelbrot_set_distributed(C: np.ndarray, params: Parameters) -> np.ndarray:
    """
    Computes the Mandelbrot set for a 2D grid of complex numbers using Dask for distributed execution.

    This function partitions the input array `C` into chunks and distributes the computation
    across multiple workers using Dask's parallel processing. Each chunk is processed using
    a Numba-accelerated vectorized Mandelbrot implementation.

    Parameters:
        C (np.ndarray): A 2D NumPy array of complex numbers representing the grid of points
                        in the complex plane.
        params: An object with the following attributes:
            - I (int): Maximum number of iterations per point.
            - T (float): Escape threshold (usually 2.0).
            - n_workers (int): Number of parallel Dask workers to use.
            - threads_per_worker (int): Number of threads per worker.

    Returns:
        np.ndarray: A 2D array of normalized escape values with the same shape as `C`.
                    Each value is in the range [0.0, 1.0], representing the rate of escape.
    """
    # define client for computations
    client = Client(threads_per_worker=params.threads_per_worker, n_workers=params.n_workers)
    # define dask array chunks
    chunksize = C.shape[0] // params.n_workers
    dask_C = da.from_array(C, chunks=chunksize)
    # carry out computation
    output_array = da.map_blocks(compute_mandelbrot_set_vectorized_wrapper, dask_C, params, dtype=C.dtype)
    return output_array.compute(scheduler="processes")
