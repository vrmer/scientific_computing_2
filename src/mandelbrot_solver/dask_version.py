from functools import partial
from dask import array as da
from dask.distributed import Client
from .numba_version import compute_mandelbrot_set_vectorized_wrapper


@profile
def compute_mandelbrot_set_distributed(C, params):
    """
    # TODO: docstring
    """
    # define client for computations
    client = Client(threads_per_worker=params.threads_per_worker, n_workers=params.n_workers)
    # define dask array chunks
    chunksize = C.shape[0] // params.n_workers
    dask_C = da.from_array(C, chunks=chunksize)
    # carry out computation
    output_array = da.map_blocks(compute_mandelbrot_set_vectorized_wrapper, dask_C, params, dtype=C.dtype)
    return output_array.compute(scheduler="processes")
