from numba import vectorize
from functools import partial
from dask import array as da
from dask.distributed import Client, LocalCluster
from src import init_meshgrid, mandelbrot_wrapper_batch, compute_mandelbrot_gu
from dask.diagnostics import ProgressBar


def main():

    ProgressBar().register()

    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)

    C = init_meshgrid(1000, 1000, (-2.0, 1.0), (1.5, -1.5))
    dask_C = da.from_array(C, chunks=(250, 250))
    
    result = da.map_blocks(mandelbrot_wrapper_batch, dask_C, dtype=float)
    final = result.compute()
    
    print(final.shape)
    
    
if __name__ == "__main__":
    main()

