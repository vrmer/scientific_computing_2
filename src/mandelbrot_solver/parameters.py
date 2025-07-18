from typing import Tuple
from dataclasses import dataclass


@dataclass
class Parameters:
    r"""
    Dataclass handling passing arguments to various functions.
    
    Parameters:
        figure_dir: directory to save Mandelbrot plots to (str)
        output_dir: directory to save the set of $\mathcal{M}(c)$ escape values for the input $c$-mesh
        p_re: dimensionality of the real values of the $c$-mesh (int)
        p_im: dimensionality of the imaginary values of the $c$-mesh (int)
        I: number of iterations to run to calculate the escape values $\mathcal{M}(c)$ (int)
        T: threshold for calculating whether a value $c$ in the $c$-mesh escapes
        real_val_lims: minimum and maximum values in $\mathfrak{I[c]}$ (list or tuple of floats)
        n_workers: number of workers used to calculate the Mandelbrot set, only used with dask (int)
        threads_per_worker: number of threads per worker used to calculate the Mandelbrot set, only used with dask (int)
    """
    # filepaths
    figure_dir: str
    output_dir: str
    
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
