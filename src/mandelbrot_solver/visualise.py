import os
import numpy as np
import matplotlib.pyplot as plt
from .parameters import Parameters


def plot_mandelbrot_set(mandelbrot_array: np.ndarray, params: Parameters) -> None:
    r"""
    Given an array of Mandelbrot sets, it visualises the $\mathcal{M}(c)$ for all points in the array.
    
    Inputs:
        mandelbrot_array: input array of computed Mandelbrot escape values (numpy array)
        params: Parameters dataclass with value limits, output directory, and backend (naive, numba, or dask)
        
    Returns:
        None
    """
    plt.imshow(mandelbrot_array, extent=(*params.real_val_lims, *params.imag_val_lims), origin="lower", cmap=plt.cm.hot)
    
    # customise labels
    plt.xlabel(r"$\mathfrak{R[c]}$")
    plt.ylabel(r"$\mathfrak{I[c]}$")
    
    # customise ticks
    plt.xticks(np.linspace(*params.real_val_lims, num=7))
    plt.yticks(np.linspace(*params.imag_val_lims, num=7))
    
    if params.figure_dir:
        os.makedirs(params.figure_dir, exist_ok=True)
        output_path = os.path.join(params.figure_dir, f"{params.backend}.pdf")
        plt.savefig(output_path)
