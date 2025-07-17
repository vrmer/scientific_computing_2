import numpy as np
import matplotlib.pyplot as plt


def plot_mandelbrot_set(mandelbrot_array: np.ndarray, outpath: str = None):
    """
    # TODO: docstring + save figure
    # TODO: Also add customisation e.g. scales
    """
    plt.imshow(mandelbrot_array, cmap=plt.cm.hot)
    
    if outpath:
        plt.savefig(outpath)
