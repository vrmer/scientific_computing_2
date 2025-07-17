import numpy as np
from typing import Tuple


def init_meshgrid(params):
    """  # TODO: add docstring
    Initialise the meshgrid used to calculate the values of the Mandelbrot set.
    """
    # initialise real and imaginary values
    real_vals = np.linspace(*params.real_val_lims, params.p_re)
    imag_vals = np.linspace(*params.imag_val_lims, params.p_im)
    
    # mesh grid (inspired by ChatGPT)
    Re, Im = np.meshgrid(real_vals, imag_vals)
    C = Re + 1j * Im
    
    return C
