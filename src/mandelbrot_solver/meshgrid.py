import numpy as np
from .parameters import Parameters


def init_meshgrid(params: Parameters) -> np.ndarray:
    r"""
    Initialise the $c$-mesh that the Mandelbrot set is calculated for.
    
    Relevant parameters from the Parameters dataclass:
        p_re: dimensionality of the real values (int)
        p_im: dimensionality of the imaginary values (int)
        real_val_lims: minimum and maximum values in $\mathfrak{R[c]}$ (list or tuple of floats)
        imag_val_lims: minimum and maximum values in $\mathfrak{I[c]}$ (list or tuple of floats)
        
    Returns:
        a $c$-mesh C limiting the target complex plane (numpy array)
    """
    # initialise real and imaginary values
    real_vals = np.linspace(*params.real_val_lims, params.p_re)
    imag_vals = np.linspace(*params.imag_val_lims, params.p_im)
    
    # mesh grid (inspired by ChatGPT)
    Re, Im = np.meshgrid(real_vals, imag_vals)
    C = Re + 1j * Im
    
    return C
