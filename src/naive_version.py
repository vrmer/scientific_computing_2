import numpy as np
from .utils import timeit


def progress(i: int, I: int):
    r""" # TODO: fill in docstring
    Helps evaluate how fast a number passed the threshold.
    
    $\mathcal{M}(c) = \dfrac{\imath(c)}{I}$
    """
    return i/I


def mandelbrot_eq(z: int|float, c: complex):
    r"""
    Quadratic complex mapping.    
    
    $z_{i+1} = z_i^2 + c$
    """
    return z**2 + c


def iteration(c: complex, I: int, T: int|float):
    r"""
    Carry out an iteration.
    
    Return $\mathcal{M}(c)$ once we pass the threshold T.
    """
    z = 0
    for i in range(1, I+1):
        z = mandelbrot_eq(z, c)
        if abs(z) > T:
            return progress(i, I)
    return progress(i, I)


@timeit
def compute_mandelbrot_set_naive(C, params):
    """
    # TODO: docstring
    """
    output_array = np.zeros(C.shape[0] * C.shape[1])
    for idx, c in enumerate(C.ravel()):
        output_array[idx] = iteration(c, params.I, params.T)
    return output_array.reshape(C.shape[0], C.shape[1])
