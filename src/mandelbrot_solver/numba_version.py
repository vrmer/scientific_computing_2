import numpy as np
from numba import vectorize
from .parameters import Parameters


@vectorize(
    ['float64(complex128, int64, float64)'],
    target="parallel"
)
def compute_mandelbrot_set_vectorized(c: complex, I: int, T: int) -> float:
    r"""
    Vectorized computation of Mandelbrot set membership using Numba's parallel ufunc.

    For each complex number $c$, this function applies the Mandelbrot recurrence
    $z_{i+1} = z_i^2 + c$ starting from $z = 0$, and returns a normalized
    escape value indicating how quickly the point escapes a given threshold.

    Optimized for performance using:
      - Squared threshold comparison to avoid square root computation.
      - Parallel execution on supported hardware via Numba.

    Parameters:
        c (complex128): Complex point to evaluate.
        I (int64): Maximum number of iterations.
        T (float64): Escape threshold (typically 2.0).

    Returns:
        float64: Normalized escape value between 0.0 and 1.0.
                 A value of 1.0 means the point did not escape.
    """
    z = 0j
    threshold = T * T  # speeding up processing not to use the square root
    for i in range(I + 1):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > threshold:
            return i / I
    return 1.0


def compute_mandelbrot_set_vectorized_wrapper(c: complex, params: Parameters) -> np.ndarray:
    """
    Wrapper function for the vectorized Mandelbrot computation that
    extracts iteration parameters from a parameter object.
    
    Used for both the Numba and the Dask implementation.

    Parameters:
        c (np.ndarray): Array of complex points to evaluate.
        params: An object with attributes:
            - I (int): Maximum number of iterations.
            - T (float): Escape threshold.

    Returns:
        np.ndarray: Array of normalized escape values for each complex input.
    """
    return compute_mandelbrot_set_vectorized(c, params.I, params.T)
    
