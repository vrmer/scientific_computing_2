from numba import njit, vectorize


@njit
def progress_jit(i: int, I: int):
    r"""
    Helps evaluate how fast a number passed the threshold.
    
    Numba version with JIT compilation.
    
    $\mathcal{M}(c) = \dfrac{\imath(c)}{I}$
    """
    return i/I

@vectorize(
    ['float64(complex128, int64, float64)'],
    target="parallel"
)
def compute_mandelbrot_set_vectorized(c, I, T):
    """
    Numba version with ufunc.
    
    Code written with a help from ChatGPT.
    # TODO: docstring
    """
    z = 0j
    threshold = T * T  # speeding up processing not to use the square root
    for i in range(I + 1):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > threshold:
            return i / I
    return 1.0


def compute_mandelbrot_set_vectorized_wrapper(c, params):
    return compute_mandelbrot_set_vectorized(c, params.I, params.T)
    
